import sys
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from functools import partial

# import GaitBase
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image, pca_image
from ..modules import GaitAlign

# =========================================================================
# Helper Functions
# =========================================================================

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim//2),
                nn.BatchNorm1d(source_dim//2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim//2, target_dim),
                )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim),
                    )
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                u_x = self.up_sampling(d_x)
                return d_x, torch.mean(self.mse(u_x, x))
            else:
                return d_x, None
        else:
            if self.Up:
                u_x = self.up_sampling(d_x)
                return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
            else:
                return torch.sigmoid(self.bn_t(d_x)), None

class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

# =========================================================================
# Main Model
# =========================================================================

class BiggerGait__SAM3DBody__Projection_Mask_Part_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        # 1. 基础参数
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]

        # 🌟 新增：加载 MHR 官方索引
        idx_path = 'pretrained_LVMs/mhr_part_indices.pt'
        if os.path.exists(idx_path):
            self.part_indices = torch.load(idx_path, map_location='cpu')
            self.msg_mgr.log_info(f"[MHR] Official part indices loaded from {idx_path}")
        else:
            self.msg_mgr.log_warning(f"[MHR] Warning: {idx_path} not found! Segmentation will fail.")
            self.part_indices = None

        # FPN Configuration
        layer_cfg = model_cfg.get("layer_config", {})
        self.layers_per_group = layer_cfg.get("layers_per_group", 2)
        
        # Hook Mask
        if "hook_mask" in layer_cfg:
            self.hook_mask = layer_cfg["hook_mask"]
        else:
            self.hook_mask = [False]*16 + [True]*16 # Default Top-16

        self.total_hooked_layers = sum(self.hook_mask)
        
        # Dimensions Check
        if self.total_hooked_layers % self.layers_per_group != 0:
            raise ValueError(f"Hook layers ({self.total_hooked_layers}) must be divisible by group size ({self.layers_per_group})")
        
        self.total_groups = self.total_hooked_layers // self.layers_per_group
        
        if self.total_groups % self.num_FPN != 0:
            raise ValueError(f"Total groups ({self.total_groups}) must be divisible by FPN heads ({self.num_FPN})")

        self.layers_per_head = self.total_hooked_layers // self.num_FPN
        input_dim = self.f4_dim * self.layers_per_head
        
        self.chunk_size = model_cfg.get("chunk_size", 96)

        # GaitNet & PreConv
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)

        # FPN Heads
        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(input_dim, affine=False),
                nn.Conv2d(input_dim, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size*2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])
        
        # Mask Branch (Keep structure but not used for projection logic)
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        
        self.init_SAM_Backbone()

    def init_SAM_Backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)
        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Error: {e}")

        self.msg_mgr.log_info(f"[SAM3D] Loading SAM 3D Body (Encoder + Decoder)...")
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')
        
        self.SAM_Engine = estimator.model
        self.Backbone = self.SAM_Engine.backbone.encoder
        
        self.SAM_Engine.cpu() # 先放 CPU

        # ================= Hook Logic =================
        self.intermediate_features = {}
        self.hook_handles = []

        def get_activation(idx_in_list):
            def hook(model, input, output):
                if isinstance(output, (list, tuple)): output = output[0]
                if isinstance(output, (list, tuple)): output = output[0]
                self.intermediate_features[idx_in_list] = output
            return hook

        all_blocks = self.Backbone.blocks

        hook_count = 0
        for layer_idx, should_hook in enumerate(self.hook_mask):
            if should_hook:
                handle = all_blocks[layer_idx].register_forward_hook(get_activation(hook_count))
                self.hook_handles.append(handle)
                hook_count += 1
        
        self.msg_mgr.log_info(f"[SAM3D] Hooked {hook_count} layers.")
        
        # Freeze Everything
        self.SAM_Engine.eval()
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        # Re-initialize SAM (since modules() init might have messed it up)
        self.init_SAM_Backbone()
        
        self.SAM_Engine.eval()
        self.SAM_Engine.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Model Count: {:.5f}M'.format(n_parameters / 1e6))

    # --- 🌟 Helper: Prepare Dummy Batch for SAM Decoder ---
    def _prepare_dummy_batch(self, image_embeddings, target_h, target_w):
        B = image_embeddings.shape[0]
        device = image_embeddings.device
        
        estimated_focal_length = max(target_h, target_w) * 1.1
        cx, cy = target_w / 2.0, target_h / 2.0
        
        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3).clone()
        cam_int[:, 0, 0] = estimated_focal_length 
        cam_int[:, 1, 1] = estimated_focal_length 
        cam_int[:, 0, 2] = cx
        cam_int[:, 1, 2] = cy
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(target_h, device=device),
            torch.arange(target_w, device=device),
            indexing='ij'
        )
        ray_x = (x_grid - cx) / estimated_focal_length
        ray_y = (y_grid - cy) / estimated_focal_length
        ray_cond = torch.stack([ray_x, ray_y], dim=0).unsqueeze(0).expand(B, 2, target_h, target_w)

        bbox_scale = torch.tensor([max(target_h, target_w)], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 1)
        bbox_center = torch.tensor([cx, cy], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        img_size = torch.tensor([float(target_w), float(target_h)], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        affine_trans = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 2, 3)

        return {
            "img": torch.zeros(B, 1, 3, target_h, target_w, device=device),
            "ori_img_size": img_size, "img_size": img_size, "bbox_center": bbox_center,
            "bbox_scale": bbox_scale, "cam_int": cam_int, "affine_trans": affine_trans,
            "ray_cond": ray_cond, 
        }

    def project_vertices_to_mask_and_depth(self, vertices, cam_t, cam_int, H_feat, W_feat, target_H, target_W):
        B, N, _ = vertices.shape
        device = vertices.device
        
        # 1. 转换到相机坐标系
        v_cam = vertices + cam_t.unsqueeze(1) 
        x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
        z_safe = z.clamp(min=1e-3) 
        
        # 2. 投影到像素坐标
        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx 
        v = (y / z_safe) * fy + cy 
        
        # 3. 计算特征图索引
        u_feat = (u / target_W * W_feat).long().clamp(0, W_feat - 1)
        v_feat = (v / target_H * H_feat).long().clamp(0, H_feat - 1)
        flat_indices = v_feat * W_feat + u_feat # [B, N]
        
        # 4. Z-Buffer 核心逻辑
        # 初始化深度图为极大值 (代表无穷远)
        depth_map = torch.full((B, H_feat * W_feat), 1e6, device=device)
        
        # 使用 scatter_reduce 获取每个像素点的最小深度 (reduce='amin')
        # 这保证了如果多个点落入同一像素，只记录离相机最近的那个深度
        depth_map.scatter_reduce_(1, flat_indices, z, reduce='amin', include_self=False)
        
        depth_map = depth_map.view(B, 1, H_feat, W_feat)
        # 有效 Mask 是深度小于初始值的地方
        mask = (depth_map < 1e5).float()
        
        return mask, depth_map

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        CHUNK_SIZE = self.chunk_size # e.g. 4
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_outs = []
        
        # 图像目标尺寸 (512, 256)
        target_h, target_w = self.image_size * 2, self.image_size 
        # 特征图尺寸 (32, 16) -> Mask 在这里生成
        h_feat, w_feat = target_h // 16, target_w // 16

        # 定义部位顺序 (必须与 SemanticPartPooling 中的逻辑一致)
        ordered_parts = ["head", "torso", "l_arm", "r_arm", "l_leg", "r_leg"]

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            # B(n*s), C, H, W
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            curr_bs = rgb_img.shape[0]
            
            with torch.no_grad():
                # 1. Resize & Backbone
                outs = self.preprocess(rgb_img, target_h, target_w)
                self.intermediate_features = {}
                _ = self.Backbone(outs)
                
                # 2. SAM Decoder Preparation
                last_hook_idx = len(self.hook_handles) - 1
                sam_emb = self.intermediate_features[last_hook_idx] # [B, 517, 1280]
                
                # DINOv3 (B,N(517),C) -> SAM (B,C,H,W)
                target_tokens = h_feat * w_feat # 512
                
                # 剔除 CLS/Registers，只保留 Spatial Tokens
                if sam_emb.shape[1] > target_tokens:
                    sam_emb = sam_emb[:, -target_tokens:, :] # [B, 512, 1280]
                
                # [B, N, C] -> [B, C, N] -> [B, C, H, W]
                sam_emb = sam_emb.transpose(1, 2).reshape(curr_bs, -1, h_feat, w_feat) # [B, 1280, 32, 16]
                
                # Prepare Inputs
                dummy_batch = self._prepare_dummy_batch(sam_emb, target_h, target_w)
                self.SAM_Engine._batch_size = curr_bs
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(curr_bs, device=rgb.device)
                self.SAM_Engine.hand_batch_idx = []
                cond_info = torch.zeros(curr_bs, 3, device=rgb.device); cond_info[:, 2] = 1.1
                dummy_kp = torch.zeros(curr_bs, 1, 3, device=rgb.device); dummy_kp[..., -1] = -2

                # 3. Run Decoder
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                     _, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_emb, 
                        keypoints=dummy_kp, 
                        condition_info=cond_info, 
                        batch=dummy_batch
                    )
                
                # 4. 生成 Mask (32x16)
                pred_verts = pose_outs[-1]['pred_vertices'] 
                pred_cam_t = pose_outs[-1]['pred_cam_t']
                cam_int = dummy_batch['cam_int']            
                
                # --- 遮挡去重逻辑 (Z-Buffer) ---
                part_depths = {}
                # 第一遍：收集每个部位的原始深度图
                if self.part_indices is not None:
                    for name, idxs in self.part_indices.items():
                        _, p_depth = self.project_vertices_to_mask_and_depth(
                            pred_verts[:, idxs, :], pred_cam_t, cam_int, 
                            h_feat, w_feat, target_h, target_w
                        )
                        part_depths[name] = p_depth

                    # 计算全局最小深度 [B, 1, 32, 16]
                    if len(part_depths) > 0:
                        all_depth_tensors = torch.cat(list(part_depths.values()), dim=1)
                        global_min_depth, _ = torch.min(all_depth_tensors, dim=1, keepdim=True)
                    else:
                        global_min_depth = torch.zeros((curr_bs, 1, h_feat, w_feat), device=rgb.device)

                    # 第二遍循环：生成去重后的 Mask
                    final_disjoint_masks = {}
                    part_summaries = {} # 用于 Visual Summary
                    part_colors = {
                        "head": [1.0, 0.0, 0.0], "torso": [0.0, 1.0, 1.0], 
                        "l_arm": [0.0, 1.0, 0.0], "r_arm": [1.0, 1.0, 0.0],
                        "l_leg": [0.0, 0.0, 1.0], "r_leg": [1.0, 0.0, 1.0]
                    }

                    for name in ordered_parts:
                        if name in part_depths:
                            p_depth = part_depths[name]
                            # 1. 深度竞争，获得基础硬掩码
                            is_closest = (p_depth == global_min_depth) & (p_depth < 1e5)
                            mask = is_closest.float()

                            # 2. 执行掩码膨胀 (Dilation)
                            # kernel_size=3 可以让每个点向四周扩充 1 像素
                            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                            
                            final_disjoint_masks[name] = mask # [B, 1, H, W]
                            
                            # 生成单部位叠加图 (仅限前5个样本)
                            if name in part_colors and curr_bs > 0:
                                m_high = F.interpolate(is_closest.float(), (target_h, target_w), mode='nearest')
                                c_vec = torch.tensor(part_colors[name], device=rgb.device).view(1, 3, 1, 1)
                                part_overlay = outs * 0.2 + (m_high * c_vec) * 0.8
                                part_summaries[f'image/part_{name}'] = part_overlay[:3].float()
                        else:
                            # 兜底：如果某个 part 没生成，给全 0
                            final_disjoint_masks[name] = torch.zeros((curr_bs, 1, h_feat, w_feat), device=rgb.device)

                    # 3. 堆叠 6 个部位 Mask [B, 6, H, W]
                    part_masks = torch.cat([final_disjoint_masks[k] for k in ordered_parts], dim=1)

                    # 4. 生成总 Mask 用于 FPN 前的背景滤除
                    generated_mask = torch.clamp(torch.sum(part_masks, dim=1, keepdim=True), 0, 1) # [B, 1, H, W]
                else:
                    raise RuntimeError("Part indices for MHR not loaded; cannot generate part masks.")

                # 5. 收集特征用于 FPN (Early Masking)
                mask_flat = generated_mask.view(curr_bs, -1, 1) # [B, 512, 1]
                
                features_to_use = []
                for i in range(len(self.hook_handles)):
                    feat = self.intermediate_features[i] 
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :] 
                    feat = feat * mask_flat # 应用总 Mask 去背景
                    features_to_use.append(feat)

            # =======================================================
            # 6. FPN Processing (Masked Features)
            # =======================================================
            processed_feat_list = []
            step = len(features_to_use) // self.num_FPN
            
            for i in range(self.num_FPN):
                sub_feats = features_to_use[i*step : (i+1)*step]
                
                # A. 拼接特征 [B, 512, C*step(4)]
                sub_app = torch.concat(sub_feats, dim=-1)
                sub_app = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * len(sub_feats), elementwise_affine=False)(sub_app)
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                
                # D. FPN Head (HumanSpace_Conv)
                # 这一步包含 Conv + Upsample (ResizeToHW)
                reduced_feat = self.HumanSpace_Conv[i](sub_app) # [B, 64, 64, 32]
                processed_feat_list.append(reduced_feat)
                
                del sub_app, sub_feats

            # 7. 拼接 FPN 输出
            human_feat = torch.concat(processed_feat_list, dim=1) # [B, Total_C, 64, 32]

            # Early Masking (ResNet 前分 Part)

            part_masks_resized = F.interpolate(part_masks, size=(self.sils_size*2, self.sils_size), mode='nearest')
            # Masking & Flatten Batch
            # Feature: [B, 1, C, H, W]
            # Mask:    [B, 6, 1, H, W]
            # Result:  [B, 6, C, H, W]
            masked_feat = human_feat.unsqueeze(1) * part_masks_resized.unsqueeze(2)

            # C. Reshape for GaitNet [n, c, s, h, w]
            # 将 (n*s) 解开，并将 Part 维度 p 融合进 batch 维度 n
            masked_feat = rearrange(masked_feat, '(n s) p c h w -> (n p) c s h w', n=n, s=s).contiguous()
            
            # # 9. GaitNet Part 1 (ResNet)
            # # Input:  [(n*6), C, s, H, W]
            # # Output: [(n*6), C_out, s, H', W']
            # outs = self.Gait_Net.test_1(masked_feat)

            # # ===============================================
            # # 🌟 核心修改：在进入 test_2 之前，消除 Part 维度 (Max Pooling)
            # # 输出: [(n p), c, s, h, w] -> [n, p, c, s, h, w]
            # outs = rearrange(outs, '(n p) c s h w -> n p c s h w', p=6)
            # # Max Pooling 聚合所有部位 -> [n, c, s, h, w]
            # outs = outs.max(dim=1)[0]
            # # ===============================================

            # all_outs.append(outs)

            # 执行 Chunk 压缩
            static_chunk_map = masked_feat.max(dim=2, keepdim=True)[0]
            all_outs.append(static_chunk_map) # 存入的是 [B, C, 1, H, W]

        # # GaitNet Part 2 (时序聚合)
        # embed_list, log_list = self.Gait_Net.test_2(
        #     torch.cat(all_outs, dim=2), # [(n*6), c, s_chunk, h, w]
        #     seqL
        # )

        # 1. 拼接所有 Part 的时序特征 -> [(n*6), c, s_total, h, w]
        full_masked_feat = torch.cat(all_outs, dim=2) 
        
        # 2. 执行时序 Max Pooling -> 得到每个部位的 Gait Entropy Map
        # 得到: [(n*6), c, 1, h, w]
        static_part_maps = full_masked_feat.max(dim=2, keepdim=True)[0]
        
        # 3. 维度重组：把 Part (6) 放到 Time 轴，Batch (n) 回归 Batch 轴
        # 结果: [n, c, 6, h, w]
        pseudo_temporal_feat = rearrange(static_part_maps, '(n p) c 1 h w -> n c p h w', p=6).contiguous()
        
        # 4. GaitNet Part 1 (ResNet)
        # 卷积核现在在 6 个部位之间滑动。输出: [n, c_out, 6, h', w']
        outs = self.Gait_Net.test_1(pseudo_temporal_feat)

        # 6. 构造适配 OpenGait 格式的 seqL (必须是包含 Tensor 的 List)
        if seqL is not None:
            new_seqL = [torch.full((n,), 6, dtype=torch.long, device=rgb.device)] # TODO may have bug
        else:
            new_seqL = None
        
        # 7. GaitNet Part 2 (逻辑完全不改)
        # test_2 内部：TP 会把 (n*6) 重新切分为 n 个样本并取 Max，cat 之后自动恢复 Batch 维度 n
        # 输出 embed_list 每个元素为: [n, c_sub, 16]
        embed_list, log_list = self.Gait_Net.test_2(outs, new_seqL)

        # 5. 拼接 FPN 结果
        embed = torch.cat(embed_list, dim=-1)
        logits = torch.cat(log_list, dim=-1)
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:3].float(),
                    **part_summaries,
                    'image/generated_3d_mask_lowres': generated_mask.view(n*s, 1, h_feat, w_feat)[:3].float(),
                },
                'inference_feat': {
                    'embeddings': embed,
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': embed,
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        return retval