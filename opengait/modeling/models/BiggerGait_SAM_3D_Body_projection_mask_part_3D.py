import sys
import os
import roma
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from functools import partial
import numpy as np
import copy

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

import torch.nn.functional as F

class GeometryOptimalTransport(nn.Module):
    def __init__(self, temperature=0.01, dist_thresh=0.2, num_iters=3):
        super().__init__()
        self.epsilon = temperature
        self.dist_thresh = dist_thresh
        self.num_iters = num_iters # 迭代次数少一点(3次)，即为"软"Sinkhorn

    def forward(self, source_feats, source_locs, target_locs, source_valid_mask=None, target_valid_mask=None):
        """
        Args:
            source_feats: [B, N, C]
            source_locs:  [B, N, 2]
            target_locs:  [B, M, 2]
            source_valid_mask: [B, N]
            target_valid_mask: [B, M]
        """
        B, N, C = source_feats.shape
        M = target_locs.shape[1]

        with torch.no_grad():
        
            # 1. 计算代价矩阵 Cost (同老方法)
            diff = target_locs.unsqueeze(2) - source_locs.unsqueeze(1)
            dist_sq = torch.sum(diff ** 2, dim=-1)

            # 2. 构建 Log-Kernel (同老方法，但在 Log 域)
            # Log_K_ij = -C_ij / epsilon
            log_K = -dist_sq / (self.epsilon + 1e-8)

            # 3. 处理 Mask (同老方法，逻辑一致)
            valid_connection = dist_sq < (self.dist_thresh ** 2)
            del diff, dist_sq # 释放内存

            if source_valid_mask is not None:
                valid_connection = valid_connection & source_valid_mask.unsqueeze(1)
            if target_valid_mask is not None:
                valid_connection = valid_connection & target_valid_mask.unsqueeze(2)
            
            # 填充 -1e9 (Log 域的 0)
            log_K = log_K.masked_fill(~valid_connection, -1e9)

            # ==========================================================
            # 4. Sinkhorn 迭代 (Log-Domain)
            # 这里的改进：我们只迭代 3 次，这是一种"部分 OT"。
            # 它比 Softmax 更锐利，但又不像完全收敛的 OT 那样死板（允许一定的质量不平衡）。
            # ==========================================================
            
            # 初始化势能
            v = torch.zeros(B, 1, N, device=source_feats.device) # Source 势能
            u = torch.zeros(B, M, 1, device=source_feats.device) # Target 势能

            for _ in range(self.num_iters):
                # 步骤 A: Target 归一化 (类似 Softmax 的行归一化)
                # u = -logsumexp(log_K + v)
                # 这一步保证了每个 Target 像素能"抢"到足够的特征
                u = -torch.logsumexp(log_K + v, dim=2, keepdim=True)
                
                # 步骤 B: Source 归一化 (列归一化)
                # v = -logsumexp(log_K + u)
                # 这一步抑制了被过度复用的 Source 像素
                v = -torch.logsumexp(log_K + u, dim=1, keepdim=True)
                
                # 【关键修正】：防止 v 在全是 Mask 的列变成 inf
                # 如果某列 Source 全是无效连接，logsumexp 结果是 -inf，v 变成 inf
                # 我们需要把这些无效列的 v 重置为 0，防止污染后续计算
                if source_valid_mask is not None:
                    v = v.masked_fill(~source_valid_mask.unsqueeze(1), 0.0)

            # 5. 计算最终 Attention Map
            # P = exp(log_K + u + v)
            attn = torch.exp(log_K + u + v)
            del log_K, u, v # 释放内存
            
            # 再次硬过滤 (双重保险，同老方法)
            # has_source = valid_connection.any(dim=-1, keepdim=True)
        
        # ==========================================================
        # 6. 特征搬运 (同老方法)
        # ==========================================================
        target_feats = torch.bmm(attn, source_feats)

        # 7. 最终清理 (同老方法)
        if target_valid_mask is not None:
            target_feats = target_feats * target_valid_mask.unsqueeze(-1).float()
            
        # 检查是否有没有来源的目标点
        # 注意：在 Sinkhorn 中，u 会自动补偿，但如果真的没有任何连接，还是需要置 0
        has_source = valid_connection.any(dim=-1, keepdim=True)
        target_feats = target_feats * has_source.float()

        return target_feats

# =========================================================================
# Main Model
# =========================================================================

class BiggerGait__SAM3DBody__Projection_Mask_Part_3D_Gaitbase_Share(BaseModel):
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

        self.branch_configs = model_cfg["branch_configs"] # 获取新的异构配置
        self.num_branches = len(self.branch_configs)
        
        # 🌟 修改 1: 为每个分支实例化独立的异构 Gait_Net
        self.Gait_Nets = nn.ModuleList()
        for b_cfg in self.branch_configs:
            # 深度拷贝全局配置，并针对当前分支修改 parts 数量
            sub_cfg = copy.deepcopy(model_cfg)
            n_parts = b_cfg['parts']
            
            # 动态调整影响 FC 和 HPP 的关键配置
            sub_cfg['SeparateFCs']['parts_num'] = n_parts
            sub_cfg['SeparateBNNecks']['parts_num'] = n_parts
            sub_cfg['bin_num'] = [n_parts] # 强制 HPP 按照当前分支指定的数量分段

            sub_cfg['vertical_pooling'] = b_cfg.get('vertical_pooling', False)
            
            self.Gait_Nets.append(Baseline_ShareTime_2B(sub_cfg))

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

        ot_temp = model_cfg.get("ot_temperature", 0.01)
        ot_dist = model_cfg.get("ot_dist_thresh", 0.2)
        ot_iters = model_cfg.get("ot_iters", 8)

        # 🌟 1. 兼容元组/列表解析: [Yaw, Pitch]
        raw_angles = model_cfg.get("target_angles", [[90.0, 0.0]])
        self.target_angles = []
        if isinstance(raw_angles, (float, int)):
            self.target_angles.append((float(raw_angles), 0.0))
        else:
            for a in raw_angles:
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    self.target_angles.append((float(a[0]), float(a[1])))
                else:
                    self.target_angles.append((float(a), 0.0))
        
        self.ot_solver = GeometryOptimalTransport(
            temperature=ot_temp, 
            dist_thresh=ot_dist, 
            num_iters=ot_iters
        )

        self.enable_visual = model_cfg.get("enable_visual", False)
        self.VIS_SAMPLE = 5 # 可视化时每个分支最多展示的样本数量
        self.msg_mgr.log_info(f"Visualization status: {self.enable_visual}")

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

    def get_source_vertex_index_map(self, vertices, cam_t, cam_int, H_feat, W_feat, target_H, target_W):
        """
        渲染 Index Map 和 Depth Map。
        Returns:
            index_map: [B, H, W] 像素对应的顶点索引
            depth_map: [B, 1, H, W] 像素对应的最小深度值
        """
        B, N_verts, _ = vertices.shape
        device = vertices.device
        
        # 1. 投影到原图相机平面
        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)
        
        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        
        u = (x / z_safe) * fx + cx 
        v = (y / z_safe) * fy + cy 
        
        # 2. 量化坐标
        u_feat = (u / target_W * W_feat).long().clamp(0, W_feat - 1)
        v_feat = (v / target_H * H_feat).long().clamp(0, H_feat - 1)
        flat_pixel_indices = v_feat * W_feat + u_feat # [B, N_verts]
        
        # 3. Z-Buffer：找出每个像素最近的深度
        depth_map_flat = torch.full((B, H_feat * W_feat), 1e6, device=device)
        depth_map_flat.scatter_reduce_(1, flat_pixel_indices, z, reduce='amin', include_self=False)
        
        # 4. 生成 Index Map
        min_depth_per_vertex = torch.gather(depth_map_flat, 1, flat_pixel_indices)
        is_visible = (z < (min_depth_per_vertex + 1e-4))
        
        index_map_flat = torch.full((B, H_feat * W_feat), -1, dtype=torch.long, device=device)
        vertex_indices = torch.arange(N_verts, device=device).unsqueeze(0).expand(B, -1)
        
        # 修正之前提到的 view 连续性问题，改用 reshape
        mask_flat = is_visible.reshape(-1) 
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * (H_feat * W_feat)
        global_pixel_indices = (flat_pixel_indices + batch_offsets).reshape(-1) 
        
        valid_pixel_indices = global_pixel_indices[mask_flat]
        valid_vertex_indices = vertex_indices.reshape(-1)[mask_flat]
        
        index_map_global = index_map_flat.reshape(-1)
        index_map_global[valid_pixel_indices] = valid_vertex_indices
        
        # 返回两个值：Index Map 和 Depth Map
        # 注意：Depth Map 恢复成 [B, 1, H, W] 以兼容你后续的 mask = (depth_map < 1e5).float()
        return index_map_global.reshape(B, H_feat, W_feat), depth_map_flat.reshape(B, 1, H_feat, W_feat)
    
    def get_pca_vis_tensor(self, feat_tensor, mask_tensor, max_samples=5):
        """
        批量提取特征的 PCA 可视化张量，用于 TensorBoard / Wandb
        """
        import numpy as np
        import torch
        from einops import rearrange

        B, C, H, W = feat_tensor.shape
        K = min(B, max_samples) # 只取前几个样本以节省时间
        
        feat = feat_tensor[:K].detach().cpu()
        mask = mask_tensor[:K].detach().cpu()
        if mask.dim() == 4:
            mask = mask.squeeze(1) # [K, H, W]
            
        flat_feat = rearrange(feat, 'k c h w -> k (h w) c').numpy()
        flat_mask = mask.view(K, -1).numpy()
        
        vis_data = {'embeddings': flat_feat, 'h': H, 'w': W}
        
        pca_res = pca_image(
            data=vis_data, 
            mask=flat_mask, 
            root=None, 
            model_name=None, 
            dataset=None, 
            n_components=3, 
            is_return=True
        ) # [1, K, 3, H, W]
        
        # [K, 3, H, W]
        pca_vis = torch.from_numpy(pca_res[0]).float() / 255.0
        
        return pca_vis.to(feat_tensor.device)
    
    def warp_features_with_ot(self, human_feat, mask_src, 
                              pred_verts, pred_keypoints, pred_cam_t, global_rot, 
                              cam_int_src, cam_int_tgt, cam_t_tgt,
                              H_feat, W_feat, target_H, target_W):
        """
        利用最优传输将 human_feat 从原视角迁移到 90度侧视视角
        关键改进：显式生成 Target Mask，防止特征泄露到背景
        """
        B, C, H, W = human_feat.shape
        device = human_feat.device
        
        # =========================================================
        # 1. Source 端几何计算 (哪些 Source 像素有效？)
        # =========================================================
        # 获取 Source Index Map: [B, H, W] (值是 vertex_idx, -1 表示无效)
        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int_src, H_feat, W_feat, target_H, target_W
        )
        
        # Source 有效性掩码：既要在原图分割 Mask 内，又要能找到对应的 Vertex
        valid_src_mask = (mask_src.squeeze(1) > 0.5) & (src_idx_map >= 0) # [B, H, W]
        
        flat_human_feat = rearrange(human_feat, 'b c h w -> b (h w) c')
        flat_src_idx_map = src_idx_map.view(B, -1)
        flat_src_mask = valid_src_mask.view(B, -1)
        
        # Gather Source Vertices 3D Coords [B, HW, 3]
        # 注意：对于无效点 (-1)，我们用 0 号顶点暂代，反正后面 mask 会把它屏蔽掉
        flat_src_verts = torch.zeros((B, H*W, 3), device=device)
        safe_indices = flat_src_idx_map.clone()
        safe_indices[safe_indices < 0] = 0
        flat_src_verts = torch.gather(pred_verts, 1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # =========================================================
        # 2. Target 端几何计算 (哪些 Target 像素有效？)
        # =========================================================
        # 为了得到 Target Mask，我们需要将 Mesh 真正旋转并投影一次
        
        midhip = (pred_keypoints[:, 9] + pred_keypoints[:, 10]) / 2.0
        centered_verts = pred_verts - midhip.unsqueeze(1) # [B, N, 3]
        
        # 构建旋转矩阵 (Current -> Canonical Side)
        rot_fix = global_rot.clone(); rot_fix[..., [0,1,2]] *= -1
        R_canon = roma.euler_to_rotmat("XYZ", rot_fix)

        # 🌟 2. 复合旋转矩阵 (Yaw 偏转 + Pitch 俯仰)
        yaw, pitch = self.target_angle
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        
        # 绕 Y 轴偏转 (Yaw)
        R_y = torch.tensor([
            [ cy, 0., sy],
            [ 0., 1., 0.],
            [-sy, 0., cy]
        ], device=device, dtype=torch.float32)
        
        # 绕 X 轴俯仰 (Pitch)
        R_p = torch.tensor([
            [ 1., 0.,  0.],
            [ 0., cp, -sp],
            [ 0., sp,  cp]
        ], device=device, dtype=torch.float32)
        
        # 组合：先偏转，再俯仰 (注意矩阵乘法顺序)
        R_side = torch.matmul(R_p, R_y).view(1, 3, 3).expand(B, 3, 3)

        # 复合旋转: R_comp @ v (在 SMPL 坐标系下)
        R_comp = torch.matmul(R_canon.transpose(1,2), R_side.transpose(1,2))
        
        # 执行旋转 (注意坐标系翻转：OpenCV -> SMPL -> Rotate -> OpenCV)
        v_tmp = centered_verts.clone(); v_tmp[...,[1,2]] *= -1 
        v_rot_smpl = torch.bmm(v_tmp, R_comp)
        v_rot_cv = v_rot_smpl.clone(); v_rot_cv[...,[1,2]] *= -1 
        
        # 关键步骤：投影生成 Target Mask
        # 我们复用 get_source_vertex_index_map，只为了拿 depth_map
        _, tgt_depth_map = self.get_source_vertex_index_map(
            v_rot_cv, cam_t_tgt, cam_int_tgt, H_feat, W_feat, target_H, target_W
        )
        # 生成 Target 有效性掩码 [B, HW]
        valid_tgt_mask = (tgt_depth_map.view(B, -1) < 1e5) 
        
        # =========================================================
        # 3. 构建 OT 坐标系 (都投影到 Target 2D 平面)
        # =========================================================
        
        # A. 计算 Source Points 在 Target 图上的 "期望落点" (Projected Source Locs)
        # 我们对 flat_src_verts (Source像素对应的3D点) 应用同样的旋转和投影
        src_centered = flat_src_verts - midhip.unsqueeze(1)
        src_tmp = src_centered.clone(); src_tmp[...,[1,2]] *= -1
        src_rot_smpl = torch.bmm(src_tmp, R_comp)
        src_rot_cv = src_rot_smpl.clone(); src_rot_cv[...,[1,2]] *= -1
        
        # 投影到 Target 2D
        v_cam_tgt = src_rot_cv + cam_t_tgt.unsqueeze(1)
        x, y, z = v_cam_tgt.unbind(-1)
        z = z.clamp(min=1e-3) # 深度保护
        
        fx, fy = cam_int_tgt[:,0,0].unsqueeze(1), cam_int_tgt[:,1,1].unsqueeze(1)
        cx, cy = cam_int_tgt[:,0,2].unsqueeze(1), cam_int_tgt[:,1,2].unsqueeze(1)
        u_tgt = (x / z) * fx + cx
        v_tgt = (y / z) * fy + cy
        
        # 归一化 Source Locs [-1, 1]
        u_norm = 2.0 * (u_tgt / target_W) - 1.0
        v_norm = 2.0 * (v_tgt / target_H) - 1.0
        projected_source_locs = torch.stack([u_norm, v_norm], dim=-1) # [B, HW, 2]
        
        # B. 构建 Target Grid Locs (实际网格坐标)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_feat, device=device),
            torch.linspace(-1, 1, W_feat, device=device),
            indexing='ij'
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 2)
        
        # =========================================================
        # 4. 执行 OT (Attention)
        # =========================================================
        # 传入所有 Mask，确保只有 (有效Source -> 有效Target) 之间产生流量
        
        transported_feats = self.ot_solver(
            flat_human_feat, 
            projected_source_locs, 
            target_grid_locs, 
            source_valid_mask=flat_src_mask, # 屏蔽无效源
            target_valid_mask=valid_tgt_mask # 屏蔽背景目标 (关键！)
        )
        
        # 恢复形状
        warped_feat = rearrange(transported_feats, 'b (h w) c -> b c h w', h=H_feat)
        
        warped_color = None
        if self.enable_visual:
            # 🌟 新增：搬运颜色流 (仅限训练可视化)
            color_grid = self._generate_color_grid(B, H_feat, W_feat, device)
            flat_color = rearrange(color_grid, 'b c h w -> b (h w) c')
            
            # 这里的 transported_feats 是特征，我们需要再算一次颜色
            # 或者直接让 ot_solver 返回 attn 矩阵。
            # 简单做法：再次调用 solver 搬运颜色
            warped_color_flat = self.ot_solver(
                flat_color, 
                projected_source_locs, 
                target_grid_locs, 
                source_valid_mask=flat_src_mask,
                target_valid_mask=valid_tgt_mask
            )
            warped_color = rearrange(warped_color_flat, 'b (h w) c -> b c h w', h=H_feat)
        
        return warped_feat, valid_tgt_mask.view(B, 1, H_feat, W_feat), warped_color
    
    def generate_mhr_tpose(self, pose_out):
        """
        利用 MHRHead 生成对应的 A-Pose (手臂自然下垂) 网格和关键点。
        在送入 MHR 前，直接在 133维 参数空间中精准修改左右肩的 Z轴 旋转。
        """
        device = pose_out['pred_vertices'].device
        B = pose_out['pred_vertices'].shape[0]

        pred_shape = pose_out['shape'].float()
        pred_scale = pose_out['scale'].float()
        pred_face = pose_out['face'].float()

        zero_global_trans = torch.zeros((B, 3), device=device, dtype=torch.float32)
        zero_global_rot = torch.zeros_like(pose_out['global_rot'], dtype=torch.float32)
        zero_hand_pose = torch.zeros_like(pose_out['hand'], dtype=torch.float32)
        
        # 1. 初始化全 0 姿态 (此时为标准 T-Pose，手臂水平)
        a_pose_body = torch.zeros_like(pose_out['body_pose'], dtype=torch.float32)
        
        # =========================================================
        # 🌟 核心修改：设置 A-Pose 角度和经过验证的 Index
        # =========================================================
        angle_rad = math.radians(-20)
        a_pose_body[:, 25] = angle_rad   # 左肩 (画面右侧)
        a_pose_body[:, 35] = angle_rad  # 右肩 (画面左侧)
        # =========================================================

        with torch.no_grad(), torch.amp.autocast(enabled=False, device_type='cuda'):
            t_pose_outputs = self.SAM_Engine.head_pose.mhr_forward(
                global_trans=zero_global_trans,
                global_rot=zero_global_rot,
                body_pose_params=a_pose_body, # 传入修改好的 A-Pose 参数
                hand_pose_params=zero_hand_pose,
                scale_params=pred_scale,
                shape_params=pred_shape,
                expr_params=pred_face,
                return_keypoints=True 
            )

        a_pose_verts = t_pose_outputs[0]
        a_pose_keypoints = t_pose_outputs[1][:, :70] 

        # 还原到 MHR 外部的 OpenCV 视角的坐标系 (翻转 Y, Z)
        a_pose_verts[..., [1, 2]] *= -1
        a_pose_keypoints[..., [1, 2]] *= -1
        
        return a_pose_verts, a_pose_keypoints

    def warp_features_with_ot_tpose(self, human_feat, mask_src, 
                                    pred_verts, pred_cam_t, 
                                    t_pose_verts, t_pose_keypoints,
                                    cam_int_src, cam_int_tgt, cam_t_tgt,
                                    H_feat, W_feat, target_H, target_W):
        """
        专用于 T-Pose 的特征迁移：
        Source: 用原姿态 (pred_verts) 找可见像素。
        Target: 将对应像素强行映射到 T-Pose (t_pose_verts) 上，并旋转渲染。
        """
        import math
        B, C, H, W = human_feat.shape
        device = human_feat.device
        
        # =========================================================
        # 1. Source 端几何计算 (依然用原网格找可见性)
        # =========================================================
        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int_src, H_feat, W_feat, target_H, target_W
        )
        # =========================================================
        # 🛠️ 插入调试代码：导出 A-Pose 覆盖率点云
        # =========================================================
        if self.training and self.enable_visual:
            # 仅在第一批数据时导出一次，避免覆写过多
            if not hasattr(self, '_debug_ply_saved'):
                self._debug_dump_apose_ply(src_idx_map, t_pose_verts, save_path="debug_apose_chunk0.ply")
                self._debug_ply_saved = True # 标记已保存
        valid_src_mask = (mask_src.squeeze(1) > 0.5) & (src_idx_map >= 0) 
        
        flat_human_feat = rearrange(human_feat, 'b c h w -> b (h w) c')
        flat_src_idx_map = src_idx_map.view(B, -1)
        flat_src_mask = valid_src_mask.view(B, -1)
        
        # 🌟 核心：获取有效像素在 T-Pose 下的 3D 绝对规范化坐标
        safe_indices = flat_src_idx_map.clone()
        safe_indices[safe_indices < 0] = 0
        flat_src_verts_tpose = torch.gather(t_pose_verts, 1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))

        # =========================================================
        # 2. Target 端几何计算 (利用 T-Pose 生成笔挺的 Mask)
        # =========================================================
        midhip = (t_pose_keypoints[:, 9] + t_pose_keypoints[:, 10]) / 2.0
        centered_tpose = t_pose_verts - midhip.unsqueeze(1) 
        
        # 🌟 移除 self.target_angle 的旋转逻辑，直接保持 0 度正视
        v_tmp = centered_tpose.clone(); v_tmp[...,[1,2]] *= -1 
        v_rot_cv = v_tmp.clone(); v_rot_cv[...,[1,2]] *= -1 
        
        _, tgt_depth_map = self.get_source_vertex_index_map(
            v_rot_cv, cam_t_tgt, cam_int_tgt, H_feat, W_feat, target_H, target_W
        )
        valid_tgt_mask = (tgt_depth_map.view(B, -1) < 1e5)

        # =========================================================
        # 3. 构建 OT 坐标系 
        # =========================================================
        src_centered = flat_src_verts_tpose - midhip.unsqueeze(1)
        src_tmp = src_centered.clone(); src_tmp[...,[1,2]] *= -1
        # 🌟 同理，无需旋转，直接进入目标坐标系
        src_rot_cv = src_tmp.clone(); src_rot_cv[...,[1,2]] *= -1
        
        v_cam_tgt = src_rot_cv + cam_t_tgt.unsqueeze(1)
        x, y, z = v_cam_tgt.unbind(-1)
        z = z.clamp(min=1e-3) 
        
        fx, fy = cam_int_tgt[:,0,0].unsqueeze(1), cam_int_tgt[:,1,1].unsqueeze(1)
        cx, cy = cam_int_tgt[:,0,2].unsqueeze(1), cam_int_tgt[:,1,2].unsqueeze(1)
        u_tgt = (x / z) * fx + cx
        v_tgt = (y / z) * fy + cy
        
        u_norm = 2.0 * (u_tgt / target_W) - 1.0
        v_norm = 2.0 * (v_tgt / target_H) - 1.0
        projected_source_locs = torch.stack([u_norm, v_norm], dim=-1)

        # =========================================================
        # 4. 执行 OT (Attention)
        # =========================================================
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_feat, device=device),
            torch.linspace(-1, 1, W_feat, device=device),
            indexing='ij'
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 2)
        
        transported_feats = self.ot_solver(
            flat_human_feat, 
            projected_source_locs, 
            target_grid_locs, 
            source_valid_mask=flat_src_mask, 
            target_valid_mask=valid_tgt_mask 
        )

        warped_feat = rearrange(transported_feats, 'b (h w) c -> b c h w', h=H_feat)
        
        warped_color = None
        if self.enable_visual:
            # 🌟 新增：搬运颜色流 (仅限训练可视化)
            color_grid = self._generate_color_grid(B, H_feat, W_feat, device)
            flat_color = rearrange(color_grid, 'b c h w -> b (h w) c')
            
            # 这里的 transported_feats 是特征，我们需要再算一次颜色
            # 或者直接让 ot_solver 返回 attn 矩阵。
            # 简单做法：再次调用 solver 搬运颜色
            warped_color_flat = self.ot_solver(
                flat_color, 
                projected_source_locs, 
                target_grid_locs, 
                source_valid_mask=flat_src_mask,
                target_valid_mask=valid_tgt_mask
            )
            warped_color = rearrange(warped_color_flat, 'b (h w) c -> b c h w', h=H_feat)
        
        return warped_feat, valid_tgt_mask.view(B, 1, H_feat, W_feat), warped_color
    
    # =========================================================================
    # 🛠️ 临时代码区：点云投影渲染器 (带 3D 实心朝向箭头)
    # =========================================================================
    def _generate_solid_arrow_pcd(self, device, num_points=4000, y_offset=-1.0):
        """
        生成一个实心的 3D 箭头点云。
        y_offset: 箭头在 Y 轴的高度。-1.0 约在脚底地面，0.0 在骨盆中心。
        """
        # 定义箭头尺寸
        w_shaft = 0.3  # 箭杆宽度
        l_shaft = 0.3   # 箭杆长度
        w_head = 0.6    # 箭头宽度
        l_head = 0.3    # 箭头长度
        
        # 计算面积用于分配点数，保证密度均匀
        area_shaft = w_shaft * l_shaft
        area_head = 0.5 * w_head * l_head
        total_area = area_shaft + area_head
        
        n_shaft = int(num_points * (area_shaft / total_area))
        n_head = num_points - n_shaft
        
        # 1. 均匀采样箭杆 (矩形)
        # x 在 [-w/2, w/2], z 在 [0, l_shaft]
        x_shaft = (torch.rand(n_shaft, device=device) - 0.5) * w_shaft
        z_shaft = torch.rand(n_shaft, device=device) * l_shaft
        
        # 2. 均匀采样箭头 (三角形)
        # 使用随机仿射组合，并折叠超出的部分保证在三角形内
        r1 = torch.rand(n_head, device=device)
        r2 = torch.rand(n_head, device=device)
        mask = (r1 + r2) > 1.0
        r1[mask] = 1.0 - r1[mask]
        r2[mask] = 1.0 - r2[mask]
        
        # 三角形顶点: 尖端 (0, l_shaft+l_head), 左下 (-w_head/2, l_shaft), 右下 (w_head/2, l_shaft)
        x_head = r1 * (-w_head/2) + r2 * (w_head/2)
        z_head = (l_shaft + l_head) - (r1 + r2) * l_head
        
        # 3. 组合为 3D 坐标
        x = torch.cat([x_shaft, x_head])
        z = torch.cat([z_shaft, z_head])
        y = torch.full_like(x, y_offset) # 压扁在同一个高度平面上
        
        arrow_pcd = torch.stack([x, y, z], dim=-1) # [N, 3]
        return arrow_pcd.unsqueeze(0) # [1, N, 3]

    def _temp_render_pcd_grid(self, pred_verts, pred_keypoints, pred_cam_t, global_rot, 
                              t_pose_verts, t_pose_keypoints,
                              cam_int_src, branch_configs_list, # 🌟 传入各分支动态参数
                              max_samples=5):
        """支持异构画幅的彩色 3D 点云投影"""
        import math
        B = min(pred_verts.shape[0], max_samples)
        device = pred_verts.device
        
        v_src = pred_verts[:B]
        kp_src = pred_keypoints[:B]
        c_int_src = cam_int_src[:B]
        g_rot = global_rot[:B]

        c_t_src = pred_cam_t[:B].clone()
        c_t_src[:, 2] += 0.5
        
        N_verts = v_src.shape[1]
        colors = torch.ones((N_verts, 3), device=device) * 0.4
        part_colors = {
            "head": [1.0, 0.3, 0.3], "torso": [0.3, 0.8, 0.8], 
            "l_arm": [0.3, 1.0, 0.3], "r_arm": [1.0, 0.8, 0.0],
            "l_leg": [0.3, 0.3, 1.0], "r_leg": [1.0, 0.3, 1.0]
        }
        if getattr(self, 'part_indices', None) is not None:
            for name, idxs in self.part_indices.items():
                if name in part_colors:
                    colors[idxs] = torch.tensor(part_colors[name], device=device)
        colors = colors.unsqueeze(0).expand(B, -1, -1) 

        # 2. 生成实心箭头点云 (自身中心设为绝对原点 0,0,0)
        num_arrow_pts = 4000
        arrow_smpl = self._generate_solid_arrow_pcd(device, num_points=num_arrow_pts, y_offset=0.0)
        arrow_smpl = arrow_smpl.expand(B, -1, -1)
        arrow_colors = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3).expand(B, num_arrow_pts, 3)

        # 🌟 渲染函数接收动态高宽
        def render_verts_to_img(verts, cam_t, cam_int, tgt_h, tgt_w, arr_verts=None):
            if arr_verts is not None:
                v_mean = verts.mean(dim=1, keepdim=True) 
                v_foot_y, _ = verts[..., 1].max(dim=-1, keepdim=True)
                v_mean[..., 1] = v_foot_y
                arr_verts = arr_verts + v_mean 
                v_render = torch.cat([verts, arr_verts], dim=1)
                c_render = torch.cat([colors, arrow_colors], dim=1)
            else:
                v_render = verts
                c_render = colors

            v_cam = v_render + cam_t.unsqueeze(1)
            x, y, z = v_cam.unbind(-1)
            z_safe = z.clamp(min=1e-3)
            
            sort_idx = torch.argsort(z_safe, dim=1, descending=True)
            x, y, z_safe = torch.gather(x, 1, sort_idx), torch.gather(y, 1, sort_idx), torch.gather(z_safe, 1, sort_idx)
            c_sorted = torch.gather(c_render, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
            
            fx, fy = cam_int[:,0,0].unsqueeze(1), cam_int[:,1,1].unsqueeze(1)
            cx, cy = cam_int[:,0,2].unsqueeze(1), cam_int[:,1,2].unsqueeze(1)
            
            u = ((x / z_safe) * fx + cx).long().clamp(0, tgt_w - 1)
            v = ((y / z_safe) * fy + cy).long().clamp(0, tgt_h - 1)
            
            canvas = torch.ones((B, 3, tgt_h, tgt_w), device=device) * 0.95 
            for b_i in range(B):
                canvas[b_i, :, v[b_i], u[b_i]] = c_sorted[b_i].T
            return canvas

        rot_fix = g_rot.clone(); rot_fix[..., [0,1,2]] *= -1
        R_canon = roma.euler_to_rotmat("XYZ", rot_fix)

        row_images = []
        
        # 🌟 遍历各分支的独立配置渲染点云
        for b_cfg in branch_configs_list:
            tgt_h = b_cfg['tgt_h']
            tgt_w = b_cfg['tgt_w']
            c_int_tgt = b_cfg['cam_int'][:B]
            c_t_tgt = b_cfg['cam_t'][:B].clone()
            c_t_tgt[:, 2] += 0.5 

            if b_cfg.get('use_tpose', False) and t_pose_verts is not None:
                v_tp = t_pose_verts[:B]
                kp_tp = t_pose_keypoints[:B]
                midhip_tp = (kp_tp[:, 9] + kp_tp[:, 10]) / 2.0
                centered_tp = v_tp - midhip_tp.unsqueeze(1)
                v_tmp = centered_tp.clone(); v_tmp[...,[1,2]] *= -1 
                v_rot_cv = v_tmp.clone(); v_rot_cv[...,[1,2]] *= -1 
                arr_cv = arrow_smpl.clone(); arr_cv[..., [1,2]] *= -1
                
                img_t = render_verts_to_img(v_rot_cv, c_t_tgt, c_int_tgt, tgt_h, tgt_w, arr_cv)
                row_images.append(img_t)
            else:
                yaw, pitch = b_cfg['angle']
                midhip = (kp_src[:, 9] + kp_src[:, 10]) / 2.0
                centered_verts = v_src - midhip.unsqueeze(1)
                
                cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
                R_y = torch.tensor([[ cy, 0., sy], [ 0., 1., 0.], [-sy, 0., cy]], device=device, dtype=torch.float32)
                R_p = torch.tensor([[ 1., 0., 0.], [ 0., cp, -sp], [ 0., sp, cp]], device=device, dtype=torch.float32)
                R_side = torch.matmul(R_p, R_y).view(1, 3, 3).expand(B, 3, 3)
                R_comp = torch.matmul(R_canon.transpose(1,2), R_side.transpose(1,2))
                
                v_tmp = centered_verts.clone(); v_tmp[...,[1,2]] *= -1 
                v_rot_cv = torch.bmm(v_tmp, R_comp).clone(); v_rot_cv[...,[1,2]] *= -1 
                arr_side_smpl = torch.bmm(arrow_smpl, R_side.transpose(1, 2))
                arr_side_cv = arr_side_smpl.clone(); arr_side_cv[..., [1,2]] *= -1
                
                img_t = render_verts_to_img(v_rot_cv, c_t_tgt, c_int_tgt, tgt_h, tgt_w, arr_side_cv)
                row_images.append(img_t)

        # 统一缩放到最大高度以便拼接
        max_h = max([img.shape[-2] for img in row_images])
        aligned_images = []
        for img in row_images:
            if img.shape[-2] != max_h:
                aligned = F.interpolate(img, size=(max_h, img.shape[-1]), mode='bilinear')
            else:
                aligned = img
            aligned_images.append(torch.cat(torch.unbind(aligned, dim=0), dim=-1))
            
        grid = torch.cat(aligned_images, dim=-2).unsqueeze(0)
        return grid
    
    def _debug_dump_apose_ply(self, src_idx_map, t_pose_verts, save_path="debug_apose_fixed_scale.ply", fixed_max=20):
        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        N_verts = t_pose_verts.shape[1]
        device = src_idx_map.device

        # 1. 统计引用频率
        valid_indices = src_idx_map[src_idx_map >= 0].long()
        ref_counts = torch.bincount(valid_indices, minlength=N_verts).float()
        
        # 2. 坐标归一化与去中心化
        verts = t_pose_verts[0].clone()
        center = verts.mean(dim=0, keepdim=True) 
        verts -= center 

        # 3. 【核心修改】：绝对颜色映射逻辑
        # 我们定义：0 为灰色，(0, fixed_max] 映射到 Jet 颜色条，>fixed_max 也是最红
        colors = torch.ones((N_verts, 3), device=device) * 0.5 # 默认灰色 (R=0.5, G=0.5, B=0.5)
        
        has_ref_mask = ref_counts > 0
        if has_ref_mask.any():
            # 将计数限制在 [0, fixed_max] 之间，然后归一化到 [0, 1]
            # 这样 20 次及以上都会变成 1.0 (最红)
            counts_clamped = torch.clamp(ref_counts[has_ref_mask], 0, fixed_max)
            norm_counts = counts_clamped / float(fixed_max)

            # 输出统计信息
            self.msg_mgr.log_info(f"[Debug] A-Pose Vertex Reference Counts: min={ref_counts.min().item()}, max={ref_counts.max().item()}, mean={ref_counts.mean().item():.2f}, fixed_max={fixed_max}")
            # 输出所有计数的分布
            unique_counts, unique_freqs = torch.unique(ref_counts, return_counts=True)
            distribution_info = ", ".join([f"{int(c.item())}:{f.item()}" for c, f in zip(unique_counts, unique_freqs)])
            self.msg_mgr.log_info(f"[Debug] A-Pose Vertex Reference Count Distribution: {distribution_info}")
            
            # 使用 Jet 色卡：蓝色(低频) -> 绿色(中频) -> 红色(高频)
            cmap = plt.get_cmap('jet')
            rgba = cmap(norm_counts.cpu().numpy())
            colors[has_ref_mask] = torch.tensor(rgba[:, :3], device=device, dtype=torch.float32)

        verts_np = verts.cpu().numpy()
        colors_np = (colors.cpu().numpy() * 255).astype(np.uint8)

        # 4. 写入文件
        with open(save_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {N_verts}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(N_verts):
                x, y, z = verts_np[i]
                # OpenCV -> 标准 3D 坐标系转换
                f.write(f"{x:.4f} {-y:.4f} {-z:.4f} {colors_np[i,0]} {colors_np[i,1]} {colors_np[i,2]}\n")
        
        self.msg_mgr.log_info(f"[Debug] A-Pose Exported. Fixed Max: {fixed_max}. "
                              f"Full Coverage: {(ref_counts >= fixed_max).sum()}/{N_verts}")

    def _generate_color_grid(self, B, H, W, device):
        """生成一个 [B, 3, H, W] 的 2D 颜色梯度网格"""
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        # R: X梯度, G: Y梯度, B: 混合或固定值
        color_grid = torch.stack([grid_x, grid_y, 1 - grid_x], dim=0) # [3, H, W]
        return color_grid.unsqueeze(0).expand(B, -1, -1, -1)
    
    def _generate_part_stripes(self, B, H, W, num_parts, device, is_vertical=False):
        """生成按 Part 数量划分的横向或竖向黑灰相间背景"""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # 🌟 根据方向切分
        if is_vertical:
            part_idx = (grid_x // (W / num_parts)).long()
        else:
            part_idx = (grid_y // (H / num_parts)).long()
            
        bg = torch.where(part_idx % 2 == 0, 0.15, 0.05).float()
        return bg.unsqueeze(0).unsqueeze(0).expand(B, 3, H, W)    # =========================================================================

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        CHUNK_SIZE = self.chunk_size 
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_outs = [[] for _ in range(self.num_branches)]
        
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
                pred_keypoints = pose_outs[-1]['pred_keypoints_3d']
                global_rot = pose_outs[-1]['global_rot']
                cam_int_src = dummy_batch['cam_int']    
                
                # --- 遮挡去重逻辑 (Z-Buffer) ---
                part_depths = {}
                # 第一遍：收集每个部位的原始深度图
                if self.part_indices is not None:
                    for name, idxs in self.part_indices.items():
                        _, p_depth = self.get_source_vertex_index_map(
                            pred_verts[:, idxs, :], pred_cam_t, cam_int_src, 
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
                            # mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
                            
                            final_disjoint_masks[name] = mask # [B, 1, H, W]
                            
                            # 生成单部位叠加图 (仅限前5个样本)
                            if self.enable_visual and name in part_colors and curr_bs > 0:
                                m_high = F.interpolate(is_closest.float(), (target_h, target_w), mode='bilinear', align_corners=False)
                                c_vec = torch.tensor(part_colors[name], device=rgb.device).view(1, 3, 1, 1)
                                part_overlay = outs * 0.2 + (m_high * c_vec) * 0.8
                                part_summaries[f'image/part_{name}'] = part_overlay[:5].float()
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
                # sub_feats = features_to_use[i*step : (i+1)*step]
                sub_feats = features_to_use[i::self.num_FPN]
                
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
            
            # 生成原图的总 Mask (用于 OT 过滤无效点)
            # 这里的 part_masks 是你之前用原始投影生成的
            full_mask_src = F.interpolate(generated_mask.float(), (self.sils_size*2, self.sils_size), mode='bilinear', align_corners=False) # [B, 1, H, W]

            # =======================================================
            #  Geometry-Guided Feature Warping (OT)
            # =======================================================
            branch_warped_feats = []
            chunk_pca_tgt_list = []
            chunk_flow_tgt_list = []
            pcd_camera_configs = [] # 用于给点云渲染传递参数

            # 生成一次 TPose 数据，防止每个配置都重复算
            t_pose_verts, t_pose_keypoints = None, None

            # 构建虚拟目标相机
            # 焦距：1.1 * max(H, W)
            focal_tgt = max(target_h, target_w) * 1.1
            cx_tgt, cy_tgt = target_w / 2.0, target_h / 2.0
            
            cam_int_tgt = torch.eye(3, device=rgb.device).unsqueeze(0).expand(curr_bs, 3, 3).clone()
            cam_int_tgt[:, 0, 0] = focal_tgt
            cam_int_tgt[:, 1, 1] = focal_tgt
            cam_int_tgt[:, 0, 2] = cx_tgt
            cam_int_tgt[:, 1, 2] = cy_tgt
            
            # 目标平移：将人放在前方 2.2 米处 (Canonical Depth)
            cam_t_tgt = torch.zeros((curr_bs, 3), device=rgb.device)
            cam_t_tgt[:, 2] = 2.2

            for b_cfg in self.branch_configs:
                branch_h_feat = int(self.sils_size * b_cfg['h_ratio'])
                self.target_angle = b_cfg['angle'] # 临时覆盖，只影响 warp_features_with_ot

                # 记录参数用于 PCD 渲染
                pcd_camera_configs.append({
                    'tgt_h': target_h, 'tgt_w': target_w,
                    'cam_int': cam_int_tgt, 'cam_t': cam_t_tgt,
                    'use_tpose': b_cfg.get('use_tpose', False),
                    'angle': b_cfg['angle']
                })

                if b_cfg.get('use_tpose', False):
                    if t_pose_verts is None:
                        t_pose_verts, t_pose_keypoints = self.generate_mhr_tpose(pose_outs[-1])
                    warp_feat, tgt_mask, tgt_color_flow = self.warp_features_with_ot_tpose(
                        human_feat, full_mask_src, 
                        pred_verts, pred_cam_t, 
                        t_pose_verts, t_pose_keypoints,
                        cam_int_src, cam_int_tgt, cam_t_tgt,
                        self.sils_size*2, self.sils_size, target_h, target_w 
                    )
                else:
                    warp_feat, tgt_mask, tgt_color_flow = self.warp_features_with_ot(
                        human_feat, full_mask_src, 
                        pred_verts, pred_keypoints, pred_cam_t, global_rot,
                        cam_int_src, cam_int_tgt, cam_t_tgt,
                        self.sils_size*2, self.sils_size, target_h, target_w 
                    )

                

                # 在图像中垂直居中切出上述输出的的tgt_h区域
                start_y = (self.sils_size*2 - branch_h_feat) // 2
                end_y = start_y + branch_h_feat
                warp_feat = warp_feat[:, :, start_y:end_y, :]

                branch_warped_feats.append(warp_feat)
                if self.training and self.enable_visual: 
                    tgt_mask = tgt_mask[:, :, start_y:end_y, :]
                    tgt_color_flow = tgt_color_flow[:, :, start_y:end_y, :]
                    # 提前取好切片并转为 float
                    curr_mask = tgt_mask[:self.VIS_SAMPLE].float() 
                    
                    pca_vis = self.get_pca_vis_tensor(warp_feat, tgt_mask, self.VIS_SAMPLE)
                    stripes = self._generate_part_stripes(self.VIS_SAMPLE, branch_h_feat, self.sils_size, b_cfg['parts'], rgb.device, is_vertical=b_cfg.get('vertical_pooling', False))
                    
                    # 🌟 修改后的运算行
                    chunk_pca_tgt_list.append(pca_vis * curr_mask + stripes * (1 - curr_mask))
                    
                    chunk_flow_tgt_list.append(tgt_color_flow)
            
            # 聚合 PCA 和其他图像
            visual_summary = {}
            if self.training and self.enable_visual:
                # 1. 聚合 PCA, flow 图像
                src_pca_batch = self.get_pca_vis_tensor(human_feat, full_mask_src, self.VIS_SAMPLE)
                src_flow_batch = (self._generate_color_grid(curr_bs, self.sils_size*2, self.sils_size, rgb.device) * full_mask_src)[:self.VIS_SAMPLE]
                chunk_pca_src = torch.cat(torch.unbind(src_pca_batch, dim=0), dim=-1).unsqueeze(0)
                chunk_flow_src = torch.cat(torch.unbind(src_flow_batch, dim=0), dim=-1).unsqueeze(0)
                
                pca_row_strips = []
                for pca in chunk_pca_tgt_list:
                    # pca 形状可能是 [5, 3, 64, 32] 或 [5, 3, 48, 32]
                    # 先把这 5 个样本左右拼成一行: [3, 动态高度, 160]
                    row_strip = torch.cat(torch.unbind(pca, dim=0), dim=-1)
                    pca_row_strips.append(row_strip)
                padding = torch.ones((3, 10, pca_row_strips[0].shape[-1]), device=rgb.device) * 0.95
                interleaved_rows = []
                for i, row in enumerate(pca_row_strips):
                    interleaved_rows.append(row)
                    if i < len(pca_row_strips) - 1:
                        interleaved_rows.append(padding)
                chunk_pca_tgt = torch.cat(interleaved_rows, dim=-2).unsqueeze(0)

                flow_row_strips = []
                for flow in chunk_flow_tgt_list:
                    row_strip = torch.cat(torch.unbind(flow[:self.VIS_SAMPLE], dim=0), dim=-1)
                    flow_row_strips.append(row_strip)
                interleaved_flow_rows = []
                for i, row in enumerate(flow_row_strips):
                    interleaved_flow_rows.append(row)
                    if i < len(flow_row_strips) - 1:
                        interleaved_flow_rows.append(padding)
                chunk_flow_grid = torch.cat(interleaved_flow_rows, dim=-2).unsqueeze(0)

                # 2. 渲染点云
                chunk_pcd_grid = self._temp_render_pcd_grid(
                    pred_verts, pred_keypoints, pred_cam_t, global_rot,
                    t_pose_verts, t_pose_keypoints,
                    cam_int_src, pcd_camera_configs, max_samples=self.VIS_SAMPLE
                )

                # 组装 visual_summary
                visual_summary = {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:self.VIS_SAMPLE].float(),
                    **part_summaries,
                    'image/generated_3d_mask_lowres': generated_mask.view(n*s, 1, h_feat, w_feat)[:self.VIS_SAMPLE].float(),
                    'image/generated_3d_mask_interpolated': full_mask_src.view(n*s, 1, self.sils_size*2, self.sils_size)[:self.VIS_SAMPLE].float(),
                    'image/pca_before_OT': chunk_pca_src,
                    'image/pca_after_OT': chunk_pca_tgt,
                    'image/point_cloud_grid': chunk_pcd_grid,
                    'image/ot_flow_reference': chunk_flow_src,
                    'image/ot_flow_warped_grid': chunk_flow_grid,
                }

            # 分别独立通过 GaitNet Part 1
            for b_idx, warp_feat in enumerate(branch_warped_feats):
                warp_feat_5d = rearrange(warp_feat, '(n s) c h w -> n c s h w', n=n, s=s).contiguous()
                
                # 🌟 优化点 1: 使用梯度检查点
                if self.training:
                    # 使用 PyTorch 原生的 checkpoint 函数
                    # 注意：需要确保输入 warp_feat_5d 有梯度（OT 输出通常是有梯度的）
                    outs = torch.utils.checkpoint.checkpoint(
                        self.Gait_Nets[b_idx].test_1, 
                        warp_feat_5d, 
                        use_reentrant=False
                    )
                else:
                    outs = self.Gait_Nets[b_idx].test_1(warp_feat_5d)
                
                all_outs[b_idx].append(outs)
                
                # 🌟 优化点 2: 及时释放不再需要的中间特征
                # 执行完 test_1 后，原本巨大的 warp_feat 已经没用了，立即手动清理
                branch_warped_feats[b_idx] = None

        # 🌟 4. 各分支分别经过 Part 2，并按 FPN 头进行特征融合 (Feature Fusion)
        embed_grouped = [[] for _ in range(self.num_FPN)]
        log_grouped = [[] for _ in range(self.num_FPN)]
        
        for b_idx in range(self.num_branches):
            branch_seq_feat = torch.cat(all_outs[b_idx], dim=2) 
            e_list, l_list = self.Gait_Nets[b_idx].test_2(branch_seq_feat, seqL)
            
            # 将当前分支的结果按 FPN 头分配到对应的组里
            for i in range(self.num_FPN):
                embed_grouped[i].append(e_list[i])
                log_grouped[i].append(l_list[i])
        
        # 将同 1 个 FPN 头在不同分支下的特征在 Part 维度 (dim=-1) 拼接
        embed_list = [torch.cat(feats, dim=-1) for feats in embed_grouped]
        log_list = [torch.cat(logits, dim=-1) for logits in log_grouped]
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.cat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.cat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': visual_summary,
                'inference_feat': {
                    'embeddings': torch.cat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': torch.cat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        return retval