# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random
from functools import partial

# import GaitBase
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image, pca_image
from ..modules import GaitAlign

# =========================================================================
# Helper Functions (Keep same as CLIP version)
# =========================================================================

def gradient_hook(grad, name, step, log):
    if torch.distributed.get_rank() == 0 and step % 100 == 0:
        log.log_info('[{}] Gradient={:.6f}'.format(step, grad.abs().mean().item()))
    return grad

def center_masked_kernel(K, mask_flat):
    N, HW, _ = K.shape
    M = mask_flat.sum(dim=1, keepdim=True).unsqueeze(2).float()
    M = torch.where(M == 0, torch.ones_like(M), M)
    sum_rows = torch.sum(K, dim=2, keepdim=True)
    sum_cols = torch.sum(K, dim=1, keepdim=True)
    row_means = (sum_rows / M) * mask_flat.unsqueeze(2)
    col_means = (sum_cols / M) * mask_flat.unsqueeze(1)
    total_mean = torch.sum(K, dim=(1, 2), keepdim=True) / (M ** 2)
    K_centered = K - row_means - col_means + total_mean
    mask_matrix = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)
    return K_centered * mask_matrix.float()

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
# Main Model: BiggerGait with SAM 3D Body (DINOv3)
# =========================================================================

class BiggerGait__SAM3DBody_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        # 1. 配置参数
        self.pretrained_lvm = model_cfg["pretrained_lvm"] # 这是一个路径，指向 sam-3d-body 源码根目录
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        
        # SAM 3D Body DINOv3 强制输入 512x512 TODO
        self.image_size = 512 
        # BiggerGait 目标 Silhouette 尺寸
        self.sils_size = model_cfg["sils_size"]

        # ViT-Huge 维度通常是 1280  TODO
        self.f4_dim = 1280 
        self.fc_dim = self.f4_dim * 4
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]

        # 2. 初始化子模块 TODO
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)

        self.Pre_Conv = nn.Sequential(
            nn.Identity(),
        )

        # FPN 适配层
        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.f4_dim*4, affine=False),
                nn.Conv2d(self.f4_dim*4, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size*2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])
        
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        
        self.t_channel = self.f4_dim
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def init_SAM_Backbone(self):
        """
        初始化 SAM 3D Body 的 Backbone (DINOv3 ViT-H)
        """
        # 动态添加路径以导入 notebook.utils
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)
            self.msg_mgr.log_info(f"[SAM3D] Added {self.pretrained_lvm} to sys.path")
        
        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Check 'pretrained_lvm' path. Error: {e}")

        self.msg_mgr.log_info(f"[SAM3D] Loading SAM 3D Body via official utils...")
        
        # 使用官方工具加载。
        # 这里假设自动下载或使用 HF 缓存。如果需要指定本地 Checkpoint，
        # 可以在 model_cfg 中添加 checkpoint_path 参数并传给 setup_sam_3d_body
        repo_id = "facebook/sam-3d-body-dinov3"
        estimator = setup_sam_3d_body(hf_repo_id=repo_id)
        
        # 提取 Backbone (Hijack)
        full_model = estimator.model
        
        # 1. 先拿到最外层的 backbone
        if hasattr(full_model, 'backbone'):
            raw_backbone = full_model.backbone
        elif hasattr(full_model, 'image_encoder'):
            raw_backbone = full_model.image_encoder
        else:
            # 兼容逻辑
            raw_backbone = full_model.backbone

        # 2. 【关键修复】剥掉包装壳，取出真正的 Transformer
        # 日志显示它藏在 .encoder 里
        if hasattr(raw_backbone, 'encoder'):
            self.Backbone = raw_backbone.encoder
            self.msg_mgr.log_info(f"[SAM3D] Found 'encoder' submodule. Unwrapping backbone...")
        else:
            self.Backbone = raw_backbone
        
        # 显存优化：删除不需要的组件 TODO
        del full_model.decoder
        del full_model.head_pose
        del full_model.prompt_encoder
        del full_model.head_camera
        del full_model
        del estimator
        
        # 确保 Backbone 在 CPU (BaseModel 会处理 device)
        self.Backbone.cpu()

        # ==================== 注册 Hook 提取特征 ====================
        self.intermediate_features = {}
        self.hook_handles = []

        def get_activation(name):
            def hook(model, input, output):
                # 1. 如果是 list 或 tuple，取第一个元素（那个 Tensor）
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # 2. 有时候会有嵌套（List 套 List），再剥一层保险 TODO ??
                if isinstance(output, (list, tuple)):
                    output = output[0]

                self.intermediate_features[name] = output
            return hook

        # 寻找 DINOv3 的 Blocks TODO DEBUG
        layers_to_hook = []
        if hasattr(self.Backbone, 'blocks'):
            layers_to_hook = self.Backbone.blocks
            self.msg_mgr.log_info(f"[SAM3D] TODO DEBUG HOOK using 'blocks' attribute.")
        elif hasattr(self.Backbone, 'layers'):
            layers_to_hook = self.Backbone.layers
            self.msg_mgr.log_info(f"[SAM3D] TODO DEBUG HOOK using 'layers' attribute.")
        else:
            # 最后的保底：看看是不是还在更深的地方（虽然概率不大了）
            raise RuntimeError(f"Could not find blocks/layers in backbone! Keys: {self.Backbone._modules.keys()}")
        
        self.msg_mgr.log_info(f"[SAM3D] Hooking {len(layers_to_hook)} layers of DINOv3.")
        
        for i, layer in enumerate(layers_to_hook):
            handle = layer.register_forward_hook(get_activation(i))
            self.hook_handles.append(handle)
            
        # 冻结 Backbone
        self.Backbone.eval()
        for param in self.Backbone.parameters():
            param.requires_grad = False
            
        self.msg_mgr.log_info(f"[SAM3D] Loaded successfully from {repo_id}")

    def init_Mask_Branch(self): # TODO
        # self.msg_mgr.log_info(f'load model from: {self.pretrained_mask_branch}')
        # load_dict = torch.load(self.pretrained_mask_branch, map_location=torch.device("cpu"))['model']
        # msg = self.Mask_Branch.load_state_dict(load_dict, strict=True)
        # n_parameters = sum(p.numel() for p in self.Mask_Branch.parameters())
        # self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        # self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        # self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_mask_branch}'")
        # self.msg_mgr.log_info('SegmentationBranch Count: {:.5f}M'.format(n_parameters / 1e6))

        # 原来的代码是加载 .pt 文件，现在直接跳过
        self.msg_mgr.log_info("=> Skip loading Mask Branch (Using Full Image Features)")
        # 保持 Mask_Branch 为随机初始化即可，反正 forward 里我们不用它
        pass

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        self.init_SAM_Backbone()
        self.init_Mask_Branch()
        
        # 确保 Mask Branch 和 Backbone 是冻结的
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Model Count: {:.5f}M'.format(n_parameters / 1e6))
        self.msg_mgr.log_info("=> init successfully")

    def preprocess(self, sils, h, w, mode='bilinear'):
        # 强制 Resize
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def forward(self, inputs):
        self.Mask_Branch.eval()
        
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # ViT-Huge 显存占用极大，使用极小的 chunk size (建议 4 或 8)
        # 如果显存依然不够，调小这里
        CHUNK_SIZE = 4 
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_outs = []
        
        # DINOv3 目标尺寸 512x512
        target_h, target_w = self.image_size, self.image_size
        
        # Patch Size = 16 (vit_h_16) -> Feature Map = 32x32
        h_feat = target_h // 16 

        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                n, s, c, h, w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                
                # 1. Resize -> 512x512
                outs = self.preprocess(rgb_img, target_h, target_w)
                
                # 2. 清空 Hook 缓存
                self.intermediate_features = {}
                
                # 3. Forward Backbone
                _ = self.Backbone(outs)
                
                # 4. 收集 Features
                # 假设我们用所有层
                num_layers = len(self.hook_handles)
                hidden_states = [self.intermediate_features[i] for i in range(num_layers)]
                
                if not hidden_states:
                    raise RuntimeError("Hook failed to capture features!")

                # 5. 处理 Feature Shape [B, N, C] -> [B, C, H, W]
                valid_layers = []
                for feat in hidden_states:
                    # 移除 CLS token (如果存在)
                    # 512x512 / 16 = 32x32 = 1024 tokens. 
                    # 如果 shape[1] == 1025，说明有 CLS
                    # if feat.shape[1] == (h_feat * h_feat) + 1:
                    #     feat = feat[:, 1:, :]

                    # [新代码] 更加鲁棒的切片逻辑
                    # 目标 Token 数：32*32 = 1024
                    target_tokens = h_feat * h_feat 
                    # 只要 Token 数大于 1024，就说明有 CLS 或 Registers
                    if feat.shape[1] > target_tokens:
                        # 取最后 target_tokens 个，这样不管前面有 1 个还是 5 个都能切干净
                        feat = feat[:, -target_tokens:, :]
                    
                    valid_layers.append(feat)

                # 使用所有层 (或根据需要切片，例如 hidden_states[1:])
                # 原 CLIP 代码用了 hidden_states[1:]，这里我们跟随该逻辑
                features_to_use = valid_layers[1:] if len(valid_layers) > 1 else valid_layers
                
                # 拼接 Feature [B, L, C] -> [B, L, C_total]
                appearance = torch.concat(features_to_use, dim=-1)
                
                # Reshape to [B, C, H, W]
                # [B, 1024, C_total] -> [B, 32*32, C_total] -> [B, C_total, 32, 32]
                appearance = rearrange(appearance, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                
                # Pre_Conv
                appearance = self.Pre_Conv(appearance)
                appearance = rearrange(appearance, 'b c h w -> b (h w) c').contiguous()
                
                # LayerNorm
                appearance = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * len(features_to_use), elementwise_affine=False)(appearance)
                
                # Reshape back for processing
                appearance = rearrange(appearance, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                
                # # 6. Mask Branch (使用最后一层特征)
                # outs_last1 = valid_layers[-1] # [B, 1024, 1280]
                # human_mask_ori, _ = self.Mask_Branch(outs_last1.contiguous().view(-1, self.f4_dim))
                
                # human_mask_ori = (human_mask_ori[:, 1] > 0.5).float()
                # # Reshape Mask [B*HW] -> [B, 1, H, W]
                # human_mask_ori = human_mask_ori.view(n*s, 1, h_feat, h_feat)
                
                # # Resize Mask back to BiggerGait sils_size (usually 64 or 128)
                # human_mask_ori = self.preprocess(human_mask_ori, self.sils_size*2, self.sils_size).detach().clone()

                # TODO 学长建议：暂时不加 Mask Branch，直接使用全图特征。
                # 做法：构造一个全为 1 的 Mask，形状与 Feature Map 对应。
                
                # h_feat = 32 (512/16)
                # 生成一个 [Batch*Seq, 1, 32, 32] 的全 1 张量
                # 注意 device 要和 appearance 保持一致
                human_mask_ori = torch.ones(
                    (n*s, 1, h_feat, h_feat), 
                    dtype=appearance.dtype, 
                    device=appearance.device
                )
                
                # 下面这行 resize 保持不变，为了兼容后续的数据流
                # Resize Mask back to BiggerGait sils_size (usually 64 or 32)
                human_mask_ori = self.preprocess(
                    human_mask_ori, 
                    self.sils_size*2, 
                    self.sils_size
                ).detach()
            
            # =================================================================
            # 7. FPN 分块处理 (Chunks) - DINOv3 适配版 (Full 32 Layers)
            # =================================================================
            
            # 在 model_cfg 中 num_FPN = 8
            # DINOv3 ViT-Huge 有 32 层
            # 32 / 8 = 4 层/Head，完美匹配 HumanSpace_Conv 的输入维度 (f4_dim * 4)

            # 1. 验证层数
            total_layers = len(valid_layers) # 应该是 32
            if total_layers % self.num_FPN != 0:
                # 只有当配置错误时才会触发，例如 num_FPN 还是 6
                # 如果触发，我们会尝试截断到最近的整数倍
                layers_to_use = (total_layers // self.num_FPN) * self.num_FPN
                features_to_use = valid_layers[-layers_to_use:]
            else:
                features_to_use = valid_layers

            # 2. 拼接所有特征 [B, N, C] -> [B, N, C_total]
            # C_total = 32 * 1280
            appearance = torch.concat(features_to_use, dim=-1)
            
            # 3. Reshape [B, C_total, H, W]
            appearance = rearrange(appearance, 'b (h w) c -> b c h w', h=h_feat).contiguous()

            # 4. Pre_Conv & Norm
            appearance = self.Pre_Conv(appearance)
            
            # Norm 在 Channel 维度
            appearance = rearrange(appearance, 'b c h w -> b (h w) c')
            appearance = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * len(features_to_use), elementwise_affine=False)(appearance)
            appearance = rearrange(appearance, 'b (h w) c -> b c h w', h=h_feat).contiguous()

            # 5. 切分给 8 个 FPN Head
            # appearance 通道数是 32 * 1280
            # split 成 8 份，每份是 4 * 1280 = 5120 通道
            human_feat = list(torch.chunk(appearance, self.num_FPN, dim=1))

            # 6. 喂给 HumanSpace_Conv (降维)
            processed_feat = []
            for i in range(self.num_FPN):
                # human_feat[i] 维度 [B, 5120, H, W]
                # HumanSpace_Conv[i] 接受 5120 (f4_dim*4) -> 输出 640 (f4_dim//2)
                processed_feat.append(self.HumanSpace_Conv[i](human_feat[i]))
            
            # 7. 拼接回 [B, C_reduced, H, W]
            # 8 * 640 -> 5120 通道 (如果后续 concat) 或者按 BiggerGait 逻辑处理
            human_feat = torch.concat(processed_feat, dim=1)

            # Apply Mask
            human_mask = human_mask_ori
            human_feat = human_feat * (human_mask > 0.5).float()
            
            # Reshape for Gait_Net [B, C, S, H, W]
            human_feat = rearrange(human_feat.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w').contiguous()

            # Gait Net Test 1
            outs = self.Gait_Net.test_1(human_feat)
            all_outs.append(outs)

        # Gait Net Test 2
        embed_list, log_list = self.Gait_Net.test_2(
            torch.cat(all_outs, dim=2),
            seqL,
        )
        
        # 组装返回值 (同 CLIP)
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask_ori.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()).clamp(0,1),
                },
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        return retval