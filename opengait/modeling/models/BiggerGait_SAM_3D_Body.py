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
        # 1. 基础参数
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]

        # ====================================================
        # 🌟 [逻辑修改] 解析层配置与数学验证
        # ====================================================
        layer_cfg = model_cfg.get("layer_config", {})
        self.layers_per_group = layer_cfg.get("layers_per_group", 2)
        
        # 1. 获取/生成 Hook Mask (32层)
        if "hook_mask" in layer_cfg:
            self.hook_mask = layer_cfg["hook_mask"]
            if len(self.hook_mask) != 32:
                raise ValueError(f"hook_mask 长度必须为 32，当前为 {len(self.hook_mask)}")
        else:
            # 默认：前16层 False，后16层 True
            self.hook_mask = [False]*16 + [True]*16
            self.msg_mgr.log_info("[Network] No hook_mask found, using default (Top-16).")

        # 2. 计算实际 Hook 的层数
        self.total_hooked_layers = sum(self.hook_mask)
        self.msg_mgr.log_info(f"[Network] Total Layers to Hook: {self.total_hooked_layers}")

        if self.total_hooked_layers == 0:
            raise ValueError("hook_mask 全为 False，没有层被选中！")

        # 3. [验证] 能否被 group 整除
        if self.total_hooked_layers % self.layers_per_group != 0:
            raise ValueError(f"Hook总层数 ({self.total_hooked_layers}) 无法被 layers_per_group ({self.layers_per_group}) 整除！")
        
        self.total_groups = self.total_hooked_layers // self.layers_per_group
        self.msg_mgr.log_info(f"[Network] Total Groups: {self.total_groups} (Size: {self.layers_per_group})")

        # 4. [验证] 总 Group 能否被 num_FPN 整除
        if self.total_groups % self.num_FPN != 0:
            raise ValueError(f"总Group数 ({self.total_groups}) 无法被 num_FPN ({self.num_FPN}) 整除！")

        # 5. 计算每个 Head 负责处理几层 (Layers Per Head)
        # 这决定了 HumanSpace_Conv 的输入通道数
        # 逻辑：总层数 / FPN数
        self.layers_per_head = self.total_hooked_layers // self.num_FPN
        input_dim = self.f4_dim * self.layers_per_head
        
        self.msg_mgr.log_info(f"[Network] === Configuration Validated ===")
        self.msg_mgr.log_info(f"          |-> FPN Heads: {self.num_FPN}")
        self.msg_mgr.log_info(f"          |-> Layers per Head: {self.layers_per_head}")
        self.msg_mgr.log_info(f"          |-> Conv Input Dim: {input_dim}")
        # ====================================================

        # 初始化下游网络
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)
        self.Pre_Conv = nn.Sequential(nn.Identity())

        # FPN 适配层 (根据计算出的 input_dim 初始化)
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
        
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        
        self.t_channel = self.f4_dim
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def init_SAM_Backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)
        
        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Error: {e}")

        self.msg_mgr.log_info(f"[SAM3D] Loading SAM 3D Body...")
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')
        
        full_model = estimator.model
        if hasattr(full_model, 'backbone'):
            raw_backbone = full_model.backbone
        elif hasattr(full_model, 'image_encoder'):
            raw_backbone = full_model.image_encoder
        else:
            raw_backbone = full_model.backbone

        if hasattr(raw_backbone, 'encoder'):
            self.Backbone = raw_backbone.encoder
        else:
            self.Backbone = raw_backbone
        
        # 清理
        del full_model.decoder
        del full_model.head_pose
        del full_model.prompt_encoder
        del full_model.head_camera
        del full_model
        del estimator
        
        self.Backbone.cpu()

        # ====================================================
        # 🌟 [逻辑修改] 根据 hook_mask 注册 Hook
        # ====================================================
        self.intermediate_features = {}
        self.hook_handles = []

        def get_activation(idx_in_list):
            # 注意：这里的 idx_in_list 是 intermediate_features 列表中的索引
            # 不是原始层号，而是第几个被 Hook 的层
            def hook(model, input, output):
                if isinstance(output, (list, tuple)): output = output[0]
                if isinstance(output, (list, tuple)): output = output[0]
                self.intermediate_features[idx_in_list] = output
            return hook

        all_blocks = []
        if hasattr(self.Backbone, 'blocks'):
            all_blocks = self.Backbone.blocks
        elif hasattr(self.Backbone, 'layers'):
            all_blocks = self.Backbone.layers
        else:
            raise RuntimeError("Cannot find blocks in Backbone")

        # 遍历所有 32 层，根据 Mask 决定是否 Hook
        hook_count = 0
        for layer_idx, should_hook in enumerate(self.hook_mask):
            if should_hook:
                # 传入 hook_count 作为存储索引，确保 features 列表是紧凑的 (0, 1, 2...)
                handle = all_blocks[layer_idx].register_forward_hook(get_activation(hook_count))
                self.hook_handles.append(handle)
                hook_count += 1
        
        self.msg_mgr.log_info(f"[SAM3D] Hooked {hook_count} layers based on mask.")
        # ====================================================

        # 冻结 & 评估模式
        self.Backbone.eval()
        for param in self.Backbone.parameters():
            param.requires_grad = False

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
        # 确保 Mask Branch 不更新
        self.Mask_Branch.eval()
        
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # 显存优化：如果依然 OOM，将此值改为 2
        CHUNK_SIZE = 4 
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_outs = []
        
        # DINOv3 固定输入分辨率
        target_h, target_w = self.image_size, self.image_size
        h_feat = target_h // 16 

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            
            # =======================================================
            # 1. Backbone 前向 (必须加 no_grad 以节省显存)
            # =======================================================
            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w)
                
                # 清空 Hook 缓存
                self.intermediate_features = {}
                
                # DINOv3 推理
                _ = self.Backbone(outs)
                
                # 收集被 Hook 的层
                num_layers = len(self.hook_handles)
                features_to_use = []
                target_tokens = h_feat * h_feat 
                
                for i in range(num_layers):
                    feat = self.intermediate_features[i]
                    # 去除 CLS Token 等多余部分，只保留 Spatial Tokens
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :]
                    features_to_use.append(feat)

            # =======================================================
            # 2. FPN 组处理 (Group Processing)
            # =======================================================
            processed_feat_list = []
            
            # 自动计算步长：例如 Hook了16层，FPN有4个，则 step=4 (即每组4层)
            # 这与 build_network 中的 input_dim 计算逻辑是完全对应的
            step = len(features_to_use) // self.num_FPN
            
            for i in range(self.num_FPN):
                # A. 切片：取出当前 Head 负责的那几层
                start_idx = i * step
                end_idx = (i + 1) * step
                sub_feats = features_to_use[start_idx : end_idx]
                
                # B. 拼接：将这几层拼在一起
                # 维度变化: [B, N, 1280] x step -> [B, N, 1280*step]
                sub_app = torch.concat(sub_feats, dim=-1)
                
                # C. 调整形状以进行卷积 [B, C, H, W]
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                
                # D. Pre_Conv (Identity)
                sub_app = self.Pre_Conv(sub_app)
                
                # E. 局部 LayerNorm (针对当前组的维度进行归一化)
                sub_app = rearrange(sub_app, 'b c h w -> b (h w) c')
                # 计算当前组的通道数，例如 1280 * 2 = 2560
                curr_dim = self.f4_dim * len(sub_feats)
                sub_app = partial(nn.LayerNorm, eps=1e-6)(curr_dim, elementwise_affine=False)(sub_app)
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                
                # F. 喂给第 i 个独立的 FPN Head
                # self.HumanSpace_Conv[i] 的输入维度在 build_network 里已经按 step 算好了
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                
                processed_feat_list.append(reduced_feat)
                
                # 释放显存
                del sub_app
                del sub_feats

            # 3. 拼接所有 Head 的输出
            human_feat = torch.concat(processed_feat_list, dim=1)
            
            # =======================================================
            # 3. 后处理 (Mask & GaitNet)
            # =======================================================
            
            # 生成全 1 Mask (跳过 Mask Branch)
            human_mask_ori = torch.ones(
                (n*s, 1, h_feat, h_feat), 
                dtype=human_feat.dtype, 
                device=human_feat.device
            )
            
            # Resize Mask 到目标尺寸
            human_mask = self.preprocess(
                human_mask_ori, 
                self.sils_size*2, 
                self.sils_size
            ).detach()
            
            # 应用 Mask
            human_feat = human_feat * (human_mask > 0.5).float()
            
            # Reshape 喂给 GaitNet
            human_feat = rearrange(human_feat.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w').contiguous()

            # GaitNet Part 1
            outs = self.Gait_Net.test_1(human_feat)
            all_outs.append(outs)

        # GaitNet Part 2 (Temporal Aggregation)
        embed_list, log_list = self.Gait_Net.test_2(
            torch.cat(all_outs, dim=2),
            seqL,
        )
        
        # 组装返回值
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()).clamp(0,1),
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