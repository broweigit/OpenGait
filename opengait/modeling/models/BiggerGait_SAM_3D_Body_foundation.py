# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import repeat,rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random
from functools import partial

# import GaitBase
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image, pca_image
from ..modules import GaitAlign

from torch.utils.checkpoint import checkpoint

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

class BiggerGait__SAM3DBody__Foundation_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        # 1. 基础参数
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)

        layer_cfg = model_cfg.get("layer_config", {})

        self.hook_mask = layer_cfg["hook_mask"] if "hook_mask" in layer_cfg else [False]*16 + [True]*16
        assert len(self.hook_mask) == 32, "hook_mask length must be 32."
        assert self.hook_mask.count(True) % self.num_FPN == 0, "Number of hook layers must be divisible by num_FPN."

        total_hooked_layers = sum(self.hook_mask) # 16
        assert total_hooked_layers > 0, "At least one layer must be hooked."

        human_conv_input_dim = self.f4_dim * (total_hooked_layers // self.num_FPN) # 1280 * 4 = 5120

        self.trans_layer_mask = layer_cfg["trans_layer_mask"] if "trans_layer_mask" in layer_cfg else [True]*6
        assert len(self.trans_layer_mask) == 6, "trans_layer_mask length must be 6."
        self.total_attn_layers = self.trans_layer_mask.count(True)

        self.num_hard_parts = model_cfg["num_hard_parts"] # 水平硬切分
        self.num_soft_parts = model_cfg["num_soft_parts"] # 软分割部分
        assert self.num_soft_parts == self.total_attn_layers * 8
        self.total_parts = self.num_hard_parts + self.num_soft_parts

        # A. Hard Stream Modules (8 Parts)
        self.Gait_Nets = nn.ModuleList([
            Baseline_ShareTime_2B(model_cfg) for _ in range(self.num_hard_parts)
        ])

        # B. Soft Stream Modules (layers * 8 Parts)
        self.Soft_Gait_Nets = nn.ModuleList([
            Baseline_ShareTime_2B(model_cfg) for _ in range(self.num_soft_parts)
        ])

        part_target_h = (self.sils_size * 2) // self.num_hard_parts # e.g. 64 // 8 = 8
        part_target_w = self.sils_size # 32

        # =======================================================
        # 1. 定义公共降维层 (Common Reducer) - Hard 和 Soft 共享
        # =======================================================
        # 输入 5120 -> 输出 640
        # 数量: num_FPN (4个)
        self.Common_HumanSpace_Reducer = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(human_conv_input_dim, affine=False),
                nn.Conv2d(human_conv_input_dim, self.f4_dim//2, kernel_size=1), # 5120 -> 640
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
            ) for _ in range(self.num_FPN)
        ])

        # =======================================================
        # 2. Hard Stream Heads (Lightweight)
        # =======================================================
        # 输入 640 -> 输出 16
        # 数量: num_FPN * num_hard_parts (4 * 8 = 32)
        # 以前这里是 Heavy Conv，现在只是一个小小的 640->16 投影
        self.Hard_Part_Heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1), # 640 -> 16
                ResizeToHW((part_target_h, part_target_w)), 
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN * self.num_hard_parts)
        ])

        # =======================================================
        # 3. Soft Stream Heads (Lightweight)
        # =======================================================
        # 这里的 num_soft_convs = num_FPN * total_attn_layers (4 * 1 = 4)
        self.num_soft_convs = self.num_FPN * self.total_attn_layers
        
        self.Soft_Part_Heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1), # 640 -> 16
                ResizeToHW((part_target_h, part_target_w)), 
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_soft_convs)
        ])
        
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        
        self.t_channel = self.f4_dim
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def init_SAM_Backbone(self):
        sys.path.insert(0, self.pretrained_lvm)
        from notebook.utils import setup_sam_3d_body
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')

        self.SAM_Engine = estimator.model
        # 1. 开启中间层输出
        self.SAM_Engine.decoder.do_interm_preds = True

        # 2. 冻结所有参数
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False
        self.SAM_Engine.eval()
        

        self.Backbone = self.SAM_Engine.backbone

        # 5. 注册 Hook (捕获 Q 和 K)
        self.hook_data = {} 
        
        def get_activation_hook(layer_idx, key_name):
            def hook(module, input, output):
                if layer_idx not in self.hook_data:
                    self.hook_data[layer_idx] = {}
                # 注意：这里 output 可能是 FP16，如果你后续计算 Attention 想用 FP32 可以在这里转
                self.hook_data[layer_idx][key_name] = output.detach().float() 
            return hook

        for i, layer in enumerate(self.SAM_Engine.decoder.layers):
            layer.cross_attn.q_proj.register_forward_hook(get_activation_hook(i, 'q'))
            layer.cross_attn.k_proj.register_forward_hook(get_activation_hook(i, 'k'))
            
        del estimator

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
    
    def _prepare_dummy_batch(self, image_embeddings):
        """构造最小限度的 Dummy Batch (符合 [B, Num_Person, ...] 格式)"""
        B, C, H, W = image_embeddings.shape
        device = image_embeddings.device
        
        # 1. 根据特征图反推输入尺寸 (假设 Patch Size = 16)
        input_h, input_w = H * 16, W * 16
        
        # 2. 构造虚拟尺寸 (用于坐标归一化)
        # 这里用计算出的 input_h/w 可能更准确，或者保持 1024 也可以
        # 为了保险起见，我们让 'img_size' 和 'ray_cond' 的尺寸对齐
        dummy_size = torch.tensor([float(input_w), float(input_h)], device=device)
        
        # [B, 1, 2]
        ori_size = dummy_size.unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        img_size = dummy_size.unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        center = (dummy_size / 2).unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        
        # Scale 设为输入图片的最大边长，模拟全图 Crop
        scale = torch.tensor([max(input_w, input_h)], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 1)
        
        # [B, 3, 3]
        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
        
        # [B, 1, 2, 3]
        affine_trans = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
        affine_trans = affine_trans.unsqueeze(0).unsqueeze(0).expand(B, 1, 2, 3)
        
        # 🌟 [关键修复] 补上 ray_cond 🌟
        # 形状: [Batch, 2, Input_H, Input_W]
        # 注意：这里不需要 Person 维度，因为它对应的是整张 Input Crop
        ray_cond = torch.zeros(B, 2, input_h, input_w, device=device)

        return {
            "img": torch.zeros(B, 1, 3, input_h, input_w, device=device),
            "ori_img_size": ori_size,
            "img_size": img_size,
            "bbox_center": center,
            "bbox_scale": scale,
            "cam_int": cam_int,
            "affine_trans": affine_trans,
            "ray_cond": ray_cond, # <--- 补上这个
        }

    def compute_attention_map(self, q_proj, k_proj, num_heads):
        """
        手动计算 Attention Map(保留Multihead)
        """
        B, N_q, C = q_proj.shape
        _, N_k, _ = k_proj.shape
        head_dim = C // num_heads
        scale = head_dim ** -0.5

        # 1. Separate Heads & Permute: [B, H, N, d]
        q = q_proj.reshape(B, N_q, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k_proj.reshape(B, N_k, num_heads, head_dim).permute(0, 2, 1, 3)

        # 2. Dot Product
        attn_logits = (q @ k.transpose(-2, -1)) * scale

        # 3. Softmax
        attn_weights = F.softmax(attn_logits, dim=-1) # [B, H, N_q, N_k]

        # # 4. Average over Heads
        # attn_weights_mean = attn_weights.mean(dim=1) # [B, N_q, N_k]
        
        return attn_weights

    def forward(self, inputs):
        # 冻结 Mask Branch
        self.Mask_Branch.eval()
        
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # 显存优化
        CHUNK_SIZE = self.chunk_size
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        # 结构: [Chunk_0_Outs, Chunk_1_Outs, ...]
        # Chunk_i_Outs: List containing results for [Hard_Parts(0-7), Soft_Parts(0-31)]
        all_chunk_outs = []
        all_attn_maps = []  # 用于存储所有 Chunk 的 Attention Maps
        
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16 # DINOv3 Patch Size = 16

        # 定义一个变量用于存储可视化用的 Mesh 数据
        vis_pose_output = None

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            
            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w)
                all_layers = self.Backbone.encoder.get_intermediate_layers(outs, n=32, reshape=True, norm=True)
                # A. 提取用于 FPN 的特征 (根据 hook_mask)
                features_to_use = [f for f, m in zip(all_layers, self.hook_mask) if m]
                # B. 提取用于 Decoder 的特征 (最后一层)
                image_embeddings = all_layers[-1] # [B, C, H, W]
                del all_layers


                self.hook_data.clear() # 清空 Hook 缓存
                dummy_batch = self._prepare_dummy_batch(image_embeddings)

                # ======= 设置 SAM Engine 状态 =======
                batch_size = image_embeddings.shape[0]
                self.SAM_Engine._batch_size = batch_size
                self.SAM_Engine._max_num_person = 1  # 告诉模型：当前 Batch 中每个样本只有 1 个人
                self.SAM_Engine.body_batch_idx = torch.arange(batch_size, device=image_embeddings.device)
                self.SAM_Engine.hand_batch_idx = []
                cond_info = None
                if self.SAM_Engine.cfg.MODEL.DECODER.CONDITION_TYPE != "none":
                    cond_info = torch.zeros(image_embeddings.shape[0], 3, device=image_embeddings.device).float()
                    cond_info[:, 2] = 1.25

                # 必须传入 keypoints 才能触发 decoder 内部的 token 初始化逻辑
                # 格式: [Batch, Num_Points, 3] -> [Batch, 1, 3]
                # 最后一个维度 3 代表 [x, y, label]
                # label = -2 表示 "Invalid Point" (仅用于占位，不产生实际提示效果)
                dummy_keypoints = torch.zeros(batch_size, 1, 3, device=image_embeddings.device)
                dummy_keypoints[:, :, -1] = -2
                
                # ====================================

                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    # 1. 捕获返回值：tokens_out 是 hidden states, pose_outs 是每一层的预测结果列表
                    tokens_out, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=image_embeddings,
                        init_estimate=None,
                        keypoints=dummy_keypoints,
                        prev_estimate=None,
                        condition_info=cond_info,
                        batch=dummy_batch
                    )
                    
                    if vis_pose_output is None:
                        # pose_outs[-1] 包含当前 chunk 所有帧的 mesh 数据 [BS, 10475, 3]
                        vis_pose_output = pose_outs[-1]

                # Compute Attention Map
                chunk_layer_maps = []
                num_heads = self.SAM_Engine.cfg.MODEL.DECODER.HEADS # 8
                sorted_layer_keys = sorted(self.hook_data.keys())

                for l_idx in sorted_layer_keys: # Layer num: 6
                    q_proj = self.hook_data[l_idx]['q'] # [B, 145, 1024] 1 + 1 + 1 + 2 + 70 + 70
                    k_proj = self.hook_data[l_idx]['k'] # [B, 512, 1024] 32x16
                    
                    # 计算原始 Attention Map Multihead -> Q @ K.T [B, H, 145, 512]
                    attn_map_full = self.compute_attention_map(q_proj, k_proj, num_heads)
                    
                    # 截取 第一 个 Body Parts
                    # Token 结构推测: [Init(1), Prev(1), Prompt(1), Hand(2), Body2D(70), Body3D(70)]
                    # self.msg_mgr.log_info(f'Layer {l_idx}: attn_map_full shape: {attn_map_full.shape}, head num: {attn_map_full.shape[1]}, total query tokens: {attn_map_full.shape[2]}')
                    attn_map_multihead = attn_map_full[:, :, :1, :] # [B, H, 70, HW]

                    # Reshape [B, 8, H, W]
                    attn_spatial = rearrange(attn_map_multihead, 'b H p (h w) -> b (H p) h w', h=h_feat, w=w_feat)
                    chunk_layer_maps.append(attn_spatial.cpu())

                # chunk_maps: [n*s, Layers, 8, h, w]
                chunk_maps = torch.stack(chunk_layer_maps, dim=1)
                
                # 还原 Batch 维度: [n, s, Layers, 8, h, w]
                # 把 s 放在 dim=1，方便后续 torch.cat(dim=1)
                chunk_maps = rearrange(chunk_maps, '(n s) l p h w -> n s l p h w', n=n, s=s)
                all_attn_maps.append(chunk_maps)

            # 这里的 chunk_maps 是在 CPU 上的 (因为 append 时用了 .cpu())
            # 我们需要把它放回 GPU 并 reshape
            # [n, s, l, p, h, w] -> [n*s, l*p(48), h, w]
            current_chunk_masks = rearrange(chunk_maps.to(rgb_img.device), 'n s l p h w -> (n s) (l p) h w')

            # FPN 组处理 (Group Processing)
            step = len(features_to_use) // self.num_FPN

            # 准备存储当前 Chunk 的处理结果 [Part_Idx][FPN_Idx]
            current_chunk_hard_inputs = [[] for _ in range(self.num_hard_parts)]
            current_chunk_soft_inputs = [[] for _ in range(self.num_soft_parts)]
            
            for i in range(self.num_FPN):
                # Group & Concat
                sub_feats = features_to_use[i*step : (i+1)*step]
                
                # [B, 1280, H, W] x step -> [B, 5120, H, W] if step = 4
                sub_app = torch.cat(sub_feats, dim=1)
                
                # LayerNorm [B, 5120, H, W] -> [B, H, W, 5120] -> LN -> [B, 5120, H, W]
                sub_app = sub_app.permute(0, 2, 3, 1) 
                sub_app = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * len(sub_feats), elementwise_affine=False)(sub_app)
                sub_app = sub_app.permute(0, 3, 1, 2).contiguous()

                # 🌟 [关键步骤] 全局统一降维 (5120 -> 640)
                # 这一步现在处理整张图，Hard 和 Soft 都用这个结果
                sub_app_reduced = self.Common_HumanSpace_Reducer[i](sub_app)

                # --- B. Stream 1: Hard Parts (水平切分) ---
                # 🌟 现在切分的是降维后的特征 [B, 640, H, W]
                reduced_hard_parts = torch.chunk(sub_app_reduced, self.num_hard_parts, dim=2)
                
                for p_idx in range(self.num_hard_parts):
                    conv_idx = i * self.num_hard_parts + p_idx
                    
                    # 通过轻量级 Head: [B, 640, h_part, w] -> [B, 16, h_target, w_target]
                    processed = self.Hard_Part_Heads[conv_idx](reduced_hard_parts[p_idx])
                    current_chunk_hard_inputs[p_idx].append(processed)

                # --- C. Stream 2: Soft Parts (Attention Masking) ---
                # 逻辑保持上次优化后的样子，使用 sub_app_reduced
                for p_idx in range(self.num_soft_parts):
                    layer_idx = p_idx // 8 
                    conv_idx = i * self.total_attn_layers + layer_idx
                    
                    mask = current_chunk_masks[:, p_idx:p_idx+1, :, :]
                    
                    # [B, 640, H, W] * Mask -> [B, 640, H, W]
                    masked_feat = sub_app_reduced * mask 
                    
                    processed = self.Soft_Part_Heads[conv_idx](masked_feat)
                    current_chunk_soft_inputs[p_idx].append(processed)

            # =======================================================
            # 4. GaitNet Test 1 (Frame Level)
            # =======================================================
            chunk_parts_outputs = [] # length = num_hard_parts + num_soft_parts

            # -> Hard Stream Forward
            for p_idx in range(self.num_hard_parts):
                part_feat = torch.cat(current_chunk_hard_inputs[p_idx], dim=1) # Concat FPN
                # Reshape: [n, c, s, h, w]
                part_feat = rearrange(part_feat.view(n, s, -1, part_feat.shape[2], part_feat.shape[3]), 
                                      'n s c h w -> n c s h w').contiguous()
                # Checkpoint 要求输入必须 requires_grad=True 才能触发
                # 通常 part_feat 来自 backbone，已经具备 grad_fn，但为了保险起见可以检查一下
                if part_feat.requires_grad:
                    out = checkpoint(self.Gait_Nets[p_idx].test_1, part_feat, use_reentrant=False)
                else:
                    # 如果是在 eval 模式或者第一层就被截断，则不需要 checkpoint
                    out = self.Gait_Nets[p_idx].test_1(part_feat)
                chunk_parts_outputs.append(out)

            # -> Soft Stream Forward
            for p_idx in range(self.num_soft_parts):
                part_feat = torch.cat(current_chunk_soft_inputs[p_idx], dim=1) # Concat FPN
                part_feat = rearrange(part_feat.view(n, s, -1, part_feat.shape[2], part_feat.shape[3]), 
                                      'n s c h w -> n c s h w').contiguous()
                # 使用 Soft Stream 的网络
                if part_feat.requires_grad:
                    out = checkpoint(self.Soft_Gait_Nets[p_idx].test_1, part_feat, use_reentrant=False)
                else:
                    out = self.Soft_Gait_Nets[p_idx].test_1(part_feat)
                chunk_parts_outputs.append(out)

            all_chunk_outs.append(chunk_parts_outputs)

        # =======================================================
        # 5. Aggregate Chunks & GaitNet Test 2 (Temporal Pooling)
        # =======================================================
        # all_chunk_outs: [Chunk0, Chunk1...]
        # ChunkX: [Part0_Tensor, ... PartN_Tensor]
        
        final_embeds_per_fpn = [[] for _ in range(self.num_FPN)]
        final_logits_per_fpn = [[] for _ in range(self.num_FPN)]

        for p_idx in range(self.total_parts):
            # 5.1 拼接时间维度 (dim=2) -> [n, c, S_total, h, w]
            part_feat_total = torch.cat([chunk[p_idx] for chunk in all_chunk_outs], dim=2)
            
            # 5.2 选择对应的网络
            if p_idx < self.num_hard_parts:
                net = self.Gait_Nets[p_idx]
            else:
                net = self.Soft_Gait_Nets[p_idx - self.num_hard_parts]

            # 5.3 Test 2: Temporal Pooling + HPP + FCs
            embed_sub, logit_sub = net.test_2(part_feat_total, seqL)
            
            # 5.4 按 FPN 层级收集 num_FPN(4) * [N, 256, bin(4)]
            for fpn_idx in range(self.num_FPN):
                final_embeds_per_fpn[fpn_idx].append(embed_sub[fpn_idx])
                final_logits_per_fpn[fpn_idx].append(logit_sub[fpn_idx])
                
        # =======================================================
        # 6. Final Concatenation
        # =======================================================
        # 组装 Embedding 和 Logits(num_FPN长的list，在 dim=-1 上拼接) 4 * [N, 256, bin(4)*(num_hard_parts+num_soft_parts)]
        embed_list = [torch.cat(parts, dim=-1) for parts in final_embeds_per_fpn]
        log_list = [torch.cat(parts, dim=-1) for parts in final_logits_per_fpn]

        # -------------------------------------------------------
        # Visualization (可视化所有层 + Jet 热力图)
        # -------------------------------------------------------
        vis_dict = {}
        if self.training and torch.distributed.get_rank() == 0:
            # try:
                # 1. 准备原图 (取第一个样本，第一帧)
                # vis_img: [3, H, W]
                vis_img = inputs[0][0][0, 0].detach().cpu()
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-6)
                vis_dict['image/0_original'] = vis_img
            # except Exception as e:
            #     self.msg_mgr.log_warning(f'Visualization Error: {e}')
            #     import traceback
            #     traceback.print_exc()

            # try:
                # 2. 准备 Attention Maps
                import matplotlib.cm as cm  # 引入 colormap
                # all_attn_maps: List of [n, s, Layers(6), Heads(8), h, w]
                # 我们需要拼接时间维度 S
                map_total_layers = torch.cat(all_attn_maps, dim=1) # [n, S_total, L, P, h, w]
                
                # 取第一个样本(n=0)，第一帧(S=0) -> [L, P, h, w]
                vis_layers_data = map_total_layers[0, 0].detach().cpu()
                
                # Layer Loop (L=6)
                num_layers = vis_layers_data.shape[0]

                # show all parts
                parts_to_show = list(range(vis_layers_data.shape[1]))
                
                # 3. 遍历每一层 (Layer Loop)
                for l_idx in range(num_layers):
                    
                    for pid in parts_to_show:
                        # 获取 raw attention map [h, w]
                        att = vis_layers_data[l_idx, pid] 
                        
                        # 插值到原图大小 [H, W]
                        att = F.interpolate(att[None, None], size=vis_img.shape[1:], mode='bilinear')[0,0]
                        
                        # 🌟 归一化 (对于 Jet 可视化非常重要)
                        # 将值映射到 0-1 之间，以便应用 colormap
                        att_norm = (att - att.min()) / (att.max() - att.min() + 1e-6)
                        
                        # 🌟 应用 Jet Colormap
                        # cm.jet(x) 返回 [H, W, 4] (RGBA)，我们需要前3个通道 (RGB)
                        # numpy -> tensor [3, H, W]
                        heatmap_np = cm.jet(att_norm.numpy())[..., :3]
                        heatmap = torch.from_numpy(heatmap_np).permute(2, 0, 1).float()
                        
                        # 叠加 (Overlay): 原图 0.4 + 热力图 0.6 (让热力图更明显一点)
                        overlay = vis_img * 0.4 + heatmap * 0.6
                        
                        # 命名格式: layer_{层号}/part_{部位号}
                        key_name = f'image/layer_{l_idx:02d}part_{pid:02d}'
                        vis_dict[key_name] = overlay.clamp(0, 1)

            # except Exception as e:
            #     self.msg_mgr.log_warning(f'Visualization Error: {e}')
            #     import traceback
            #     traceback.print_exc()

            # # =======================================================
            # # 新增: 3D Mesh 可视化渲染
            # # =======================================================
            # try:
                import matplotlib.pyplot as plt
                plt.switch_backend('Agg')
                import numpy as np
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                # 1. 获取第一个样本的数据
                # Vertices: [18439, 3] (注意: mhr_forward 内部可能已经做了坐标系翻转，通常 Y 轴向下)
                mesh_verts = vis_pose_output["pred_vertices"][0].detach().cpu().numpy()

                # [B, 3]
                cam_t = vis_pose_output["pred_cam_t"][0].detach().cpu().numpy()
                mesh_verts = mesh_verts + cam_t

                # 2. 创建 Matplotlib 画布
                fig = plt.figure(figsize=(4, 4), dpi=100)
                ax = fig.add_subplot(111, projection='3d')

                # 3. 渲染 (推荐使用散点图 scatter，速度快且不易崩)
                # 降采样：每 5 个点画 1 个
                step = 5 
                # c=mesh_verts[::step, 2]: 根据深度(Z轴)设置颜色深浅，增加立体感
                ax.scatter(mesh_verts[::step, 0], mesh_verts[::step, 1], mesh_verts[::step, 2], 
                           s=1, c=mesh_verts[::step, 2], cmap='viridis', alpha=0.5)

                # 4. 强制坐标轴比例一致 (Fix Aspect Ratio)
                # 否则Matplotlib 会自动拉伸坐标轴，导致人变形
                x_min, x_max = mesh_verts[:, 0].min(), mesh_verts[:, 0].max()
                y_min, y_max = mesh_verts[:, 1].min(), mesh_verts[:, 1].max()
                z_min, z_max = mesh_verts[:, 2].min(), mesh_verts[:, 2].max()

                max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0

                mid_x = (x_max + x_min) * 0.5
                mid_y = (y_max + y_min) * 0.5
                mid_z = (z_max + z_min) * 0.5

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                # 5. 设置视角
                # elev=-90, azim=-90 是为了配合 vert[..., 1,2] *= -1 后的坐标系
                # 如果你发现人是倒着的，可以尝试改成 elev=90
                ax.view_init(elev=-90, azim=-90) 
                ax.set_axis_off() # 隐藏坐标轴

                # 6. Canvas 转 Tensor (修复版 API)
                fig.canvas.draw()
                
                # 使用 buffer_rgba 获取数据
                buf = np.array(fig.canvas.renderer.buffer_rgba())
                
                # 去掉 Alpha 通道并归一化 [H, W, 4] -> [3, H, W]
                mesh_img_tensor = torch.from_numpy(buf[..., :3]).float().permute(2, 0, 1) / 255.0
                
                vis_dict['image/3d_mesh_preview'] = mesh_img_tensor
                
                plt.close(fig)

            # except Exception as e:
            #     import traceback
            #     err_msg = traceback.format_exc()
            #     self.msg_mgr.log_warning(f'Mesh Vis Error:\n{err_msg}')

        # 组装返回值
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': vis_dict,
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