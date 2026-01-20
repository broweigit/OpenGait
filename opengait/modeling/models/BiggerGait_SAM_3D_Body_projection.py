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

def compute_pca_rgb(features):
    """
    简易 PCA 将高维特征降维到 3 RGB 通道用于可视化
    Args:
        features: [N, C] or [H, W, C]
    Returns:
        rgb: [N, 3] or [H, W, 3] (Normalized 0-1)
    """
    shape = features.shape
    x = features.reshape(-1, shape[-1]) # [N_pixels, C]
    
    # 1. Center data
    mean = torch.mean(x, dim=0)
    x_centered = x - mean
    
    # 2. PCA using SVD (use lowrank for speed on GPU)
    # U, S, V = torch.pca_lowrank(x_centered, q=3, center=False, niter=2)
    try:
        _, _, V = torch.linalg.svd(x_centered.float(), full_matrices=False)
        components = V[:3] # [3, C]
        projected = torch.matmul(x_centered.float(), components.T) # [N, 3]
    except:
        # Fallback if SVD fails
        projected = x_centered[:, :3]

    # 3. Normalize to [0, 1] for RGB
    p_min = projected.min(dim=0, keepdim=True)[0]
    p_max = projected.max(dim=0, keepdim=True)[0]
    rgb = (projected - p_min) / (p_max - p_min + 1e-6)
    
    return rgb.reshape(*shape[:-1], 3)

# =========================================================================
# Main Model: BiggerGait with SAM 3D Body (DINOv3)
# =========================================================================

class BiggerGait__SAM3DBody__Projection_Gaitbase_Share(BaseModel):
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
        assert 32 % self.num_FPN == 0, "num_FPN must divide layer count 32."

        total_hooked_layers = sum(self.hook_mask) # 16
        assert total_hooked_layers > 0, "At least one layer must be hooked."

        human_conv_input_dim = self.f4_dim * (total_hooked_layers // self.num_FPN) # 1280 * 4 = 5120

        self.trans_layer_mask = layer_cfg["trans_layer_mask"] if "trans_layer_mask" in layer_cfg else [True]*6
        assert len(self.trans_layer_mask) == 6, "trans_layer_mask length must be 6."
        self.total_attn_layers = self.trans_layer_mask.count(True)

        self.num_parts = model_cfg["num_parts"] # 水平硬切分

        # A. Hard Stream Modules (8 Parts)
        self.Gait_Nets = nn.ModuleList([
            Baseline_ShareTime_2B(model_cfg) for _ in range(self.num_parts)
        ])

        part_target_h = (self.sils_size * 2) // self.num_parts # e.g. 64 // 8 = 8
        part_target_w = self.sils_size # 32

        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(human_conv_input_dim, affine=False),
                nn.Conv2d(human_conv_input_dim, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                ResizeToHW((part_target_h, part_target_w)), # 强制统一尺寸
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
    
    def get_vertex_features(self, image_embeddings, vertices, cam_t, cam_int, img_size):
        """
        将 2D DINO 特征反向投影到 3D 顶点上
        Args:
            image_embeddings: [B, C, H_feat, W_feat] (DINO 特征)
            vertices: [B, N_verts, 3] (SAM Decoder 输出的顶点,通常是 Canonical Space)
            cam_t: [B, 3] (相机平移)
            cam_int: [B, 3, 3] (相机内参)
            img_size: (W, H) 原始图片尺寸，用于归一化坐标
        Returns:
            vertex_features: [B, C, N_verts]
        """
        B, N_v, _ = vertices.shape
        
        # 1. World -> Camera Space (应用平移)
        # vertices 通常是 [x, y, z]，cam_t 也是 [tx, ty, tz]
        v_cam = vertices + cam_t.unsqueeze(1) # [B, N_verts, 3]
        
        # 2. Camera -> Image Plane (透视投影)
        # u = fx * (x/z) + cx
        # v = fy * (y/z) + cy
        x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
        
        # 避免除以 0
        z = z.clamp(min=1e-3)
        
        fx = cam_int[:, 0, 0].unsqueeze(1)
        fy = cam_int[:, 1, 1].unsqueeze(1)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
        
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy
        
        # 3. Normalize to [-1, 1] for grid_sample
        # grid_sample 要求坐标范围是 [-1, 1]
        # u_norm = 2 * (u / W) - 1
        # v_norm = 2 * (v / H) - 1
        W_img, H_img = img_size
        u_norm = 2.0 * (u / W_img) - 1.0
        v_norm = 2.0 * (v / H_img) - 1.0
        
        # Stack coordinate grid [B, N_verts, 1, 2] (u, v)
        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(2)
        
        # 4. Grid Sample (双线性插值采样)
        # image_embeddings: [B, C, Hf, Wf]
        # output: [B, C, N_verts, 1]
        sampled_feats = F.grid_sample(
            image_embeddings, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', # 超出边界的点(被遮挡或在图外)填0
            align_corners=False
        )
        
        # Remove last dim -> [B, C, N_verts]
        return sampled_feats.squeeze(-1)

    def forward(self, inputs):
        # 冻结 Mask Branch
        self.Mask_Branch.eval()
        
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # 显存优化
        CHUNK_SIZE = self.chunk_size
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16 # DINOv3 Patch Size = 16

        # 存储用于可视化的数据 (只存第一个 Batch 的第一帧)
        vis_data = {
            "fpn_feats_2d": [], # List of [H, W, C]
            "fpn_feats_3d": [], # List of [N, C]
            "vertices": None,
            "cam_t": None,
            "vis_img": inputs[0][0][0, 0].detach().cpu()
        }

        has_captured_vis = False

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

                # 获取当前 Chunk 的 3D 信息
                cur_vertices = pose_outs[-1]['pred_vertices'] # [B, N, 3]
                cur_cam_t = pose_outs[-1]['pred_cam_t']
                cur_cam_int = dummy_batch['cam_int']

                # =======================================================
                # FPN Grouping & Back-Projection (按组处理)
                # =======================================================
                if not has_captured_vis:
                    vis_data["vertices"] = cur_vertices[0].detach().cpu()
                    vis_data["cam_t"] = cur_cam_t[0].detach().cpu()

                # features_to_use 包含 16 层。 num_FPN = 4。 step = 4。
                step = len(features_to_use) // self.num_FPN
                
                for i in range(self.num_FPN):
                    # 1. Construct FPN Feature Map (2D)
                    # Concat 4 layers: [B, 1280, H, W] * 4 -> [B, 5120, H, W]
                    sub_feats = features_to_use[i*step : (i+1)*step]
                    sub_app = torch.cat(sub_feats, dim=1) 
                    
                    # 2. Back-Project to 3D Vertices
                    # Input: [B, 5120, H, W], Output: [B, 5120, N]
                    # 注意: 特征图尺寸是 DINO 输出尺寸 (h_feat, w_feat)
                    vertex_feats = self.get_vertex_features(
                        sub_app, 
                        cur_vertices, 
                        cur_cam_t, 
                        cur_cam_int, 
                        (w_feat, h_feat) # 特征图对应的原始图尺寸比例
                    )
                    
                    # 3. Capture for Visualization (Only first frame)
                    if not has_captured_vis:
                        # Store 2D [H, W, 5120]
                        feat_2d_vis = sub_app[0].permute(1, 2, 0).detach().cpu()
                        vis_data["fpn_feats_2d"].append(feat_2d_vis)
                        
                        # Store 3D [N, 5120]
                        feat_3d_vis = vertex_feats[0].permute(1, 0).detach().cpu()
                        vis_data["fpn_feats_3d"].append(feat_3d_vis)

                if not has_captured_vis:
                    has_captured_vis = True
            
            # 停止后续计算，我们只做可视化
            break # 如果不需要遍历所有 chunk 也可以 break

        # -------------------------------------------------------
        # Visualization TODO: 在这里可视化PCA降维
        # -------------------------------------------------------
        vis_dict = {}
        if self.training and torch.distributed.get_rank() == 0:
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')
            import numpy as np
            
            # 0. 原始图像
            vis_img = vis_data["vis_img"]
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-6)
            vis_dict['vis/00_original'] = vis_img
            
            # Loop over FPN groups
            for idx in range(self.num_FPN):
                # --- 2D Visualization (PCA -> RGB) ---
                feat_2d = vis_data["fpn_feats_2d"][idx] # [H, W, 5120]
                rgb_2d = compute_pca_rgb(feat_2d) # [H, W, 3]
                
                # Upsample to match original image size for display
                rgb_2d_tensor = rgb_2d.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
                rgb_2d_resized = F.interpolate(rgb_2d_tensor, size=vis_img.shape[1:], mode='nearest')[0]
                vis_dict[f'vis/fpn_{idx}_2d_pca'] = rgb_2d_resized
                
                # --- 3D Visualization (PCA -> Colored Point Cloud) ---
                feat_3d = vis_data["fpn_feats_3d"][idx] # [N, 5120]
                rgb_3d = compute_pca_rgb(feat_3d).numpy() # [N, 3]
                
                # Prepare Mesh Data
                mesh_verts = vis_data["vertices"].numpy() # [N, 3]
                cam_t_val = vis_data["cam_t"].numpy()
                mesh_verts_world = mesh_verts + cam_t_val
                
                # Plot
                fig = plt.figure(figsize=(4, 4), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                
                step = 2 # 采样步长
                # c argument takes RGB values [N, 3]
                ax.scatter(
                    mesh_verts_world[::step, 0], 
                    mesh_verts_world[::step, 1], 
                    mesh_verts_world[::step, 2],
                    s=2, 
                    c=rgb_3d[::step], # Apply PCA Colors here!
                    alpha=0.8
                )
                
                # Fix Aspect Ratio
                x_min, x_max = mesh_verts_world[:, 0].min(), mesh_verts_world[:, 0].max()
                y_min, y_max = mesh_verts_world[:, 1].min(), mesh_verts_world[:, 1].max()
                z_min, z_max = mesh_verts_world[:, 2].min(), mesh_verts_world[:, 2].max()
                max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
                mid_x, mid_y, mid_z = (x_max+x_min)*0.5, (y_max+y_min)*0.5, (z_max+z_min)*0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.view_init(elev=-90, azim=-90)
                ax.set_axis_off()
                
                fig.canvas.draw()
                buf = np.array(fig.canvas.renderer.buffer_rgba())
                mesh_img_tensor = torch.from_numpy(buf[..., :3]).float().permute(2, 0, 1) / 255.0
                vis_dict[f'vis/fpn_{idx}_3d_pca'] = mesh_img_tensor
                plt.close(fig)

        # Return Zeros for Training
        # 构造假的返回值以防止 Crash
        B_total = rgb.shape[0] * rgb.shape[1] # n * s
        # dummy_embed: [n, 256, p] -> zero
        dummy_embed = torch.zeros(inputs[1].shape[0], 256, self.num_FPN * self.num_parts).to(rgb.device)
        dummy_logits = torch.zeros(inputs[1].shape[0], self.num_unknown, self.num_FPN * self.num_parts).to(rgb.device)

        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': dummy_embed, 'labels': labs},
                    'softmax': {'logits': dummy_logits, 'labels': labs},
                },
                'visual_summary': vis_dict,
                'inference_feat': {
                    'embeddings': dummy_embed,
                }
            }
        else:
            retval = {
                'inference_feat': {'embeddings': dummy_embed}
            }
        
        return retval