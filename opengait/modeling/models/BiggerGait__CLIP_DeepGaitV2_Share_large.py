# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
from kornia import morphology as morph
import random

# import GaitBase & DINOv2_small
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.DINOv2 import vit_small, vit_large, vit_giant
# from .BigGait_utils.DINOv2_Prompt import vit_small, vit_large, vit_giant
from .BigGait_utils.save_img import save_image, pca_image
from ..modules import GaitAlign

from functools import partial

def gradient_hook(grad, name, step, log):
    if torch.distributed.get_rank() == 0 and step % 100 == 0:
        log.log_info('[{}] Gradient={:.6f}'.format(step, grad.abs().mean().item()))
        log.log_info('[{}] :384 Gradient={:.6f}'.format(step, grad[:,:96].abs().mean().item()))
        log.log_info('[{}] 384:768 Gradient={:.6f}'.format(step, grad[:,96:192].abs().mean().item()))
        log.log_info('[{}] 768:1152 Gradient={:.6f}'.format(step, grad[:,192:288].abs().mean().item()))
        log.log_info('[{}] -384: Gradient={:.6f}'.format(step, grad[:,-96:].abs().mean().item()))
    return grad

# ######################################## CKA ###########################################
def center_masked_kernel(K, mask_flat):
    """
    K:        [N, HW, HW] 原始 Gram 矩阵
    mask_flat:[N, HW]     空间位置掩码 值为0或1
    Return:   [N, HW, HW] 中心化后的 Gram 矩阵
    """
    N, HW, _ = K.shape

    # 有效位置数 M [N, 1, 1]
    M = mask_flat.sum(dim=1, keepdim=True).unsqueeze(2).float()
    M = torch.where(M == 0, torch.ones_like(M), M)  # 防止除零

    # 均值计算
    sum_rows = torch.sum(K, dim=2, keepdim=True)
    sum_cols = torch.sum(K, dim=1, keepdim=True)
    row_means = (sum_rows / M) * mask_flat.unsqueeze(2)
    col_means = (sum_cols / M) * mask_flat.unsqueeze(1)
    total_mean = torch.sum(K, dim=(1, 2), keepdim=True) / (M ** 2)

    # 双重中心化
    K_centered = K - row_means - col_means + total_mean

    # 应用掩码
    mask_matrix = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)
    return K_centered * mask_matrix.float()

def linear_CKA_2d(F1, F2, mask=None, eps=1e-5):
    """
    F1/F2: [N, C, H, W] 输入特征
    mask:  [N, 1, H, W] 0/1 掩码，可为 None 表示使用全图
    Return: CKA 相似度标量
    """
    N, C1, H, W = F1.shape
    _, C2, _, _ = F2.shape
    HW = H * W

    # reshape
    X = F1.view(N, C1, HW).transpose(1, 2)  # [N, HW, C1]
    Y = F2.view(N, C2, HW).transpose(1, 2)  # [N, HW, C2]

    # 处理掩码
    if mask is None:
        mask_flat = torch.ones((N, HW), dtype=torch.float32, device=F1.device)
    else:
        mask_flat = mask.view(N, HW).float()
        assert torch.all((mask_flat == 0) | (mask_flat == 1)), "mask must be 0/1 only"
    mask_matrix = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)  # [N, HW, HW]
    
    # 通道标准化 (避免数值过大)
    X = (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + eps)
    Y = (Y - Y.mean(dim=2, keepdim=True)) / (Y.std(dim=2, keepdim=True) + eps)

    # 计算缩放后的 Gram 矩阵
    K = torch.bmm(X, X.transpose(1, 2)) / C1  # [N, HW, HW]
    L = torch.bmm(Y, Y.transpose(1, 2)) / C2
    K, L = K * mask_matrix, L * mask_matrix

    # 中心化
    Kc = center_masked_kernel(K, mask_flat)
    Lc = center_masked_kernel(L, mask_flat)

    # 计算 CKA 分数
    hsic = (Kc * Lc).sum(dim=(1, 2))
    norm_K = (Kc ** 2).sum(dim=(1, 2))
    norm_L = (Lc ** 2).sum(dim=(1, 2))
    # if norm_K * norm_L <=0:
    if (norm_K * norm_L < 0).any():
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    if torch.isnan(norm_K).any() or torch.isnan(norm_L).any():
        print("NaN detected in norm_K or norm_L")
        print("norm_K:", norm_K)
        print("norm_L:", norm_L)

    cka = hsic / (torch.sqrt(norm_K * norm_L) + eps)
    
    # 处理全0掩码样本
    valid = mask_flat.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=F1.device)

    return (cka * valid.float()).sum() / valid.sum().float()

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
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                return d_x, None
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


# ######################################## Affine ###########################################

# 0 Background; 1 Hand; 2 Head; 3 Leg; 4 Body; 5 Feet; 6 Shoulder;
def find_channel_centroid(parsing_map, channel_idx):
    """计算指定通道的质心坐标"""
    # 提取目标通道并二值化 [N, H, W]
    mask = (parsing_map[:, channel_idx] > 0.5).float()
    
    # 生成坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(parsing_map.size(2), device=parsing_map.device),
        torch.arange(parsing_map.size(3), device=parsing_map.device),
        indexing='ij'
    )
    
    # 批量计算加权坐标 [N, 2]
    total = mask.sum(dim=(1,2)) + 1e-6  # 防零除
    centroid_x = (mask * x_coords).sum(dim=(1,2)) / total
    centroid_y = (mask * y_coords).sum(dim=(1,2)) / total
    
    return torch.stack([centroid_x, centroid_y], dim=1)  # [N, 2]


def calculate_rotation_angle(p1, p2):
    """计算两点间旋转角度（弧度）"""
    delta = p2 - p1
    angle = torch.atan2(delta[:,0], delta[:,1])  # atan2(dx, dy)
    return angle  # [N]


def rotate_feature_maps(feature_maps, angle):
    """执行旋转操作"""
    N, _, H, W = feature_maps.shape
    # angle_deg = torch.rad2deg(angle)
    
    # 构建旋转矩阵
    rotation_matrix = torch.zeros(N, 2, 3, device=feature_maps.device)
    rotation_matrix[:,0,0] = rotation_matrix[:,1,1] = torch.cos(angle)
    rotation_matrix[:,0,1] = -torch.sin(angle)
    rotation_matrix[:,1,0] = torch.sin(angle)
    
    # 生成仿射网格
    grid = F.affine_grid(rotation_matrix, [N, 1, H, W], align_corners=False)
    
    # 执行双线性插值
    rotated = F.grid_sample(
        feature_maps.float(), grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=False
    )
    
    return rotated


# 0 Background; 1 Hand; 2 Head; 3 Leg; 4 Body; 5 Feet; 6 Shoulder;
# align_human_vertical(parsing_maps=parsing_feat, feature_maps=parsing_feat, mask=mask, align_fn=self.Gait_Align, ratios=ratios)
def align_human_vertical(parsing_maps, feature_maps, mask, align_fn, ratios):
    """人体垂直对齐主函数"""
    # 输入参数检查
    assert parsing_maps.dim() == 4, "Input must be NCHW tensor"
    
    # Step 1: 获取关键点
    # upper_centers = (find_channel_centroid(parsing_maps, 2) + find_channel_centroid(parsing_maps, 4)) / 2  # Head + Body
    upper_centers = find_channel_centroid(parsing_maps, 4)  # Body
    lower_centers = find_channel_centroid(parsing_maps, 3)  # Leg
    
    # Step 2: 有效性验证
    valid_mask = ~torch.isnan(upper_centers).any(1) & ~torch.isnan(lower_centers).any(1)
    
    # Step 3: 计算旋转角度
    rotation_angles = torch.zeros(parsing_maps.size(0), device=parsing_maps.device)
    rotation_angles[valid_mask] = -calculate_rotation_angle(
        upper_centers[valid_mask], 
        lower_centers[valid_mask]
    )
    
    feature_maps = F.pad(feature_maps, (10, 10, 20, 20), mode='constant', value=0)
    mask = F.pad(mask, (10, 10, 20, 20), mode='constant', value=0)
    feature_maps = F.interpolate(feature_maps, scale_factor=2, mode='bilinear')
    mask = F.interpolate(mask, scale_factor=2, mode='bilinear')

    feature_maps = rotate_feature_maps(feature_maps, rotation_angles)
    mask = rotate_feature_maps(mask, rotation_angles)
    feature_maps = F.interpolate(feature_maps, scale_factor=0.5, mode='bilinear')
    mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')

    feature_maps = align_fn(feature_maps, mask, ratios) # [n, c, h, w]
    mask = align_fn(mask, mask, ratios) # [n, c, h, w]
    return feature_maps, mask

# ######################################## BigGait ###########################################

def padding_resize(x, ratios, target_h, target_w):
    n,h,w = x.size(0),target_h, target_w
    ratios = ratios.view(-1)
    need_w = (h * ratios).int()
    need_padding_mask = need_w < w
    pad_left = torch.where(need_padding_mask, (w - need_w) // 2, torch.tensor(0).to(x.device))
    pad_right = torch.where(need_padding_mask, w - need_w - pad_left, torch.tensor(0).to(x.device)).tolist()
    need_w = need_w.tolist()
    pad_left = pad_left.tolist()
    x = torch.concat([F.pad(F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False), (pad_left[i], pad_right[i]))  if need_padding_mask[i] else F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False)[...,pad_left[i]:pad_left[i]+w]  for i in range(n)], dim=0)
    return x

class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)


from transformers import CLIPVisionModel, CLIPVisionConfig
import torch
class BiggerGait__CLIPL_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]

        # set feature dim
        self.f4_dim = model_cfg['source_dim']
        self.fc_dim = self.f4_dim*4
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]

        # init submodules
        # self.Gait_Net = Baseline_FPN(model_cfg)
        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)
        # self.Gait_Net = Baseline_ShareTime(model_cfg)

        self.Pre_Conv = nn.Sequential(
            nn.Identity(),
        )

        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.f4_dim*4, affine=False),
                nn.Conv2d(self.f4_dim*4, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                # nn.Upsample(scale_factor=(self.sils_size // (self.image_size // 14)), mode='bilinear', align_corners=False),
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


    def init_CLIP(self):
        config = CLIPVisionConfig.from_pretrained( self.pretrained_lvm + "/config.json")
        config.output_hidden_states = True
        self.Backbone = CLIPVisionModel.from_pretrained(self.pretrained_lvm, config=config)
        self.msg_mgr.log_info(f'load model from: {self.pretrained_lvm}')

    def init_Mask_Branch(self):
        self.msg_mgr.log_info(f'load model from: {self.pretrained_mask_branch}')
        load_dict = torch.load(self.pretrained_mask_branch, map_location=torch.device("cpu"))['model']
        msg = self.Mask_Branch.load_state_dict(load_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.Mask_Branch.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_mask_branch}'")
        self.msg_mgr.log_info('SegmentationBranch Count: {:.5f}M'.format(n_parameters / 1e6))

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

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('Expect Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        
        self.init_CLIP()
        self.init_Mask_Branch()
        
        # # Cal GFlops
        if self.training:
            from fvcore.nn import FlopCountAnalysis
            self.eval()
            with torch.no_grad():
                device = torch.distributed.get_rank()
                inputs = ([[torch.randn((1,1,3,448,224),dtype=torch.float32).to(device), torch.rand(1,dtype=torch.float32).to(device)], None, None, None, None],)
                flops = FlopCountAnalysis(self.to(device), inputs).total()  / 1e9   # GFLOPs 
                # self.msg_mgr.log_info(f"{flops:.2f} GFLOPs")
            self.train()
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)
        
        n_parameters = sum(p.numel() for p in self.parameters())
        if self.training:
            self.msg_mgr.log_info('All Backbone Count: {:.5f}M, {:.2f} GFLOPs'.format(n_parameters / 1e6, flops))
        else:
            self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
            
        self.msg_mgr.log_info("=> init successfully")

    # resize image
    def preprocess(self, sils, h, w, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def get_synced_r(self):
        if self.training:
            device = torch.device("cuda")
            if torch.distributed.get_rank() == 0:
                r_tensor = torch.clamp(torch.rand(1, device=device) * 2 - 0.5, min=0, max=1)
            else:
                r_tensor = torch.tensor([0.0], device=device)
            torch.distributed.broadcast(r_tensor, src=0)
            r = r_tensor.item()
        else:
            r = 0.5  # 非训练模式使用固定值
        return r

# ############################# For Mask Train ##############################

    # def forward(self, inputs):
    #     if self.training and self.iteration<=500:
    #         self.Mask_Branch.train()
    #         self.Mask_Branch.requires_grad_(True)
    #     else:
    #         self.Mask_Branch.eval()
    #         self.Mask_Branch.requires_grad_(False)

    #     ipts, labs, ty, vi, seqL = inputs
    #     rgb, ratios = ipts[0], ipts[1] # CCPG
    #     # rgb, ratios = ipts[1], ipts[0] # CASIA-B, SUSTech1K
    #     del ipts

    #     rgb_chunks = torch.chunk(rgb, (rgb.size(1)//96)+1, dim=1)
    #     all_outs = []
    #     loss_mse_list = []
    #     for _, rgb_img in enumerate(rgb_chunks):
    #         with torch.no_grad():
    #             # get RGB
    #             n,s,c,h,w = rgb_img.size()
    #             rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
    #             outs = self.preprocess(rgb_img, self.image_size, self.image_size)
    #             # outs = self.Backbone(outs, is_training=True) # [ns,h*w,c]

    #             outputs = self.Backbone(pixel_values=outs)
    #             hidden_states = outputs.hidden_states  # 是一个 tuple，包含所有层

    #             outs_last1 = hidden_states[-1][:, 1:].contiguous() # [ns,h*w,c]
    #             outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) (h w) c').contiguous()
    #             outs_last1 = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs_last1)
    #             del hidden_states
            
    #         human_mask_ori, loss_mse = self.Mask_Branch(outs_last1.view(-1, self.f4_dim))
    #         loss_mse_list.append(loss_mse)
    #         human_mask_ori = (human_mask_ori[:,1] > 0.5).float()
    #         human_mask_ori = human_mask_ori.view(n*s, 1, self.image_size//14, self.image_size//14)
    #         human_mask_ori = self.preprocess(human_mask_ori, self.sils_size*2, self.sils_size).detach().clone()
    #         del outs_last1
        
    #     if self.training:
    #         retval = {
    #             'training_feat': {
    #                 'shape_mse': F.relu(torch.stack(loss_mse_list).mean() - 0.7),
    #             },
    #             'visual_summary': {
    #                 'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
    #                 'image/human_mask': self.min_max_norm(human_mask_ori.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()).clamp(0,1),
    #             },
    #         }
    #     else:
    #         retval = {
    #             'training_feat': {},
    #             'visual_summary': {},
    #         }
    #     return retval

# # ############################# For Train ##############################

    def forward(self, inputs):
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)

        ipts, labs, ty, vi, seqL = inputs
        # rgb, ratios = ipts[0], ipts[1] # CCPG
        # rgb, ratios = ipts[1], ipts[0] # CASIA-B, SUSTech1K
        rgb = ipts[0]# CCGR
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//10)+1, dim=1)
        all_outs = []
        # loss_mse_list = []
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                # get RGB
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                outs = self.preprocess(rgb_img, self.image_size, self.image_size)
                # outs = self.Backbone(outs, is_training=True) # [ns,h*w,c]

                outputs = self.Backbone(pixel_values=outs)
                hidden_states = outputs.hidden_states  # 是一个 tuple，包含所有层

                outs_last1 = hidden_states[-1][:, 1:].contiguous() # [ns,h*w,c]
                outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) (h w) c').contiguous()
                outs_last1 = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs_last1)
                
                # Pre_Conv
                appearance = torch.concat(hidden_states[1:], dim=-1)[:, 1:].contiguous() # [ns,h*w,c]
                appearance = rearrange(appearance.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                appearance = self.Pre_Conv(appearance) # n,384,h,w
                appearance = rearrange(appearance, 'n c h w -> n (h w) c').contiguous()
                # appearance = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*self.num_FPN, elementwise_affine=False)(appearance)
                appearance = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*24, elementwise_affine=False)(appearance)
                appearance = rearrange(appearance.view(n, s, self.image_size//14, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                del hidden_states
            
                human_mask_ori, _ = self.Mask_Branch(outs_last1.view(-1, self.f4_dim))
                human_mask_ori = (human_mask_ori[:,1] > 0.5).float()
                human_mask_ori = human_mask_ori.view(n*s, 1, self.image_size//14, self.image_size//14)
                human_mask_ori = self.preprocess(human_mask_ori, self.sils_size*2, self.sils_size).detach().clone()
                del outs_last1

            human_feat = list(torch.chunk(appearance, 24, dim=1))
            human_feat = [torch.cat(human_feat[i:i+4], dim=1) for i in range(0, 24, 4)] # 6G
            for i in range(self.num_FPN):
                human_feat[i] = self.HumanSpace_Conv[i](human_feat[i])
            human_feat = torch.concat(human_feat, dim=1)

            # human_feat = list(torch.chunk(appearance, self.num_FPN, dim=1))
            # for i in range(self.num_FPN):
            #     human_feat[i] = self.HumanSpace_Conv[i](human_feat[i])
            # human_feat = torch.concat(human_feat, dim=1)

            # human_feat = list(torch.chunk(appearance, 24, dim=1))
            # human_feat = human_feat[1::2]

            # # human_feat = human_feat[-7:-1]
            # # human_feat = human_feat[:]
            # t = torch.tensor(list(range(self.num_FPN))).to(appearance).view(1,-1).repeat(n*s,1)
            # for i in range(self.num_FPN):
            #     temb = get_timestep_embedding(t[:,i], self.t_channel, max_timesteps=self.num_FPN)
            #     temb = self.temb_proj(temb)
            #     human_feat[i] = self.HumanSpace_Conv[i](human_feat[i] + temb[:,:,None,None])
            # human_feat = torch.concat(human_feat, dim=1)

            # Human Space Align:
            # ratios = torch.ones((n*s), device=human_feat.device, dtype=human_feat.dtype) / 2.0
            # human_feat, human_mask = align_human_vertical(parsing_maps=parsing_feat, feature_maps=human_feat, mask=human_mask_ori.float().clone(), align_fn=self.Gait_Align, ratios=ratios)
            human_mask = human_mask_ori
            human_feat = human_feat * (human_mask > 0.5).float()
            human_feat = rearrange(human_feat.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w').contiguous()

            # get embeding
            outs = self.Gait_Net.test_1(human_feat)
            all_outs.append(outs)

        # # vis
        # if self.training:
        #     vis_num = min(5, n*s)
        #     human_feat = rearrange(human_feat, 'n c s h w -> (n s) (h w) c').contiguous().detach()
        #     human_feat = torch.chunk(human_feat, 4, dim=-1)
        #     try:
        #         # foreground = human_mask.view(n*s, self.sils_size*2*self.sils_size, 1)
        #         foreground = torch.ones_like(human_mask).to(human_mask).view(n*s, self.sils_size*2*self.sils_size, 1)
        #         vis_mask = foreground.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()
        #         vis_human = pca_image(data={'embeddings':torch.concat(human_feat,dim=-1).view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True, img_h=self.sils_size*2, img_w=self.sils_size) # n s c h w
        #         vis_human_0 = pca_image(data={'embeddings':human_feat[0].view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True, img_h=self.sils_size*2, img_w=self.sils_size) # n s c h w
        #         vis_human_1 = pca_image(data={'embeddings':human_feat[1].view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True, img_h=self.sils_size*2, img_w=self.sils_size) # n s c h w
        #         vis_human_2 = pca_image(data={'embeddings':human_feat[2].view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True, img_h=self.sils_size*2, img_w=self.sils_size) # n s c h w
        #         vis_human_3 = pca_image(data={'embeddings':human_feat[3].view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True, img_h=self.sils_size*2, img_w=self.sils_size) # n s c h w
        #     except:
        #         vis_human = torch.ones_like(foreground).view(vis_num,1,1,self.sils_size*2, self.sils_size).detach().cpu().numpy()
        #         vis_human_0 = torch.ones_like(foreground).view(vis_num,1,1,self.sils_size*2, self.sils_size).detach().cpu().numpy()
        #         vis_human_1 = torch.ones_like(foreground).view(vis_num,1,1,self.sils_size*2, self.sils_size).detach().cpu().numpy()
        #         vis_human_2 = torch.ones_like(foreground).view(vis_num,1,1,self.sils_size*2, self.sils_size).detach().cpu().numpy()
        #         vis_human_3 = torch.ones_like(foreground).view(vis_num,1,1,self.sils_size*2, self.sils_size).detach().cpu().numpy()

        # get embeding
        embed_list, log_list = self.Gait_Net.test_2(
                                        torch.cat(all_outs, dim=2),
                                        seqL,
                                        )
        
        if self.training:
            retval = {
                'training_feat': {
                    # 'shape_mse': torch.stack(loss_mse_list).mean(),
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask_ori.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()).clamp(0,1),
                    # 'image/human_feat': self.min_max_norm(rearrange(torch.from_numpy(vis_human).float(), 'n s c h w -> (n s) c h w').contiguous()),
                    # 'image/human_feat_0': self.min_max_norm(rearrange(torch.from_numpy(vis_human_0).float(), 'n s c h w -> (n s) c h w').contiguous()),
                    # 'image/human_feat_1': self.min_max_norm(rearrange(torch.from_numpy(vis_human_1).float(), 'n s c h w -> (n s) c h w').contiguous()),
                    # 'image/human_feat_2': self.min_max_norm(rearrange(torch.from_numpy(vis_human_2).float(), 'n s c h w -> (n s) c h w').contiguous()),
                    # 'image/human_feat_3': self.min_max_norm(rearrange(torch.from_numpy(vis_human_3).float(), 'n s c h w -> (n s) c h w').contiguous()),
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
