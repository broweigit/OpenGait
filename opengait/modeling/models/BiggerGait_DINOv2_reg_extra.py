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

from .BigGait_utils.BigGait_GaitBase_reg import *
from .BigGait_utils.save_img import save_image, pca_image
from functools import partial

# ######################################## BiggerGait ###########################################

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        self.down_sampling = nn.Linear(source_dim, target_dim)
        self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()

    def forward(self, x, mse=True):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        d_x = F.softmax(d_x, dim=1)
        if mse:
            u_x = self.up_sampling(d_x)
            return d_x, torch.mean(self.mse(u_x, x))
        else:
            return d_x, None

class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
    
class RegisterContextInjection(nn.Module):
    """
    使用 Cross Attention 将 Register 信息注入到 Spatial 特征中 (带位置编码版)
    """
    def __init__(self, dim, height=64, width=32, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # [新增] 可学习的位置编码
        # 形状与 Spatial Feature 对应 [1, C, H, W]
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, height, width))
        
        # 线性映射
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        
        # 零初始化的门控系数
        self.gamma = nn.Parameter(torch.zeros(1))

        # 初始化位置编码 (使用截断正态分布，数值很小，不影响 Zero-Init 策略)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_spatial, x_reg):
        """
        x_spatial: [N, C, H, W]
        x_reg:     [N, C, R, 1]
        """
        B, C, H, W = x_spatial.shape
        _, _, R, _ = x_reg.shape

        # [新增] 注入位置信息
        # 这样 Query 就包含了 "我是什么内容" + "我在什么位置"
        x_spatial_with_pos = x_spatial + self.pos_embed

        # 1. 生成 Q, K, V
        # Q 来自带位置信息的 Spatial
        q = self.to_q(x_spatial_with_pos).view(B, self.num_heads, C // self.num_heads, H * W)
        
        # K, V 来自 Register (Register 没有位置概念，不需要加 PE)
        k = self.to_k(x_reg).view(B, self.num_heads, C // self.num_heads, R)
        v = self.to_v(x_reg).view(B, self.num_heads, C // self.num_heads, R)
        
        # 2. Attention 计算 (Q @ K.T)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 3. 聚合信息 (Attn @ V)
        out = (attn @ v.transpose(-2, -1))
        
        # 4. 还原形状
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        
        # 5. 投影
        out = self.proj(out)
        
        # Result = Spatial + (gamma * Context)
        # 注意：这里加回的是原始 x_spatial，而不是带 pos 的，保持残差纯净
        return x_spatial + self.gamma * out

class BiggerGait__DINOv2_Reg_Extra(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]

        # set feature dim
        self.f4_dim = model_cfg['source_dim']
        self.num_unknown = model_cfg["num_unknown"]

        # set layer / group / gait_head number
        self.total_layer_num = model_cfg["total_layer_num"] # total layer number is 12
        self.group_layer_num = model_cfg["group_layer_num"] # each group have 2 layers
        self.head_num = model_cfg["head_num"] # 2 gait heads
        assert self.total_layer_num % self.group_layer_num == 0
        assert (self.total_layer_num // self.group_layer_num) % self.head_num == 0
        self.num_FPN = self.total_layer_num // self.group_layer_num

        self.Gait_Net = Baseline_Share(model_cfg)

        total_dim = self.num_unknown * self.num_FPN
        
        # [修改] 实例化时传入 H, W
        # sils_size 通常是 32 (宽度)，高度通常是 64 (sils_size * 2)
        # 你的 ResizeToHW 写的是 (self.sils_size*2, self.sils_size)
        self.Reg_Injector = RegisterContextInjection(
            dim=total_dim, 
            height=self.sils_size * 2, 
            width=self.sils_size
        )

        # [1] Spatial 专用卷积 (保持不变)
        self.HumanSpace_Conv = nn.ModuleList([ 
                nn.Sequential(
                    nn.BatchNorm2d(self.f4_dim*self.group_layer_num, affine=False),
                    nn.Conv2d(self.f4_dim*self.group_layer_num, self.f4_dim//2, kernel_size=1),
                    nn.BatchNorm2d(self.f4_dim//2, affine=False),
                    nn.GELU(),
                    nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                    ResizeToHW((self.sils_size*2, self.sils_size)), # Spatial 需要 Resize
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid()
                ) for _ in range(self.num_FPN)
            ])
        
        # [2] Register 专用卷积 (新增!)
        # 结构类似，但独立权重，且不需要 Resize
        self.Register_Conv = nn.ModuleList([ 
                nn.Sequential(
                    nn.BatchNorm2d(self.f4_dim*self.group_layer_num, affine=False),
                    nn.Conv2d(self.f4_dim*self.group_layer_num, self.f4_dim//2, kernel_size=1),
                    nn.BatchNorm2d(self.f4_dim//2, affine=False),
                    nn.GELU(),
                    nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                    # Remove ResizeToHW: Register 保持 [4, 1] 即可
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid()
                ) for _ in range(self.num_FPN)
            ])
            
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])

    def init_DINOv2(self):
        # from transformers import Dinov2Config, Dinov2Model
        # from transformers.modeling_outputs import BaseModelOutputWithPooling
        # config = Dinov2Config.from_pretrained(self.pretrained_lvm + "/config.json")
        # self.Backbone = Dinov2Model.from_pretrained(
        #     self.pretrained_lvm, 
        #     config=config,
        # )
        # TODO changed to "with-register" model
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(self.pretrained_lvm + "/config.json")
        self.Backbone = AutoModel.from_pretrained(
            self.pretrained_lvm, 
            config=config,
        )
        self.Backbone.cpu()
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
        
        self.init_DINOv2()
        self.init_Mask_Branch()
        
        # # Cal GFlops
        if self.training:
            from fvcore.nn import FlopCountAnalysis
            self.eval()
            with torch.no_grad():
                device = torch.distributed.get_rank()
                inputs = ([[torch.randn((1,1,3,448,224),dtype=torch.float32).to(device), torch.rand(1,dtype=torch.float32).to(device)], None, None, None, None],)
                # flops = FlopCountAnalysis(self.to(device), inputs).total()  / 1e9   # GFLOPs 
            self.train()
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)
        
        n_parameters = sum(p.numel() for p in self.parameters())
        # if self.training:
        #     self.msg_mgr.log_info('All Backbone Count: {:.5f}M, {:.2f} GFLOPs'.format(n_parameters / 1e6, flops))
        # else:
        #     self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
            
        self.msg_mgr.log_info("=> init successfully")

    # resize image
    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size*2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

# # ############################# For Train ##############################

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # adjust gpu
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//96)+1, dim=1)
        all_outs = []
        
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                # get RGB
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                outs = self.preprocess(rgb_img, self.image_size)
                
                # DINO Output: [ns, 517, 384] (1 CLS + 4 Reg + 512 Patch)
                outs = self.Backbone(outs,output_hidden_states=True).hidden_states[1:] 

                # ================= [1. 源头分流与重组 (关键修正)] =================
                
                spatial_with_cls_list = [] # 用于模拟原版: [CLS + Patch]
                reg_list_raw = []          # 独立的 Register
                
                num_registers = 4
                token_offset = 1 + num_registers # 5

                for layer in outs:
                    # 1. 提取 CLS
                    cls_token = layer[:, 0:1]
                    
                    # 2. 提取 Register
                    reg_tokens = layer[:, 1:5]
                    reg_tokens = reg_tokens.permute(0, 2, 1).unsqueeze(-1) # [ns, 384, 4, 1]
                    reg_list_raw.append(reg_tokens)

                    # 3. 提取 Patches
                    patches = layer[:, 5:]
                    
                    # 4. [核心操作] 拼凑出不含 Register 的组合
                    # 这样后续的 LayerNorm 就会基于 [CLS, Patch] 计算统计量
                    # 这与你跑通的那个“物理剔除版”逻辑完全一致！
                    spatial_with_cls_list.append(torch.cat([cls_token, patches], dim=1))

                # ================= [2. Spatial 分支 (复刻成功版)] =================
                
                # 2.1 Concat [CLS+Patch]
                spatial_cat = torch.cat(spatial_with_cls_list, dim=-1) # [ns, 513, 4608]
                
                # 2.2 LayerNorm (不含 Register!)
                spatial_normed = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*len(outs), elementwise_affine=False)(spatial_cat)
                
                # 2.3 去掉 CLS, 保留 Patch
                spatial_normed = spatial_normed[:, 1:] # [ns, 512, 4608]
                
                # 2.4 Chunk & Reshape
                spatial_list_tokens = list(torch.chunk(spatial_normed, self.total_layer_num, dim=-1))
                
                spatial_list = []
                for spa_tok in spatial_list_tokens:
                    spa = rearrange(spa_tok.view(n*s, self.image_size//7, self.image_size//14, self.f4_dim), 
                                    'ns h w c -> ns c h w')
                    spatial_list.append(spa)

                # 2.5 Mask Generation (同样基于 [CLS+Patch] 的 Norm 结果)
                # 这样 Mask 的生成逻辑也和成功版一致
                last_layer_spatial_with_cls = spatial_with_cls_list[-1]
                mask_normed = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(last_layer_spatial_with_cls)
                mask_in = mask_normed[:, 1:].mean(dim=1) # 去掉 CLS 后求均值
                
                human_mask, _ = self.Mask_Branch(mask_in, mse=False)
                human_mask = (human_mask[:,1] > 0.5).float() 
                human_mask = human_mask.view(n*s, 1, 1, 1) 
                human_mask = self.preprocess(human_mask, self.sils_size).detach().clone()

                # ================= [3. Register 分支 (独立 Norm)] =================
                
                # Register 数值很大，必须自己 Norm，不能污染 Spatial，也不能被 Spatial 影响
                reg_cat = torch.cat(reg_list_raw, dim=1) # [ns, 4608, 4, 1]
                
                # Permute for LayerNorm
                reg_cat = reg_cat.permute(0, 2, 3, 1) 
                reg_normed = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*len(outs), elementwise_affine=False)(reg_cat)
                reg_normed = reg_normed.permute(0, 3, 1, 2) # [ns, 4608, 4, 1]

                reg_list = list(torch.chunk(reg_normed, self.total_layer_num, dim=1))

            # ================= [4. FPN (独立卷积权重)] =================
            
            spatial_groups = [torch.cat(spatial_list[i:i+self.group_layer_num], dim=1).contiguous() for i in range(0, self.total_layer_num, self.group_layer_num)]
            reg_groups = [torch.cat(reg_list[i:i+self.group_layer_num], dim=1).contiguous() for i in range(0, self.total_layer_num, self.group_layer_num)]
            
            spatial_feats = []
            reg_feats = []

            for i in range(self.num_FPN):
                # A. Spatial 分支: 使用 HumanSpace_Conv
                spatial_feats.append(self.HumanSpace_Conv[i](spatial_groups[i]))
                
                # B. Register 分支: 使用 Register_Conv (独立权重，无 Resize)
                reg_feats.append(self.Register_Conv[i](reg_groups[i]))

            # ================= [5. 整理与 Cross Attention] =================
            
            # 5.1 整理 Spatial
            spatial_intermediates = torch.concat(spatial_feats, dim=1)
            spatial_intermediates = spatial_intermediates * (human_mask > 0.5).to(spatial_intermediates)
            spatial_intermediates = rearrange(spatial_intermediates.view(n, s, -1, self.sils_size*2, self.sils_size), 
                                            'n s c h w -> n c s h w').contiguous()

            # 5.2 整理 Register
            reg_intermediates = torch.concat(reg_feats, dim=1)
            reg_intermediates = rearrange(reg_intermediates.view(n, s, -1, 4, 1), 
                                        'n s c h w -> n c s h w').contiguous()

           # ================= [5.3 Cross Attention 注入 (修正变量覆盖Bug)] =================
            
            # spatial_intermediates: [n, c, s, h, w]
            # reg_intermediates:     [n, c, s, 4, 1]
            
            # [FIX] 使用不同的变量名，不要覆盖 RGB 图片的 n, c, h, w
            n_feat, c_feat, s_feat, h_feat, w_feat = spatial_intermediates.size()
            _, _, _, r_tokens, _ = reg_intermediates.size()

            # 1. 合并 N 和 S -> [n*s, c, h, w]
            # 使用 _feat 后缀的变量
            spa_in = spatial_intermediates.permute(0, 2, 1, 3, 4).contiguous().view(n_feat*s_feat, c_feat, h_feat, w_feat)
            reg_in = reg_intermediates.permute(0, 2, 1, 3, 4).contiguous().view(n_feat*s_feat, c_feat, r_tokens, 1)
            
            # 2. 调用 Injector
            # 输出: [n*s, c, h, w]
            spa_out = self.Reg_Injector(spa_in, reg_in)
            
            # 3. 还原维度 -> [n, c, s, h, w]
            # 使用 _feat 后缀的变量还原
            spatial_enriched = spa_out.view(n_feat, s_feat, c_feat, h_feat, w_feat).permute(0, 2, 1, 3, 4).contiguous()

            # =======================================================================

            # 5.4 送入原始 GaitNet
            outs = self.Gait_Net.test_1(spatial_enriched)
            all_outs.append(outs)

        # ================= [6. 时序聚合] =================
        # GaitBase 回滚到原始版本，只接收一个输入
        full_spatial = torch.cat(all_outs, dim=2)
        embed_list, log_list = self.Gait_Net.test_2(full_spatial, seqL)
        
        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n*s, c, h, w)[:5].float(),
                    'image/human_mask': self.min_max_norm(human_mask.view(n*s, -1, self.sils_size*2, self.sils_size)[:5].float()),
                },
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': torch.concat(embed_list, dim=-1),
                }
            }
        return retval