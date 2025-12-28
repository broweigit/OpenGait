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

from .BigGait_utils.BigGait_GaitBase import *
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

class BiggerGait__DINOv3_Huge(BaseModel):
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

        self.gradient_checkpointing = model_cfg.get("gradient_checkpointing", False)
        self.chunk_size = model_cfg.get("chunk_size", 96)

        self.Gait_Net = Baseline_Share(model_cfg)

        self.HumanSpace_Conv = nn.ModuleList([ 
                nn.Sequential(
                    nn.BatchNorm2d(self.f4_dim*self.group_layer_num, affine=False),
                    nn.Conv2d(self.f4_dim*self.group_layer_num, self.f4_dim//2, kernel_size=1),
                    nn.BatchNorm2d(self.f4_dim//2, affine=False),
                    nn.GELU(),
                    nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                    ResizeToHW((self.sils_size*2, self.sils_size)),
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid()
                ) for _ in range(self.num_FPN)
            ])
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])

        # ================== [新增] Mask 适配器 ==================
        # 目的：将 Huge 的 1280 维映射回 384 维，以便复用 MaskBranch
        # 384 是 MaskBranch 预训练时的输入维度
        self.mask_adapter = nn.Sequential(
            nn.Linear(1280, 384),
            nn.ReLU() # 可选，加个非线性
        )
        # 初始化这个线性层
        nn.init.xavier_uniform_(self.mask_adapter[0].weight)
        nn.init.constant_(self.mask_adapter[0].bias, 0)
        # =======================================================

    def init_DINOv2(self):
        from transformers import Dinov2Config, Dinov2Model
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        config = Dinov2Config.from_pretrained(self.pretrained_lvm + "/config.json")
        self.Backbone = Dinov2Model.from_pretrained(
            self.pretrained_lvm, 
            config=config,
            ignore_mismatched_sizes=True
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
        # ================== 修改点：调整 Chunk Size ==================
        # DINOv2-Small 原版是 96。
        # DINOv2-Large 参数量是 Small 的 4 倍以上，建议改成 8 或者 4。
        # 如果还是 OOM，就改成 1 (即作者说的“一帧一帧输入”)。
        CHUNK_SIZE = self.chunk_size
        
        # 逻辑：将总帧数切分成更小的块
        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // CHUNK_SIZE) + 1, dim=1)
        # ==========================================================
        all_outs = []
        for _, rgb_img in enumerate(rgb_chunks):
            with torch.no_grad():
                # get RGB
                n,s,c,h,w = rgb_img.size()
                rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
                outs = self.preprocess(rgb_img, self.image_size)
                # outs = self.Backbone(outs,output_hidden_states=True).hidden_states[1:] # [ns,h*w,c]
                # ================== [修改 1] 获取 Huge 所有层并按段采样 ==================
                # full_outs: [B, 32, 1280] (假设不包含 Input Embedding，或者你切片后只剩32层)
                # 注意：hidden_states 通常包含 (Input, Layer 0, ... Layer 31)，共33个。
                # 之前的代码写的是 hidden_states[1:]，所以 full_outs 长度刚好是 32。
                full_outs = self.Backbone(outs, output_hidden_states=True).hidden_states[1:]

                total_depth = len(full_outs) # 32
                target_num = self.total_layer_num # 12

                # [核心逻辑] 均匀切分，取每段末尾
                # 解释：(i+1) 代表第几段，乘以比例算出物理位置，转int向下取整，-1转为索引
                sampled_indices = [
                    int((i + 1) * total_depth / target_num) - 1 
                    for i in range(target_num)
                ]

                outs = [full_outs[i] for i in sampled_indices]
                # ===================================================================

                # ================== [修正 1] 动态计算特征图尺寸 ==================
                # 不要再用 //7 或 //14 了，直接算
                patch_size = 16 
                h_feat = (self.image_size * 2) // patch_size  # 512 // 16 = 32
                w_feat = self.image_size // patch_size        # 256 // 16 = 16
                # =============================================================

                intermediates = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim*len(outs), elementwise_affine=False)(torch.concat(outs, dim=-1))[:,1:]
                # intermediates = rearrange(intermediates.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                # ================== [修正 2] 使用正确的尺寸 Reshape ==================
                # 原代码: self.image_size//7, self.image_size//14
                # 新代码: h_feat, w_feat
                intermediates = rearrange(
                    intermediates.view(n, s, h_feat, w_feat, -1), 
                    'n s h w c -> (n s) c h w'
                ).contiguous()
                intermediates = list(torch.chunk(intermediates, self.total_layer_num, dim=1))

                # # human_mask = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs[-1])[:,1:].contiguous()
                # # human_mask, _ = self.Mask_Branch(human_mask.view(-1, self.f4_dim), mse=False)
                # # ================== [修改 2] Mask Branch 适配 ==================
                # # outs[-1] 现在是 Huge 的最后一层 (1280维)
                # human_mask_feat = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim, elementwise_affine=False)(outs[-1])[:,1:]
                
                # # [新增] 维度压缩 1280 -> 384
                # # 注意 view 的操作，先压扁喂给 Linear，再处理
                # # 输入: [B*S*L, 1280]
                # human_mask_feat = self.mask_adapter(human_mask_feat.reshape(-1, self.f4_dim)) 
                
                # # 现在的 human_mask_feat 是 [B*S*L, 384]，完美喂给原始 MaskBranch
                # human_mask, _ = self.Mask_Branch(human_mask_feat, mse=False)
                # # ==============================================================
                # human_mask = (human_mask[:,1] > 0.5).float() # check which is the foreground at first!!!   0 or 1; 50%;
                # # human_mask = human_mask.view(n*s, 1, self.image_size//7, self.image_size//14)
                # human_mask = human_mask.view(n*s, 1, h_feat, w_feat)
                # human_mask = self.preprocess(human_mask, self.sils_size).detach().clone()

                # ================== [修改点] 全 1 Mask ==================
                # 不再计算 outs[-1]，不再跑 Mask_Adapter 和 Mask_Branch
                human_mask = torch.ones(
                    (n * s, 1, h_feat, w_feat),
                    dtype=intermediates[0].dtype,
                    device=intermediates[0].device
                )
                
                # Resize 到 (64, 32)
                human_mask = self.preprocess(human_mask, self.sils_size).detach().clone()
                # =======================================================

            intermediates = [torch.cat(intermediates[i:i+self.group_layer_num], dim=1).contiguous() for i in range(0, self.total_layer_num, self.group_layer_num)]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            intermediates = torch.concat(intermediates, dim=1)
            intermediates = intermediates * (human_mask > 0.5).to(intermediates)
            intermediates = rearrange(intermediates.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> n c s h w').contiguous()

            outs = self.Gait_Net.test_1(intermediates)
            all_outs.append(outs)

        embed_list, log_list = self.Gait_Net.test_2(
                                        torch.cat(all_outs, dim=2),
                                        seqL,
                                        )
        
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
