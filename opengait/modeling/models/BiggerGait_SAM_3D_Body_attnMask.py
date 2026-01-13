# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import repeat, rearrange
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
# Helper Functions
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

# =========================================================================
# Main Model: BiggerGait with SAM 3D Body (DINOv3) + Semantic Pooling
# =========================================================================

class BiggerGait__SAM3DBody__AttnMask(BaseModel):
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
        
        total_hooked_layers = sum(self.hook_mask) 
        human_conv_input_dim = self.f4_dim * (total_hooked_layers // self.num_FPN) # 1280 * 4 = 5120

        # Hook layers for Attention
        self.trans_layer_mask = layer_cfg["trans_layer_mask"] if "trans_layer_mask" in layer_cfg else [True]*6
        
        # 选择用于 Semantic Pooling 的层索引 (对应 hook_data 中的 key)
        self.semantic_layer_idx = model_cfg.get("semantic_layer_idx", 2) 

        self.Gait_Net = Baseline_Share(model_cfg)

        part_target_h = self.sils_size * 2
        part_target_w = self.sils_size

        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(human_conv_input_dim, affine=False),
                nn.Conv2d(human_conv_input_dim, self.f4_dim//2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim//2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim//2, self.num_unknown, kernel_size=1),
                # ResizeToHW((part_target_h, part_target_w)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])

        # 保持 Mask Branch 结构以免报错
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])

    def init_SAM_Backbone(self):
        sys.path.insert(0, self.pretrained_lvm)
        from notebook.utils import setup_sam_3d_body
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')

        self.SAM_Engine = estimator.model
        self.SAM_Engine.decoder.do_interm_preds = True

        for param in self.SAM_Engine.parameters():
            param.requires_grad = False
        self.SAM_Engine.eval()

        self.Backbone = self.SAM_Engine.backbone

        self.hook_data = {} 
        
        def get_activation_hook(layer_idx, key_name):
            def hook(module, input, output):
                if layer_idx not in self.hook_data:
                    self.hook_data[layer_idx] = {}
                self.hook_data[layer_idx][key_name] = output.detach().float() 
            return hook

        for i, layer in enumerate(self.SAM_Engine.decoder.layers):
            layer.cross_attn.q_proj.register_forward_hook(get_activation_hook(i, 'q'))
            layer.cross_attn.k_proj.register_forward_hook(get_activation_hook(i, 'k'))
            
        del estimator

    def init_Mask_Branch(self):
        self.msg_mgr.log_info("=> Skip loading Mask Branch (Using Full Image Features)")
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
        
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Model Count: {:.5f}M'.format(n_parameters / 1e6))
        self.msg_mgr.log_info("=> init successfully")

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def _prepare_dummy_batch(self, image_embeddings):
        B, C, H, W = image_embeddings.shape
        device = image_embeddings.device
        input_h, input_w = H * 16, W * 16
        dummy_size = torch.tensor([float(input_w), float(input_h)], device=device)
        ori_size = dummy_size.unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        img_size = dummy_size.unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        center = (dummy_size / 2).unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
        scale = torch.tensor([max(input_w, input_h)], device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, 1)
        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
        affine_trans = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
        affine_trans = affine_trans.unsqueeze(0).unsqueeze(0).expand(B, 1, 2, 3)
        ray_cond = torch.zeros(B, 2, input_h, input_w, device=device)

        return {
            "img": torch.zeros(B, 1, 3, input_h, input_w, device=device),
            "ori_img_size": ori_size,
            "img_size": img_size,
            "bbox_center": center,
            "bbox_scale": scale,
            "cam_int": cam_int,
            "affine_trans": affine_trans,
            "ray_cond": ray_cond,
        }

    def compute_attention_map(self, q_proj, k_proj, num_heads):
        B, N_q, C = q_proj.shape
        _, N_k, _ = k_proj.shape
        head_dim = C // num_heads
        scale = head_dim ** -0.5
        q = q_proj.reshape(B, N_q, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k_proj.reshape(B, N_k, num_heads, head_dim).permute(0, 2, 1, 3)
        attn_logits = (q @ k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1) # [B, H, N_q, N_k]
        return attn_weights

    def forward(self, inputs):
        self.Mask_Branch.eval()
        
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        CHUNK_SIZE = self.chunk_size
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_chunk_outs = []    # Hard Parts Outputs
        all_attn_maps = []
        
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16 

        vis_pose_output = None

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            
            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w)
                all_layers = self.Backbone.encoder.get_intermediate_layers(outs, n=32, reshape=True, norm=True)
                features_to_use = [f for f, m in zip(all_layers, self.hook_mask) if m]
                image_embeddings = all_layers[-1] 
                del all_layers

                self.hook_data.clear()
                dummy_batch = self._prepare_dummy_batch(image_embeddings)
                
                batch_size = image_embeddings.shape[0]
                self.SAM_Engine._batch_size = batch_size
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(batch_size, device=image_embeddings.device)
                self.SAM_Engine.hand_batch_idx = []
                
                cond_info = None
                if self.SAM_Engine.cfg.MODEL.DECODER.CONDITION_TYPE != "none":
                    cond_info = torch.zeros(image_embeddings.shape[0], 3, device=image_embeddings.device).float()
                    cond_info[:, 2] = 1.25

                dummy_keypoints = torch.zeros(batch_size, 1, 3, device=image_embeddings.device)
                dummy_keypoints[:, :, -1] = -2
                
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    tokens_out, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=image_embeddings,
                        init_estimate=None,
                        keypoints=dummy_keypoints,
                        prev_estimate=None,
                        condition_info=cond_info,
                        batch=dummy_batch
                    )
                    if vis_pose_output is None:
                        vis_pose_output = pose_outs[-1]

                # ========================================================
                # 🌟 获取并处理 Attention Maps (For Semantic Pooling)
                # ========================================================
                # 假设 hook_data key 是 sorted index，例如 0,1,2,3,4,5
                sorted_layer_keys = sorted(self.hook_data.keys())
                
                # 获取指定层 (e.g., Layer 1) 的 Q 和 K
                target_l_idx = sorted_layer_keys[self.semantic_layer_idx]
                
                q_proj = self.hook_data[target_l_idx]['q'] 
                k_proj = self.hook_data[target_l_idx]['k'] 
                
                num_heads = self.SAM_Engine.cfg.MODEL.DECODER.HEADS
                
                # 计算 Attn: [B, H, N_query(145), N_key(HW)]
                with torch.no_grad():
                    attn_map_full = self.compute_attention_map(q_proj, k_proj, num_heads)
                
                # 提取 Body Tokens (70-42=29)不包含左右手部分对应的 Attention
                # Token Index: 5 (Init+Prompt等) : 5+29 (Body2D)
                # 我们取 Body 2D/3D tokens。通常 Body 2D tokens (index 5:75) 对特征图响应最强
                KEY_HAND = list(range(21, 63))
                body_token_indices = [i for i in range(5, 5 + 70) if i not in KEY_HAND]
                semantic_attn_heads = attn_map_full[:, :, [0] + body_token_indices, :]

                # 🌟 关键操作：Head Average
                # 我们需要 [B, 29, HW] 的权重用于 BMM
                semantic_attn = semantic_attn_heads.mean(dim=1) 
                
                # 保存用于可视化 (Reshape back to spatial)
                attn_spatial_vis = rearrange(semantic_attn[:, :, :], 'b p (h w) -> b p h w', h=h_feat, w=w_feat)
                all_attn_maps.append(rearrange(attn_spatial_vis.detach().cpu(), '(n s) p h w -> n s p h w', n=n, s=s))

            # semantic_attn [B, 29, 2048]
            semantic_attn = rearrange(semantic_attn, 'b k (c h w) -> (b k) c h w', h=h_feat, w=w_feat, c=1)
            # semantic_attn = F.interpolate(semantic_attn, (h_feat * 2, w_feat * 2), mode='bilinear')
            min_val = semantic_attn.amin(dim=(2, 3), keepdim=True) # [B, K, 1, 1]
            max_val = semantic_attn.amax(dim=(2, 3), keepdim=True)
            semantic_attn = (semantic_attn - min_val) / (max_val - min_val + 1e-6)
            semantic_attn = (semantic_attn > 0.3).float()
            semantic_attn = rearrange(semantic_attn, '(b k) c h w -> b k c h w', b=n*s, k=29) # [B, 29, 1, 64, 32]

            # ========================================================
            # FPN Feature Aggregation
            # ========================================================
            step = len(features_to_use) // self.num_FPN
            chunk_out = []
            for i in range(self.num_FPN):
                sub_app = torch.cat(features_to_use[i*step : (i+1)*step], dim=1) # [B, 5120, H, W]
                
                # LayerNorm
                sub_app = sub_app.permute(0, 2, 3, 1) 
                sub_app = partial(nn.LayerNorm, eps=1e-6)(self.f4_dim * step, elementwise_affine=False)(sub_app)
                sub_app = sub_app.permute(0, 3, 1, 2).contiguous() # [B, 5120, H, W]

                sub_feat = self.HumanSpace_Conv[i](sub_app) # [B, num_unknown(16), 64, 32]

                # semantic attn masking
                sub_feat = sub_feat[:, None] * semantic_attn # [B, 29, 16, 64, 32]
                chunk_out.append(sub_feat)

            # GaitNet
            part_feat = torch.cat(chunk_out, dim=2) # [B, 29, 16*4, 64, 32]
            part_feat = rearrange(part_feat, '(n s) p c h w -> (n p) c s h w', n=n, s=s).contiguous()
            if part_feat.requires_grad:
                out = checkpoint(self.Gait_Net.test_1, part_feat, use_reentrant=False)
            else:
                out = self.Gait_Net.test_1(part_feat) # [(n p), 16*4, s, 32, 16]

            out = rearrange(out, '(n p) c s h w -> n p c s h w', n=n, p=29).contiguous()
            out = out.sum(dim=1) # [n, 16*4, s, 32, 16]
            all_chunk_outs.append(out) # chunks * [n, 16*4, s, 32, 16]

        # =======================================================
        # Temporal Pooling
        # =======================================================
        
        embed_list, log_list = self.Gait_Net.test_2(torch.cat(all_chunk_outs, dim=2), seqL)

        # -------------------------------------------------------
        # Visualization (可视化)
        # -------------------------------------------------------
        with torch.no_grad():
            vis_dict = {}
            if self.training and torch.distributed.get_rank() == 0:
                import matplotlib.cm as cm
                vis_img = inputs[0][0][0, 0].detach().cpu()
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-6)
                vis_dict['image/0_original'] = vis_img

                # all_attn_maps: List of [n, s, P=8, h, w]
                map_total = torch.cat(all_attn_maps, dim=1) # [n, S, P, h, w]
                vis_data = map_total[0, 0].detach().cpu() # [P, h, w]

                for pid in range(vis_data.shape[0]):
                    att = vis_data[pid]
                    att = F.interpolate(att[None, None], size=vis_img.shape[1:], mode='bilinear')[0,0]
                    att_norm = (att - att.min()) / (att.max() - att.min() + 1e-6)
                    heatmap_np = cm.jet(att_norm.numpy())[..., :3]
                    heatmap = torch.from_numpy(heatmap_np).permute(2, 0, 1).float()
                    overlay = vis_img * 0.4 + heatmap * 0.6
                    vis_dict[f'image/semantic_part_{pid:02d}'] = overlay.clamp(0, 1)

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