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
from functools import partial
from torch.utils.checkpoint import checkpoint
import copy

# import GaitBase
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import save_image

# =========================================================================
# Helper Modules
# =========================================================================

class DirectSemanticAdapter(nn.Module):
    """
    将 Transformer Decoder 的 Token 输出适配到 GaitNet 输入格式
    Input: [B, N_tokens, Dim_in]
    Output: [B, Dim_out, N_tokens, 1]
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.LayerNorm(out_dim), 
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, N, D]
        x = self.projector(x) # [B, N, out_dim]
        # Reshape to [B, C, Part, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x

class SimpleSemanticHead(nn.Module):
    def __init__(self, model_cfg):
        super(SimpleSemanticHead, self).__init__()
        
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.TP = PackSequenceWrapper(torch.max)
        
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        
        self.num_FPN = model_cfg['num_FPN']

        self.Heads = nn.ModuleList([
            nn.ModuleDict({
                'FCs': SeparateFCs(**model_cfg['SeparateFCs']),
                'BNNecks': SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            })
            for _ in range(self.num_FPN)
        ])

    def test_2(self, x, seqL):
        # x shape: [n, c_total, s, 28, 1] 
        
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        
        embed_list = []
        log_list = []
        
        for i in range(self.num_FPN):
            feat = x_list[i] # [n, c, s, 28, 1]
            
            # 1. Temporal Pooling
            feat = self.TP(feat, seqL, options={"dim": 2})[0] # [n, c, 28, 1]
            
            # 2. Horizontal Pooling 
            feat = self.HPP(feat) # [n, c, p] (p=14 or 28, depending on bin_num)
            
            # 3. FC & BNNeck
            head = self.Heads[i]
            embed_1 = head['FCs'](feat)
            _, logits = head['BNNecks'](embed_1)
            
            embed_list.append(embed_1)
            log_list.append(logits)
            
        return embed_list, log_list

# =========================================================================
# Main Model: Pure Semantic Tokens (No Hard Branch)
# =========================================================================

class BiggerGait__SAM3DBody__PureSemantic(BaseModel):
    def build_network(self, model_cfg):
        # 1. 基础参数
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.image_size = model_cfg["image_size"]
        self.chunk_size = model_cfg.get("chunk_size", 96)
        
        # =================================================================
        # Pure Semantic Stream Setup
        # =================================================================
        # 1. 读取配置中的 num_FPN
        self.num_FPN = model_cfg['num_FPN'] # 假设配置为 2 或 3
        
        # 2. 计算分组策略
        total_decoder_layers = 6 # SAM-3D-Body 只有 6 层
        
        if total_decoder_layers % self.num_FPN != 0:
            raise ValueError(f"SAM Decoder layers (6) must be divisible by num_FPN ({self.num_FPN})")
            
        self.layers_per_level = total_decoder_layers // self.num_FPN
        
        decoder_dim = 1024 
        adapter_in_dim = decoder_dim * self.layers_per_level
        
        # 3. 初始化 Adapters
        self.Semantic_Adapters = nn.ModuleList([
            DirectSemanticAdapter(
                in_dim=adapter_in_dim, 
                out_dim=model_cfg['SeparateFCs']['in_channels'] # 512
            ) for _ in range(self.num_FPN)
        ])

        # 4. 配置 Head
        # 直接传入 model_cfg，不做任何魔改
        # SimpleSemanticHead 会自己去读 model_cfg['bin_num'] 和 ['num_FPN']
        # 并根据这些配置来建立 HPP 和 FC
        self.Semantic_Head = SimpleSemanticHead(model_cfg)

        self.Mask_Branch = nn.Identity()

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
        
        # 🌟 Hook Strategy: 只 Hook Decoder，完全不再 Hook Encoder
        self.token_buffer = {}
        
        # 提前定义好需要保留的 Body Token 索引
        KEY_HAND = list(range(21, 63))
        self.body_indices = torch.tensor([0] + [i for i in range(5, 5 + 70) if i not in KEY_HAND], dtype=torch.long)
        
        def get_layer_output_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                
                # 立即切片，只保留 Body Tokens，扔掉 Image Tokens
                if self.body_indices.device != out.device:
                    self.body_indices = self.body_indices.to(out.device)
                
                body_out = out[:, self.body_indices, :]
                
                # 显存优化：FP16/BF16
                self.token_buffer[layer_idx] = body_out.detach()
                
            return hook

        for i, layer in enumerate(self.SAM_Engine.decoder.layers):
            layer.register_forward_hook(get_layer_output_hook(i))
            
        del estimator

    def init_parameters(self):
        for m in self.Semantic_Adapters.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.constant_(m.weight.data, 1.0)

        self.init_SAM_Backbone()
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.msg_mgr.log_info("=> init SAM Backbone (Pure Semantic) successfully")

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def _prepare_dummy_batch(self, image_embeddings):
        B, C, H, W = image_embeddings.shape
        device = image_embeddings.device
        input_h, input_w = H * 16, W * 16
        return {
            "img": torch.zeros(B, 1, 3, input_h, input_w, device=device),
            "ori_img_size": torch.tensor([float(input_w), float(input_h)], device=device).view(1,1,2).expand(B,1,2),
            "img_size": torch.tensor([float(input_w), float(input_h)], device=device).view(1,1,2).expand(B,1,2),
            "bbox_center": (torch.tensor([float(input_w), float(input_h)], device=device)/2).view(1,1,2).expand(B,1,2),
            "bbox_scale": torch.tensor([max(input_w, input_h)], device=device).view(1,1,1).expand(B,1,1),
            "cam_int": torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3),
            "affine_trans": torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).view(1,1,2,3).expand(B,1,2,3),
            "ray_cond": torch.zeros(B, 2, input_h, input_w, device=device),
        }

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        CHUNK_SIZE = self.chunk_size
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_chunk_outs_semantic = [] 
        
        target_h, target_w = self.image_size * 2, self.image_size

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            
            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w)
                
                # 🌟 显存优化终极奥义：不再 Hook Encoder 的中间层！
                # 只需要最后一层输出给 Decoder 用即可
                # get_intermediate_layers 会返回 list，我们只取最后一层
                # 注意：SAM Backbone 的 forward 可能有点不同，这里沿用之前证实可用的接口
                # 但不使用 hook_mask，直接拿 output
                
                # 这里的 intermediate_layers 实际上是 SAM Encoder 的输出
                enc_out = self.Backbone.forward(outs)
                # DINOv3 forward 返回字典或 Tensor，通常需要 forward_features
                # 为了兼容性，我们还是用 get_intermediate_layers 但只取最后一层，且不用 hook mask
                
                # 更稳妥的方式：直接调 Encoder forward，DINOv3 encoder 通常输出 output dict
                # 但为了少改动，我们沿用之前的接口，但 n=1
                all_layers = self.Backbone.encoder.get_intermediate_layers(outs, n=1, reshape=True, norm=True)
                image_embeddings = all_layers[-1]
                del all_layers # 立即释放

                # 清空 Hook Buffer
                self.token_buffer.clear()
                dummy_batch = self._prepare_dummy_batch(image_embeddings)
                
                batch_size = image_embeddings.shape[0]
                self.SAM_Engine._batch_size = batch_size
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(batch_size, device=image_embeddings.device)
                self.SAM_Engine.hand_batch_idx = []
                
                cond_info = torch.zeros(batch_size, 3, device=image_embeddings.device)
                cond_info[:, 2] = 1.25 # focal length assumption

                dummy_kps = torch.zeros(batch_size, 1, 3, device=image_embeddings.device)
                dummy_kps[:, :, -1] = -2
                
                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    # 这一步会触发 Decoder Hook，填充 token_buffer
                    _ = self.SAM_Engine.forward_decoder(
                        image_embeddings=image_embeddings,
                        init_estimate=None,
                        keypoints=dummy_kps,
                        prev_estimate=None,
                        condition_info=cond_info,
                        batch=dummy_batch
                    )
                
                # 立即释放巨大的 Image Embedding
                del image_embeddings
                del enc_out

            # ========================================================
            # Direct Token Processing
            # ========================================================
            chunk_semantic_out = []
            
            # 根据计算出的分组策略进行遍历
            for lvl in range(self.num_FPN):
                # 动态计算该组包含哪些层
                start_layer = lvl * self.layers_per_level
                end_layer = (lvl + 1) * self.layers_per_level
                
                layers_to_concat = []
                for i in range(start_layer, end_layer):
                    if i in self.token_buffer:
                        layers_to_concat.append(self.token_buffer[i])
                    else:
                        ref = list(self.token_buffer.values())[0]
                        layers_to_concat.append(torch.zeros_like(ref))
                
                # Concat: [B, 28, 1024 * layers_per_level]
                concat_tokens = torch.cat(layers_to_concat, dim=-1) 
                
                # Adapter: [B, 512, 28, 1]
                # 注意：Adapter 始终输出 28 (Body Token Num)，后续由 HPP 负责 resize 到 bin_num
                concat_tokens.requires_grad_(True)
                adapted_tokens = checkpoint(self.Semantic_Adapters[lvl], concat_tokens, use_reentrant=False)
                
                chunk_semantic_out.append(adapted_tokens)

            # Concat FPN levels: [B, 512*num_FPN, 28, 1]
            semantic_feat_total = torch.cat(chunk_semantic_out, dim=1)
            
            # Reshape for TP: [n, c, s, h, w]
            semantic_feat_total = rearrange(semantic_feat_total.view(n, s, -1, semantic_feat_total.shape[2], semantic_feat_total.shape[3]),
                                            'n s c h w -> n c s h w').contiguous()
            
            all_chunk_outs_semantic.append(semantic_feat_total)

        # =======================================================
        # Temporal Pooling & Merging
        # =======================================================
        
        # [n, c, S_total, 28, 1]
        sem_feat_final = torch.cat(all_chunk_outs_semantic, dim=2)
        
        # Head Forward
        sem_embed_list, sem_logit_list = self.Semantic_Head.test_2(sem_feat_final, seqL)
        
        # Concat all FPN levels
        # Result: [n, c_total, 28]
        final_embed = torch.cat(sem_embed_list, dim=-1)
        final_log = torch.cat(sem_logit_list, dim=-1)

        # -------------------------------------------------------
        # Visualization
        # -------------------------------------------------------
        vis_dict = {}

        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': final_embed, 'labels': labs},
                    'softmax': {'logits': final_log, 'labels': labs},
                },
                'visual_summary': vis_dict,
                'inference_feat': {
                    'embeddings': final_embed
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': final_embed
                }
            }
        return retval