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

class BiggerGait__SAM3DBody__Query_Gaitbase_Share(BaseModel):
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

        self.chunk_size = model_cfg.get("chunk_size", 96)

        # 初始化下游网络
        # self.Gait_Net = Baseline_ShareTime_2B(model_cfg)
        self.Gait_Net = Baseline_Semantic_2B(model_cfg)
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
        
        self.SAM_Engine = estimator.model
        
        try:
            self.Backbone = self.SAM_Engine.backbone.encoder
            self.msg_mgr.log_info(f"[SAM3D] Backbone set to SAM_Engine.backbone.encoder")
        except AttributeError:
            raise RuntimeError("Could not find Backbone encoder in SAM Engine")

        try:
            self.Decoder = self.SAM_Engine.decoder
            self.msg_mgr.log_info(f"[SAM3D] Decoder set to SAM_Engine.decoder")
            # 强制关闭中间监督，防止因缺少回调函数而 Crash
            self.Decoder.do_interm_preds = False
            # 强制关闭 Keypoint Token Update (如果有的话)，同样防止 Crash
            self.Decoder.keypoint_token_update = False
        except AttributeError:
            raise RuntimeError("Could not find Decoder in SAM Engine")
        
        # 4. 提取 Keypoint Queries (Parts)
        # 直接提取预训练好的 Embeddings
        # shape: [70, Dim]
        # 注意：这里我们使用 .clone() 并且注册为 Buffer，意味着我们将其视为"常量先验"，不进行更新
        # 如果你想微调这些 Query，可以改为 nn.Parameter
        part_query_data = self.SAM_Engine.keypoint_embedding.weight.data.clone()
        self.register_buffer('fixed_part_queries', part_query_data)

        # =======================================================
        # 🌟 [关键修正] 提取 K_proj (图像投影) 和 Q_proj (语义投影)
        # =======================================================
        self.feat_k_proj = None
        self.feat_q_proj = None
        
        try:
            # 路径: Decoder -> 最后一层 -> Cross Attention
            last_layer = self.Decoder.layers[-1]
            
            if hasattr(last_layer, 'cross_attn'):
                cross_attn = last_layer.cross_attn
                
                # 1. 提取 K_proj (1280 -> 512)
                if hasattr(cross_attn, 'k_proj'):
                    self.feat_k_proj = cross_attn.k_proj
                
                # 2. 提取 Q_proj (1024 -> 512)
                if hasattr(cross_attn, 'q_proj'):
                    self.feat_q_proj = cross_attn.q_proj
                    
                if self.feat_k_proj and self.feat_q_proj:
                    self.msg_mgr.log_info(f"[SAM3D] Reusing Projections: K={self.feat_k_proj}, Q={self.feat_q_proj}")
                else:
                    raise AttributeError("Missing k_proj or q_proj in CrossAttn.")
            else:
                raise AttributeError("No cross_attn found.")
                
        except Exception as e:
            self.msg_mgr.log_warning(f"[SAM3D] Projection extraction failed: {e}")
            raise RuntimeError("Failed to extract k_proj and q_proj from Decoder.")
        
        self.msg_mgr.log_info(f"[SAM3D] Extracted {self.fixed_part_queries.shape[0]} Part Queries. (No MHR)")

        # 5. 注册 FPN Hook (保持不变)
        self.intermediate_features = {}
        self.hook_handles = []
        
        def get_activation(idx_in_list):
            def hook(model, input, output):
                if isinstance(output, (list, tuple)): output = output[0]
                self.intermediate_features[idx_in_list] = output
            return hook

        target_blocks = None
        if hasattr(self.Backbone, 'blocks'):
            target_blocks = self.Backbone.blocks
        elif hasattr(self.Backbone, 'layers'):
            target_blocks = self.Backbone.layers
        
        if target_blocks:
            hook_count = 0
            for layer_idx, should_hook in enumerate(self.hook_mask):
                if should_hook and layer_idx < len(target_blocks):
                    target_blocks[layer_idx].register_forward_hook(get_activation(hook_count))
                    self.hook_handles.append(None)
                    hook_count += 1
            self.msg_mgr.log_info(f"[SAM3D] Hooked {hook_count} layers inside Backbone.")

        # 6. 冻结所有组件
        self.SAM_Engine.eval()
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False
            
        # 清理
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

    def forward(self, inputs):
        # 确保 Mask Branch 不更新
        self.Mask_Branch.eval()
        
        ipts, labs, ty, vi, seqL = inputs
        rgb = ipts[0]
        del ipts

        # 显存优化：如果依然 OOM，将此值改为 2
        CHUNK_SIZE = self.chunk_size
        rgb_chunks = torch.chunk(rgb, (rgb.size(1)//CHUNK_SIZE)+1, dim=1)
        
        all_outs = []
        # [新增] 用于收集每个 Chunk 的 Attention Map
        all_attn_maps = []
        
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            
            # =======================================================
            # 1. Backbone 前向 (必须加 no_grad 以节省显存)
            # =======================================================
            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w) # ns c H W -> B 3 512 256
                
                # 清空 Hook 缓存
                self.intermediate_features = {}
                
                # DINOv3 推理
                _ = self.Backbone(outs) # lay B L C -> 16 B 32*16+1+4=517 1280

                # [关键修改] 从 Hook 中提取最后一层特征 (3D Tensor)
                # self.hook_handles 记录了注册了多少个 hook
                # 对应的 key 是 0 到 len-1
                # backbone_feat_3d: [B, L, Dim] (例如 [4, 650, 1280])
                backbone_feat = self.intermediate_features[len(self.hook_handles) - 1]
                
                # =======================================================
                #  手动组装 Decoder 输入 (Parts Only)
                # =======================================================
                
                # A. 准备 Image Embedding: [B, Dim, H_feat, W_feat]
                # 计算 Patch 数量
                # DINOv3 ViT-Huge 通常 patch_size = 16
                patch_size = 16 
                feat_h = target_h // patch_size
                feat_w = target_w // patch_size
                num_spatial = feat_h * feat_w
                
                # 截取 Spatial Tokens (去掉前面的 Register Tokens)
                # 如果 backbone_feat 长度正好等于 num_spatial，说明没有 register token
                # 如果大于，说明前面有 register token
                if backbone_feat.shape[1] >= num_spatial:
                    img_tokens = backbone_feat[:, -num_spatial:, :] # 取最后 N 个

                # Reshape 为 Decoder 要求的 [B, C, H, W]
                img_emb = rearrange(img_tokens, 'b (h w) c -> b c h w', h=feat_h, w=feat_w)
                
                # B. 准备 Token Embedding (Queries): [B, N_queries, Dim]
                bs = img_emb.shape[0]
                
                # 只使用 Part Queries，扩展到 Batch 维度
                # [70, Dim] -> [B, 70, Dim]
                all_queries = repeat(self.fixed_part_queries, 'n d -> b n d', b=bs)
                
                # C. 调用 Decoder
                try:
                    # PromptableDecoder 只需要这两个主要参数就能跑
                    # 它内部会利用 CrossAttention 让 queries 去查询 image_embedding
                    semantic_out = self.Decoder(
                        token_embedding=all_queries,
                        image_embedding=img_emb
                    )
                    
                    # semantic_out: [B, 70, Dim]
                    # 这里的 70 就是 70 个关键点对应的特征
                    
                    if isinstance(semantic_out, tuple):
                        semantic_out = semantic_out[0]
                        
                    # 2. 🌟 计算 Attention Map (Soft Semantic Mask)
                    # 我们计算 更新后的Query 与 图像特征 的相似度
                    # Map = Softmax(Q @ K.T / sqrt(dim))

                    # 2. 🌟 计算 Attention Map (双投影修正版)
                    
                    # [Step 1] 投影 Image Key (K)
                    # [B, HW, 1280] -> [B, HW, 512]
                    k_feats = self.feat_k_proj(img_tokens)
                    
                    # [Step 2] 投影 Semantic Query (Q)
                    # [B, 70, 1024] -> [B, 70, 512]
                    q_feats = self.feat_q_proj(semantic_out)
                    
                    # [Step 3] 验证维度并计算
                    dim_k = k_feats.shape[-1]
                    dim_q = q_feats.shape[-1]
                    
                    if dim_k != dim_q:
                        raise ValueError(f"Proj dim mismatch: K={dim_k}, Q={dim_q}")

                    # Dot Product: [B, 70, 512] @ [B, 512, HW] -> [B, 70, HW]
                    raw_attn = torch.matmul(q_feats, k_feats.transpose(1, 2))
                    raw_attn = raw_attn / (dim_q ** 0.5)
                    
                    # Softmax & Reshape
                    attn_map = F.softmax(raw_attn, dim=-1) # [B, 70, HW]
                    attn_map_spatial = rearrange(attn_map, 'b p (h w) -> b p h w', h=feat_h, w=feat_w)
                    
                    # self.msg_mgr.log_info(f"Generated Attention Map: {attn_map_spatial.shape}")
                    # [新增] 收集 Attention Map
                    # 我们需要将其 Reshape 回 [n, p, s, h, w] 以便后续在 s 维度拼接
                    # 注意：这里必须使用当前 chunk 的 s
                    current_map = rearrange(attn_map_spatial, '(n s) p h w -> n p s h w', n=n, s=s)
                    all_attn_maps.append(current_map)
                    
                except Exception as e:
                    self.msg_mgr.log_warning(f"Manual Decoder/Attention Failed: {type(e).__name__}: {e}")
                    print(f"Q: {semantic_out.shape}, K_raw: {img_tokens.shape}")
                
                # 收集被 Hook 的层
                num_layers = len(self.hook_handles) # 16
                features_to_use = []
                target_tokens = h_feat * w_feat # 512
                
                for i in range(num_layers):
                    feat = self.intermediate_features[i] # B L C -> B 517 1280
                    # 去除 CLS Token 等多余部分，只保留 Spatial Tokens
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :]
                    features_to_use.append(feat) # lay B L C -> 16 B 512 1280

            # =======================================================
            # 2. FPN 组处理 (Group Processing)
            # =======================================================
            processed_feat_list = []
            
            # 自动计算步长：例如 Hook了16层，FPN有4个，则 step=4 (即每组4层)
            # 这与 build_network 中的 input_dim 计算逻辑是完全对应的
            step = len(features_to_use) // self.num_FPN # 16 / 4 = 4
            
            for i in range(self.num_FPN):
                # A. 切片：取出当前 Head 负责的那几层
                start_idx = i * step
                end_idx = (i + 1) * step
                sub_feats = features_to_use[start_idx : end_idx]
                
                # B. 拼接：将这几层拼在一起
                # 维度变化: [B, N, 1280] x step -> [B, N, 1280*step]
                sub_app = torch.concat(sub_feats, dim=-1) # B 512 1280*4=5120
                
                # C. 调整形状以进行卷积 [B, C, H, W]
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous() # B 5120 32 16
                
                # D. Pre_Conv (Identity)
                sub_app = self.Pre_Conv(sub_app)
                
                # E. 局部 LayerNorm (针对当前组的维度进行归一化)
                sub_app = rearrange(sub_app, 'b c h w -> b (h w) c') # B 512 5120
                # 计算当前组的通道数，例如 1280 * 2 = 2560
                curr_dim = self.f4_dim * len(sub_feats) # 1280 * 4 = 5120
                sub_app = partial(nn.LayerNorm, eps=1e-6)(curr_dim, elementwise_affine=False)(sub_app)
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous() # B 5120 32 16
                
                # F. 喂给第 i 个独立的 FPN Head
                # self.HumanSpace_Conv[i] 的输入维度在 build_network 里已经按 step 算好了
                reduced_feat = self.HumanSpace_Conv[i](sub_app) # B num_unknown -> B 16 64 32
                
                processed_feat_list.append(reduced_feat) # *4
                
                # 释放显存
                del sub_app
                del sub_feats

            # 3. 拼接所有 Head 的输出
            human_feat = torch.concat(processed_feat_list, dim=1) # B num_unknown*num_FPN H W -> B 16*4=64 64 32
            
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

        # # GaitNet Part 2 (Temporal Aggregation)
        # embed_list, log_list = self.Gait_Net.test_2(
        #     torch.cat(all_outs, dim=2),
        #     seqL,
        # )
        
        # 1. 拼接特征 (在时间维度 s 上, dim=2)
        # [n, c, S_total, h, w]
        feat_total = torch.cat(all_outs, dim=2)
        
        # 2. 拼接 Attention Maps (在时间维度 s 上, dim=2)
        # [n, p, S_total, h, w]
        map_total = torch.cat(all_attn_maps, dim=2)
        
        # 3. 调用新的 test_2 (Semantic Pooling + Temporal Pooling)
        # 传入特征和对应的 Attention Map
        embed_list, log_list = self.Gait_Net.test_2(
            feat_total,
            map_total, 
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