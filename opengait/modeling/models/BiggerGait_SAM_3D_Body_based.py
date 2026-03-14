import sys

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from functools import partial

from ..base_model import BaseModel
from .BigGait_utils.BigGait_GaitBase import *


class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim // 2),
                nn.BatchNorm1d(source_dim // 2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim // 2, target_dim),
            )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim // 2),
                    nn.BatchNorm1d(source_dim // 2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim // 2, source_dim),
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
            return d_x, None
        if self.Up:
            u_x = self.up_sampling(d_x)
            return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
        return torch.sigmoid(self.bn_t(d_x)), None


class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)


class BiggerGait__SAM3DBody__Based_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)

        layer_cfg = model_cfg.get("layer_config", {})
        self.hook_mask = layer_cfg.get("hook_mask", [False] * 16 + [True] * 16)
        if len(self.hook_mask) != 32:
            raise ValueError(f"hook_mask length must be 32, got {len(self.hook_mask)}")
        self.hook_sample_type = layer_cfg.get("hook_sample_type", "chunk")

        self.total_hooked_layers = sum(self.hook_mask)
        if self.total_hooked_layers == 0:
            raise ValueError("hook_mask selects no layers.")
        if self.total_hooked_layers % self.num_FPN != 0:
            raise ValueError(
                f"Hook layers ({self.total_hooked_layers}) must be divisible by FPN heads ({self.num_FPN})"
            )

        self.layers_per_head = self.total_hooked_layers // self.num_FPN
        input_dim = self.f4_dim * self.layers_per_head

        self.Gait_Net = Baseline_ShareTime_2B(model_cfg)
        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(input_dim, affine=False),
                nn.Conv2d(input_dim, self.f4_dim // 2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim // 2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim // 2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size * 2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])

        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        self.init_SAM_Backbone()

    def init_SAM_Backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)

        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Error: {e}")

        self.msg_mgr.log_info(f"[SAM3D] Loading SAM 3D Body (Encoder + Decoder)...")
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')

        self.SAM_Engine = estimator.model
        if hasattr(self.SAM_Engine, 'backbone'):
            raw_backbone = self.SAM_Engine.backbone
        elif hasattr(self.SAM_Engine, 'image_encoder'):
            raw_backbone = self.SAM_Engine.image_encoder
        else:
            raise RuntimeError("Cannot find backbone in SAM Engine")

        if hasattr(raw_backbone, 'encoder'):
            self.Backbone = raw_backbone.encoder
        else:
            self.Backbone = raw_backbone

        self.SAM_Engine.cpu()
        self.intermediate_features = {}
        self.hook_handles = []

        def get_activation(idx_in_list):
            def hook(model, input, output):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if isinstance(output, (list, tuple)):
                    output = output[0]
                self.intermediate_features[idx_in_list] = output
            return hook

        if hasattr(self.Backbone, 'blocks'):
            all_blocks = self.Backbone.blocks
        elif hasattr(self.Backbone, 'layers'):
            all_blocks = self.Backbone.layers
        else:
            raise RuntimeError("Cannot find blocks in Backbone")

        hook_count = 0
        for layer_idx, should_hook in enumerate(self.hook_mask):
            if should_hook:
                handle = all_blocks[layer_idx].register_forward_hook(get_activation(hook_count))
                self.hook_handles.append(handle)
                hook_count += 1

        self.msg_mgr.log_info(f"[SAM3D] Hooked {hook_count} layers.")
        self.SAM_Engine.eval()
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        self.init_SAM_Backbone()
        self.SAM_Engine.eval()
        self.SAM_Engine.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Model Count: {:.5f}M'.format(n_parameters / 1e6))

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        all_outs = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()

            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w)
                self.intermediate_features = {}
                _ = self.Backbone(outs)

                target_tokens = h_feat * w_feat
                features_to_use = []
                for i in range(len(self.hook_handles)):
                    feat = self.intermediate_features[i]
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :]
                    features_to_use.append(feat)

            processed_feat_list = []
            step = len(features_to_use) // self.num_FPN
            for i in range(self.num_FPN):
                if self.hook_sample_type == 'interleave':
                    sub_feats = features_to_use[i::self.num_FPN]
                elif self.hook_sample_type == 'chunk':
                    start_idx = i * step
                    end_idx = (i + 1) * step
                    sub_feats = features_to_use[start_idx:end_idx]
                else:
                    raise ValueError(f"Invalid hook_sample_type: {self.hook_sample_type}")

                sub_app = torch.concat(sub_feats, dim=-1)
                curr_dim = self.f4_dim * len(sub_feats)
                sub_app = partial(nn.LayerNorm, eps=1e-6)(
                    curr_dim, elementwise_affine=False
                )(sub_app)
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_feat = torch.concat(processed_feat_list, dim=1)
            human_feat = rearrange(
                human_feat.view(n, s, -1, self.sils_size * 2, self.sils_size),
                'n s c h w -> n c s h w'
            ).contiguous()

            outs = self.Gait_Net.test_1(human_feat)
            all_outs.append(outs)

        embed_list, log_list = self.Gait_Net.test_2(torch.cat(all_outs, dim=2), seqL)

        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n * s, c, h, w)[:5].float(),
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
