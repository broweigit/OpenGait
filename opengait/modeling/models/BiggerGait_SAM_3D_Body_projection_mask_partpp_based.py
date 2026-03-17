import os
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


class BiggerGait__SAM3DBody__Projection_Mask_PartPP_Based_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)
        self.part_indices_path = model_cfg["part_indices_path"]

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

        self._load_part_indices(model_cfg)

        self.Gait_Net = Baseline_PartPP_ShareTime_2B(model_cfg)
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

    def _load_part_indices(self, model_cfg):
        if not os.path.exists(self.part_indices_path):
            raise FileNotFoundError(f"Cannot find part indices file: {self.part_indices_path}")

        self.part_indices = torch.load(self.part_indices_path, map_location='cpu')
        self.ordered_parts = list(self.part_indices.keys())
        self.num_parts = len(self.ordered_parts)

        expected_parts = model_cfg['SeparateFCs']['parts_num']
        if self.num_parts != expected_parts:
            raise ValueError(
                f"Loaded {self.num_parts} parts from {self.part_indices_path}, "
                f"but SeparateFCs.parts_num is {expected_parts}"
            )

        max_vertex_idx = max(int(indices.max().item()) for indices in self.part_indices.values()) + 1
        vertex_to_part = torch.full((max_vertex_idx,), -1, dtype=torch.long)
        for part_idx, part_name in enumerate(self.ordered_parts):
            vertex_to_part[self.part_indices[part_name].long()] = part_idx
        self.register_buffer("vertex_to_part", vertex_to_part, persistent=False)
        self.msg_mgr.log_info(
            f"[MHR] Loaded {self.num_parts} parts from {self.part_indices_path}"
        )

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

    def _prepare_dummy_batch(self, image_embeddings, target_h, target_w):
        bsz = image_embeddings.shape[0]
        device = image_embeddings.device

        estimated_focal_length = max(target_h, target_w) * 1.1
        cx, cy = target_w / 2.0, target_h / 2.0

        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(bsz, 3, 3).clone()
        cam_int[:, 0, 0] = estimated_focal_length
        cam_int[:, 1, 1] = estimated_focal_length
        cam_int[:, 0, 2] = cx
        cam_int[:, 1, 2] = cy

        y_grid, x_grid = torch.meshgrid(
            torch.arange(target_h, device=device),
            torch.arange(target_w, device=device),
            indexing='ij'
        )
        ray_x = (x_grid - cx) / estimated_focal_length
        ray_y = (y_grid - cy) / estimated_focal_length
        ray_cond = torch.stack([ray_x, ray_y], dim=0).unsqueeze(0).expand(bsz, 2, target_h, target_w)

        bbox_scale = torch.tensor([max(target_h, target_w)], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 1)
        bbox_center = torch.tensor([cx, cy], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 2)
        img_size = torch.tensor([float(target_w), float(target_h)], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 2)
        affine_trans = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 2, 3)

        return {
            "img": torch.zeros(bsz, 1, 3, target_h, target_w, device=device),
            "ori_img_size": img_size,
            "img_size": img_size,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_int": cam_int,
            "affine_trans": affine_trans,
            "ray_cond": ray_cond,
        }

    def project_vertices_to_mask(self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w):
        bsz, _, _ = vertices.shape
        device = vertices.device

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
        z = z.clamp(min=1e-3)

        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy

        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)

        mask = torch.zeros(bsz, 1, h_feat, w_feat, device=device)
        flat_indices = v_feat * w_feat + u_feat
        ones = torch.ones_like(flat_indices, dtype=torch.float)
        mask_flat = mask.view(bsz, -1)
        mask_flat.scatter_(1, flat_indices, ones)
        return mask_flat.view(bsz, 1, h_feat, w_feat)

    def get_source_vertex_index_map(self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w):
        bsz, num_verts, _ = vertices.shape
        device = vertices.device

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy

        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)
        flat_pixel_indices = v_feat * w_feat + u_feat

        depth_map_flat = torch.full((bsz, h_feat * w_feat), 1e6, device=device)
        depth_map_flat.scatter_reduce_(1, flat_pixel_indices, z, reduce='amin', include_self=False)

        min_depth_per_vertex = torch.gather(depth_map_flat, 1, flat_pixel_indices)
        is_visible = z < (min_depth_per_vertex + 1e-4)

        index_map_flat = torch.full((bsz, h_feat * w_feat), -1, dtype=torch.long, device=device)
        vertex_indices = torch.arange(num_verts, device=device).unsqueeze(0).expand(bsz, -1)

        mask_flat = is_visible.reshape(-1)
        batch_offsets = torch.arange(bsz, device=device).unsqueeze(1) * (h_feat * w_feat)
        global_pixel_indices = (flat_pixel_indices + batch_offsets).reshape(-1)

        valid_pixel_indices = global_pixel_indices[mask_flat]
        valid_vertex_indices = vertex_indices.reshape(-1)[mask_flat]

        index_map_global = index_map_flat.reshape(-1)
        index_map_global[valid_pixel_indices] = valid_vertex_indices
        return index_map_global.reshape(bsz, h_feat, w_feat), depth_map_flat.reshape(bsz, 1, h_feat, w_feat)

    def build_part_label_map(self, pred_verts, pred_cam_t, cam_int, part_h, part_w, target_h, target_w):
        source_vertex_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int, part_h, part_w, target_h, target_w
        )

        valid = source_vertex_map >= 0
        lookup_index = source_vertex_map.clamp(min=0)
        in_range = lookup_index < self.vertex_to_part.shape[0]
        lookup_index = lookup_index.clamp(max=self.vertex_to_part.shape[0] - 1)
        part_ids = self.vertex_to_part[lookup_index]

        valid = valid & in_range & (part_ids >= 0)
        return torch.where(valid, part_ids + 1, torch.zeros_like(part_ids))

    def resize_part_labels(self, part_labels, out_h, out_w):
        resized = F.interpolate(
            part_labels.float().unsqueeze(1),
            size=(out_h, out_w),
            mode='nearest'
        )
        return resized.squeeze(1).long()

    def assign_nearest_parts(self, part_labels):
        bsz, h, w = part_labels.shape
        device = part_labels.device

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        coords = torch.stack([yy, xx], dim=-1).view(-1, 2).float()
        flat_labels = part_labels.view(bsz, -1).clone()

        for b_idx in range(bsz):
            valid = flat_labels[b_idx] > 0
            if valid.all():
                continue
            if not valid.any():
                flat_labels[b_idx].fill_(1)
                continue

            bg_idx = (~valid).nonzero(as_tuple=False).squeeze(1)
            fg_idx = valid.nonzero(as_tuple=False).squeeze(1)
            nearest_idx = torch.cdist(
                coords[bg_idx].unsqueeze(0),
                coords[fg_idx].unsqueeze(0)
            ).squeeze(0).argmin(dim=1)
            flat_labels[b_idx, bg_idx] = flat_labels[b_idx, fg_idx[nearest_idx]]

        return flat_labels.view(bsz, h, w).long() - 1

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
        all_part_labels = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
            curr_bs = rgb_img.shape[0]

            with torch.no_grad():
                outs = self.preprocess(rgb_img, target_h, target_w)
                self.intermediate_features = {}
                _ = self.Backbone(outs)

                last_hook_idx = len(self.hook_handles) - 1
                sam_emb = self.intermediate_features[last_hook_idx]
                target_tokens = h_feat * w_feat
                if sam_emb.shape[1] > target_tokens:
                    sam_emb = sam_emb[:, -target_tokens:, :]
                sam_emb = sam_emb.transpose(1, 2).reshape(curr_bs, -1, h_feat, w_feat)

                dummy_batch = self._prepare_dummy_batch(sam_emb, target_h, target_w)
                self.SAM_Engine._batch_size = curr_bs
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(curr_bs, device=rgb.device)
                self.SAM_Engine.hand_batch_idx = []
                cond_info = torch.zeros(curr_bs, 3, device=rgb.device)
                cond_info[:, 2] = 1.1
                dummy_kp = torch.zeros(curr_bs, 1, 3, device=rgb.device)
                dummy_kp[..., -1] = -2

                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    _, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_emb,
                        keypoints=dummy_kp,
                        condition_info=cond_info,
                        batch=dummy_batch
                    )

                pred_verts = pose_outs[-1]['pred_vertices']
                pred_cam_t = pose_outs[-1]['pred_cam_t']
                cam_int = dummy_batch['cam_int']
                generated_mask = self.project_vertices_to_mask(
                    pred_verts, pred_cam_t, cam_int, h_feat, w_feat, target_h, target_w
                )
                part_labels_highres = self.build_part_label_map(
                    pred_verts, pred_cam_t, cam_int,
                    self.sils_size * 2, self.sils_size, target_h, target_w
                )

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
            human_mask = self.preprocess(
                generated_mask, self.sils_size * 2, self.sils_size
            ).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            human_feat = rearrange(
                human_feat.view(n, s, -1, self.sils_size * 2, self.sils_size),
                'n s c h w -> n c s h w'
            ).contiguous()

            outs = self.Gait_Net.test_1(human_feat)
            out_h, out_w = outs.shape[-2:]

            part_labels_lowres = self.resize_part_labels(part_labels_highres, out_h, out_w)
            part_labels_lowres = self.assign_nearest_parts(part_labels_lowres)
            all_part_labels.append(part_labels_lowres.view(n, s, out_h, out_w))
            all_outs.append(outs)

        all_outs = torch.cat(all_outs, dim=2)
        all_part_labels = torch.cat(all_part_labels, dim=1)
        embed_list, log_list = self.Gait_Net.test_2(all_outs, seqL, all_part_labels)

        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.concat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n * s, c, h, w)[:5].float(),
                    'image/generated_3d_mask_lowres': generated_mask.view(n * s, 1, h_feat, w_feat)[:5].float(),
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
