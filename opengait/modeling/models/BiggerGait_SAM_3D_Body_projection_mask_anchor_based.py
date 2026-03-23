import os
import sys

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from functools import partial
from PIL import Image, ImageDraw

from ..base_model import BaseModel
from .BigGait_utils.BigGait_GaitBase import *
from .BigGait_utils.save_img import pca_image


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


class BiggerGait__SAM3DBody__Projection_Mask_Anchor_Based_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)
        self.anchor_pt_path = model_cfg["anchor_pt_path"]
        self.anchor_depth_tol = model_cfg.get("anchor_depth_tol", 0.02)
        self.projection_zoom_ratio = model_cfg.get("projection_zoom_ratio", 1.0)
        self.anchor_normal_offset_scale = model_cfg.get("anchor_normal_offset_scale", 0.0)
        self.debug_anchor_vis = model_cfg.get("debug_anchor_vis", False)

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

        self._load_anchor_indices(model_cfg)

        self.Gait_Net = Baseline_AnchorMasked_ShareTime_2B(model_cfg)
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

    def _load_anchor_indices(self, model_cfg):
        if not os.path.exists(self.anchor_pt_path):
            raise FileNotFoundError(f"Cannot find anchor file: {self.anchor_pt_path}")
        anchor_data = torch.load(self.anchor_pt_path, map_location='cpu')

        if isinstance(anchor_data, torch.Tensor):
            if anchor_data.dim() != 2 or anchor_data.shape[1] != 2:
                raise ValueError(
                    f"Tensor anchor file should be [num_anchors, 2], got {tuple(anchor_data.shape)}"
                )
            anchor_patch_labels = anchor_data[:, 0].long()
            sampled_vertex_indices = anchor_data[:, 1].long()
        elif isinstance(anchor_data, dict) and 'sampled_vertex_indices' in anchor_data:
            sampled_vertex_indices = anchor_data['sampled_vertex_indices'].long()
            anchor_patch_labels = torch.arange(sampled_vertex_indices.numel(), dtype=torch.long)
        else:
            raise KeyError(
                f"Unsupported anchor file format in {self.anchor_pt_path}. "
                "Expected {'sampled_vertex_indices': ...} or a [num_anchors, 2] tensor."
            )

        num_patches = int(anchor_patch_labels.max().item()) + 1
        expected_parts = model_cfg['SeparateFCs']['parts_num']
        if num_patches != expected_parts:
            raise ValueError(
                f"Loaded {num_patches} anchor patches from {self.anchor_pt_path}, "
                f"but SeparateFCs.parts_num is {expected_parts}"
            )
        self.register_buffer("sampled_vertex_indices", sampled_vertex_indices, persistent=False)
        self.register_buffer("anchor_patch_labels", anchor_patch_labels, persistent=False)
        self.num_anchors = sampled_vertex_indices.numel()
        self.num_anchor_patches = num_patches
        self.msg_mgr.log_info(
            f"[Anchor] Loaded {self.num_anchors} sampled vertices grouped into "
            f"{self.num_anchor_patches} patches from {self.anchor_pt_path}"
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
        if hasattr(estimator, 'faces'):
            faces = estimator.faces
        elif hasattr(estimator.model, 'head_pose') and hasattr(estimator.model.head_pose, 'faces'):
            faces = estimator.model.head_pose.faces
        else:
            raise RuntimeError("Cannot find mesh faces in SAM estimator")
        if not isinstance(faces, torch.Tensor):
            faces = torch.as_tensor(faces)
        self.register_buffer("mesh_faces", faces.long(), persistent=False)

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

    def _get_projection_cam_int(self, cam_int):
        if self.projection_zoom_ratio == 1.0:
            return cam_int
        proj_cam_int = cam_int.clone()
        proj_cam_int[:, 0, 0] = proj_cam_int[:, 0, 0] * self.projection_zoom_ratio
        proj_cam_int[:, 1, 1] = proj_cam_int[:, 1, 1] * self.projection_zoom_ratio
        return proj_cam_int

    def _project_vertices_to_feature_coords(self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w):
        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy

        u_cont = (u / target_w * w_feat).clamp(0, w_feat - 1e-4)
        v_cont = (v / target_h * h_feat).clamp(0, h_feat - 1e-4)
        return u_cont, v_cont

    def _compute_vertex_normals(self, vertices):
        bsz, num_verts, _ = vertices.shape
        faces = self.mesh_faces.to(vertices.device)

        v0 = vertices[:, faces[:, 0], :]
        v1 = vertices[:, faces[:, 1], :]
        v2 = vertices[:, faces[:, 2], :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        vertex_normals = torch.zeros_like(vertices)
        vertex_normals_flat = vertex_normals.reshape(bsz * num_verts, 3)
        batch_offsets = (torch.arange(bsz, device=vertices.device) * num_verts).view(bsz, 1)
        face_idx = faces.unsqueeze(0) + batch_offsets.unsqueeze(-1)

        vertex_normals_flat.index_add_(0, face_idx[:, :, 0].reshape(-1), face_normals.reshape(-1, 3))
        vertex_normals_flat.index_add_(0, face_idx[:, :, 1].reshape(-1), face_normals.reshape(-1, 3))
        vertex_normals_flat.index_add_(0, face_idx[:, :, 2].reshape(-1), face_normals.reshape(-1, 3))
        vertex_normals = vertex_normals_flat.view(bsz, num_verts, 3)
        return F.normalize(vertex_normals, p=2, dim=-1, eps=1e-6)

    def _should_log_visual_summary(self):
        if not self.training:
            return False
        log_iter = self.engine_cfg.get('log_iter', None)
        if not log_iter:
            return False
        return ((self.iteration + 1) % log_iter) == 0

    def _build_anchor_debug_overlay_batch(self, feat_map, anchor_u_cont, anchor_v_cont, visible_mask, max_frames=5):
        feat_map = feat_map.detach().float().cpu()
        anchor_u_cont = anchor_u_cont.detach().float().cpu()
        anchor_v_cont = anchor_v_cont.detach().float().cpu()
        visible_mask = visible_mask.detach().bool().cpu()

        num_frames = min(max_frames, feat_map.shape[0])
        vis_frames = []
        scale = 8
        gray_r = 0.6
        white_r = 0.9

        for idx in range(num_frames):
            curr_feat = feat_map[idx]
            _, height, width = curr_feat.shape
            feat_np = rearrange(curr_feat.numpy(), 'c h w -> 1 (h w) c')
            mask_np = np.ones((1, height * width), dtype=np.uint8)
            pca_img = pca_image(
                data={'embeddings': feat_np, 'h': height, 'w': width},
                mask=mask_np,
                root=None,
                model_name=None,
                dataset=None,
                n_components=3,
                is_return=True,
            )[0, 0].transpose(1, 2, 0)

            overlay = Image.fromarray(pca_img).resize((width * scale, height * scale), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(overlay)

            for uu, vv in zip(anchor_u_cont[idx].tolist(), anchor_v_cont[idx].tolist()):
                cx = uu * scale
                cy = vv * scale
                draw.ellipse((cx - gray_r, cy - gray_r, cx + gray_r, cy + gray_r), fill=(150, 150, 150))

            for uu, vv, vis in zip(anchor_u_cont[idx].tolist(), anchor_v_cont[idx].tolist(), visible_mask[idx].tolist()):
                if vis:
                    cx = uu * scale
                    cy = vv * scale
                    draw.ellipse((cx - white_r, cy - white_r, cx + white_r, cy + white_r), fill=(255, 255, 255))

            vis_np = np.asarray(overlay, dtype=np.float32) / 255.0
            vis_frames.append(torch.from_numpy(vis_np).permute(2, 0, 1))

        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _build_feature_norm_vis_batch(self, feat_map, max_frames=5):
        feat_map = feat_map.detach().float().cpu()
        num_frames = min(max_frames, feat_map.shape[0])
        vis_frames = []

        for idx in range(num_frames):
            curr_feat = feat_map[idx]
            norm_map = torch.linalg.vector_norm(curr_feat, ord=2, dim=0, keepdim=True)
            min_val = norm_map.min()
            max_val = norm_map.max()
            if (max_val - min_val) > 1e-6:
                norm_map = (norm_map - min_val) / (max_val - min_val)
            else:
                norm_map = torch.zeros_like(norm_map)
            vis_frames.append(norm_map)

        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

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

        u_cont = (u / target_w * w_feat).clamp(0, w_feat - 1e-4)
        v_cont = (v / target_h * h_feat).clamp(0, h_feat - 1e-4)
        u_feat = u_cont.long()
        v_feat = v_cont.long()

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

        u_cont = (u / target_w * w_feat).clamp(0, w_feat - 1e-4)
        v_cont = (v / target_h * h_feat).clamp(0, h_feat - 1e-4)
        u_feat = u_cont.long()
        v_feat = v_cont.long()
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

    def extract_anchor_features(
        self,
        feat_map,
        anchor_vertices,
        cam_t,
        cam_int,
        dense_depth_map,
        h_feat,
        w_feat,
        target_h,
        target_w,
        return_debug=False
    ):
        bsz, channels, _, _ = feat_map.shape
        device = feat_map.device

        v_cam = anchor_vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy

        u_cont = (u / target_w * w_feat).clamp(0, w_feat - 1e-4)
        v_cont = (v / target_h * h_feat).clamp(0, h_feat - 1e-4)
        u_feat = u_cont.long()
        v_feat = v_cont.long()
        flat_pixel_indices = v_feat * w_feat + u_feat

        front_valid = torch.ones_like(z, dtype=torch.bool)

        sparse_depth_flat = torch.full((bsz, h_feat * w_feat), 1e6, device=device)
        masked_z = torch.where(front_valid, z, torch.full_like(z, 1e6))
        sparse_depth_flat.scatter_reduce_(1, flat_pixel_indices, masked_z, reduce='amin', include_self=False)
        sparse_min_depth = torch.gather(sparse_depth_flat, 1, flat_pixel_indices)
        is_visible = front_valid & (z <= (sparse_min_depth + 1e-4))

        feat_flat = rearrange(feat_map, 'b c h w -> b (h w) c').contiguous()
        anchor_feats = feat_map.new_zeros(bsz, self.num_anchors, channels)
        anchor_valid = torch.zeros(bsz, self.num_anchors, dtype=torch.bool, device=device)

        for b_idx in range(bsz):
            visible_idx = is_visible[b_idx].nonzero(as_tuple=False).squeeze(1)
            if visible_idx.numel() == 0:
                continue
            pixel_idx = flat_pixel_indices[b_idx, visible_idx]
            anchor_feats[b_idx, visible_idx] = feat_flat[b_idx, pixel_idx]
            anchor_valid[b_idx, visible_idx] = True

        if return_debug:
            unique_cells = []
            for b_idx in range(bsz):
                unique_cells.append(torch.unique(flat_pixel_indices[b_idx]).numel())
            unique_cells = torch.tensor(unique_cells, device=device, dtype=torch.long)
            debug_info = {
                'u_cont': u_cont,
                'v_cont': v_cont,
                'u_feat': u_feat,
                'v_feat': v_feat,
                'flat_pixel_indices': flat_pixel_indices,
                'front_valid': front_valid,
                'is_visible': is_visible,
                'unique_cells': unique_cells,
            }
            return anchor_feats, anchor_valid, debug_info
        return anchor_feats, anchor_valid

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        num_rgb_chunks = len(rgb_chunks)
        should_log_anchor_vis = self.debug_anchor_vis and self._should_log_visual_summary()
        anchor_overlay_summary = None
        cnn_layer2_norm_summary = None
        all_anchor_feats = []
        all_anchor_valid = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        for chunk_idx, rgb_img in enumerate(rgb_chunks):
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
                proj_cam_int = self._get_projection_cam_int(cam_int)
                if self.anchor_normal_offset_scale != 0.0:
                    vertex_normals = self._compute_vertex_normals(pred_verts)
                else:
                    vertex_normals = None
                generated_mask = self.project_vertices_to_mask(
                    pred_verts, pred_cam_t, cam_int, h_feat, w_feat, target_h, target_w
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
                sub_app = partial(nn.LayerNorm, eps=1e-6)(curr_dim, elementwise_affine=False)(sub_app)
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_feat = torch.concat(processed_feat_list, dim=1)
            human_mask = self.preprocess(generated_mask, self.sils_size * 2, self.sils_size).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            human_feat = rearrange(
                human_feat.view(n, s, -1, self.sils_size * 2, self.sils_size),
                'n s c h w -> n c s h w'
            ).contiguous()

            debug_test_1 = should_log_anchor_vis and (chunk_idx == num_rgb_chunks - 1)
            test_1_outputs = self.Gait_Net.test_1(human_feat, return_debug=debug_test_1)
            if debug_test_1:
                outs, gait_debug = test_1_outputs
                layer2_feat = rearrange(gait_debug['layer2_feat'], 'n c s h w -> (n s) c h w').contiguous()
                cnn_layer2_norm_summary = self._build_feature_norm_vis_batch(layer2_feat[:5])
            else:
                outs = test_1_outputs
            out_h, out_w = outs.shape[-2:]

            dense_depth_map = self.get_source_vertex_index_map(
                pred_verts, pred_cam_t, cam_int, out_h, out_w, target_h, target_w
            )[1]
            anchor_vertices = pred_verts[:, self.sampled_vertex_indices, :]
            if vertex_normals is not None:
                anchor_normals = vertex_normals[:, self.sampled_vertex_indices, :]
                anchor_vertices_for_projection = anchor_vertices + self.anchor_normal_offset_scale * anchor_normals
            else:
                anchor_vertices_for_projection = anchor_vertices
            anchor_feat_map = rearrange(outs, 'n c s h w -> (n s) c h w').contiguous()
            debug_extract = should_log_anchor_vis and (chunk_idx == num_rgb_chunks - 1)
            extract_outputs = self.extract_anchor_features(
                anchor_feat_map,
                anchor_vertices_for_projection,
                pred_cam_t,
                proj_cam_int,
                dense_depth_map,
                out_h,
                out_w,
                target_h,
                target_w,
                return_debug=debug_extract
            )
            if debug_extract:
                anchor_feats_flat, anchor_valid_flat, anchor_debug = extract_outputs
            else:
                anchor_feats_flat, anchor_valid_flat = extract_outputs

            if debug_extract:
                anchor_overlay_summary = self._build_anchor_debug_overlay_batch(
                    anchor_feat_map[:5],
                    anchor_debug['u_cont'][:5],
                    anchor_debug['v_cont'][:5],
                    anchor_valid_flat[:5],
                )

            anchor_feats = rearrange(anchor_feats_flat, '(n s) k c -> n c s k', n=n, s=s).contiguous()
            anchor_valid = rearrange(anchor_valid_flat, '(n s) k -> n s k', n=n, s=s).contiguous()

            all_anchor_feats.append(anchor_feats)
            all_anchor_valid.append(anchor_valid)

        all_anchor_feats = torch.cat(all_anchor_feats, dim=2)
        all_anchor_valid = torch.cat(all_anchor_valid, dim=1)
        embed_list, log_list = self.Gait_Net.test_2(
            all_anchor_feats, seqL, all_anchor_valid, self.anchor_patch_labels
        )
        embeddings = torch.concat(embed_list, dim=-1)

        if self.training:
            visual_summary = {
                'image/rgb_img': rgb_img.view(n * s, c, h, w)[:5].float(),
                'image/generated_3d_mask_lowres': generated_mask.view(n * s, 1, h_feat, w_feat)[:5].float(),
            }
            if anchor_overlay_summary is not None:
                visual_summary['image/anchor_overlay_pca'] = anchor_overlay_summary.float()
            if cnn_layer2_norm_summary is not None:
                visual_summary['image/cnn_layer2_l2norm'] = cnn_layer2_norm_summary.float()
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embeddings, 'labels': labs},
                    'softmax': {'logits': torch.concat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': visual_summary,
                'inference_feat': {
                    'embeddings': embeddings,
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)},
                }
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': embeddings,
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)},
                }
            }
        return retval
