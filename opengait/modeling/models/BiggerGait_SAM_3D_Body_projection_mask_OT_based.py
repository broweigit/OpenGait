import copy
import math
import sys

import numpy as np
import roma
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from torch.nn import functional as F
from functools import partial

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


class GeometryOptimalTransport(nn.Module):
    def __init__(self, temperature=0.01, dist_thresh=0.2, num_iters=8):
        super().__init__()
        self.epsilon = temperature
        self.dist_thresh = dist_thresh
        self.num_iters = num_iters

    def forward(self, source_feats, source_locs, target_locs, source_valid_mask=None, target_valid_mask=None):
        bsz, _, _ = source_feats.shape

        with torch.no_grad():
            diff = target_locs.unsqueeze(2) - source_locs.unsqueeze(1)
            dist_sq = torch.sum(diff ** 2, dim=-1)

            log_k = -dist_sq / (self.epsilon + 1e-8)
            valid_connection = dist_sq < (self.dist_thresh ** 2)
            del diff, dist_sq

            if source_valid_mask is not None:
                valid_connection = valid_connection & source_valid_mask.unsqueeze(1)
            if target_valid_mask is not None:
                valid_connection = valid_connection & target_valid_mask.unsqueeze(2)

            log_k = log_k.masked_fill(~valid_connection, -1e9)

            src_count = source_feats.shape[1]
            tgt_count = target_locs.shape[1]
            v = torch.zeros(bsz, 1, src_count, device=source_feats.device)
            u = torch.zeros(bsz, tgt_count, 1, device=source_feats.device)

            for _ in range(self.num_iters):
                u = -torch.logsumexp(log_k + v, dim=2, keepdim=True)
                v = -torch.logsumexp(log_k + u, dim=1, keepdim=True)
                if source_valid_mask is not None:
                    v = v.masked_fill(~source_valid_mask.unsqueeze(1), 0.0)

            attn = torch.exp(log_k + u + v)
            has_source = valid_connection.any(dim=-1, keepdim=True)

        target_feats = torch.bmm(attn, source_feats)
        if target_valid_mask is not None:
            target_feats = target_feats * target_valid_mask.unsqueeze(-1).float()
        target_feats = target_feats * has_source.float()
        return target_feats


class BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)
        self.debug_pca_vis = model_cfg.get("debug_pca_vis", False)

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

        self.branch_configs = model_cfg["branch_configs"]
        self.num_branches = len(self.branch_configs)
        if self.num_branches == 0:
            raise ValueError("branch_configs must contain at least one branch.")
        for branch_cfg in self.branch_configs:
            if "yaw" not in branch_cfg:
                raise ValueError("Each branch config must contain a yaw field.")

        self.Gait_Nets = nn.ModuleList([
            Baseline_ShareTime_2B(copy.deepcopy(model_cfg)) for _ in range(self.num_branches)
        ])
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
        self.ot_solver = GeometryOptimalTransport(
            temperature=model_cfg.get("ot_temperature", 0.01),
            dist_thresh=model_cfg.get("ot_dist_thresh", 0.2),
            num_iters=model_cfg.get("ot_iters", 8),
        )
        self.init_SAM_Backbone()

    def init_SAM_Backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)

        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Error: {e}")

        self.msg_mgr.log_info("[SAM3D] Loading SAM 3D Body (Encoder + Decoder)...")
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

    def build_target_camera(self, batch_size, device, target_h, target_w):
        focal_tgt = max(target_h, target_w) * 1.1
        cx_tgt, cy_tgt = target_w / 2.0, target_h / 2.0

        cam_int_tgt = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3).clone()
        cam_int_tgt[:, 0, 0] = focal_tgt
        cam_int_tgt[:, 1, 1] = focal_tgt
        cam_int_tgt[:, 0, 2] = cx_tgt
        cam_int_tgt[:, 1, 2] = cy_tgt

        cam_t_tgt = torch.zeros((batch_size, 3), device=device)
        cam_t_tgt[:, 2] = 2.2
        return cam_int_tgt, cam_t_tgt

    def generate_mhr_apose(self, pose_out):
        device = pose_out['pred_vertices'].device
        batch_size = pose_out['pred_vertices'].shape[0]

        pred_shape = pose_out['shape'].float()
        pred_scale = pose_out['scale'].float()
        pred_face = pose_out['face'].float()

        zero_global_trans = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        zero_global_rot = torch.zeros_like(pose_out['global_rot'], dtype=torch.float32)
        zero_hand_pose = torch.zeros_like(pose_out['hand'], dtype=torch.float32)
        apose_body = torch.zeros_like(pose_out['body_pose'], dtype=torch.float32)

        angle_rad = math.radians(-20.0)
        apose_body[:, 25] = angle_rad
        apose_body[:, 35] = angle_rad

        with torch.no_grad(), torch.amp.autocast(enabled=False, device_type='cuda'):
            apose_outputs = self.SAM_Engine.head_pose.mhr_forward(
                global_trans=zero_global_trans,
                global_rot=zero_global_rot,
                body_pose_params=apose_body,
                hand_pose_params=zero_hand_pose,
                scale_params=pred_scale,
                shape_params=pred_shape,
                expr_params=pred_face,
                return_keypoints=True
            )

        apose_verts = apose_outputs[0]
        apose_keypoints = apose_outputs[1][:, :70]
        apose_verts[..., [1, 2]] *= -1
        apose_keypoints[..., [1, 2]] *= -1
        return apose_verts, apose_keypoints

    def build_branch_geometry(self, branch_cfg, pose_out):
        use_apose = branch_cfg.get("use_apose", False)
        yaw = float(branch_cfg.get("yaw", 0.0))

        if use_apose:
            if not hasattr(self, "_apose_cache"):
                self._apose_cache = {}
            cache_key = id(pose_out)
            if cache_key not in self._apose_cache:
                self._apose_cache = {cache_key: self.generate_mhr_apose(pose_out)}
            branch_verts, branch_keypoints = self._apose_cache[cache_key]
            apply_global_rot_alignment = False
        else:
            branch_verts = pose_out['pred_vertices']
            branch_keypoints = pose_out['pred_keypoints_3d']
            apply_global_rot_alignment = True

        return {
            "verts": branch_verts,
            "keypoints": branch_keypoints,
            "yaw": yaw,
            "apply_global_rot_alignment": apply_global_rot_alignment,
        }

    def rotate_branch_geometry(self, verts, keypoints, global_rot, yaw, apply_global_rot_alignment):
        batch_size = verts.shape[0]
        device = verts.device

        midhip = (keypoints[:, 9] + keypoints[:, 10]) / 2.0
        centered_verts = verts - midhip.unsqueeze(1)

        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        r_yaw = torch.tensor(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 3).expand(batch_size, 3, 3)

        if apply_global_rot_alignment:
            rot_fix = global_rot.clone()
            rot_fix[..., [0, 1, 2]] *= -1
            r_canon = roma.euler_to_rotmat("XYZ", rot_fix)
            r_comp = torch.matmul(r_canon.transpose(1, 2), r_yaw.transpose(1, 2))
        else:
            r_comp = r_yaw.transpose(1, 2)

        verts_tmp = centered_verts.clone()
        verts_tmp[..., [1, 2]] *= -1
        rotated_smpl = torch.bmm(verts_tmp, r_comp)
        rotated_cv = rotated_smpl.clone()
        rotated_cv[..., [1, 2]] *= -1
        return rotated_cv, midhip, r_comp

    def warp_features_with_ot(
        self,
        human_feat,
        mask_src,
        pred_verts,
        branch_verts,
        branch_keypoints,
        pred_cam_t,
        global_rot,
        cam_int_src,
        cam_int_tgt,
        cam_t_tgt,
        h_feat,
        w_feat,
        target_h,
        target_w,
        yaw,
        apply_global_rot_alignment,
    ):
        bsz, _, _, _ = human_feat.shape
        device = human_feat.device

        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int_src, h_feat, w_feat, target_h, target_w
        )
        valid_src_mask = (mask_src.squeeze(1) > 0.5) & (src_idx_map >= 0)

        flat_human_feat = rearrange(human_feat, 'b c h w -> b (h w) c')
        flat_src_idx_map = src_idx_map.view(bsz, -1)
        flat_src_mask = valid_src_mask.view(bsz, -1)

        safe_indices = flat_src_idx_map.clone()
        safe_indices[safe_indices < 0] = 0
        flat_src_verts = torch.gather(branch_verts, 1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))

        v_rot_cv, midhip, r_comp = self.rotate_branch_geometry(
            branch_verts, branch_keypoints, global_rot, yaw, apply_global_rot_alignment
        )

        _, tgt_depth_map = self.get_source_vertex_index_map(
            v_rot_cv, cam_t_tgt, cam_int_tgt, h_feat, w_feat, target_h, target_w
        )
        valid_tgt_mask = tgt_depth_map.view(bsz, -1) < 1e5

        src_centered = flat_src_verts - midhip.unsqueeze(1)
        src_tmp = src_centered.clone()
        src_tmp[..., [1, 2]] *= -1
        src_rot_smpl = torch.bmm(src_tmp, r_comp)
        src_rot_cv = src_rot_smpl.clone()
        src_rot_cv[..., [1, 2]] *= -1

        v_cam_tgt = src_rot_cv + cam_t_tgt.unsqueeze(1)
        x, y, z = v_cam_tgt.unbind(-1)
        z = z.clamp(min=1e-3)

        fx, fy = cam_int_tgt[:, 0, 0].unsqueeze(1), cam_int_tgt[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int_tgt[:, 0, 2].unsqueeze(1), cam_int_tgt[:, 1, 2].unsqueeze(1)
        u_tgt = (x / z) * fx + cx
        v_tgt = (y / z) * fy + cy

        u_norm = 2.0 * (u_tgt / target_w) - 1.0
        v_norm = 2.0 * (v_tgt / target_h) - 1.0
        projected_source_locs = torch.stack([u_norm, v_norm], dim=-1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h_feat, device=device),
            torch.linspace(-1, 1, w_feat, device=device),
            indexing='ij'
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz, -1, 2)

        transported_feats = self.ot_solver(
            flat_human_feat,
            projected_source_locs,
            target_grid_locs,
            source_valid_mask=flat_src_mask,
            target_valid_mask=valid_tgt_mask,
        )

        warped_feat = rearrange(transported_feats, 'b (h w) c -> b c h w', h=h_feat)
        return warped_feat, valid_tgt_mask.view(bsz, 1, h_feat, w_feat), tgt_depth_map

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def _should_log_visual_summary(self):
        if not self.training:
            return False
        log_iter = self.engine_cfg.get('log_iter', None)
        if not log_iter:
            return False
        return ((self.iteration + 1) % log_iter) == 0

    def _build_pca_vis_batch(self, feat_map, max_frames=5):
        feat_map = feat_map.detach().float().cpu()
        num_frames = min(max_frames, feat_map.shape[0])
        vis_frames = []
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
            )[0, 0]
            vis_frames.append(torch.from_numpy(pca_img).float() / 255.0)
        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _build_feature_norm_vis_batch(self, rgb_frames, feat_map, max_frames=5):
        rgb_frames = rgb_frames.detach().float().cpu()
        feat_map = feat_map.detach().float().cpu()
        num_frames = min(max_frames, feat_map.shape[0])
        vis_frames = []

        for idx in range(num_frames):
            rgb_frame = rgb_frames[idx]
            curr_feat = feat_map[idx]
            norm_map = torch.linalg.vector_norm(curr_feat, ord=2, dim=0, keepdim=True)
            min_val = norm_map.min()
            max_val = norm_map.max()
            if (max_val - min_val) > 1e-6:
                norm_map = (norm_map - min_val) / (max_val - min_val)
            else:
                norm_map = torch.zeros_like(norm_map)

            norm_map = F.interpolate(
                norm_map.unsqueeze(0),
                size=rgb_frame.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            if rgb_frame.min() < 0 or rgb_frame.max() > 1:
                rgb_min = rgb_frame.amin(dim=(-2, -1), keepdim=True)
                rgb_max = rgb_frame.amax(dim=(-2, -1), keepdim=True)
                rgb_frame = (rgb_frame - rgb_min) / (rgb_max - rgb_min + 1e-6)
            else:
                rgb_frame = rgb_frame.clamp(0, 1)

            x = norm_map[0].clamp(0, 1)
            heat_r = (1.5 - torch.abs(4.0 * x - 3.0)).clamp(0.0, 1.0)
            heat_g = (1.5 - torch.abs(4.0 * x - 2.0)).clamp(0.0, 1.0)
            heat_b = (1.5 - torch.abs(4.0 * x - 1.0)).clamp(0.0, 1.0)
            heat_map = torch.stack([heat_r, heat_g, heat_b], dim=0)
            alpha = 0.7 * norm_map.pow(0.85)
            vis_frame = rgb_frame * (1.0 - alpha) + heat_map * alpha
            vis_frames.append(vis_frame.clamp(0, 1))

        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _build_feature_norm_on_depth_vis_batch(self, depth_maps, feat_map, max_frames=5):
        depth_maps = depth_maps.detach().float().cpu()
        feat_map = feat_map.detach().float().cpu()
        num_frames = min(max_frames, feat_map.shape[0], depth_maps.shape[0])
        vis_frames = []
        invalid_gray = 0.25

        for idx in range(num_frames):
            depth_map = depth_maps[idx]
            curr_feat = feat_map[idx]
            if depth_map.dim() == 2:
                depth_map = depth_map.unsqueeze(0)

            valid_mask = depth_map[0] < 1e5
            inv_depth = torch.zeros_like(depth_map[0])
            if valid_mask.any():
                valid_depth = depth_map[0][valid_mask].clamp(min=1e-6)
                inv_valid_depth = 1.0 / valid_depth
                inv_min = inv_valid_depth.min()
                inv_max = inv_valid_depth.max()
                if (inv_max - inv_min) > 1e-6:
                    inv_depth[valid_mask] = (inv_valid_depth - inv_min) / (inv_max - inv_min)
                else:
                    inv_depth[valid_mask] = 1.0

            base_frame = inv_depth.unsqueeze(0).repeat(3, 1, 1)
            base_frame[:, ~valid_mask] = invalid_gray

            norm_map = torch.linalg.vector_norm(curr_feat, ord=2, dim=0, keepdim=True)
            min_val = norm_map.min()
            max_val = norm_map.max()
            if (max_val - min_val) > 1e-6:
                norm_map = (norm_map - min_val) / (max_val - min_val)
            else:
                norm_map = torch.zeros_like(norm_map)

            norm_map = F.interpolate(
                norm_map.unsqueeze(0),
                size=base_frame.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            x = norm_map[0].clamp(0, 1)
            heat_r = (1.5 - torch.abs(4.0 * x - 3.0)).clamp(0.0, 1.0)
            heat_g = (1.5 - torch.abs(4.0 * x - 2.0)).clamp(0.0, 1.0)
            heat_b = (1.5 - torch.abs(4.0 * x - 1.0)).clamp(0.0, 1.0)
            heat_map = torch.stack([heat_r, heat_g, heat_b], dim=0)
            alpha = 0.7 * norm_map.pow(0.85)
            vis_frame = base_frame * (1.0 - alpha) + heat_map * alpha
            vis_frames.append(vis_frame.clamp(0, 1))

        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _stack_branch_vis(self, branch_vis_list):
        valid_vis = [vis for vis in branch_vis_list if vis is not None]
        if not valid_vis:
            return None
        num_frames = min(vis.shape[0] for vis in valid_vis)
        return torch.cat([vis[:num_frames] for vis in valid_vis], dim=2)

    def _stack_fpn_vis(self, fpn_vis_list):
        valid_vis = [vis for vis in fpn_vis_list if vis is not None]
        if not valid_vis:
            return None
        num_frames = min(vis.shape[0] for vis in valid_vis)
        return torch.cat([vis[:num_frames] for vis in valid_vis], dim=3)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        branch_layer1_norm = [None] * self.num_branches
        branch_pca_before_cnn = [None] * self.num_branches
        branch_pca_after_cnn = [None] * self.num_branches
        branch_layer2_norm = [None] * self.num_branches
        branch_layer3_norm = [None] * self.num_branches
        branch_layer4_norm = [None] * self.num_branches
        branch_target_depth = [None] * self.num_branches
        all_outs = [[] for _ in range(self.num_branches)]
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        num_rgb_chunks = len(rgb_chunks)
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

                pose_out = pose_outs[-1]
                self._apose_cache = {}
                pred_verts = pose_out['pred_vertices']
                pred_cam_t = pose_out['pred_cam_t']
                global_rot = pose_out['global_rot']
                cam_int_src = dummy_batch['cam_int']
                _, src_depth_map = self.get_source_vertex_index_map(
                    pred_verts, pred_cam_t, cam_int_src, h_feat, w_feat, target_h, target_w
                )
                generated_mask = (src_depth_map < 1e5).float()

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

            cam_int_tgt, cam_t_tgt = self.build_target_camera(curr_bs, rgb.device, target_h, target_w)
            branch_warped_feats = []
            for b_idx, branch_cfg in enumerate(self.branch_configs):
                branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
                warp_feat, _, tgt_depth_map = self.warp_features_with_ot(
                    human_feat,
                    human_mask.float(),
                    pred_verts,
                    branch_geo["verts"],
                    branch_geo["keypoints"],
                    pred_cam_t,
                    global_rot,
                    cam_int_src,
                    cam_int_tgt,
                    cam_t_tgt,
                    self.sils_size * 2,
                    self.sils_size,
                    target_h,
                    target_w,
                    branch_geo["yaw"],
                    branch_geo["apply_global_rot_alignment"],
                )
                branch_warped_feats.append(warp_feat)
                branch_target_depth[b_idx] = tgt_depth_map

            debug_test_1 = should_log_pca_vis and (chunk_idx == num_rgb_chunks - 1)
            for b_idx, warp_feat in enumerate(branch_warped_feats):
                warp_feat_5d = rearrange(warp_feat, '(n s) c h w -> n c s h w', n=n, s=s).contiguous()
                if debug_test_1:
                    pca_before_vis = []
                    for in_chunk in torch.chunk(warp_feat_5d, self.num_FPN, dim=1):
                        pca_before_vis.append(
                            self._build_pca_vis_batch(
                                rearrange(in_chunk, 'n c s h w -> (n s) c h w').contiguous()[:5]
                            )
                        )
                    branch_pca_before_cnn[b_idx] = self._stack_fpn_vis(pca_before_vis)
                    outs, gait_debug = self.Gait_Nets[b_idx].test_1(
                        warp_feat_5d, return_debug=True
                    )
                    layer1_vis = []
                    layer2_vis = []
                    layer3_vis = []
                    layer4_vis = []
                    pca_vis = []
                    for i in range(self.num_FPN):
                        layer1_feat = rearrange(
                            gait_debug['layer1_feat_list'][i], 'n c s h w -> (n s) c h w'
                        ).contiguous()
                        layer2_feat = rearrange(
                            gait_debug['layer2_feat_list'][i], 'n c s h w -> (n s) c h w'
                        ).contiguous()
                        layer3_feat = rearrange(
                            gait_debug['layer3_feat_list'][i], 'n c s h w -> (n s) c h w'
                        ).contiguous()
                        layer4_feat = rearrange(
                            gait_debug['layer4_feat_list'][i], 'n c s h w -> (n s) c h w'
                        ).contiguous()
                        layer1_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(
                                branch_target_depth[b_idx][:5], layer1_feat[:5]
                            )
                        )
                        layer2_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(
                                branch_target_depth[b_idx][:5], layer2_feat[:5]
                            )
                        )
                        layer3_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(
                                branch_target_depth[b_idx][:5], layer3_feat[:5]
                            )
                        )
                        layer4_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(
                                branch_target_depth[b_idx][:5], layer4_feat[:5]
                            )
                        )
                    for out_chunk in torch.chunk(outs, self.num_FPN, dim=1):
                        pca_vis.append(
                            self._build_pca_vis_batch(
                                rearrange(out_chunk, 'n c s h w -> (n s) c h w').contiguous()[:5]
                            )
                        )
                    branch_layer1_norm[b_idx] = self._stack_fpn_vis(layer1_vis)
                    branch_layer2_norm[b_idx] = self._stack_fpn_vis(layer2_vis)
                    branch_layer3_norm[b_idx] = self._stack_fpn_vis(layer3_vis)
                    branch_layer4_norm[b_idx] = self._stack_fpn_vis(layer4_vis)
                    branch_pca_after_cnn[b_idx] = self._stack_fpn_vis(pca_vis)
                elif self.training:
                    outs = torch.utils.checkpoint.checkpoint(
                        self.Gait_Nets[b_idx].test_1,
                        warp_feat_5d,
                        use_reentrant=False,
                    )
                else:
                    outs = self.Gait_Nets[b_idx].test_1(warp_feat_5d)
                all_outs[b_idx].append(outs)

        embed_grouped = [[] for _ in range(self.num_FPN)]
        log_grouped = [[] for _ in range(self.num_FPN)]

        for b_idx in range(self.num_branches):
            branch_seq_feat = torch.cat(all_outs[b_idx], dim=2)
            e_list, l_list = self.Gait_Nets[b_idx].test_2(branch_seq_feat, seqL)
            for i in range(self.num_FPN):
                embed_grouped[i].append(e_list[i])
                log_grouped[i].append(l_list[i])

        embed_list = [torch.cat(feats, dim=-1) for feats in embed_grouped]
        log_list = [torch.cat(logits, dim=-1) for logits in log_grouped]
        cnn_layer1_norm_summary = self._stack_branch_vis(branch_layer1_norm)
        pca_before_cnn_summary = self._stack_branch_vis(branch_pca_before_cnn)
        pca_after_cnn_summary = self._stack_branch_vis(branch_pca_after_cnn)
        cnn_layer2_norm_summary = self._stack_branch_vis(branch_layer2_norm)
        cnn_layer3_norm_summary = self._stack_branch_vis(branch_layer3_norm)
        cnn_layer4_norm_summary = self._stack_branch_vis(branch_layer4_norm)

        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': torch.cat(embed_list, dim=-1), 'labels': labs},
                    'softmax': {'logits': torch.cat(log_list, dim=-1), 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n * s, c, h, w)[:5].float(),
                    'image/generated_3d_mask_lowres': generated_mask.view(n * s, 1, h_feat, w_feat)[:5].float(),
                    'image/generated_3d_mask_interpolated': human_mask.view(n * s, 1, self.sils_size * 2, self.sils_size)[:5].float(),
                },
                'inference_feat': {
                    'embeddings': torch.cat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
            if cnn_layer1_norm_summary is not None:
                retval['visual_summary']['image/cnn_layer1_l2norm'] = cnn_layer1_norm_summary.float()
            if pca_before_cnn_summary is not None:
                retval['visual_summary']['image/pca_before_cnn'] = pca_before_cnn_summary.float()
            if pca_after_cnn_summary is not None:
                retval['visual_summary']['image/pca_after_cnn'] = pca_after_cnn_summary.float()
            if cnn_layer2_norm_summary is not None:
                retval['visual_summary']['image/cnn_layer2_l2norm'] = cnn_layer2_norm_summary.float()
            if cnn_layer3_norm_summary is not None:
                retval['visual_summary']['image/cnn_layer3_l2norm'] = cnn_layer3_norm_summary.float()
            if cnn_layer4_norm_summary is not None:
                retval['visual_summary']['image/cnn_layer4_l2norm'] = cnn_layer4_norm_summary.float()
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': torch.cat(embed_list, dim=-1),
                    **{f'embeddings_{i}': embed_list[i] for i in range(self.num_FPN)}
                }
            }
        return retval
