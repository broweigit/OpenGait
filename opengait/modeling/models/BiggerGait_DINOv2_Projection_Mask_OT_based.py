import copy
import math
import sys
import time

import numpy as np
import roma
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

from .BiggerGait_DINOv2_Projection_Mask_based import BiggerGait__DINOv2__Projection_Mask_Based


class GeometryOptimalTransport(nn.Module):
    def __init__(
        self,
        temperature=0.01,
        dist_thresh=0.2,
        num_iters=8,
        topk_support=0,
        sparse_rebalance_iters=None,
    ):
        super().__init__()
        self.epsilon = temperature
        self.dist_thresh = dist_thresh
        self.num_iters = num_iters
        self.topk_support = int(topk_support) if topk_support else 0
        if sparse_rebalance_iters is None or int(sparse_rebalance_iters) <= 0:
            self.sparse_rebalance_iters = num_iters
        else:
            self.sparse_rebalance_iters = int(sparse_rebalance_iters)

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

            if self.topk_support > 0:
                src_count = source_feats.shape[1]
                topk = min(self.topk_support, src_count)
                if topk < src_count:
                    masked_log_k = log_k.masked_fill(~valid_connection, -1e9)
                    topk_idx = masked_log_k.topk(k=topk, dim=2, largest=True).indices
                    sparse_support = torch.zeros_like(valid_connection)
                    sparse_support.scatter_(2, topk_idx, True)
                    valid_connection = valid_connection & sparse_support
                    del sparse_support, masked_log_k, topk_idx

            log_k = log_k.masked_fill(~valid_connection, -1e9)

            src_count = source_feats.shape[1]
            tgt_count = target_locs.shape[1]
            v = torch.zeros(bsz, 1, src_count, device=source_feats.device)
            u = torch.zeros(bsz, tgt_count, 1, device=source_feats.device)
            sinkhorn_iters = self.sparse_rebalance_iters if self.topk_support > 0 else self.num_iters

            for _ in range(sinkhorn_iters):
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


class BiggerGait__DINOv2__Projection_Mask_OT_Based(BiggerGait__DINOv2__Projection_Mask_Based):
    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.pretrained_sam3d_root = model_cfg["pretrained_sam3d_root"]
        self.require_full_sam_decoder = model_cfg.get("require_full_sam_decoder", True)

        self.branch_configs = model_cfg["branch_configs"]
        self.num_branches = len(self.branch_configs)
        if self.num_branches == 0:
            raise ValueError("branch_configs must contain at least one branch.")
        for branch_cfg in self.branch_configs:
            if "yaw" not in branch_cfg:
                raise ValueError("Each branch config must contain a yaw field.")

        first_gait_net = self.Gait_Net
        del self.Gait_Net
        self.Gait_Nets = nn.ModuleList(
            [first_gait_net] + [copy.deepcopy(first_gait_net) for _ in range(self.num_branches - 1)]
        )
        self.ot_topk_support = int(model_cfg.get("ot_topk_support", 0) or 0)
        self.ot_sparse_rebalance_iters = int(
            model_cfg.get("ot_sparse_rebalance_iters", model_cfg.get("ot_iters", 8))
        )
        self.ot_solver = GeometryOptimalTransport(
            temperature=model_cfg.get("ot_temperature", 0.01),
            dist_thresh=model_cfg.get("ot_dist_thresh", 0.2),
            num_iters=model_cfg.get("ot_iters", 8),
            topk_support=self.ot_topk_support,
            sparse_rebalance_iters=self.ot_sparse_rebalance_iters,
        )
        if self.ot_topk_support > 0:
            self.msg_mgr.log_info(
                f"[OT] Sparse support enabled: topk={self.ot_topk_support}, "
                f"rebalance_iters={self.ot_sparse_rebalance_iters}"
            )

    def init_sam_pose_engine(self):
        if self.pretrained_sam3d_root not in sys.path:
            sys.path.insert(0, self.pretrained_sam3d_root)

        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as exc:
            raise ImportError(f"Cannot import setup_sam_3d_body from {self.pretrained_sam3d_root}. Error: {exc}")

        self.msg_mgr.log_info("[SAM3D] Loading pose engine for offline OT geometry...")
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device="cpu")
        self.SAM_Engine = estimator.model
        self.SAM_Engine.cpu()
        self.SAM_Engine.eval()
        self.SAM_Engine.requires_grad_(False)

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
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.init_sam_pose_engine()

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        self.msg_mgr.log_info("=> init successfully")

    def _stack_full_pose_frames(self, sam_frames, device):
        if not sam_frames:
            raise ValueError("Empty SAM frame list.")
        if "pose_outs" not in sam_frames[0]:
            msg = (
                "This OT branch needs full offline SAM decoder dumps with `pose_outs`, not lite-only "
                "`pred_vertices/pred_cam_t/cam_int`. Please preprocess with `--save-mode full`."
            )
            if self.require_full_sam_decoder:
                raise ValueError(msg)
            raise ValueError(msg)

        final_pose_frames = []
        for frame in sam_frames:
            pose_outs = frame["pose_outs"]
            final_pose_frames.append(pose_outs[-1] if isinstance(pose_outs, (list, tuple)) else pose_outs)

        required_pose_keys = [
            "pred_vertices",
            "pred_cam_t",
            "pred_keypoints_3d",
            "global_rot",
            "shape",
            "scale",
            "face",
            "body_pose",
            "hand",
        ]
        pose_out = {}
        for key in required_pose_keys:
            if key not in final_pose_frames[0]:
                raise KeyError(f"Missing `{key}` in offline SAM decoder pose_outs.")
            pose_out[key] = torch.stack(
                [torch.as_tensor(frame[key], dtype=torch.float32) for frame in final_pose_frames],
                dim=0,
            ).to(device)

        cam_int = torch.stack(
            [torch.as_tensor(frame["cam_int"], dtype=torch.float32) for frame in sam_frames],
            dim=0,
        ).to(device)
        return pose_out, cam_int

    def get_source_vertex_index_map(self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w):
        bsz, num_verts, _ = vertices.shape
        device = vertices.device

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx = cam_int[:, 0, 0].unsqueeze(1)
        fy = cam_int[:, 1, 1].unsqueeze(1)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
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
        device = pose_out["pred_vertices"].device
        batch_size = pose_out["pred_vertices"].shape[0]

        pred_shape = pose_out["shape"].float()
        pred_scale = pose_out["scale"].float()
        pred_face = pose_out["face"].float()

        zero_global_trans = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        zero_global_rot = torch.zeros_like(pose_out["global_rot"], dtype=torch.float32)
        zero_hand_pose = torch.zeros_like(pose_out["hand"], dtype=torch.float32)
        apose_body = torch.zeros_like(pose_out["body_pose"], dtype=torch.float32)

        angle_rad = math.radians(-20.0)
        apose_body[:, 25] = angle_rad
        apose_body[:, 35] = angle_rad

        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.amp.autocast(enabled=False, device_type=amp_device_type):
            apose_outputs = self.SAM_Engine.head_pose.mhr_forward(
                global_trans=zero_global_trans,
                global_rot=zero_global_rot,
                body_pose_params=apose_body,
                hand_pose_params=zero_hand_pose,
                scale_params=pred_scale,
                shape_params=pred_shape,
                expr_params=pred_face,
                return_keypoints=True,
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
            cache_key = id(pose_out)
            if getattr(self, "_apose_cache_key", None) != cache_key:
                self._apose_cache = self.generate_mhr_apose(pose_out)
                self._apose_cache_key = cache_key
            branch_verts, branch_keypoints = self._apose_cache
            apply_global_rot_alignment = False
        else:
            branch_verts = pose_out["pred_vertices"]
            branch_keypoints = pose_out["pred_keypoints_3d"]
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

        cy = math.cos(math.radians(yaw))
        sy = math.sin(math.radians(yaw))
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
        flat_flip_flags=None,
    ):
        bsz, _, _, _ = human_feat.shape
        device = human_feat.device

        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int_src, h_feat, w_feat, target_h, target_w
        )
        if flat_flip_flags is not None and torch.any(flat_flip_flags):
            src_idx_map = src_idx_map.clone()
            src_idx_map[flat_flip_flags] = torch.flip(src_idx_map[flat_flip_flags], dims=[-1])

        valid_src_mask = (mask_src.squeeze(1) > 0.5) & (src_idx_map >= 0)

        flat_human_feat = rearrange(human_feat, "b c h w -> b (h w) c")
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

        fx = cam_int_tgt[:, 0, 0].unsqueeze(1)
        fy = cam_int_tgt[:, 1, 1].unsqueeze(1)
        cx = cam_int_tgt[:, 0, 2].unsqueeze(1)
        cy = cam_int_tgt[:, 1, 2].unsqueeze(1)
        u_tgt = (x / z) * fx + cx
        v_tgt = (y / z) * fy + cy

        u_norm = 2.0 * (u_tgt / target_w) - 1.0
        v_norm = 2.0 * (v_tgt / target_h) - 1.0
        projected_source_locs = torch.stack([u_norm, v_norm], dim=-1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h_feat, device=device),
            torch.linspace(-1, 1, w_feat, device=device),
            indexing="ij",
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1).reshape(
            bsz, -1, 2
        )

        transported_feats = self.ot_solver(
            flat_human_feat,
            projected_source_locs,
            target_grid_locs,
            source_valid_mask=flat_src_mask,
            target_valid_mask=valid_tgt_mask,
        )

        warped_feat = rearrange(transported_feats, "b (h w) c -> b c h w", h=h_feat)
        return warped_feat, valid_tgt_mask.view(bsz, 1, h_feat, w_feat), tgt_depth_map

    def forward(self, inputs):
        timing_info = {
            "model_hflip": 0.0,
            "model_rgb_preprocess": 0.0,
            "model_dino": 0.0,
            "model_sam_unpack": 0.0,
            "model_project_mask": 0.0,
            "model_humanspace": 0.0,
            "model_ot": 0.0,
            "model_gait_head": 0.0,
        }

        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        sam_decoder = ipts[1]
        del ipts
        model_start = self._perf_now(rgb.device)

        if self.training and self.sync_hflip_prob > 0:
            flip_flags = torch.rand(rgb.size(0), device=rgb.device) < self.sync_hflip_prob
        else:
            flip_flags = torch.zeros(rgb.size(0), device=rgb.device, dtype=torch.bool)

        num_chunks = (rgb.size(1) // self.chunk_size) + 1
        rgb_chunks = torch.chunk(rgb, num_chunks, dim=1)
        chunk_lengths = [chunk.size(1) for chunk in rgb_chunks]

        all_outs = [[] for _ in range(self.num_branches)]
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = self.image_size // 7, self.image_size // 14
        last_rgb_img = None
        last_generated_mask = None
        last_human_mask = None
        last_target_masks = None

        seq_start = 0
        for rgb_chunk, chunk_len in zip(rgb_chunks, chunk_lengths):
            seq_end = seq_start + chunk_len
            sam_chunk = [seq[seq_start:seq_end] for seq in sam_decoder]
            seq_start = seq_end

            n, s, c, h, w = rgb_chunk.size()
            flat_sam_frames = [frame for seq in sam_chunk for frame in seq]
            if len(flat_sam_frames) != n * s:
                raise RuntimeError(f"SAM frame count mismatch: expected {n * s}, got {len(flat_sam_frames)}")

            flat_flip_flags = flip_flags.unsqueeze(1).expand(-1, s).reshape(-1)

            with torch.no_grad():
                stage_start = self._perf_now(rgb.device)
                rgb_img = rearrange(rgb_chunk, "n s c h w -> (n s) c h w").contiguous()
                rgb_img = self._apply_sequence_hflip(rgb_img, flip_flags, s)
                timing_info["model_hflip"] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                rgb_img = self._gpu_rgb_preprocess(rgb_img)
                timing_info["model_rgb_preprocess"] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                outs = self.preprocess(rgb_img, self.image_size)
                outs = self.Backbone(outs, output_hidden_states=True).hidden_states[1:]
                intermediates = partial(nn.LayerNorm, eps=1e-6)(
                    self.f4_dim * len(outs), elementwise_affine=False
                )(torch.concat(outs, dim=-1))[:, 1:]
                intermediates = rearrange(
                    intermediates.view(n, s, h_feat, w_feat, -1),
                    "n s h w c -> (n s) c h w",
                ).contiguous()
                intermediates = list(torch.chunk(intermediates, self.total_layer_num, dim=1))
                timing_info["model_dino"] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                pose_out, cam_int_src = self._stack_full_pose_frames(flat_sam_frames, rgb.device)
                cam_int_src = self._resize_cam_int(cam_int_src, target_h, target_w)
                timing_info["model_sam_unpack"] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                _, src_depth_map = self.get_source_vertex_index_map(
                    pose_out["pred_vertices"],
                    pose_out["pred_cam_t"],
                    cam_int_src,
                    h_feat,
                    w_feat,
                    target_h,
                    target_w,
                )
                generated_mask = (src_depth_map < 1e5).float()
                generated_mask = self._apply_sequence_hflip(generated_mask, flip_flags, s)
                timing_info["model_project_mask"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            intermediates = [
                torch.cat(intermediates[i : i + self.group_layer_num], dim=1).contiguous()
                for i in range(0, self.total_layer_num, self.group_layer_num)
            ]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            human_feat = torch.concat(intermediates, dim=1)
            human_mask = self.preprocess(generated_mask, self.sils_size)
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            timing_info["model_humanspace"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            self._apose_cache = None
            self._apose_cache_key = None
            cam_int_tgt, cam_t_tgt = self.build_target_camera(rgb_img.shape[0], rgb.device, target_h, target_w)
            branch_warped_feats = []
            branch_target_masks = []
            for branch_cfg in self.branch_configs:
                branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
                warp_feat, tgt_valid_mask, _ = self.warp_features_with_ot(
                    human_feat,
                    human_mask.float(),
                    pose_out["pred_vertices"],
                    branch_geo["verts"],
                    branch_geo["keypoints"],
                    pose_out["pred_cam_t"],
                    pose_out["global_rot"],
                    cam_int_src,
                    cam_int_tgt,
                    cam_t_tgt,
                    self.sils_size * 2,
                    self.sils_size,
                    target_h,
                    target_w,
                    branch_geo["yaw"],
                    branch_geo["apply_global_rot_alignment"],
                    flat_flip_flags=flat_flip_flags,
                )
                branch_warped_feats.append(warp_feat)
                branch_target_masks.append(tgt_valid_mask)
            timing_info["model_ot"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            for b_idx, warp_feat in enumerate(branch_warped_feats):
                warp_feat_5d = rearrange(warp_feat, "(n s) c h w -> n c s h w", n=n, s=s).contiguous()
                outs = self.Gait_Nets[b_idx].test_1(warp_feat_5d)
                all_outs[b_idx].append(outs)
            timing_info["model_gait_head"] += self._perf_now(rgb.device) - stage_start

            last_rgb_img = rgb_img
            last_generated_mask = generated_mask
            last_human_mask = human_mask
            last_target_masks = branch_target_masks

        stage_start = self._perf_now(rgb.device)
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
        timing_info["model_gait_head"] += self._perf_now(rgb.device) - stage_start

        embeddings = torch.cat(embed_list, dim=-1)
        logits = torch.cat(log_list, dim=-1)
        model_end = self._perf_now(rgb.device)
        timed_sum = sum(timing_info.values())
        timing_info["model_misc"] = max(model_end - model_start - timed_sum, 0.0)

        if self.training:
            visual_summary = {
                "image/rgb_img": last_rgb_img[:5].float(),
                "image/generated_3d_mask_lowres": last_generated_mask[:5].float(),
                "image/generated_3d_mask_interpolated": last_human_mask[:5].float(),
            }
            if last_target_masks:
                visual_summary["image/ot_target_mask_branch0"] = last_target_masks[0][:5].float()
            retval = {
                "training_feat": {
                    "triplet": {"embeddings": embeddings, "labels": labs},
                    "softmax": {"logits": logits, "labels": labs},
                },
                "visual_summary": visual_summary,
                "inference_feat": {
                    "embeddings": embeddings,
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
                "timing_info": timing_info,
            }
        else:
            retval = {
                "training_feat": {},
                "visual_summary": {},
                "inference_feat": {
                    "embeddings": embeddings,
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
                "timing_info": timing_info,
            }
        return retval
