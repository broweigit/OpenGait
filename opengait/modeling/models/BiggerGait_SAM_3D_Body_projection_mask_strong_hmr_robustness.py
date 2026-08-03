"""Cached full-set robustness evaluation for the submitted PuppetGait A4 model.

The ordinary A4 forward is intentionally left unchanged.  This evaluation-only
subclass runs the frozen LVM, SAM 3D Body decoder, feature reduction, and the
original-view branch once per RGB chunk.  It then batches several controlled
HMR perturbations through only the canonical branch.  The class introduces no
parameters, so an A4 checkpoint can be restored strictly.
"""

import math
import zlib
from functools import partial

import roma
import torch
import torch.nn as nn
from einops import rearrange

from .BiggerGait_SAM_3D_Body_projection_mask_OT_based_SparseTopK4 import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_StrongHMRRobustness_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share
):
    """A4 with cached, strongly controlled HMR perturbations at test time."""

    # MHR body-pose layout from sam_3d_body.models.modules.mhr_utils.  Hand
    # entries (62--115), translations (124--129), and jaw (130--132) are
    # deliberately excluded from articulated-body perturbations.
    _MHR_3DOF_GROUPS = (
        (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17),
        (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29),
        (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55),
    )
    _MHR_1DOF_INDICES = tuple(
        index for index in (
            1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43,
            47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61,
            62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84,
            89, 90, 94, 95, 101, 102, 104, 105, 107, 108,
            110, 111, 116, 117, 118, 119, 120, 121, 122, 123,
        )
        if not 62 <= index <= 115
    )

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        cfg = model_cfg.get("strong_hmr_robustness_eval", {})
        self.strong_hmr_cfg = cfg
        self.strong_hmr_enabled = bool(cfg.get("enabled", False))
        self.variant_microbatch_size = int(cfg.get("variant_microbatch_size", 3))
        self.canonical_frame_chunk_size = int(
            cfg.get("canonical_frame_chunk_size", 8)
        )
        if self.variant_microbatch_size <= 0:
            raise ValueError("variant_microbatch_size must be positive.")
        if self.canonical_frame_chunk_size <= 0:
            raise ValueError("canonical_frame_chunk_size must be positive.")
        if self.strong_hmr_enabled:
            if self.num_branches != 2:
                raise ValueError(
                    "Strong HMR robustness requires exactly the original and "
                    "canonical A4 branches."
                )
            if not self._is_original_view_branch(self.branch_configs[0]):
                raise ValueError("branch_configs[0] must be the original-view branch.")
            if self._is_original_view_branch(self.branch_configs[1]):
                raise ValueError("branch_configs[1] must be the canonical branch.")
            self.msg_mgr.log_info(
                "[StrongHMR] Cached full-set evaluation enabled: one LVM/HMR "
                f"pass per chunk, chunk_size={self.chunk_size}, canonical "
                f"variant_microbatch_size={self.variant_microbatch_size}, "
                f"canonical_frame_chunk_size={self.canonical_frame_chunk_size}."
            )

    def _variant_specs(self):
        specs = [{"name": "clean", "type": "clean", "value": 0.0}]
        for value in self.strong_hmr_cfg.get("joint_max_degrees", [15.0, 30.0, 45.0]):
            value = float(value)
            specs.append({
                "name": f"joint_max_{self._format_number(value)}deg",
                "type": "joint",
                "value": value,
            })
        for value in self.strong_hmr_cfg.get("shape_noise_stds", [0.5, 1.0]):
            value = float(value)
            specs.append({
                "name": f"shape_std_{self._format_number(value)}",
                "type": "shape",
                "value": value,
            })
        for value in self.strong_hmr_cfg.get("projection_scales", [0.75, 1.25]):
            value = float(value)
            specs.append({
                "name": f"projection_scale_{self._format_number(value)}",
                "type": "projection_scale",
                "value": value,
            })
        for value in self.strong_hmr_cfg.get(
            "global_orientation_max_degrees", [15.0, 30.0, 45.0]
        ):
            value = float(value)
            specs.append({
                "name": f"global_orientation_max_{self._format_number(value)}deg",
                "type": "global_orientation",
                "value": value,
            })

        requested = self.strong_hmr_cfg.get("variants")
        if requested is not None:
            by_name = {spec["name"]: spec for spec in specs}
            unknown = [name for name in requested if name not in by_name]
            if unknown:
                raise ValueError(
                    f"Unknown strong HMR variants {unknown}; available={sorted(by_name)}"
                )
            specs = [by_name[name] for name in requested]
        if not specs or specs[0]["name"] != "clean":
            raise ValueError("Strong HMR variants must begin with clean.")
        return specs

    @staticmethod
    def _format_number(value):
        return f"{float(value):g}".replace(".", "p")

    def _stable_sequence_seed(self, labs, sequence_types, views):
        if int(labs.numel()) != 1 or len(sequence_types) != 1 or len(views) != 1:
            raise ValueError(
                "Cached strong-HMR evaluation requires one sequence per GPU. "
                "Set evaluator batch_size equal to world_size."
            )
        token = f"{int(labs.reshape(-1)[0])}|{sequence_types[0]}|{views[0]}"
        token_hash = zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF
        return int(self.strong_hmr_cfg.get("seed", 7890)) + token_hash

    @staticmethod
    def _random_unit_vectors(shape, device, generator):
        vectors = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _build_noise_cache(self, total_frames, device, seed):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        group_count = len(self._MHR_3DOF_GROUPS)
        one_dof_count = len(self._MHR_1DOF_INDICES)
        return {
            "joint_axes": self._random_unit_vectors(
                (total_frames, group_count, 3), device, generator
            ),
            "joint_amounts": torch.empty(
                total_frames, group_count, 1, device=device, dtype=torch.float32
            ).uniform_(-1.0, 1.0, generator=generator),
            "joint_1d_amounts": torch.empty(
                total_frames, one_dof_count, device=device, dtype=torch.float32
            ).uniform_(-1.0, 1.0, generator=generator),
            # One systematic shape error per sequence, deliberately reused by
            # every frame so shape error is not confounded with temporal jitter.
            "shape_noise": torch.randn(
                1, 45, device=device, dtype=torch.float32, generator=generator
            ),
            "global_axes": self._random_unit_vectors(
                (total_frames, 3), device, generator
            ),
            "global_amounts": torch.empty(
                total_frames, 1, device=device, dtype=torch.float32
            ).uniform_(-1.0, 1.0, generator=generator),
        }

    def _apply_joint_perturbation(self, pose_out, max_degrees, noise, frame_slice):
        perturbed = dict(pose_out)
        body_pose = pose_out["body_pose"].float().clone()
        axes = noise["joint_axes"][frame_slice]
        amounts = noise["joint_amounts"][frame_slice]
        angles = amounts * math.radians(float(max_degrees))
        delta_rot = roma.rotvec_to_rotmat(axes * angles)

        for group_index, parameter_indices in enumerate(self._MHR_3DOF_GROUPS):
            clean_euler = body_pose[:, list(parameter_indices)]
            clean_rot = roma.euler_to_rotmat("XYZ", clean_euler)
            composed = torch.matmul(delta_rot[:, group_index], clean_rot)
            body_pose[:, list(parameter_indices)] = roma.rotmat_to_euler(
                "XYZ", composed
            )

        one_dof_indices = list(self._MHR_1DOF_INDICES)
        one_dof_delta = (
            noise["joint_1d_amounts"][frame_slice]
            * math.radians(float(max_degrees))
        )
        updated_1d = body_pose[:, one_dof_indices] + one_dof_delta
        body_pose[:, one_dof_indices] = torch.atan2(
            updated_1d.sin(), updated_1d.cos()
        )
        perturbed["body_pose"] = body_pose
        return perturbed

    @staticmethod
    def _apply_shape_perturbation(pose_out, noise_std, noise):
        perturbed = dict(pose_out)
        perturbed["shape"] = (
            pose_out["shape"].float()
            + noise["shape_noise"] * float(noise_std)
        )
        return perturbed

    @staticmethod
    def _apply_global_orientation_perturbation(
        pose_out, max_degrees, noise, frame_slice
    ):
        perturbed = dict(pose_out)
        global_rot = pose_out["global_rot"].float()
        angles = (
            noise["global_amounts"][frame_slice]
            * math.radians(float(max_degrees))
        )
        delta = roma.rotvec_to_rotmat(noise["global_axes"][frame_slice] * angles)
        clean = roma.euler_to_rotmat("xyz", global_rot)
        perturbed["global_rot"] = roma.rotmat_to_euler(
            "xyz", torch.matmul(delta, clean)
        )
        return perturbed

    def _rebuild_pose_variants(self, parameter_variants):
        """Rebuild multiple variant meshes in one MHR body-model invocation."""
        if not parameter_variants:
            return []
        batch_size = parameter_variants[0]["global_rot"].shape[0]
        keys = ("global_rot", "body_pose", "hand", "scale", "shape", "face")
        stacked = {
            key: torch.cat([variant[key].float() for variant in parameter_variants], dim=0)
            for key in keys
        }
        zero_global_trans = torch.zeros_like(stacked["global_rot"])
        device_type = stacked["global_rot"].device.type
        with torch.no_grad(), torch.amp.autocast(enabled=False, device_type=device_type):
            vertices, keypoints = self.SAM_Engine.head_pose.mhr_forward(
                global_trans=zero_global_trans,
                global_rot=stacked["global_rot"],
                body_pose_params=stacked["body_pose"],
                hand_pose_params=stacked["hand"],
                scale_params=stacked["scale"],
                shape_params=stacked["shape"],
                expr_params=stacked["face"],
                return_keypoints=True,
            )
        vertices[..., [1, 2]] *= -1
        keypoints = keypoints[:, :70]
        keypoints[..., [1, 2]] *= -1
        vertex_chunks = vertices.float().split(batch_size, dim=0)
        keypoint_chunks = keypoints.float().split(batch_size, dim=0)
        rebuilt = []
        for variant, variant_vertices, variant_keypoints in zip(
            parameter_variants, vertex_chunks, keypoint_chunks
        ):
            output = dict(variant)
            output["pred_vertices"] = variant_vertices
            output["pred_keypoints_3d"] = variant_keypoints
            rebuilt.append(output)
        return rebuilt

    def _build_variant_pose_group(self, specs, clean_pose, noise, frame_slice):
        parameter_variants = []
        rebuild_positions = []
        outputs = [None] * len(specs)
        projection_scales = []
        for position, spec in enumerate(specs):
            variant_type = spec["type"]
            projection_scales.append(
                float(spec["value"]) if variant_type == "projection_scale" else 1.0
            )
            if variant_type in ("clean", "projection_scale"):
                outputs[position] = clean_pose
                continue
            if variant_type == "joint":
                params = self._apply_joint_perturbation(
                    clean_pose, spec["value"], noise, frame_slice
                )
            elif variant_type == "shape":
                params = self._apply_shape_perturbation(
                    clean_pose, spec["value"], noise
                )
            elif variant_type == "global_orientation":
                params = self._apply_global_orientation_perturbation(
                    clean_pose, spec["value"], noise, frame_slice
                )
            else:
                raise ValueError(f"Unsupported strong HMR variant type: {variant_type}")
            parameter_variants.append(params)
            rebuild_positions.append(position)

        rebuilt = self._rebuild_pose_variants(parameter_variants)
        for position, pose_out in zip(rebuild_positions, rebuilt):
            outputs[position] = pose_out
        return outputs, projection_scales

    @staticmethod
    def _project_vertices(vertices, cam_t, cam_int, projection_scale=1.0):
        v_cam = vertices.float() + cam_t.float().unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)
        fx = cam_int[:, 0, 0].float().unsqueeze(1)
        fy = cam_int[:, 1, 1].float().unsqueeze(1)
        cx = cam_int[:, 0, 2].float().unsqueeze(1)
        cy = cam_int[:, 1, 2].float().unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy
        valid = torch.isfinite(v_cam).all(dim=-1) & (z > 1e-3)

        positive_inf = torch.full_like(u, float("inf"))
        negative_inf = torch.full_like(u, float("-inf"))
        u_min = torch.where(valid, u, positive_inf).amin(dim=1)
        u_max = torch.where(valid, u, negative_inf).amax(dim=1)
        v_min = torch.where(valid, v, positive_inf).amin(dim=1)
        v_max = torch.where(valid, v, negative_inf).amax(dim=1)
        has_valid = valid.any(dim=1)
        u_min = torch.where(has_valid, u_min, torch.zeros_like(u_min))
        u_max = torch.where(has_valid, u_max, torch.ones_like(u_max))
        v_min = torch.where(has_valid, v_min, torch.zeros_like(v_min))
        v_max = torch.where(has_valid, v_max, torch.ones_like(v_max))
        center_u = ((u_min + u_max) * 0.5).unsqueeze(1)
        center_v = ((v_min + v_max) * 0.5).unsqueeze(1)

        if torch.is_tensor(projection_scale):
            scale = projection_scale.to(u).reshape(-1, 1)
        else:
            scale = torch.full(
                (u.shape[0], 1), float(projection_scale), device=u.device, dtype=u.dtype
            )
        u = center_u + scale * (u - center_u)
        v = center_v + scale * (v - center_v)
        bbox_height = (v_max - v_min).clamp_min(1e-6) * scale.squeeze(1).abs()
        return u, v, z, valid, bbox_height

    def _source_vertex_index_map_scaled(
        self,
        vertices,
        cam_t,
        cam_int,
        h_feat,
        w_feat,
        target_h,
        target_w,
        projection_scale,
    ):
        batch_size, vertex_count, _ = vertices.shape
        u, v, z, _, _ = self._project_vertices(
            vertices, cam_t, cam_int, projection_scale
        )
        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)
        flat_pixels = v_feat * w_feat + u_feat

        depth = torch.full(
            (batch_size, h_feat * w_feat), 1e6, device=vertices.device
        )
        depth.scatter_reduce_(1, flat_pixels, z, reduce="amin", include_self=False)
        minimum_depth = torch.gather(depth, 1, flat_pixels)
        visible = z < (minimum_depth + 1e-4)

        index_map = torch.full(
            (batch_size, h_feat * w_feat),
            -1,
            dtype=torch.long,
            device=vertices.device,
        )
        vertex_indices = torch.arange(vertex_count, device=vertices.device).unsqueeze(0)
        vertex_indices = vertex_indices.expand(batch_size, -1)
        batch_offsets = torch.arange(batch_size, device=vertices.device).unsqueeze(1)
        batch_offsets = batch_offsets * (h_feat * w_feat)
        global_pixels = (flat_pixels + batch_offsets).reshape(-1)
        visible_flat = visible.reshape(-1)
        index_map.reshape(-1)[global_pixels[visible_flat]] = (
            vertex_indices.reshape(-1)[visible_flat]
        )
        return (
            index_map.reshape(batch_size, h_feat, w_feat),
            depth.reshape(batch_size, 1, h_feat, w_feat),
        )

    def _warp_canonical_batch(
        self,
        human_feat,
        source_mask,
        pose_out,
        cam_int_src,
        projection_scales,
        target_h,
        target_w,
    ):
        batch_size = human_feat.shape[0]
        h_feat, w_feat = source_mask.shape[-2:]
        pred_verts = pose_out["pred_vertices"]
        pred_cam_t = pose_out["pred_cam_t"]
        global_rot = pose_out["global_rot"]
        scales = torch.as_tensor(
            projection_scales,
            device=pred_verts.device,
            dtype=torch.float32,
        )
        src_idx_map, _ = self._source_vertex_index_map_scaled(
            pred_verts,
            pred_cam_t,
            cam_int_src,
            h_feat,
            w_feat,
            target_h,
            target_w,
            scales,
        )
        valid_src_mask = (source_mask.squeeze(1) > 0.5) & (src_idx_map >= 0)
        flat_human_feat = rearrange(human_feat, "b c h w -> b (h w) c")
        flat_indices = src_idx_map.reshape(batch_size, -1)
        safe_indices = flat_indices.clamp_min(0)

        canonical_cfg = self.branch_configs[1]
        self._apose_cache = {}
        branch_geo = self.build_branch_geometry(canonical_cfg, pose_out)
        branch_verts = branch_geo["verts"]
        branch_keypoints = branch_geo["keypoints"]
        yaw = branch_geo["yaw"]
        apply_global_rot_alignment = branch_geo["apply_global_rot_alignment"]

        source_surface_vertices = torch.gather(
            branch_verts,
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        rotated_vertices, midhip, rotation = self.rotate_branch_geometry(
            branch_verts,
            branch_keypoints,
            global_rot,
            yaw,
            apply_global_rot_alignment,
        )
        cam_int_tgt, cam_t_tgt = self.build_target_camera(
            batch_size, pred_verts.device, target_h, target_w
        )
        _, target_depth = self.get_source_vertex_index_map(
            rotated_vertices,
            cam_t_tgt,
            cam_int_tgt,
            h_feat,
            w_feat,
            target_h,
            target_w,
        )
        valid_target_mask = target_depth.reshape(batch_size, -1) < 1e5

        centered = source_surface_vertices - midhip.unsqueeze(1)
        centered_smpl = centered.clone()
        centered_smpl[..., [1, 2]] *= -1
        canonical_smpl = torch.bmm(centered_smpl, rotation)
        canonical_cv = canonical_smpl.clone()
        canonical_cv[..., [1, 2]] *= -1
        target_camera_vertices = canonical_cv + cam_t_tgt.unsqueeze(1)
        x, y, z = target_camera_vertices.unbind(-1)
        z = z.clamp(min=1e-3)
        fx = cam_int_tgt[:, 0, 0].unsqueeze(1)
        fy = cam_int_tgt[:, 1, 1].unsqueeze(1)
        cx = cam_int_tgt[:, 0, 2].unsqueeze(1)
        cy = cam_int_tgt[:, 1, 2].unsqueeze(1)
        target_u = (x / z) * fx + cx
        target_v = (y / z) * fy + cy
        projected_source_locs = torch.stack(
            [2.0 * target_u / target_w - 1.0, 2.0 * target_v / target_h - 1.0],
            dim=-1,
        )

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h_feat, device=pred_verts.device),
            torch.linspace(-1, 1, w_feat, device=pred_verts.device),
            indexing="ij",
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1)
        target_grid_locs = target_grid_locs.unsqueeze(0).expand(batch_size, -1, -1, -1)
        target_grid_locs = target_grid_locs.reshape(batch_size, -1, 2)

        transported = self.ot_solver(
            flat_human_feat,
            projected_source_locs,
            target_grid_locs,
            source_valid_mask=valid_src_mask.reshape(batch_size, -1),
            target_valid_mask=valid_target_mask,
        )
        return rearrange(transported, "b (h w) c -> b c h w", h=h_feat)

    def _geometry_diagnostics(
        self,
        clean_pose,
        variant_pose,
        cam_int_src,
        projection_scale,
        target_h,
        target_w,
    ):
        clean_verts = clean_pose["pred_vertices"]
        variant_verts = variant_pose["pred_vertices"]
        clean_cam_t = clean_pose["pred_cam_t"]
        variant_cam_t = variant_pose["pred_cam_t"]
        clean_u, clean_v, _, clean_valid, clean_bbox_height = self._project_vertices(
            clean_verts, clean_cam_t, cam_int_src, 1.0
        )
        variant_u, variant_v, _, variant_valid, variant_bbox_height = self._project_vertices(
            variant_verts, variant_cam_t, cam_int_src, projection_scale
        )

        common = clean_valid & variant_valid
        displacement = torch.sqrt(
            (variant_u - clean_u).square() + (variant_v - clean_v).square()
        )
        common_count = common.sum(dim=1).clamp_min(1)
        reprojection_error = (
            (displacement * common).sum(dim=1) / common_count
        ) / clean_bbox_height.clamp_min(1e-6)

        clean_map, _ = self._source_vertex_index_map_scaled(
            clean_verts,
            clean_cam_t,
            cam_int_src,
            self.sils_size * 2,
            self.sils_size,
            target_h,
            target_w,
            1.0,
        )
        variant_map, _ = self._source_vertex_index_map_scaled(
            variant_verts,
            variant_cam_t,
            cam_int_src,
            self.sils_size * 2,
            self.sils_size,
            target_h,
            target_w,
            projection_scale,
        )
        clean_support = clean_map >= 0
        variant_support = variant_map >= 0
        intersection = (clean_support & variant_support).reshape(clean_verts.shape[0], -1)
        union = (clean_support | variant_support).reshape(clean_verts.shape[0], -1)
        support_iou = intersection.sum(dim=1).float() / union.sum(dim=1).clamp_min(1)

        clean_kp = clean_pose["pred_keypoints_3d"]
        variant_kp = variant_pose["pred_keypoints_3d"]
        clean_pelvis = (clean_kp[:, 9] + clean_kp[:, 10]) * 0.5
        variant_pelvis = (variant_kp[:, 9] + variant_kp[:, 10]) * 0.5
        clean_centered = clean_verts - clean_pelvis.unsqueeze(1)
        variant_centered = variant_verts - variant_pelvis.unsqueeze(1)
        clean_body_height = (
            clean_centered[..., 1].amax(dim=1) - clean_centered[..., 1].amin(dim=1)
        ).clamp_min(1e-6)
        pve = (variant_centered - clean_centered).norm(dim=-1).mean(dim=1)
        pve = pve / clean_body_height

        bbox_scale_ratio = variant_bbox_height / clean_bbox_height.clamp_min(1e-6)
        return {
            "projected_support_iou": support_iou,
            "vertex_reprojection_error": reprojection_error,
            "pelvis_aligned_pve": pve,
            "projected_bbox_scale_ratio": bbox_scale_ratio,
        }

    @staticmethod
    def _aggregate_frame_values(values, sequence_lengths):
        output = []
        start = 0
        for length in sequence_lengths:
            end = start + int(length)
            output.append(values[start:end].mean())
            start = end
        if start != values.numel():
            raise RuntimeError(
                f"Diagnostic frame count mismatch: consumed {start}, got {values.numel()}."
            )
        return torch.stack(output).reshape(-1, 1)

    @staticmethod
    def _concat_pose_outputs(pose_outputs):
        required = (
            "pred_vertices", "pred_keypoints_3d", "pred_cam_t", "global_rot",
            "body_pose", "hand", "scale", "shape", "face",
        )
        return {
            key: torch.cat([pose[key] for pose in pose_outputs], dim=0)
            for key in required
        }


    @staticmethod
    def _slice_pose_output(pose_output, frame_slice, frame_count):
        """Slice per-frame MHR tensors without dropping auxiliary entries."""
        return {
            key: (
                value[frame_slice]
                if torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == frame_count
                else value
            )
            for key, value in pose_output.items()
        }

    def forward(self, inputs):
        if not self.strong_hmr_enabled:
            return super().forward(inputs)
        if self.training:
            raise RuntimeError("Strong HMR robustness is evaluation-only.")

        ipts, labs, sequence_types, views, seq_l = inputs
        rgb = ipts[0]
        if rgb.shape[0] != 1 or seq_l is None:
            raise ValueError(
                "Strong HMR robustness requires all_ordered inference with a "
                "single concatenated sequence per GPU."
            )
        sequence_lengths = [int(value) for value in seq_l.reshape(-1).tolist()]
        if len(sequence_lengths) != 1:
            raise ValueError(
                "Set evaluator sampler.batch_size equal to world_size so each "
                "GPU receives exactly one sequence."
            )
        total_frames = int(rgb.shape[1])
        if sum(sequence_lengths) != total_frames:
            raise RuntimeError("seqL does not match the concatenated RGB frames.")

        specs = self._variant_specs()
        variant_names = [spec["name"] for spec in specs]
        seed = self._stable_sequence_seed(labs, sequence_types, views)
        noise = self._build_noise_cache(total_frames, rgb.device, seed)
        original_chunks = []
        canonical_chunks = {name: [] for name in variant_names}
        diagnostic_chunks = {
            name: {
                "projected_support_iou": [],
                "vertex_reprojection_error": [],
                "pelvis_aligned_pve": [],
                "projected_bbox_scale_ratio": [],
            }
            for name in variant_names
        }

        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16
        # torch.split makes chunk_size an actual maximum, unlike the older
        # approximately-even torch.chunk formulation.
        rgb_chunks = torch.split(rgb, self.chunk_size, dim=1)
        frame_start = 0

        for rgb_chunk in rgb_chunks:
            _, sequence_frames, channels, input_h, input_w = rgb_chunk.shape
            flat_rgb = rearrange(rgb_chunk, "n s c h w -> (n s) c h w").contiguous()
            current_frames = flat_rgb.shape[0]
            frame_slice = slice(frame_start, frame_start + current_frames)

            with torch.no_grad():
                backbone_input = self.prepare_backbone_input(
                    flat_rgb,
                    target_h,
                    target_w,
                    sequence_batch=1,
                    sequence_length=sequence_frames,
                    sequence_lengths=seq_l,
                )
                backbone_input = self._cast_floating_to_module_dtype(
                    backbone_input, self.Backbone
                )
                self.intermediate_features = {}
                _ = self.Backbone(backbone_input)
                last_hook_index = len(self.hook_handles) - 1
                sam_embedding = self.intermediate_features[last_hook_index]
                target_tokens = h_feat * w_feat
                if sam_embedding.shape[1] > target_tokens:
                    sam_embedding = sam_embedding[:, -target_tokens:, :]
                sam_embedding = sam_embedding.transpose(1, 2).reshape(
                    current_frames, -1, h_feat, w_feat
                ).float()

                dummy_batch = self._prepare_dummy_batch(
                    sam_embedding, target_h, target_w
                )
                self.SAM_Engine._batch_size = current_frames
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(
                    current_frames, device=rgb.device
                )
                self.SAM_Engine.hand_batch_idx = []
                condition_info = torch.zeros(current_frames, 3, device=rgb.device)
                condition_info[:, 2] = 1.1
                dummy_keypoints = torch.zeros(
                    current_frames, 1, 3, device=rgb.device
                )
                dummy_keypoints[..., -1] = -2
                with torch.amp.autocast(enabled=False, device_type=rgb.device.type):
                    _, pose_outputs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_embedding,
                        keypoints=dummy_keypoints,
                        condition_info=condition_info,
                        batch=dummy_batch,
                    )
                clean_pose = self._cast_floating_dtype(
                    pose_outputs[-1], torch.float32
                )
                cam_int_src = dummy_batch["cam_int"].float()
                _, clean_depth = self.get_source_vertex_index_map(
                    clean_pose["pred_vertices"],
                    clean_pose["pred_cam_t"],
                    cam_int_src,
                    h_feat,
                    w_feat,
                    target_h,
                    target_w,
                )
                clean_mask = (clean_depth < 1e5).float()

                features_to_use = []
                for hook_index in range(len(self.hook_handles)):
                    feature = self.intermediate_features[hook_index]
                    if feature.shape[1] > target_tokens:
                        feature = feature[:, -target_tokens:, :]
                    features_to_use.append(feature.float())

            processed_features = []
            step = len(features_to_use) // self.num_FPN
            for fpn_index in range(self.num_FPN):
                if self.hook_sample_type == "interleave":
                    sub_features = features_to_use[fpn_index::self.num_FPN]
                elif self.hook_sample_type == "chunk":
                    start = fpn_index * step
                    end = (fpn_index + 1) * step
                    sub_features = features_to_use[start:end]
                else:
                    raise ValueError(f"Invalid hook_sample_type: {self.hook_sample_type}")
                appearance = torch.cat(sub_features, dim=-1)
                appearance = partial(nn.LayerNorm, eps=1e-6)(
                    self.f4_dim * len(sub_features), elementwise_affine=False
                )(appearance)
                appearance = rearrange(
                    appearance, "b (h w) c -> b c h w", h=h_feat
                ).contiguous()
                processed_features.append(self.HumanSpace_Conv[fpn_index](appearance))

            human_feat = torch.cat(processed_features, dim=1)
            human_mask = self.preprocess(
                clean_mask, self.sils_size * 2, self.sils_size
            ).detach()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)

            original_5d = rearrange(
                human_feat, "(n s) c h w -> n c s h w", n=1, s=sequence_frames
            ).contiguous()
            original_chunks.append(self.Gait_Nets[0].test_1(original_5d))

            # The frozen LVM/HMR pass above benefits from a large frame chunk.
            # OT builds a quadratic 2048x2048 token-distance matrix, however,
            # so canonical variants use an independent frame microbatch. This
            # changes only execution grouping, not the resulting embeddings.
            for canonical_start in range(
                0, current_frames, self.canonical_frame_chunk_size
            ):
                canonical_end = min(
                    canonical_start + self.canonical_frame_chunk_size,
                    current_frames,
                )
                local_slice = slice(canonical_start, canonical_end)
                absolute_slice = slice(
                    frame_start + canonical_start,
                    frame_start + canonical_end,
                )
                local_frames = canonical_end - canonical_start
                clean_pose_local = self._slice_pose_output(
                    clean_pose, local_slice, current_frames
                )
                human_feat_local = human_feat[local_slice]
                human_mask_local = human_mask[local_slice]
                cam_int_src_local = cam_int_src[local_slice]

                for group_start in range(
                    0, len(specs), self.variant_microbatch_size
                ):
                    group_specs = specs[
                        group_start:group_start + self.variant_microbatch_size
                    ]
                    group_poses, group_scales = self._build_variant_pose_group(
                        group_specs, clean_pose_local, noise, absolute_slice
                    )
                    for spec, pose_out, projection_scale in zip(
                        group_specs, group_poses, group_scales
                    ):
                        diagnostics = self._geometry_diagnostics(
                            clean_pose_local,
                            pose_out,
                            cam_int_src_local,
                            projection_scale,
                            target_h,
                            target_w,
                        )
                        for key, value in diagnostics.items():
                            diagnostic_chunks[spec["name"]][key].append(value)

                    group_size = len(group_specs)
                    pose_batch = self._concat_pose_outputs(group_poses)
                    feature_batch = torch.cat(
                        [human_feat_local] * group_size, dim=0
                    )
                    mask_batch = torch.cat(
                        [human_mask_local] * group_size, dim=0
                    )
                    camera_batch = torch.cat(
                        [cam_int_src_local] * group_size, dim=0
                    )
                    scale_batch = torch.tensor(
                        group_scales,
                        device=human_feat.device,
                        dtype=torch.float32,
                    ).repeat_interleave(local_frames)
                    canonical = self._warp_canonical_batch(
                        feature_batch,
                        mask_batch,
                        pose_batch,
                        camera_batch,
                        scale_batch,
                        target_h,
                        target_w,
                    )
                    canonical_5d = rearrange(
                        canonical,
                        "(v s) c h w -> v c s h w",
                        v=group_size,
                        s=local_frames,
                    ).contiguous()
                    canonical_output = self.Gait_Nets[1].test_1(canonical_5d)
                    for local_index, spec in enumerate(group_specs):
                        canonical_chunks[spec["name"]].append(
                            canonical_output[local_index:local_index + 1]
                        )
            frame_start += current_frames
            del self.intermediate_features

        if frame_start != total_frames:
            raise RuntimeError("Not all RGB frames were consumed.")

        original_sequence = torch.cat(original_chunks, dim=2)
        original_embeddings, _ = self.Gait_Nets[0].test_2(
            original_sequence, seq_l
        )
        inference_feat = {}
        for spec in specs:
            name = spec["name"]
            canonical_sequence = torch.cat(canonical_chunks[name], dim=2)
            canonical_embeddings, _ = self.Gait_Nets[1].test_2(
                canonical_sequence, seq_l
            )
            fused_per_fpn = [
                torch.cat([original, canonical], dim=-1)
                for original, canonical in zip(
                    original_embeddings, canonical_embeddings
                )
            ]
            embedding = torch.cat(fused_per_fpn, dim=-1)
            if name == "clean":
                inference_feat["embeddings"] = embedding
            else:
                inference_feat[f"embeddings_{name}"] = embedding

            for diagnostic_name, chunks in diagnostic_chunks[name].items():
                frame_values = torch.cat(chunks, dim=0)
                inference_feat[f"{diagnostic_name}_{name}"] = (
                    self._aggregate_frame_values(frame_values, sequence_lengths)
                )

        return {
            "training_feat": {},
            "visual_summary": {},
            "inference_feat": inference_feat,
        }
