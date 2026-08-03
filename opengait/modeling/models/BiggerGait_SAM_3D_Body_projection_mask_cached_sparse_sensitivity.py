"""Cached Sparse-OT sensitivity evaluation for the submitted PuppetGait A4.

This evaluation-only subclass preserves the submitted SparseTopK algorithm and
checkpoint.  For each sequence it executes the frozen LVM, SAM3D decoder,
feature reducer, and original-view branch once.  Clean canonical geometry is
also constructed once per frame microbatch; only transport and the canonical
gait branch are repeated for the requested solver settings.
"""

import torch
from einops import rearrange

from .BiggerGait_SAM_3D_Body_projection_mask_strong_hmr_robustness import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_StrongHMRRobustness_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_CachedSparseSensitivity_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_StrongHMRRobustness_Gaitbase_Share
):
    """A4 with cache-equivalent, multi-setting Sparse-OT test inference."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        cfg = model_cfg.get("cached_sparse_sensitivity_eval", {})
        self.cached_sparse_cfg = cfg
        self.cached_sparse_enabled = bool(cfg.get("enabled", False))
        self.sparse_sensitivity_cfg = cfg
        self.variant_microbatch_size = int(cfg.get("variant_microbatch_size", 11))
        self.canonical_frame_chunk_size = int(
            cfg.get("canonical_frame_chunk_size", 8)
        )
        if self.variant_microbatch_size <= 0:
            raise ValueError("variant_microbatch_size must be positive.")
        if self.canonical_frame_chunk_size <= 0:
            raise ValueError("canonical_frame_chunk_size must be positive.")
        if self.cached_sparse_enabled:
            if self.num_branches != 2:
                raise ValueError(
                    "Cached Sparse sensitivity requires exactly the original "
                    "and canonical A4 branches."
                )
            if not self._is_original_view_branch(self.branch_configs[0]):
                raise ValueError("branch_configs[0] must be the original branch.")
            if self._is_original_view_branch(self.branch_configs[1]):
                raise ValueError("branch_configs[1] must be canonical.")
            # Reuse the thoroughly tested cached forward in the parent class.
            self.strong_hmr_enabled = True
            self.strong_hmr_cfg = cfg
            names = [spec["name"] for spec in self._variant_specs()]
            self.msg_mgr.log_info(
                "[CachedSparseSensitivity] One LVM/HMR/original pass and one "
                "clean-geometry construction per frame microbatch; only OT "
                "and the canonical branch repeat. Effective post-TopK "
                f"normalization iterations are tested. variants={names}, "
                f"chunk_size={self.chunk_size}, canonical_frame_chunk_size="
                f"{self.canonical_frame_chunk_size}."
            )

    def _variant_specs(self):
        variants = self._sparse_sensitivity_variants()
        requested = self.cached_sparse_cfg.get("variants")
        if requested is not None:
            by_name = {variant["name"]: variant for variant in variants}
            unknown = [name for name in requested if name not in by_name]
            if unknown:
                raise ValueError(
                    f"Unknown cached Sparse variants {unknown}; "
                    f"available={sorted(by_name)}"
                )
            variants = [by_name[name] for name in requested]
        if not variants or variants[0]["name"] != "current":
            raise ValueError("Cached Sparse variants must begin with current.")
        return [
            {"name": variant["name"], "type": "clean", "value": 0.0,
             "settings": variant}
            for variant in variants
        ]

    @staticmethod
    def _build_noise_cache(total_frames, device, seed):
        # Geometry is unchanged across transport settings.
        return {}

    def _build_variant_pose_group(self, specs, clean_pose, noise, frame_slice):
        self._active_transport_specs = specs
        return [clean_pose] * len(specs), [1.0] * len(specs)

    @staticmethod
    def _concat_pose_outputs(pose_outputs):
        # Every variant uses exactly the same clean mesh.  Avoid materializing
        # V copies of all per-vertex tensors; _warp_canonical_batch expands only
        # the much smaller transported feature maps after OT.
        return pose_outputs[0]

    @staticmethod
    def _geometry_diagnostics(
        clean_pose, variant_pose, cam_int_src, projection_scale, target_h, target_w
    ):
        batch_size = clean_pose["pred_vertices"].shape[0]
        device = clean_pose["pred_vertices"].device
        ones = torch.ones(batch_size, device=device, dtype=torch.float32)
        zeros = torch.zeros_like(ones)
        return {
            "projected_support_iou": ones,
            "vertex_reprojection_error": zeros,
            "pelvis_aligned_pve": zeros,
            "projected_bbox_scale_ratio": ones,
        }

    def _prepare_clean_transport(
        self, human_feat, source_mask, pose_out, cam_int_src, target_h, target_w
    ):
        batch_size = human_feat.shape[0]
        h_feat, w_feat = source_mask.shape[-2:]
        pred_verts = pose_out["pred_vertices"]
        pred_cam_t = pose_out["pred_cam_t"]
        global_rot = pose_out["global_rot"]

        src_idx_map, _ = self._source_vertex_index_map_scaled(
            pred_verts,
            pred_cam_t,
            cam_int_src,
            h_feat,
            w_feat,
            target_h,
            target_w,
            1.0,
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
        source_surface_vertices = torch.gather(
            branch_verts,
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        rotated_vertices, midhip, rotation = self.rotate_branch_geometry(
            branch_verts,
            branch_keypoints,
            global_rot,
            branch_geo["yaw"],
            branch_geo["apply_global_rot_alignment"],
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
        target_u = (
            (x / z) * cam_int_tgt[:, 0, 0].unsqueeze(1)
            + cam_int_tgt[:, 0, 2].unsqueeze(1)
        )
        target_v = (
            (y / z) * cam_int_tgt[:, 1, 1].unsqueeze(1)
            + cam_int_tgt[:, 1, 2].unsqueeze(1)
        )
        projected_source_locs = torch.stack(
            [2.0 * target_u / target_w - 1.0,
             2.0 * target_v / target_h - 1.0],
            dim=-1,
        )

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h_feat, device=pred_verts.device),
            torch.linspace(-1, 1, w_feat, device=pred_verts.device),
            indexing="ij",
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1)
        target_grid_locs = target_grid_locs.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        ).reshape(batch_size, -1, 2)
        return (
            flat_human_feat,
            projected_source_locs,
            target_grid_locs,
            valid_src_mask.reshape(batch_size, -1),
            valid_target_mask,
            h_feat,
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
        specs = self._active_transport_specs
        clean_batch_size = pose_out["pred_vertices"].shape[0]
        # The parent forward repeats only feature/mask/camera tensors.  Slice
        # their first clean block; the mesh itself was not duplicated.
        prepared = self._prepare_clean_transport(
            human_feat[:clean_batch_size],
            source_mask[:clean_batch_size],
            pose_out,
            cam_int_src[:clean_batch_size],
            target_h,
            target_w,
        )
        (
            flat_human_feat,
            projected_source_locs,
            target_grid_locs,
            valid_src_mask,
            valid_target_mask,
            h_feat,
        ) = prepared

        outputs = []
        try:
            for spec in specs:
                self._set_ot_settings(spec["settings"])
                transported = self.ot_solver(
                    flat_human_feat,
                    projected_source_locs,
                    target_grid_locs,
                    source_valid_mask=valid_src_mask,
                    target_valid_mask=valid_target_mask,
                )
                outputs.append(
                    rearrange(transported, "b (h w) c -> b c h w", h=h_feat)
                )
        finally:
            self._set_ot_settings(self._default_ot_settings)
        return torch.cat(outputs, dim=0)

    def forward(self, inputs):
        if not self.cached_sparse_enabled:
            return super().forward(inputs)
        output = super().forward(inputs)
        inference_feat = output["inference_feat"]
        if "embeddings_current" not in inference_feat:
            raise RuntimeError("Cached Sparse evaluation did not produce current.")
        inference_feat["embeddings"] = inference_feat["embeddings_current"]
        return output
