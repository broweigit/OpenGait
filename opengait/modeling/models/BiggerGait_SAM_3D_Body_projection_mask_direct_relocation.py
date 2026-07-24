"""Hard direct-relocation ablation for PuppetGait.

This module keeps the full SparseTopK4 model pipeline and two-branch encoder,
but replaces GAOT in the canonical branch with deterministic source-to-target
assignment.  It is a reconstruction of Table 2(c)'s Direct setting because the
original training implementation was not preserved in the repository.
"""

import torch
from einops import rearrange

from .BiggerGait_SAM_3D_Body_projection_mask_OT_based_SparseTopK4 import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_DirectRelocation_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share
):
    """Relocate each valid source token to one discretized target grid cell.

    When multiple source tokens collide at a target cell, the token(s) with
    minimum target-camera depth are retained.  Exact depth ties are averaged.
    The original-view branch still bypasses relocation, exactly as in the full
    SparseTopK4 model.
    """

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.msg_mgr.log_info(
            "[DirectRelocation] Canonical branches use hard source-to-target "
            "assignment; GAOT weights are disabled."
        )

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
        # Preserve the original-view branch behavior from SparseTopK4.
        if branch_verts is None or yaw is None:
            _, src_depth_map = self.get_source_vertex_index_map(
                pred_verts,
                pred_cam_t,
                cam_int_src,
                h_feat,
                w_feat,
                target_h,
                target_w,
            )
            return human_feat, mask_src, src_depth_map

        bsz, channels, _, _ = human_feat.shape

        # Attach every valid source feature-grid cell to its visible source
        # vertex, then look up the same indexed vertex on the target puppet.
        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts,
            pred_cam_t,
            cam_int_src,
            h_feat,
            w_feat,
            target_h,
            target_w,
        )
        valid_src_mask = (mask_src.squeeze(1) > 0.5) & (src_idx_map >= 0)
        flat_source_feats = rearrange(human_feat, "b c h w -> b (h w) c")
        flat_src_idx_map = src_idx_map.reshape(bsz, -1)
        flat_src_mask = valid_src_mask.reshape(bsz, -1)

        safe_vertex_indices = flat_src_idx_map.clamp_min(0)
        flat_src_verts = torch.gather(
            branch_verts,
            1,
            safe_vertex_indices.unsqueeze(-1).expand(-1, -1, 3),
        )

        target_verts, midhip, rotation = self.rotate_branch_geometry(
            branch_verts,
            branch_keypoints,
            global_rot,
            yaw,
            apply_global_rot_alignment,
        )
        _, target_depth_map = self.get_source_vertex_index_map(
            target_verts,
            cam_t_tgt,
            cam_int_tgt,
            h_feat,
            w_feat,
            target_h,
            target_w,
        )
        target_valid_mask = target_depth_map.reshape(bsz, -1) < 1e5

        # Apply exactly the same target pose/yaw transform used by GAOT.
        source_centered = flat_src_verts - midhip.unsqueeze(1)
        source_smpl = source_centered.clone()
        source_smpl[..., [1, 2]] *= -1
        source_rotated = torch.bmm(source_smpl, rotation)
        source_rotated[..., [1, 2]] *= -1

        source_in_target_camera = source_rotated + cam_t_tgt.unsqueeze(1)
        x, y, z_raw = source_in_target_camera.unbind(-1)
        z_safe = z_raw.clamp(min=1e-3)
        fx = cam_int_tgt[:, 0, 0].unsqueeze(1)
        fy = cam_int_tgt[:, 1, 1].unsqueeze(1)
        cx = cam_int_tgt[:, 0, 2].unsqueeze(1)
        cy = cam_int_tgt[:, 1, 2].unsqueeze(1)
        u_target = (x / z_safe) * fx + cx
        v_target = (y / z_safe) * fy + cy

        # Direct relocation discretizes the continuous projection instead of
        # constructing a target-by-source transport matrix.
        u_grid = u_target / target_w * w_feat
        v_grid = v_target / target_h * h_feat
        in_bounds = (
            (u_grid >= 0)
            & (u_grid < w_feat)
            & (v_grid >= 0)
            & (v_grid < h_feat)
            & (z_raw > 1e-3)
            & torch.isfinite(z_raw)
        )
        u_index = u_grid.long().clamp(0, w_feat - 1)
        v_index = v_grid.long().clamp(0, h_feat - 1)
        target_indices = v_index * w_feat + u_index
        target_count = h_feat * w_feat

        target_valid_at_source = torch.gather(
            target_valid_mask, 1, target_indices
        )
        valid_assignment = flat_src_mask & in_bounds & target_valid_at_source

        # Resolve source collisions using the canonical-camera z-buffer rule.
        depth_for_reduce = torch.where(
            valid_assignment,
            z_raw,
            torch.full_like(z_raw, 1e6),
        )
        minimum_depth = torch.full(
            (bsz, target_count),
            1e6,
            device=human_feat.device,
            dtype=z_raw.dtype,
        )
        minimum_depth.scatter_reduce_(
            1,
            target_indices,
            depth_for_reduce,
            reduce="amin",
            include_self=True,
        )
        winning_depth = torch.gather(minimum_depth, 1, target_indices)
        winners = valid_assignment & (z_raw <= winning_depth + 1e-4)

        relocated_sum = torch.zeros(
            (bsz, target_count, channels),
            device=human_feat.device,
            dtype=flat_source_feats.dtype,
        )
        relocated_sum.scatter_add_(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, channels),
            flat_source_feats * winners.unsqueeze(-1).to(flat_source_feats),
        )
        relocated_count = torch.zeros(
            (bsz, target_count, 1),
            device=human_feat.device,
            dtype=flat_source_feats.dtype,
        )
        relocated_count.scatter_add_(
            1,
            target_indices.unsqueeze(-1),
            winners.unsqueeze(-1).to(flat_source_feats),
        )
        relocated_feats = relocated_sum / relocated_count.clamp_min(1.0)
        relocated_valid_mask = (
            relocated_count.squeeze(-1) > 0
        ) & target_valid_mask
        relocated_feats = relocated_feats * relocated_valid_mask.unsqueeze(-1).to(
            relocated_feats
        )

        warped_feat = rearrange(
            relocated_feats,
            "b (h w) c -> b c h w",
            h=h_feat,
            w=w_feat,
        )
        return (
            warped_feat,
            relocated_valid_mask.reshape(bsz, 1, h_feat, w_feat),
            target_depth_map,
        )
