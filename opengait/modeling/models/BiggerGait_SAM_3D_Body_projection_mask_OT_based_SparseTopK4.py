import torch
import torch.nn as nn
import torch.utils.checkpoint

from .BiggerGait_SAM_3D_Body_projection_mask_OT_based import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share,
)


class GeometryOptimalTransportSparseTopK(nn.Module):
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


class BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share
):
    def build_network(self, model_cfg):
        super().build_network(model_cfg)

        branch_desc = [
            "original_view" if self._is_original_view_branch(cfg) else f"yaw={float(cfg.get('yaw', 0.0))}"
            for cfg in self.branch_configs
        ]
        self.msg_mgr.log_info(f"[OT] Branch configs: {branch_desc}")

        self.ot_topk_support = int(model_cfg.get("ot_topk_support", 0) or 0)
        self.ot_sparse_rebalance_iters = int(
            model_cfg.get("ot_sparse_rebalance_iters", model_cfg.get("ot_iters", 8))
        )
        self.ot_solver = GeometryOptimalTransportSparseTopK(
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

    @staticmethod
    def _is_original_view_branch(branch_cfg):
        if bool(branch_cfg.get("original_view", False)) or bool(branch_cfg.get("no_ot", False)):
            return True
        yaw = branch_cfg.get("yaw", 0.0)
        if yaw is None:
            return True
        if isinstance(yaw, str):
            return yaw.strip().lower() in {
                "none",
                "null",
                "raw",
                "original",
                "original_view",
                "identity",
                "no_rotate",
                "no-rotate",
            }
        return False

    def build_branch_geometry(self, branch_cfg, pose_out):
        if self._is_original_view_branch(branch_cfg):
            return {
                "use_original_view": True,
                "verts": None,
                "keypoints": None,
                "yaw": None,
                "apply_global_rot_alignment": False,
            }

        branch_geo = super().build_branch_geometry(branch_cfg, pose_out)
        branch_geo["use_original_view"] = False
        return branch_geo

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
        if branch_verts is None or yaw is None:
            _, src_depth_map = self.get_source_vertex_index_map(
                pred_verts, pred_cam_t, cam_int_src, h_feat, w_feat, target_h, target_w
            )
            return human_feat, mask_src, src_depth_map

        return super().warp_features_with_ot(
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
        )
