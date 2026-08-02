"""Local one-to-one feature placement for a trained PuppetGait A4 model.

This diagnostic keeps the complete A4 architecture and replaces only its
parameter-free Sparse OT placement operator.  Visible source tokens and valid
canonical cells form a local bipartite graph.  Sparse minimum-cost matching
assigns each source to at most one target and each target to at most one source,
so collisions can move to nearby free cells without feature averaging or
source duplication.
"""

import numpy as np
import torch
import torch.nn as nn

from .BiggerGait_SAM_3D_Body_projection_mask_OT_based_SparseTopK4 import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share,
)


class GeometryLocalOneToOneMatching(nn.Module):
    """Sparse exact local assignment with optional unmatched source tokens."""

    def __init__(self, dist_thresh=0.2, candidate_count=16):
        super().__init__()
        self.dist_thresh = float(dist_thresh)
        self.candidate_count = int(candidate_count)
        if self.dist_thresh <= 0:
            raise ValueError("dist_thresh must be positive")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")

    def _solve_frame(self, candidate_targets, candidate_costs, candidate_valid,
                     source_valid, target_count):
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import min_weight_full_bipartite_matching
        except ImportError as exc:
            raise ImportError(
                "Local one-to-one matching requires scipy. Install scipy in "
                "the OpenGait environment."
            ) from exc

        valid_sources = torch.nonzero(source_valid, as_tuple=False).flatten()
        matched_targets = torch.full_like(source_valid, -1, dtype=torch.long)
        source_count = int(valid_sources.numel())
        if source_count == 0:
            return matched_targets

        target_np = candidate_targets[valid_sources].detach().cpu().numpy()
        cost_np = candidate_costs[valid_sources].detach().float().cpu().numpy()
        valid_np = candidate_valid[valid_sources].detach().cpu().numpy()

        row_grid = np.broadcast_to(
            np.arange(source_count, dtype=np.int64)[:, None], target_np.shape
        )
        real_rows = row_grid[valid_np]
        real_cols = target_np[valid_np].astype(np.int64, copy=False)
        # Sparse scipy matrices drop numerical zeros, so keep every legal edge
        # strictly positive without changing its ordering.
        real_costs = cost_np[valid_np].astype(np.float64, copy=False) + 1e-8

        # One private dummy target per source permits unmatched tokens.  The
        # penalty is large enough to maximize cardinality before distance.
        dummy_rows = np.arange(source_count, dtype=np.int64)
        dummy_cols = target_count + dummy_rows
        max_total_real_cost = source_count * (self.dist_thresh ** 2 + 1e-6)
        dummy_costs = np.full(
            source_count, max_total_real_cost + 1.0, dtype=np.float64
        )

        rows = np.concatenate((real_rows, dummy_rows))
        cols = np.concatenate((real_cols, dummy_cols))
        costs = np.concatenate((real_costs, dummy_costs))
        graph = coo_matrix(
            (costs, (rows, cols)),
            shape=(source_count, target_count + source_count),
        ).tocsr()
        row_ind, col_ind = min_weight_full_bipartite_matching(graph)

        real_match = col_ind < target_count
        if np.any(real_match):
            matched_source_rows = torch.from_numpy(row_ind[real_match]).to(
                valid_sources.device
            )
            matched_source_indices = valid_sources[matched_source_rows]
            matched_target_indices = torch.from_numpy(
                col_ind[real_match].astype(np.int64, copy=False)
            ).to(valid_sources.device)
            matched_targets[matched_source_indices] = matched_target_indices
        return matched_targets

    def forward(self, source_feats, source_locs, target_locs,
                source_valid_mask=None, target_valid_mask=None):
        batch_size, source_count, channels = source_feats.shape
        target_count = target_locs.shape[1]
        device = source_feats.device

        if source_valid_mask is None:
            source_valid_mask = torch.ones(
                batch_size, source_count, dtype=torch.bool, device=device
            )
        else:
            source_valid_mask = source_valid_mask.bool()
        if target_valid_mask is None:
            target_valid_mask = torch.ones(
                batch_size, target_count, dtype=torch.bool, device=device
            )
        else:
            target_valid_mask = target_valid_mask.bool()

        with torch.no_grad():
            diff = source_locs.unsqueeze(2) - target_locs.unsqueeze(1)
            dist_sq = torch.sum(diff.square(), dim=-1)
            valid_connection = (
                (dist_sq < self.dist_thresh ** 2)
                & source_valid_mask.unsqueeze(2)
                & target_valid_mask.unsqueeze(1)
            )
            candidate_source_mask = valid_connection.any(dim=2)
            masked_cost = dist_sq.masked_fill(~valid_connection, float("inf"))
            candidate_count = min(self.candidate_count, target_count)
            candidate_costs, candidate_targets = torch.topk(
                masked_cost,
                k=candidate_count,
                dim=2,
                largest=False,
                sorted=True,
            )
            candidate_valid = torch.isfinite(candidate_costs)

            matched_targets = torch.stack(
                [
                    self._solve_frame(
                        candidate_targets[b],
                        candidate_costs[b],
                        candidate_valid[b],
                        source_valid_mask[b],
                        target_count,
                    )
                    for b in range(batch_size)
                ],
                dim=0,
            )
            retained_source_mask = matched_targets >= 0
            safe_targets = matched_targets.clamp_min(0)
            target_occupancy = torch.zeros(
                batch_size, target_count, dtype=torch.long, device=device
            )
            target_occupancy.scatter_add_(
                1, safe_targets, retained_source_mask.long()
            )
            supported_target_mask = target_occupancy > 0

            self.last_supported_target_mask = supported_target_mask.detach()
            self.last_valid_source_mask = source_valid_mask.detach()
            self.last_candidate_source_mask = candidate_source_mask.detach()
            self.last_retained_source_mask = retained_source_mask.detach()
            self.last_transport_edge_count = retained_source_mask.sum(1).detach()

        target_feats = torch.zeros(
            batch_size,
            target_count,
            channels,
            dtype=source_feats.dtype,
            device=device,
        )
        target_feats.scatter_add_(
            1,
            safe_targets.unsqueeze(-1).expand(-1, -1, channels),
            source_feats * retained_source_mask.unsqueeze(-1).to(source_feats),
        )
        target_feats = target_feats * supported_target_mask.unsqueeze(-1).to(
            target_feats
        )
        return target_feats


class BiggerGait__SAM3DBody__Projection_Mask_LocalOneToOne_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share
):
    """A4 with Sparse OT replaced by parameter-free local matching."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.ot_solver = GeometryLocalOneToOneMatching(
            dist_thresh=model_cfg.get(
                "matching_dist_thresh", model_cfg.get("ot_dist_thresh", 0.2)
            ),
            candidate_count=model_cfg.get("matching_candidate_count", 16),
        )
        self.msg_mgr.log_info(
            "[LocalOneToOne] Sparse OT replaced by exact sparse min-cost "
            f"matching: candidates={self.ot_solver.candidate_count}, "
            f"distance={self.ot_solver.dist_thresh}."
        )
