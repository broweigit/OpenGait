import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.robustness_eval_cfg = model_cfg.get("robustness_eval", {}) or {}
        self.robustness_eval_enabled = bool(
            self.robustness_eval_cfg.get("enabled", False)
        )
        self.sparse_sensitivity_cfg = model_cfg.get(
            "sparse_sensitivity_eval", {}
        ) or {}
        self.sparse_sensitivity_enabled = bool(
            self.sparse_sensitivity_cfg.get("enabled", False)
        )
        if self.robustness_eval_enabled and self.sparse_sensitivity_enabled:
            raise ValueError(
                "robustness_eval and sparse_sensitivity_eval are mutually exclusive."
            )
        self._active_robustness_variant = {"type": "clean", "name": "clean"}
        self._robustness_batch_index = 0
        self._active_robustness_seed = int(
            self.robustness_eval_cfg.get("seed", 7890)
        )

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
        self._default_ot_settings = {
            "topk": self.ot_solver.topk_support,
            "distance": self.ot_solver.dist_thresh,
            "iterations": self.ot_solver.sparse_rebalance_iters,
            "temperature": self.ot_solver.epsilon,
        }
        if self.ot_topk_support > 0:
            self.msg_mgr.log_info(
                f"[OT] Sparse support enabled: topk={self.ot_topk_support}, "
                f"rebalance_iters={self.ot_sparse_rebalance_iters}"
            )
        if self.robustness_eval_enabled:
            self.msg_mgr.log_info(
                "[Robustness] One-pass evaluation enabled: low resolution, "
                "occlusion, frame-wise pose jitter, and temporal parameter mean."
            )
        if self.sparse_sensitivity_enabled:
            self.msg_mgr.log_info(
                "[OT] One-pass Sparse sensitivity evaluation enabled."
            )

    def _robustness_variants(self):
        lowres_scales = self.robustness_eval_cfg.get("lowres_scales", [0.5, 0.25])
        occlusion_fractions = self.robustness_eval_cfg.get(
            "occlusion_fractions", [0.4, 0.6, 0.8]
        )
        pose_jitter_degrees = self.robustness_eval_cfg.get(
            "pose_jitter_degrees", [5.0, 10.0]
        )
        if len(lowres_scales) != 2 or len(pose_jitter_degrees) != 2:
            raise ValueError(
                "robustness_eval requires exactly two lowres_scales and two "
                "pose_jitter_degrees."
            )

        variants = [{"type": "clean", "name": "clean"}]
        for scale in lowres_scales:
            scale = float(scale)
            if not 0.0 < scale <= 1.0:
                raise ValueError(f"Invalid robustness low-resolution scale: {scale}")
            divisor = int(round(1.0 / scale))
            variants.append({
                "type": "lowres",
                "name": f"lowres_x{divisor}",
                "scale": scale,
            })
        for fraction in occlusion_fractions:
            fraction = float(fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"Invalid occlusion fraction: {fraction}")
            percent = int(round(fraction * 100.0))
            variants.append({
                "type": "occlusion",
                "name": f"occlusion_{percent}pct",
                "fraction": fraction,
            })
        for degrees in pose_jitter_degrees:
            degrees = float(degrees)
            if degrees < 0.0:
                raise ValueError(f"Invalid pose jitter: {degrees} degrees")
            degree_name = (
                str(int(degrees))
                if degrees.is_integer()
                else str(degrees).replace(".", "p")
            )
            variants.append({
                "type": "pose_jitter",
                "name": f"pose_jitter_{degree_name}deg",
                "std_degrees": degrees,
            })
        if self.robustness_eval_cfg.get("temporal_parameter_mean", True):
            variants.append({
                "type": "temporal_mean",
                "name": "temporal_parameter_mean",
            })
        return variants

    def _sparse_sensitivity_variants(self):
        default = dict(self._default_ot_settings)
        variants = [{"name": "current", **default}]

        def append_variant(name, **updates):
            settings = dict(default)
            settings.update(updates)
            if settings != default:
                variants.append({"name": name, **settings})

        for topk in self.sparse_sensitivity_cfg.get("topk_values", [1, 2, 4, 8]):
            append_variant(f"topk_{int(topk)}", topk=int(topk))
        for distance in self.sparse_sensitivity_cfg.get(
            "distance_values", [0.1, 0.2, 0.3]
        ):
            value = float(distance)
            append_variant(
                f"distance_{str(value).replace('.', 'p')}", distance=value
            )
        for iterations in self.sparse_sensitivity_cfg.get(
            "iteration_values", [1, 2, 4, 8]
        ):
            append_variant(
                f"iterations_{int(iterations)}", iterations=int(iterations)
            )
        for temperature in self.sparse_sensitivity_cfg.get(
            "temperature_values", [0.005, 0.01, 0.02]
        ):
            value = float(temperature)
            append_variant(
                f"temperature_{str(value).replace('.', 'p')}",
                temperature=value,
            )
        return variants

    def prepare_backbone_input(
        self,
        rgb_img,
        target_h,
        target_w,
        sequence_batch=None,
        sequence_length=None,
        sequence_lengths=None,
    ):
        backbone_input = super().prepare_backbone_input(
            rgb_img,
            target_h,
            target_w,
            sequence_batch=sequence_batch,
            sequence_length=sequence_length,
            sequence_lengths=sequence_lengths,
        )
        variant = self._active_robustness_variant
        if variant.get("type") == "lowres":
            scale = float(variant["scale"])
            low_h = max(1, int(round(target_h * scale)))
            low_w = max(1, int(round(target_w * scale)))
            lowres = F.interpolate(
                backbone_input,
                size=(low_h, low_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            return F.interpolate(
                lowres,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        if variant.get("type") == "occlusion":
            fraction = float(variant["fraction"])
            band_h = max(1, int(round(target_h * fraction)))
            band_top = (target_h - band_h) // 2
            band_bottom = band_top + band_h
            occluded = backbone_input.clone()
            black = torch.tensor(
                [
                    -0.485 / 0.229,
                    -0.456 / 0.224,
                    -0.406 / 0.225,
                ],
                device=occluded.device,
                dtype=occluded.dtype,
            ).view(1, 3, 1, 1)
            occluded[:, :, band_top:band_bottom, :] = black
            return occluded
        return backbone_input

    @staticmethod
    def _mhr_rotation_parameter_indices(device):
        # MHR body_pose has 3-DoF and 1-DoF rotation entries interleaved with
        # six translation entries (124..129). Only rotations are perturbed.
        indices_3d = [
            (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17),
            (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29),
            (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55),
            (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82),
            (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106),
            (114, 98, 109), (115, 99, 103),
        ]
        indices_1d = [
            1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43,
            47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62,
            63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90,
            94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116,
            117, 118, 119, 120, 121, 122, 123,
        ]
        flattened_3d = [index for group in indices_3d for index in group]
        return torch.tensor(flattened_3d + indices_1d, dtype=torch.long, device=device)

    def _rebuild_pose_geometry(self, pose_out):
        zero_global_trans = torch.zeros_like(
            pose_out["global_rot"], dtype=torch.float32
        )
        pose_device_type = pose_out["global_rot"].device.type
        with torch.no_grad(), torch.amp.autocast(
            enabled=False, device_type=pose_device_type
        ):
            vertices, keypoints = self.SAM_Engine.head_pose.mhr_forward(
                global_trans=zero_global_trans,
                global_rot=pose_out["global_rot"].float(),
                body_pose_params=pose_out["body_pose"].float(),
                hand_pose_params=pose_out["hand"].float(),
                scale_params=pose_out["scale"].float(),
                shape_params=pose_out["shape"].float(),
                expr_params=pose_out["face"].float(),
                return_keypoints=True,
            )
        vertices[..., [1, 2]] *= -1
        keypoints = keypoints[:, :70]
        keypoints[..., [1, 2]] *= -1
        rebuilt = dict(pose_out)
        rebuilt["pred_vertices"] = vertices.float()
        rebuilt["pred_keypoints_3d"] = keypoints.float()
        return rebuilt

    @staticmethod
    def _normalize_sequence_lengths(sequence_lengths, total_frames):
        if sequence_lengths is None:
            return [int(total_frames)]
        if torch.is_tensor(sequence_lengths):
            values = sequence_lengths.detach().cpu().reshape(-1).tolist()
        else:
            values = torch.as_tensor(sequence_lengths).reshape(-1).tolist()
        values = [int(value) for value in values]
        if any(value <= 0 for value in values):
            raise ValueError(f"Invalid sequence lengths: {values}")
        if sum(values) != int(total_frames):
            raise ValueError(
                f"seqL sums to {sum(values)}, but input contains {total_frames} frames."
            )
        return values

    def _collect_temporal_parameters(self, pose_out):
        keys = [
            "global_rot", "body_pose", "shape", "scale", "hand", "face",
            "pred_cam_t",
        ]
        chunk = {}
        for key in keys:
            chunk[key] = pose_out[key].float().detach()
        self._temporal_pose_chunks.append(chunk)

    @staticmethod
    def _circular_mean(values, dim):
        return torch.atan2(values.sin().mean(dim=dim), values.cos().mean(dim=dim))

    def _finalize_temporal_parameter_mean(self):
        if not self._temporal_pose_chunks:
            raise RuntimeError("No pose chunks were collected for temporal mean.")
        keys = self._temporal_pose_chunks[0].keys()
        full = {
            key: torch.cat([chunk[key] for chunk in self._temporal_pose_chunks], dim=0)
            for key in keys
        }
        total_frames = next(iter(full.values())).shape[0]
        if sum(self._current_sequence_lengths) != total_frames:
            raise RuntimeError(
                "Collected temporal-parameter frames do not match current seqL."
            )

        per_frame = {key: [] for key in keys}
        start = 0
        for length in self._current_sequence_lengths:
            end = start + length
            sequence_values = {key: value[start:end] for key, value in full.items()}
            means = {
                key: value.mean(dim=0, keepdim=True)
                for key, value in sequence_values.items()
            }
            means["global_rot"] = self._circular_mean(
                sequence_values["global_rot"], dim=0
            ).unsqueeze(0)
            body_mean = means["body_pose"]
            rotation_indices = self._mhr_rotation_parameter_indices(body_mean.device)
            body_mean[:, rotation_indices] = self._circular_mean(
                sequence_values["body_pose"][:, rotation_indices], dim=0
            ).unsqueeze(0)
            means["body_pose"] = body_mean
            for key, mean_value in means.items():
                per_frame[key].append(mean_value.expand(length, *mean_value.shape[1:]))
            start = end

        self._temporal_pose_per_frame = {
            key: torch.cat(values, dim=0) for key, values in per_frame.items()
        }
        self._temporal_pose_cursor = 0

    def _apply_temporal_parameter_mean(
        self, pose_out, sequence_batch, sequence_length
    ):
        if not hasattr(self, "_temporal_pose_per_frame"):
            raise RuntimeError("Temporal parameter mean cache has not been built.")
        rebuilt = dict(pose_out)
        chunk_frames = int(sequence_batch) * int(sequence_length)
        start = self._temporal_pose_cursor
        end = start + chunk_frames
        for key, values in self._temporal_pose_per_frame.items():
            rebuilt[key] = values[start:end]
        if end > next(iter(self._temporal_pose_per_frame.values())).shape[0]:
            raise RuntimeError("Temporal mean cache was exhausted before forward ended.")
        self._temporal_pose_cursor = end
        return self._rebuild_pose_geometry(rebuilt)

    def _reset_mesh_quality_stats(self):
        self._mesh_quality_chunks = []

    def _record_mesh_quality(self, pose_out, sequence_batch, sequence_length):
        vertices = pose_out["pred_vertices"].float()
        cam_t = pose_out["pred_cam_t"].float()
        target_h, target_w = self.image_size * 2, self.image_size
        focal = max(target_h, target_w) * 1.1

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        finite_vertices = torch.isfinite(v_cam).all(dim=-1)
        positive_depth = z > 1e-3
        z_safe = z.clamp(min=1e-3)
        u = (x / z_safe) * focal + target_w / 2.0
        v = (y / z_safe) * focal + target_h / 2.0
        in_frame = (
            finite_vertices
            & positive_depth
            & (u >= 0.0)
            & (u < target_w)
            & (v >= 0.0)
            & (v < target_h)
        )
        in_frame_ratio = in_frame.float().mean(dim=1)
        finite_frame = finite_vertices.all(dim=1) & torch.isfinite(cam_t).all(dim=1)
        minimum_ratio = float(
            self.robustness_eval_cfg.get("mesh_failure_min_in_frame_ratio", 0.1)
        )
        failure = (~finite_frame) | (in_frame_ratio < minimum_ratio)
        self._mesh_quality_chunks.append({
            "failure": failure.float(),
            "in_frame_ratio": in_frame_ratio,
        })

    def _finalize_mesh_quality_stats(self):
        if not self._mesh_quality_chunks:
            raise RuntimeError("No mesh-quality chunks were recorded.")
        failures = torch.cat(
            [chunk["failure"] for chunk in self._mesh_quality_chunks], dim=0
        )
        ratios = torch.cat(
            [chunk["in_frame_ratio"] for chunk in self._mesh_quality_chunks], dim=0
        )
        if failures.numel() != sum(self._current_sequence_lengths):
            raise RuntimeError("Mesh diagnostic frames do not match current seqL.")
        failure_per_sequence = []
        ratio_per_sequence = []
        start = 0
        for length in self._current_sequence_lengths:
            end = start + length
            failure_per_sequence.append(failures[start:end].mean())
            ratio_per_sequence.append(ratios[start:end].mean())
            start = end
        return {
            "mesh_failure_rate": torch.stack(failure_per_sequence).unsqueeze(-1),
            "mesh_in_frame_vertex_ratio": torch.stack(ratio_per_sequence).unsqueeze(-1),
        }

    def perturb_pose_out_for_eval(
        self,
        pose_out,
        sequence_batch,
        sequence_length,
        chunk_index=0,
        sequence_lengths=None,
    ):
        variant = self._active_robustness_variant
        variant_type = variant.get("type")
        if variant_type == "collect_temporal_mean":
            self._collect_temporal_parameters(pose_out)
            return pose_out
        if variant_type == "temporal_mean":
            pose_out = self._apply_temporal_parameter_mean(
                pose_out, sequence_batch, sequence_length
            )
        elif variant_type == "pose_jitter":
            body_pose = pose_out["body_pose"].float().clone()
            global_rot = pose_out["global_rot"].float().clone()
            expected_frames = int(sequence_batch) * int(sequence_length)
            rotation_indices = self._mhr_rotation_parameter_indices(body_pose.device)
            generator = torch.Generator(device=body_pose.device)
            generator.manual_seed(
                self._active_robustness_seed + int(chunk_index) * 1000003
            )
            frame_noise = torch.randn(
                (expected_frames, 3 + rotation_indices.numel()),
                device=body_pose.device,
                dtype=body_pose.dtype,
                generator=generator,
            )
            noise_scale = math.radians(float(variant["std_degrees"]))
            global_rot += frame_noise[:, :3] * noise_scale
            body_pose[:, rotation_indices] += frame_noise[:, 3:] * noise_scale
            jittered = dict(pose_out)
            jittered["global_rot"] = global_rot
            jittered["body_pose"] = body_pose
            pose_out = self._rebuild_pose_geometry(jittered)

        self._record_mesh_quality(pose_out, sequence_batch, sequence_length)
        return pose_out

    def _set_ot_settings(self, settings):
        self.ot_solver.topk_support = int(settings["topk"])
        self.ot_solver.dist_thresh = float(settings["distance"])
        self.ot_solver.sparse_rebalance_iters = int(settings["iterations"])
        self.ot_solver.epsilon = float(settings["temperature"])

    def _forward_sparse_sensitivity(self, inputs):
        inference_feat = {}
        try:
            for variant in self._sparse_sensitivity_variants():
                self._set_ot_settings(variant)
                retval = super().forward(inputs)
                embedding = retval["inference_feat"]["embeddings"]
                inference_feat[f"embeddings_{variant['name']}"] = embedding
                if variant["name"] == "current":
                    inference_feat["embeddings"] = embedding
                del retval
        finally:
            self._set_ot_settings(self._default_ot_settings)
        return {
            "training_feat": {},
            "visual_summary": {},
            "inference_feat": inference_feat,
        }

    def _forward_robustness(self, inputs):
        inference_feat = {}
        base_seed = int(self.robustness_eval_cfg.get("seed", 7890))
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self._active_robustness_seed = (
            base_seed + rank * 10000019 + self._robustness_batch_index * 1000033
        )
        rgb = inputs[0][0]
        if inputs[-1] is None or int(rgb.shape[0]) != 1:
            raise ValueError(
                "robustness_eval requires an all_ordered inference sampler so "
                "that seqL supplies the exact boundaries of concatenated sequences."
            )
        total_frames = int(rgb.shape[1])
        self._current_sequence_lengths = self._normalize_sequence_lengths(
            inputs[-1], total_frames
        )
        try:
            for variant in self._robustness_variants():
                if variant["type"] == "temporal_mean":
                    self._temporal_pose_chunks = []
                    self._active_robustness_variant = {
                        "type": "collect_temporal_mean",
                        "name": "collect_temporal_mean",
                    }
                    collect_retval = super().forward(inputs)
                    del collect_retval
                    self._finalize_temporal_parameter_mean()

                self._reset_mesh_quality_stats()
                self._active_robustness_variant = variant
                retval = super().forward(inputs)
                embedding = retval["inference_feat"]["embeddings"]
                inference_feat[f"embeddings_{variant['name']}"] = embedding
                if variant["type"] == "clean":
                    inference_feat["embeddings"] = embedding
                for stat_name, stat_value in self._finalize_mesh_quality_stats().items():
                    inference_feat[f"{stat_name}_{variant['name']}"] = stat_value
                del retval
        finally:
            self._active_robustness_variant = {"type": "clean", "name": "clean"}
            self._robustness_batch_index += 1
            self._temporal_pose_chunks = []
            if hasattr(self, "_temporal_pose_per_frame"):
                del self._temporal_pose_per_frame
            if hasattr(self, "_temporal_pose_cursor"):
                del self._temporal_pose_cursor
        return {
            "training_feat": {},
            "visual_summary": {},
            "inference_feat": inference_feat,
        }

    def forward(self, inputs):
        if not self.robustness_eval_enabled and not self.sparse_sensitivity_enabled:
            return super().forward(inputs)
        if self.training:
            raise RuntimeError("Variant evaluation modes are evaluation-only.")
        if self.robustness_eval_enabled:
            return self._forward_robustness(inputs)
        return self._forward_sparse_sensitivity(inputs)

    def _log_gflops_if_training(self):
        if not self.training:
            return

        was_training = self.training
        device = torch.device("cuda", torch.distributed.get_rank()) if torch.cuda.is_available() else torch.device("cpu")
        profile_inputs = (
            [[
                torch.randn((1, 1, 3, self.image_size * 2, self.image_size), dtype=torch.float32, device=device)
            ], None, None, None, None],
        )

        try:
            from fvcore.nn import FlopCountAnalysis

            self.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(self.to(device), profile_inputs).total() / 1e9
            self.msg_mgr.log_info(f"[Profile] GFLOPs: {flops:.2f}")
        except Exception as exc:
            try:
                from torch.profiler import ProfilerActivity, profile

                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)

                self.eval()
                with torch.no_grad():
                    with profile(
                        activities=activities,
                        with_flops=True,
                        record_shapes=False,
                        profile_memory=False,
                    ) as prof:
                        _ = self(profile_inputs[0])
                flops = sum((getattr(evt, "flops", 0) or 0) for evt in prof.key_averages()) / 1e9
                self.msg_mgr.log_info(f"[Profile] GFLOPs: {flops:.2f}")
            except Exception as fallback_exc:
                self.msg_mgr.log_warning(
                    f"[Profile] GFLOPs profiling failed: {exc}; fallback failed: {fallback_exc}"
                )
        finally:
            self.train(was_training)

    def init_parameters(self):
        super().init_parameters()
        self._log_gflops_if_training()

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
