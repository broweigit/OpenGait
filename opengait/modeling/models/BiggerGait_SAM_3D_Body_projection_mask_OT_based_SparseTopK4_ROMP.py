import math
import time

import numpy as np
import roma
import torch
import torch.utils.checkpoint
from einops import rearrange
from functools import partial
from torch.nn import functional as F

from utils import list2var, np2var
from .BiggerGait_SAM_3D_Body_projection_mask_OT_based_SparseTopK4 import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share
):
    """SparseTopK4 ablation that keeps SAM3D LVM features but uses offline ROMP geometry."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.romp_hip_indices = tuple(model_cfg.get("romp_hip_indices", [1, 2]))
        self.romp_require_verts_camed_org = bool(model_cfg.get("romp_require_verts_camed_org", True))
        self.msg_mgr.log_info(
            "[ROMP] Using offline ROMP geometry for 3D projection; SAM3D backbone/LVM is unchanged."
        )
        self.msg_mgr.log_info(f"[ROMP] Hip indices for midhip: {self.romp_hip_indices}")

    def inputs_pretreament(self, inputs):
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. "
                f"But got {len(seqs_batch)} and {len(seq_trfs)}"
            )
        if len(seqs_batch) != 2:
            raise ValueError(
                "ROMP branch expects two modalities: RGB pkl and ROMP geometry pkl. "
                f"Got {len(seqs_batch)} modalities."
            )

        pretreat_start = time.perf_counter()
        requires_grad = bool(self.training)

        rgb_start = time.perf_counter()
        rgb = np2var(
            np.asarray([seq_trfs[0](seq) for seq in seqs_batch[0]], dtype=np.float32),
            requires_grad=requires_grad,
        ).float()
        rgb_time = time.perf_counter() - rgb_start

        # Keep ROMP frame dictionaries as Python objects. np.asarray would turn
        # the frame-list into object arrays and make nested dict handling brittle.
        geom_start = time.perf_counter()
        romp_geometry = [seq_trfs[1](seq) for seq in seqs_batch[1]]
        geom_time = time.perf_counter() - geom_start

        labs = list2var(labs_batch).long()
        seqL = np2var(seqL_batch).int() if seqL_batch is not None else None
        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            rgb = rgb[:, :seqL_sum]
            romp_geometry = [seq[:seqL_sum] for seq in romp_geometry]

        pretreat_total = time.perf_counter() - pretreat_start
        self._last_pretreat_timing = {
            "scalar/time/input_pretreat_rgb": rgb_time,
            "scalar/time/input_pretreat_romp": geom_time,
            "scalar/time/input_pretreat_misc": max(pretreat_total - rgb_time - geom_time, 0.0),
        }
        return [rgb, romp_geometry], labs, typs_batch, vies_batch, seqL

    @staticmethod
    def _extract_pose_frame(frame):
        pose = frame.get("pose_outs", frame)
        if isinstance(pose, (list, tuple)):
            pose = pose[-1]
        return pose

    @staticmethod
    def _frame_value(frame, key, default=None):
        if key in frame:
            return frame[key]
        pose = BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share._extract_pose_frame(frame)
        return pose.get(key, default)

    def _stack_romp_frames(self, frames, device):
        pose_frames = [self._extract_pose_frame(frame) for frame in frames]

        pred_vertices = torch.stack([
            torch.as_tensor(self._frame_value(frame, "pred_vertices"), dtype=torch.float32)
            for frame in frames
        ], dim=0).to(device)
        pred_cam_t = torch.stack([
            torch.as_tensor(self._frame_value(frame, "pred_cam_t"), dtype=torch.float32)
            for frame in frames
        ], dim=0).to(device)
        pred_keypoints = torch.stack([
            torch.as_tensor(self._frame_value(frame, "pred_keypoints_3d"), dtype=torch.float32)
            for frame in frames
        ], dim=0).to(device)

        verts_camed_org_list = [pose.get("verts_camed_org", None) for pose in pose_frames]
        if any(value is None for value in verts_camed_org_list):
            if self.romp_require_verts_camed_org:
                raise KeyError(
                    "ROMP geometry pkl is missing pose_outs['verts_camed_org']. "
                    "Regenerate it with the updated preprocess_romp_smpl_geometry_dataset.py."
                )
            verts_camed_org = None
        else:
            verts_camed_org = torch.stack([
                torch.as_tensor(value, dtype=torch.float32) for value in verts_camed_org_list
            ], dim=0).to(device)

        global_rot_values = []
        for pose in pose_frames:
            value = pose.get("global_rot", None)
            if value is None:
                value = np.zeros((3,), dtype=np.float32)
            global_rot_values.append(torch.as_tensor(value, dtype=torch.float32))
        global_rot = torch.stack(global_rot_values, dim=0).to(device)

        cam_int_values = []
        for frame in frames:
            value = frame.get("cam_int", None)
            if value is None:
                value = self._frame_value(frame, "cam_int", None)
            if value is None:
                value = np.eye(3, dtype=np.float32)
            cam_int_values.append(torch.as_tensor(value, dtype=torch.float32))
        cam_int = torch.stack(cam_int_values, dim=0).to(device)

        pose_out = {
            "pred_vertices": pred_vertices,
            "pred_cam_t": pred_cam_t,
            "pred_keypoints_3d": pred_keypoints,
            "global_rot": global_rot,
            "global_rot_type": "axis_angle",
            "cam_int": cam_int,
        }
        if verts_camed_org is not None:
            pose_out["verts_camed_org"] = verts_camed_org
        return pose_out

    def get_romp_source_vertex_index_map(self, verts_camed_org, h_feat, w_feat, target_h, target_w):
        bsz, num_verts, _ = verts_camed_org.shape
        device = verts_camed_org.device

        u = verts_camed_org[..., 0]
        v = verts_camed_org[..., 1]
        depth = -verts_camed_org[..., 2]

        in_bounds = (u >= 0) & (u < target_w) & (v >= 0) & (v < target_h)
        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)
        flat_pixel_indices = v_feat * w_feat + u_feat

        depth_for_reduce = depth.masked_fill(~in_bounds, 1e6)
        depth_map_flat = torch.full((bsz, h_feat * w_feat), 1e6, device=device)
        depth_map_flat.scatter_reduce_(1, flat_pixel_indices, depth_for_reduce, reduce="amin", include_self=False)

        min_depth_per_vertex = torch.gather(depth_map_flat, 1, flat_pixel_indices)
        is_visible = in_bounds & (depth_for_reduce < (min_depth_per_vertex + 1e-4))

        index_map_flat = torch.full((bsz, h_feat * w_feat), -1, dtype=torch.long, device=device)
        vertex_indices = torch.arange(num_verts, device=device).unsqueeze(0).expand(bsz, -1)
        mask_flat = is_visible.reshape(-1)
        batch_offsets = torch.arange(bsz, device=device).unsqueeze(1) * (h_feat * w_feat)
        global_pixel_indices = (flat_pixel_indices + batch_offsets).reshape(-1)

        index_map_global = index_map_flat.reshape(-1)
        index_map_global[global_pixel_indices[mask_flat]] = vertex_indices.reshape(-1)[mask_flat]
        return index_map_global.reshape(bsz, h_feat, w_feat), depth_map_flat.reshape(bsz, 1, h_feat, w_feat)

    def build_branch_geometry(self, branch_cfg, pose_out):
        if bool(branch_cfg.get("use_apose", False)):
            raise ValueError(
                "ROMP ablation uses SMPL topology and cannot reuse SAM3D/MHR A-pose generation. "
                "Set use_apose: false for all branches."
            )
        return super().build_branch_geometry(branch_cfg, pose_out)

    def rotate_branch_geometry(self, verts, keypoints, global_rot, yaw, apply_global_rot_alignment):
        batch_size = verts.shape[0]
        device = verts.device
        left_idx, right_idx = self.romp_hip_indices
        if keypoints.shape[1] <= max(left_idx, right_idx):
            midhip = verts.mean(dim=1)
        else:
            midhip = (keypoints[:, left_idx] + keypoints[:, right_idx]) / 2.0
        centered_verts = verts - midhip.unsqueeze(1)

        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        r_yaw = torch.tensor(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 3).expand(batch_size, 3, 3)

        if apply_global_rot_alignment:
            r_canon = roma.rotvec_to_rotmat(-global_rot.float())
            r_comp = torch.matmul(r_canon.transpose(1, 2), r_yaw.transpose(1, 2))
        else:
            r_comp = r_yaw.transpose(1, 2)

        verts_tmp = centered_verts.clone()
        verts_tmp[..., [1, 2]] *= -1
        rotated_smpl = torch.bmm(verts_tmp, r_comp)
        rotated_cv = rotated_smpl.clone()
        rotated_cv[..., [1, 2]] *= -1
        return rotated_cv, midhip, r_comp

    def warp_features_with_romp_ot(
        self,
        human_feat,
        mask_src,
        pred_verts,
        source_verts_camed_org,
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
            _, src_depth_map = self.get_romp_source_vertex_index_map(
                source_verts_camed_org, h_feat, w_feat, target_h, target_w
            )
            return human_feat, mask_src, src_depth_map

        bsz = human_feat.shape[0]
        src_idx_map, _ = self.get_romp_source_vertex_index_map(
            source_verts_camed_org, h_feat, w_feat, target_h, target_w
        )
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
            torch.linspace(-1, 1, h_feat, device=human_feat.device),
            torch.linspace(-1, 1, w_feat, device=human_feat.device),
            indexing="ij",
        )
        target_grid_locs = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(
            bsz, -1, -1, -1
        ).reshape(bsz, -1, 2)

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
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        romp_geometry = ipts[1]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        chunk_lengths = [chunk.size(1) for chunk in rgb_chunks]
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
        seq_start = 0
        for chunk_idx, (rgb_img, chunk_len) in enumerate(zip(rgb_chunks, chunk_lengths)):
            seq_end = seq_start + chunk_len
            geom_chunk = [seq[seq_start:seq_end] for seq in romp_geometry]
            seq_start = seq_end

            n, s, c, h, w = rgb_img.size()
            flat_geom_frames = [frame for seq in geom_chunk for frame in seq]
            if len(flat_geom_frames) != n * s:
                raise RuntimeError(f"ROMP frame count mismatch: expected {n * s}, got {len(flat_geom_frames)}")

            rgb_img = rearrange(rgb_img, "n s c h w -> (n s) c h w").contiguous()
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

                pose_out = self._stack_romp_frames(flat_geom_frames, rgb.device)
                self._apose_cache = {}
                pred_verts = pose_out["pred_vertices"]
                pred_cam_t = pose_out["pred_cam_t"]
                global_rot = pose_out["global_rot"]
                cam_int_src = pose_out["cam_int"]
                source_verts_camed_org = pose_out["verts_camed_org"]
                _, src_depth_map = self.get_romp_source_vertex_index_map(
                    source_verts_camed_org, h_feat, w_feat, target_h, target_w
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
                if self.hook_sample_type == "interleave":
                    sub_feats = features_to_use[i::self.num_FPN]
                elif self.hook_sample_type == "chunk":
                    start_idx = i * step
                    end_idx = (i + 1) * step
                    sub_feats = features_to_use[start_idx:end_idx]
                else:
                    raise ValueError(f"Invalid hook_sample_type: {self.hook_sample_type}")

                sub_app = torch.concat(sub_feats, dim=-1)
                curr_dim = self.f4_dim * len(sub_feats)
                sub_app = partial(torch.nn.LayerNorm, eps=1e-6)(
                    curr_dim, elementwise_affine=False
                )(sub_app)
                sub_app = rearrange(sub_app, "b (h w) c -> b c h w", h=h_feat).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_feat = torch.concat(processed_feat_list, dim=1)
            human_mask = self.preprocess(generated_mask, self.sils_size * 2, self.sils_size).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)

            cam_int_tgt, cam_t_tgt = self.build_target_camera(curr_bs, rgb.device, target_h, target_w)
            branch_warped_feats = []
            for b_idx, branch_cfg in enumerate(self.branch_configs):
                branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
                warp_feat, _, tgt_depth_map = self.warp_features_with_romp_ot(
                    human_feat,
                    human_mask.float(),
                    pred_verts,
                    source_verts_camed_org,
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
                warp_feat_5d = rearrange(warp_feat, "(n s) c h w -> n c s h w", n=n, s=s).contiguous()
                if debug_test_1:
                    pca_before_vis = []
                    for in_chunk in torch.chunk(warp_feat_5d, self.num_FPN, dim=1):
                        pca_before_vis.append(
                            self._build_pca_vis_batch(
                                rearrange(in_chunk, "n c s h w -> (n s) c h w").contiguous()[:5]
                            )
                        )
                    branch_pca_before_cnn[b_idx] = self._stack_fpn_vis(pca_before_vis)
                    outs, gait_debug = self.Gait_Nets[b_idx].test_1(warp_feat_5d, return_debug=True)
                    layer1_vis = []
                    layer2_vis = []
                    layer3_vis = []
                    layer4_vis = []
                    pca_vis = []
                    for i in range(self.num_FPN):
                        layer1_feat = rearrange(
                            gait_debug["layer1_feat_list"][i], "n c s h w -> (n s) c h w"
                        ).contiguous()
                        layer2_feat = rearrange(
                            gait_debug["layer2_feat_list"][i], "n c s h w -> (n s) c h w"
                        ).contiguous()
                        layer3_feat = rearrange(
                            gait_debug["layer3_feat_list"][i], "n c s h w -> (n s) c h w"
                        ).contiguous()
                        layer4_feat = rearrange(
                            gait_debug["layer4_feat_list"][i], "n c s h w -> (n s) c h w"
                        ).contiguous()
                        layer1_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(branch_target_depth[b_idx][:5], layer1_feat[:5])
                        )
                        layer2_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(branch_target_depth[b_idx][:5], layer2_feat[:5])
                        )
                        layer3_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(branch_target_depth[b_idx][:5], layer3_feat[:5])
                        )
                        layer4_vis.append(
                            self._build_feature_norm_on_depth_vis_batch(branch_target_depth[b_idx][:5], layer4_feat[:5])
                        )
                    for out_chunk in torch.chunk(outs, self.num_FPN, dim=1):
                        pca_vis.append(
                            self._build_pca_vis_batch(
                                rearrange(out_chunk, "n c s h w -> (n s) c h w").contiguous()[:5]
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
                "training_feat": {
                    "triplet": {"embeddings": torch.cat(embed_list, dim=-1), "labels": labs},
                    "softmax": {"logits": torch.cat(log_list, dim=-1), "labels": labs},
                },
                "visual_summary": {
                    "image/rgb_img": rgb_img.view(n * s, c, h, w)[:5].float(),
                    "image/generated_3d_mask_lowres": generated_mask.view(n * s, 1, h_feat, w_feat)[:5].float(),
                    "image/generated_3d_mask_interpolated": human_mask.view(
                        n * s, 1, self.sils_size * 2, self.sils_size
                    )[:5].float(),
                },
                "inference_feat": {
                    "embeddings": torch.cat(embed_list, dim=-1),
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
            }
            if cnn_layer1_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer1_l2norm"] = cnn_layer1_norm_summary.float()
            if pca_before_cnn_summary is not None:
                retval["visual_summary"]["image/pca_before_cnn"] = pca_before_cnn_summary.float()
            if pca_after_cnn_summary is not None:
                retval["visual_summary"]["image/pca_after_cnn"] = pca_after_cnn_summary.float()
            if cnn_layer2_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer2_l2norm"] = cnn_layer2_norm_summary.float()
            if cnn_layer3_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer3_l2norm"] = cnn_layer3_norm_summary.float()
            if cnn_layer4_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer4_l2norm"] = cnn_layer4_norm_summary.float()
        else:
            retval = {
                "training_feat": {},
                "visual_summary": {},
                "inference_feat": {
                    "embeddings": torch.cat(embed_list, dim=-1),
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
            }
        return retval
