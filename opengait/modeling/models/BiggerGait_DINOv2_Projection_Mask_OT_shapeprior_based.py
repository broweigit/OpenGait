import copy
import time

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .BiggerGait_DINOv2_Projection_Mask_based import ResizeToHW
from .BiggerGait_DINOv2_Projection_Mask_OT_based import (
    BiggerGait__DINOv2__Projection_Mask_OT_Based,
)


class BiggerGait__DINOv2__Projection_Mask_OT_ShapePrior_Based(
    BiggerGait__DINOv2__Projection_Mask_OT_Based
):
    """OT model with a silhouette-supervised shape-prior branch.

    The extra branch mirrors the appearance OT path, but first passes each DINO
    FPN group through a silhouette bottleneck before reconstructing a
    HumanSpace-ready feature map.
    """

    def build_network(self, model_cfg):
        super().build_network(model_cfg)

        shape_cfg = model_cfg.get("shape_prior_cfg", {})
        self.shape_prior_dim = int(shape_cfg.get("prior_dim", 64))
        self.shape_prior_detach_bottleneck = bool(shape_cfg.get("detach_bottleneck", True))
        self.shape_prior_use_layernorm = bool(shape_cfg.get("use_layernorm", True))
        self.shape_prior_use_silhouette_loss = bool(shape_cfg.get("use_silhouette_loss", True))

        self.shape_fc_dim = int(self.f4_dim * self.group_layer_num)
        hidden_dim = max(self.shape_fc_dim // 2, self.shape_prior_dim)
        human_hidden_dim = max(self.shape_fc_dim // 4, self.num_unknown)

        self.Shape_Branch1_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.shape_fc_dim, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, self.shape_prior_dim, kernel_size=1),
                    nn.BatchNorm2d(self.shape_prior_dim),
                    nn.GELU(),
                )
                for _ in range(self.num_FPN)
            ]
        )

        self.Shape_Branch2_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.shape_prior_dim, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, self.shape_fc_dim, kernel_size=1),
                )
                for _ in range(self.num_FPN)
            ]
        )

        self.Shape_Head_list = nn.ModuleList(
            [
                nn.Sequential(
                    ResizeToHW((self.sils_size * 2, self.sils_size)),
                    nn.Conv2d(self.shape_prior_dim, 1, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )
                for _ in range(self.num_FPN)
            ]
        )

        self.Shape_HumanSpace_Conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(self.shape_fc_dim, affine=False),
                    nn.Conv2d(self.shape_fc_dim, human_hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(human_hidden_dim, affine=False),
                    nn.GELU(),
                    nn.Conv2d(human_hidden_dim, self.num_unknown, kernel_size=1),
                    ResizeToHW((self.sils_size * 2, self.sils_size)),
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid(),
                )
                for _ in range(self.num_FPN)
            ]
        )

        self.Shape_Gait_Nets = nn.ModuleList([copy.deepcopy(net) for net in self.Gait_Nets])
        self.msg_mgr.log_info(
            "[ShapePrior] Enabled silhouette prior branch: "
            f"prior_dim={self.shape_prior_dim}, detach_bottleneck={self.shape_prior_detach_bottleneck}, "
            f"use_layernorm={self.shape_prior_use_layernorm}, "
            f"use_silhouette_loss={self.shape_prior_use_silhouette_loss}."
        )

    def _shape_reconstruct_norm(self, x):
        if not self.shape_prior_use_layernorm:
            return x
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        x = F.layer_norm(x, (self.shape_fc_dim,), eps=1.0e-6)
        return rearrange(x, "b h w c -> b c h w").contiguous()

    def _run_shape_prior(self, grouped_intermediates):
        shape_feats = []
        shape_probs = []
        for i, feat in enumerate(grouped_intermediates):
            shape_latent = self.Shape_Branch1_list[i](feat)
            shape_probs.append(self.Shape_Head_list[i](shape_latent))
            branch2_input = shape_latent.detach() if self.shape_prior_detach_bottleneck else shape_latent
            reconstructed = self.Shape_Branch2_list[i](branch2_input)
            reconstructed = self._shape_reconstruct_norm(reconstructed)
            shape_feats.append(self.Shape_HumanSpace_Conv[i](reconstructed))
        return torch.cat(shape_feats, dim=1), torch.stack(shape_probs, dim=0).mean(dim=0)

    def _warp_branch_features(
        self,
        feat,
        mask,
        pose_out,
        cam_int_src,
        cam_int_tgt,
        cam_t_tgt,
        target_h,
        target_w,
        flat_flip_flags,
    ):
        branch_warped_feats = []
        branch_target_masks = []
        for branch_cfg in self.branch_configs:
            branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
            if branch_geo["use_original_view"]:
                warp_feat = feat
                tgt_valid_mask = mask.float()
            else:
                warp_feat, tgt_valid_mask, _ = self.warp_features_with_ot(
                    feat,
                    mask.float(),
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
        return branch_warped_feats, branch_target_masks

    def forward(self, inputs):
        timing_info = {
            "model_hflip": 0.0,
            "model_rgb_preprocess": 0.0,
            "model_dino": 0.0,
            "model_sam_unpack": 0.0,
            "model_project_mask": 0.0,
            "model_humanspace": 0.0,
            "model_shape_prior": 0.0,
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

        app_outs = [[] for _ in range(self.num_branches)]
        shape_outs = [[] for _ in range(self.num_branches)]
        shape_pred_masks = []
        shape_target_masks = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = self.image_size // 7, self.image_size // 14
        last_rgb_img = None
        last_generated_mask = None
        last_human_mask = None
        last_shape_prob = None
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
                intermediates = F.layer_norm(
                    torch.cat(outs, dim=-1),
                    (self.f4_dim * len(outs),),
                    eps=1.0e-6,
                )[:, 1:]
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

            grouped_intermediates = [
                torch.cat(intermediates[i : i + self.group_layer_num], dim=1).contiguous()
                for i in range(0, self.total_layer_num, self.group_layer_num)
            ]

            stage_start = self._perf_now(rgb.device)
            app_feats = [self.HumanSpace_Conv[i](feat) for i, feat in enumerate(grouped_intermediates)]
            human_feat = torch.cat(app_feats, dim=1)
            human_mask = self.preprocess(generated_mask, self.sils_size)
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            timing_info["model_humanspace"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            shape_feat, shape_prob = self._run_shape_prior(grouped_intermediates)
            shape_feat = shape_feat * (human_mask > 0.5).to(shape_feat)
            shape_pred_masks.append(shape_prob)
            shape_target_masks.append(human_mask.detach().float())
            timing_info["model_shape_prior"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            self._apose_cache = None
            self._apose_cache_key = None
            cam_int_tgt, cam_t_tgt = self.build_target_camera(rgb_img.shape[0], rgb.device, target_h, target_w)
            app_warped_feats, branch_target_masks = self._warp_branch_features(
                human_feat,
                human_mask,
                pose_out,
                cam_int_src,
                cam_int_tgt,
                cam_t_tgt,
                target_h,
                target_w,
                flat_flip_flags,
            )
            shape_warped_feats, _ = self._warp_branch_features(
                shape_feat,
                human_mask,
                pose_out,
                cam_int_src,
                cam_int_tgt,
                cam_t_tgt,
                target_h,
                target_w,
                flat_flip_flags,
            )
            timing_info["model_ot"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            for b_idx, warp_feat in enumerate(app_warped_feats):
                warp_feat_5d = rearrange(warp_feat, "(n s) c h w -> n c s h w", n=n, s=s).contiguous()
                outs = self.Gait_Nets[b_idx].test_1(warp_feat_5d)
                app_outs[b_idx].append(outs)
            for b_idx, warp_feat in enumerate(shape_warped_feats):
                warp_feat_5d = rearrange(warp_feat, "(n s) c h w -> n c s h w", n=n, s=s).contiguous()
                outs = self.Shape_Gait_Nets[b_idx].test_1(warp_feat_5d)
                shape_outs[b_idx].append(outs)
            timing_info["model_gait_head"] += self._perf_now(rgb.device) - stage_start

            last_rgb_img = rgb_img
            last_generated_mask = generated_mask
            last_human_mask = human_mask
            last_shape_prob = shape_prob
            last_target_masks = branch_target_masks

        stage_start = self._perf_now(rgb.device)
        app_embed_grouped = [[] for _ in range(self.num_FPN)]
        app_log_grouped = [[] for _ in range(self.num_FPN)]
        shape_embed_grouped = [[] for _ in range(self.num_FPN)]
        shape_log_grouped = [[] for _ in range(self.num_FPN)]
        for b_idx in range(self.num_branches):
            branch_seq_feat = torch.cat(app_outs[b_idx], dim=2)
            e_list, l_list = self.Gait_Nets[b_idx].test_2(branch_seq_feat, seqL)
            shape_seq_feat = torch.cat(shape_outs[b_idx], dim=2)
            se_list, sl_list = self.Shape_Gait_Nets[b_idx].test_2(shape_seq_feat, seqL)
            for i in range(self.num_FPN):
                app_embed_grouped[i].append(e_list[i])
                app_log_grouped[i].append(l_list[i])
                shape_embed_grouped[i].append(se_list[i])
                shape_log_grouped[i].append(sl_list[i])

        app_embed_list = [torch.cat(feats, dim=-1) for feats in app_embed_grouped]
        app_log_list = [torch.cat(logits, dim=-1) for logits in app_log_grouped]
        shape_embed_list = [torch.cat(feats, dim=-1) for feats in shape_embed_grouped]
        shape_log_list = [torch.cat(logits, dim=-1) for logits in shape_log_grouped]
        embed_list = [torch.cat([app_embed_list[i], shape_embed_list[i]], dim=-1) for i in range(self.num_FPN)]
        log_list = [torch.cat([app_log_list[i], shape_log_list[i]], dim=-1) for i in range(self.num_FPN)]
        timing_info["model_gait_head"] += self._perf_now(rgb.device) - stage_start

        embeddings = torch.cat(embed_list, dim=-1)
        logits = torch.cat(log_list, dim=-1)
        shape_pred = torch.cat(shape_pred_masks, dim=0).clamp(1.0e-6, 1.0 - 1.0e-6)
        shape_target = torch.cat(shape_target_masks, dim=0)

        model_end = self._perf_now(rgb.device)
        timed_sum = sum(timing_info.values())
        timing_info["model_misc"] = max(model_end - model_start - timed_sum, 0.0)

        if self.training:
            visual_summary = {
                "image/rgb_img": last_rgb_img[:5].float(),
                "image/generated_3d_mask_lowres": last_generated_mask[:5].float(),
                "image/generated_3d_mask_interpolated": last_human_mask[:5].float(),
                "image/shape_prior_prob": last_shape_prob[:5].float(),
            }
            if last_target_masks:
                visual_summary["image/ot_target_mask_branch0"] = last_target_masks[0][:5].float()
            training_feat = {
                "triplet": {"embeddings": embeddings, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            }
            if self.shape_prior_use_silhouette_loss:
                training_feat["shape_silhouette"] = {"logits": shape_pred, "labels": shape_target}
            retval = {
                "training_feat": training_feat,
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
