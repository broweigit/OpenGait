import torch
from einops import rearrange
from torch.nn import functional as F

from .BiggerGait_DINOv2_Projection_Mask_OT_shapeprior_based import (
    BiggerGait__DINOv2__Projection_Mask_OT_ShapePrior_Based,
)


class BiggerGait__DINOv2__Projection_Mask_OT_ShapePrior_Pretrain_Based(
    BiggerGait__DINOv2__Projection_Mask_OT_ShapePrior_Based
):
    """Shape-prior OT model with an explicit silhouette-prior pretraining stage.

    Stage 1 only trains the prior translator from frozen DINO features to a
    SAM3D-projected silhouette and back to DINO feature space. Stage 2 freezes
    that translator and trains the gait branches.
    """

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        pretrain_cfg = model_cfg.get("shape_prior_pretrain_cfg", {})
        self.shape_prior_pretrain_iters = int(pretrain_cfg.get("pretrain_iters", 5000))
        self.shape_prior_reconstruction_weight = float(
            pretrain_cfg.get("reconstruction_loss_weight", 0.1)
        )
        self.shape_prior_freeze_after_pretrain = bool(
            pretrain_cfg.get("freeze_translator_after_pretrain", True)
        )
        self.shape_prior_disable_sil_loss_after_pretrain = bool(
            pretrain_cfg.get("disable_silhouette_loss_after_pretrain", True)
        )
        self._shape_prior_translator_frozen = False
        self.msg_mgr.log_info(
            "[ShapePriorPretrain] Enabled two-stage training: "
            f"pretrain_iters={self.shape_prior_pretrain_iters}, "
            f"reconstruction_loss_weight={self.shape_prior_reconstruction_weight:.4f}, "
            f"freeze_translator_after_pretrain={self.shape_prior_freeze_after_pretrain}, "
            f"disable_silhouette_loss_after_pretrain={self.shape_prior_disable_sil_loss_after_pretrain}."
        )

    def _shape_prior_translator_modules(self):
        return [
            self.Shape_Branch1_list,
            self.Shape_Head_list,
            self.Shape_Branch2_list,
        ]

    def _freeze_shape_prior_translator(self):
        if not self._shape_prior_translator_frozen:
            for module in self._shape_prior_translator_modules():
                module.requires_grad_(False)
            self._shape_prior_translator_frozen = True
            self.msg_mgr.log_info(
                "[ShapePriorPretrain] Frozen Shape_Branch1/Shape_Head/Shape_Branch2 "
                f"at iteration {getattr(self, 'iteration', 0)}."
            )
        for module in self._shape_prior_translator_modules():
            module.eval()

    def _run_shape_prior_with_reconstruction(self, grouped_intermediates, return_shape_feat):
        shape_feats = []
        shape_probs = []
        reconstruction_losses = []
        for i, feat in enumerate(grouped_intermediates):
            shape_latent = self.Shape_Branch1_list[i](feat)
            shape_probs.append(self.Shape_Head_list[i](shape_latent))

            branch2_input = shape_latent.detach() if self.shape_prior_detach_bottleneck else shape_latent
            reconstructed = self.Shape_Branch2_list[i](branch2_input)
            reconstructed = self._shape_reconstruct_norm(reconstructed)

            target = self._shape_reconstruct_norm(feat.detach())
            reconstruction_losses.append(F.mse_loss(reconstructed.float(), target.float()))

            if return_shape_feat:
                shape_feats.append(self.Shape_HumanSpace_Conv[i](reconstructed))

        shape_prob = torch.stack(shape_probs, dim=0).mean(dim=0)
        reconstruction_loss = torch.stack(reconstruction_losses).mean()
        if return_shape_feat:
            return torch.cat(shape_feats, dim=1), shape_prob, reconstruction_loss
        return None, shape_prob, reconstruction_loss

    def _is_shape_prior_pretraining(self):
        return self.training and self.iteration < self.shape_prior_pretrain_iters

    def _forward_shape_prior_pretrain(self, inputs):
        timing_info = {
            "model_hflip": 0.0,
            "model_rgb_preprocess": 0.0,
            "model_dino": 0.0,
            "model_sam_unpack": 0.0,
            "model_project_mask": 0.0,
            "model_shape_prior": 0.0,
            "model_misc": 0.0,
        }

        ipts, _labs, _, _, _seqL = inputs
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

        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = self.image_size // 7, self.image_size // 14
        shape_pred_masks = []
        shape_target_masks = []
        reconstruction_losses = []
        last_rgb_img = None
        last_generated_mask = None
        last_human_mask = None
        last_shape_prob = None

        seq_start = 0
        for rgb_chunk, chunk_len in zip(rgb_chunks, chunk_lengths):
            seq_end = seq_start + chunk_len
            sam_chunk = [seq[seq_start:seq_end] for seq in sam_decoder]
            seq_start = seq_end

            n, s, _c, _h, _w = rgb_chunk.size()
            flat_sam_frames = [frame for seq in sam_chunk for frame in seq]
            if len(flat_sam_frames) != n * s:
                raise RuntimeError(f"SAM frame count mismatch: expected {n * s}, got {len(flat_sam_frames)}")

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
                human_mask = self.preprocess(generated_mask, self.sils_size)
                timing_info["model_project_mask"] += self._perf_now(rgb.device) - stage_start

            grouped_intermediates = [
                torch.cat(intermediates[i : i + self.group_layer_num], dim=1).contiguous()
                for i in range(0, self.total_layer_num, self.group_layer_num)
            ]

            stage_start = self._perf_now(rgb.device)
            _, shape_prob, reconstruction_loss = self._run_shape_prior_with_reconstruction(
                grouped_intermediates, return_shape_feat=False
            )
            shape_pred_masks.append(shape_prob)
            shape_target_masks.append(human_mask.detach().float())
            reconstruction_losses.append(reconstruction_loss)
            timing_info["model_shape_prior"] += self._perf_now(rgb.device) - stage_start

            last_rgb_img = rgb_img
            last_generated_mask = generated_mask
            last_human_mask = human_mask
            last_shape_prob = shape_prob

        shape_pred = torch.cat(shape_pred_masks, dim=0).clamp(1.0e-6, 1.0 - 1.0e-6)
        shape_target = torch.cat(shape_target_masks, dim=0)
        reconstruction_loss = torch.stack(reconstruction_losses).mean()

        model_end = self._perf_now(rgb.device)
        timed_sum = sum(timing_info.values())
        timing_info["model_misc"] = max(model_end - model_start - timed_sum, 0.0)

        training_feat = {}
        if self.shape_prior_use_silhouette_loss:
            training_feat["shape_silhouette"] = {"logits": shape_pred, "labels": shape_target}
        if self.shape_prior_reconstruction_weight > 0:
            training_feat["shape_reconstruction_loss"] = (
                reconstruction_loss * self.shape_prior_reconstruction_weight
            )

        return {
            "training_feat": training_feat,
            "visual_summary": {
                "image/rgb_img": last_rgb_img[:5].float(),
                "image/generated_3d_mask_lowres": last_generated_mask[:5].float(),
                "image/generated_3d_mask_interpolated": last_human_mask[:5].float(),
                "image/shape_prior_prob": last_shape_prob[:5].float(),
                "scalar/shape_prior_stage": torch.tensor(0.0, device=rgb.device),
            },
            "inference_feat": {},
            "timing_info": timing_info,
        }

    def forward(self, inputs):
        if self._is_shape_prior_pretraining():
            return self._forward_shape_prior_pretrain(inputs)

        if self.shape_prior_freeze_after_pretrain:
            self._freeze_shape_prior_translator()
        if self.shape_prior_disable_sil_loss_after_pretrain:
            self.shape_prior_use_silhouette_loss = False

        retval = super().forward(inputs)
        if self.training:
            retval["visual_summary"]["scalar/shape_prior_stage"] = torch.tensor(
                1.0, device=inputs[0][0].device
            )
        return retval
