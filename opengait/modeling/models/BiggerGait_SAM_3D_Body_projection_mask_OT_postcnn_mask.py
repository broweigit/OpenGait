from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .BiggerGait_SAM_3D_Body_projection_mask_OT_based import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_OT_PostCNN_Mask_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share
):
    @staticmethod
    def _mask_to_vis(mask, max_frames=5):
        mask = mask.detach().float()
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.repeat(1, 3, 1, 1)
        return mask[:max_frames].clamp(0, 1)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        branch_pca_after_cnn = [None] * self.num_branches
        branch_pca_after_ot = [None] * self.num_branches
        branch_layer2_norm = [None] * self.num_branches
        branch_target_depth = [None] * self.num_branches
        branch_postot_target_mask = [None] * self.num_branches
        all_outs = [[] for _ in range(self.num_branches)]
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        num_rgb_chunks = len(rgb_chunks)
        for chunk_idx, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
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
                sam_emb = sam_emb.transpose(1, 2).reshape(curr_bs, -1, h_feat, w_feat)

                dummy_batch = self._prepare_dummy_batch(sam_emb, target_h, target_w)
                self.SAM_Engine._batch_size = curr_bs
                self.SAM_Engine._max_num_person = 1
                self.SAM_Engine.body_batch_idx = torch.arange(curr_bs, device=rgb.device)
                self.SAM_Engine.hand_batch_idx = []
                cond_info = torch.zeros(curr_bs, 3, device=rgb.device)
                cond_info[:, 2] = 1.1
                dummy_kp = torch.zeros(curr_bs, 1, 3, device=rgb.device)
                dummy_kp[..., -1] = -2

                with torch.amp.autocast(enabled=False, device_type="cuda"):
                    _, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_emb,
                        keypoints=dummy_kp,
                        condition_info=cond_info,
                        batch=dummy_batch,
                    )

                pose_out = pose_outs[-1]
                self._apose_cache = {}
                pred_verts = pose_out["pred_vertices"]
                pred_cam_t = pose_out["pred_cam_t"]
                global_rot = pose_out["global_rot"]
                cam_int_src = dummy_batch["cam_int"]
                _, src_depth_map = self.get_source_vertex_index_map(
                    pred_verts, pred_cam_t, cam_int_src, h_feat, w_feat, target_h, target_w
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
                sub_app = partial(nn.LayerNorm, eps=1e-6)(
                    curr_dim, elementwise_affine=False
                )(sub_app)
                sub_app = rearrange(sub_app, "b (h w) c -> b c h w", h=h_feat).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_feat = torch.concat(processed_feat_list, dim=1)
            human_mask = self.preprocess(
                generated_mask, self.sils_size * 2, self.sils_size
            ).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            human_feat_5d = rearrange(
                human_feat, "(n s) c h w -> n c s h w", n=n, s=s
            ).contiguous()

            cam_int_tgt, cam_t_tgt = self.build_target_camera(
                curr_bs, rgb.device, target_h, target_w
            )
            debug_test_1 = should_log_pca_vis and (chunk_idx == num_rgb_chunks - 1)
            for b_idx, branch_cfg in enumerate(self.branch_configs):
                branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
                if debug_test_1:
                    outs, gait_debug = self.Gait_Nets[b_idx].test_1(
                        human_feat_5d, return_debug=True
                    )
                    layer2_feat = rearrange(
                        gait_debug["layer2_feat"], "n c s h w -> (n s) c h w"
                    ).contiguous()
                    branch_layer2_norm[b_idx] = self._build_feature_norm_vis_batch(
                        rgb_img[:5], layer2_feat[:5]
                    )
                    branch_pca_after_cnn[b_idx] = self._build_pca_vis_batch(
                        rearrange(outs, "n c s h w -> (n s) c h w").contiguous()[:5]
                    )
                elif self.training:
                    outs = torch.utils.checkpoint.checkpoint(
                        self.Gait_Nets[b_idx].test_1,
                        human_feat_5d,
                        use_reentrant=False,
                    )
                else:
                    outs = self.Gait_Nets[b_idx].test_1(human_feat_5d)

                post_cnn_feat = rearrange(
                    outs, "n c s h w -> (n s) c h w"
                ).contiguous()
                out_h, out_w = post_cnn_feat.shape[-2:]
                src_mask_postcnn = F.interpolate(
                    generated_mask, size=(out_h, out_w), mode="nearest"
                ).detach().clone()

                warped_post_cnn_feat, target_valid_mask, tgt_depth_map = self.warp_features_with_ot(
                    post_cnn_feat,
                    src_mask_postcnn.float(),
                    pred_verts,
                    branch_geo["verts"],
                    branch_geo["keypoints"],
                    pred_cam_t,
                    global_rot,
                    cam_int_src,
                    cam_int_tgt,
                    cam_t_tgt,
                    out_h,
                    out_w,
                    target_h,
                    target_w,
                    branch_geo["yaw"],
                    branch_geo["apply_global_rot_alignment"],
                )
                branch_target_depth[b_idx] = tgt_depth_map
                outs = rearrange(
                    warped_post_cnn_feat, "(n s) c h w -> n c s h w", n=n, s=s
                ).contiguous()

                if debug_test_1:
                    branch_pca_after_ot[b_idx] = self._build_pca_vis_batch(
                        warped_post_cnn_feat[:5]
                    )
                    branch_postot_target_mask[b_idx] = self._mask_to_vis(
                        target_valid_mask
                    )
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
        pca_after_cnn_summary = self._stack_branch_vis(branch_pca_after_cnn)
        pca_after_ot_summary = self._stack_branch_vis(branch_pca_after_ot)
        cnn_layer2_norm_summary = self._stack_branch_vis(branch_layer2_norm)
        post_ot_target_mask_summary = self._stack_branch_vis(branch_postot_target_mask)

        if self.training:
            retval = {
                "training_feat": {
                    "triplet": {"embeddings": torch.cat(embed_list, dim=-1), "labels": labs},
                    "softmax": {"logits": torch.cat(log_list, dim=-1), "labels": labs},
                },
                "visual_summary": {
                    "image/rgb_img": rgb_img.view(n * s, c, h, w)[:5].float(),
                    "image/generated_3d_mask_lowres": generated_mask.view(
                        n * s, 1, h_feat, w_feat
                    )[:5].float(),
                    "image/generated_3d_mask_interpolated": human_mask.view(
                        n * s, 1, self.sils_size * 2, self.sils_size
                    )[:5].float(),
                },
                "inference_feat": {
                    "embeddings": torch.cat(embed_list, dim=-1),
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
            }
            if pca_after_cnn_summary is not None:
                retval["visual_summary"]["image/pca_after_cnn"] = pca_after_cnn_summary.float()
            if pca_after_ot_summary is not None:
                retval["visual_summary"]["image/pca_after_ot"] = pca_after_ot_summary.float()
            if cnn_layer2_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer2_l2norm"] = (
                    cnn_layer2_norm_summary.float()
                )
            if post_ot_target_mask_summary is not None:
                retval["visual_summary"]["image/post_ot_target_mask"] = (
                    post_ot_target_mask_summary.float()
                )
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
