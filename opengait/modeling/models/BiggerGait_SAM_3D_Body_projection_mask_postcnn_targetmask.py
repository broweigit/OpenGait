from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .BiggerGait_SAM_3D_Body_projection_mask_based import (
    BiggerGait__SAM3DBody__Projection_Mask_Based_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_PostCNN_TargetMask_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_Based_Gaitbase_Share
):
    def _apply_post_cnn_target_mask(self, outs, target_valid_mask):
        if target_valid_mask.dim() == 3:
            target_valid_mask = target_valid_mask.unsqueeze(1)
        elif target_valid_mask.dim() != 4:
            raise ValueError(
                f"Expected target_valid_mask as [ns, h, w] or [ns, 1, h, w], got {target_valid_mask.shape}"
            )

        n, _, s, h, w = outs.shape
        resized_mask = F.interpolate(
            target_valid_mask.float(), size=(h, w), mode="nearest"
        )
        resized_mask = (resized_mask > 0.5).float()
        resized_mask = rearrange(
            resized_mask, "(n s) c h w -> n c s h w", n=n, s=s
        ).contiguous()
        masked_outs = outs * resized_mask.to(dtype=outs.dtype)
        return masked_outs, resized_mask

    @staticmethod
    def _masked_hpp(x, mask, bin_num):
        n, c = x.size()[:2]
        features = []
        mask = mask.float()
        for b in bin_num:
            feat_bin = x.view(n, c, b, -1)
            mask_bin = mask.view(n, 1, b, -1)
            valid_count = mask_bin.sum(-1).clamp_min(1.0)
            feat_mean = (feat_bin * mask_bin).sum(-1) / valid_count

            valid_part = mask_bin.sum(-1) > 0
            feat_max = feat_bin.masked_fill(
                ~valid_part.unsqueeze(-1).expand(-1, -1, -1, feat_bin.size(-1)),
                -100.0,
            ).max(-1)[0]
            feat_max = torch.where(
                valid_part.expand(-1, c, -1), feat_max, torch.zeros_like(feat_max)
            )
            features.append(feat_mean + feat_max)
        return torch.cat(features, dim=-1)

    def _masked_single_test_2(self, gait_single, outs, mask, seqL):
        outs = gait_single.TP(outs, seqL, options={"dim": 2})[0]
        mask = gait_single.TP(mask, seqL, options={"dim": 2})[0]
        if gait_single.vertical_pooling:
            outs = outs.transpose(2, 3).contiguous()
            mask = mask.transpose(2, 3).contiguous()
        outs = self._masked_hpp(outs, mask, gait_single.HPP.bin_num)
        embed_1 = gait_single.FCs(outs)
        _, logits = gait_single.BNNecks(embed_1)
        return embed_1, logits

    def _masked_test_2(self, gait_net, x, mask, seqL):
        x_list = torch.chunk(x, gait_net.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(gait_net.num_FPN):
            embed_1, logits = self._masked_single_test_2(
                gait_net.Gait_List[i], x_list[i], mask, seqL
            )
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list

    def _stack_fpn_vis(self, fpn_vis_list):
        valid_vis = [vis for vis in fpn_vis_list if vis is not None]
        if not valid_vis:
            return None
        num_frames = min(vis.shape[0] for vis in valid_vis)
        return torch.cat([vis[:num_frames] for vis in valid_vis], dim=3)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        layer1_norm_summary = None
        pca_before_cnn_summary = None
        pca_after_cnn_summary = None
        layer2_norm_summary = None
        layer3_norm_summary = None
        layer4_norm_summary = None
        post_cnn_mask_summary = None
        all_outs = []
        all_masks = []
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

                pred_verts = pose_outs[-1]["pred_vertices"]
                pred_cam_t = pose_outs[-1]["pred_cam_t"]
                cam_int = dummy_batch["cam_int"]
                generated_mask = self.project_vertices_to_mask(
                    pred_verts, pred_cam_t, cam_int, h_feat, w_feat, target_h, target_w
                )

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
                sub_app = rearrange(
                    sub_app, "b (h w) c -> b c h w", h=h_feat
                ).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_feat = torch.concat(processed_feat_list, dim=1)
            human_mask = self.preprocess(
                generated_mask, self.sils_size * 2, self.sils_size
            ).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            human_feat_5d = rearrange(
                human_feat.view(n, s, -1, self.sils_size * 2, self.sils_size),
                "n s c h w -> n c s h w",
            ).contiguous()

            debug_test_1 = should_log_pca_vis and (chunk_idx == num_rgb_chunks - 1)
            if debug_test_1:
                pca_before_vis = []
                for in_chunk in torch.chunk(human_feat_5d, self.num_FPN, dim=1):
                    pca_before_vis.append(
                        self._build_pca_vis_batch(
                            rearrange(
                                in_chunk, "n c s h w -> (n s) c h w"
                            ).contiguous()[:5]
                        )
                    )
                pca_before_cnn_summary = self._stack_fpn_vis(pca_before_vis)

                outs, gait_debug = self.Gait_Net.test_1(human_feat_5d, return_debug=True)

                layer1_vis = []
                layer2_vis = []
                layer3_vis = []
                layer4_vis = []
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
                        self._build_feature_norm_vis_batch(rgb_img[:5], layer1_feat[:5])
                    )
                    layer2_vis.append(
                        self._build_feature_norm_vis_batch(rgb_img[:5], layer2_feat[:5])
                    )
                    layer3_vis.append(
                        self._build_feature_norm_vis_batch(rgb_img[:5], layer3_feat[:5])
                    )
                    layer4_vis.append(
                        self._build_feature_norm_vis_batch(rgb_img[:5], layer4_feat[:5])
                    )
                layer1_norm_summary = self._stack_fpn_vis(layer1_vis)
                layer2_norm_summary = self._stack_fpn_vis(layer2_vis)
                layer3_norm_summary = self._stack_fpn_vis(layer3_vis)
                layer4_norm_summary = self._stack_fpn_vis(layer4_vis)
            elif self.training:
                outs = torch.utils.checkpoint.checkpoint(
                    self.Gait_Net.test_1,
                    human_feat_5d,
                    use_reentrant=False,
                )
            else:
                outs = self.Gait_Net.test_1(human_feat_5d)

            outs, post_cnn_mask = self._apply_post_cnn_target_mask(outs, human_mask)
            if debug_test_1:
                pca_after_vis = []
                for out_chunk in torch.chunk(outs, self.num_FPN, dim=1):
                    pca_after_vis.append(
                        self._build_pca_vis_batch(
                            rearrange(
                                out_chunk, "n c s h w -> (n s) c h w"
                            ).contiguous()[:5]
                        )
                    )
                pca_after_cnn_summary = self._stack_fpn_vis(pca_after_vis)
                post_cnn_mask_summary = (
                    rearrange(post_cnn_mask, "n c s h w -> (n s) c h w")
                    .contiguous()[:5]
                    .repeat(1, 3, 1, 1)
                )

            all_outs.append(outs)
            all_masks.append(post_cnn_mask)

        embed_list, log_list = self._masked_test_2(
            self.Gait_Net,
            torch.cat(all_outs, dim=2),
            torch.cat(all_masks, dim=2),
            seqL,
        )

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
            if layer1_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer1_l2norm"] = (
                    layer1_norm_summary.float()
                )
            if pca_before_cnn_summary is not None:
                retval["visual_summary"]["image/pca_before_cnn"] = (
                    pca_before_cnn_summary.float()
                )
            if pca_after_cnn_summary is not None:
                retval["visual_summary"]["image/pca_after_cnn"] = (
                    pca_after_cnn_summary.float()
                )
            if layer2_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer2_l2norm"] = (
                    layer2_norm_summary.float()
                )
            if layer3_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer3_l2norm"] = (
                    layer3_norm_summary.float()
                )
            if layer4_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer4_l2norm"] = (
                    layer4_norm_summary.float()
                )
            if post_cnn_mask_summary is not None:
                retval["visual_summary"]["image/post_cnn_target_mask"] = (
                    post_cnn_mask_summary.float()
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
