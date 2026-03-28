import copy
import os
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .BigGait_utils.BigGait_GaitBase import Baseline_AnchorMasked_ShareTime_2B
from .BiggerGait_SAM_3D_Body_projection_mask_OT_based import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_OT_Anchor_Pooling_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share
):
    def build_network(self, model_cfg):
        self.freeze_pre_test2 = model_cfg.get("freeze_pre_test2", True)
        self.reset_iteration_after_restore = model_cfg.get(
            "reset_iteration_after_restore", True
        )
        self.anchor_pt_path = model_cfg["anchor_pt_path"]
        self.anchor_depth_tol = model_cfg.get("anchor_depth_tol", 0.02)
        self.anchor_sampling_mode = model_cfg.get("anchor_sampling_mode", "bilinear")
        self.anchor_visibility_mode = model_cfg.get("anchor_visibility_mode", "nearest")
        if self.anchor_visibility_mode not in {"nearest", "all"}:
            raise ValueError(
                f"Unsupported anchor_visibility_mode: {self.anchor_visibility_mode}. "
                "Expected 'nearest' or 'all'."
            )

        super().build_network(model_cfg)
        self._load_anchor_indices(model_cfg)
        self.Gait_Nets = nn.ModuleList(
            [
                Baseline_AnchorMasked_ShareTime_2B(copy.deepcopy(model_cfg))
                for _ in range(self.num_branches)
            ]
        )
        if self.freeze_pre_test2:
            self._freeze_before_test2()

    def _load_anchor_indices(self, model_cfg):
        if not os.path.exists(self.anchor_pt_path):
            raise FileNotFoundError(f"Cannot find anchor file: {self.anchor_pt_path}")
        anchor_data = torch.load(self.anchor_pt_path, map_location="cpu")

        if isinstance(anchor_data, torch.Tensor):
            if anchor_data.dim() != 2 or anchor_data.shape[1] != 2:
                raise ValueError(
                    f"Tensor anchor file should be [num_anchors, 2], got {tuple(anchor_data.shape)}"
                )
            anchor_patch_labels = anchor_data[:, 0].long()
            sampled_vertex_indices = anchor_data[:, 1].long()
        elif isinstance(anchor_data, dict) and "sampled_vertex_indices" in anchor_data:
            sampled_vertex_indices = anchor_data["sampled_vertex_indices"].long()
            anchor_patch_labels = torch.arange(
                sampled_vertex_indices.numel(), dtype=torch.long
            )
        else:
            raise KeyError(
                f"Unsupported anchor file format in {self.anchor_pt_path}. "
                "Expected {'sampled_vertex_indices': ...} or a [num_anchors, 2] tensor."
            )

        num_patches = int(anchor_patch_labels.max().item()) + 1
        expected_parts = model_cfg["SeparateFCs"]["parts_num"]
        if num_patches != expected_parts:
            raise ValueError(
                f"Loaded {num_patches} anchor patches from {self.anchor_pt_path}, "
                f"but SeparateFCs.parts_num is {expected_parts}"
            )

        self.register_buffer(
            "sampled_vertex_indices", sampled_vertex_indices, persistent=False
        )
        self.register_buffer(
            "anchor_patch_labels", anchor_patch_labels, persistent=False
        )
        self.num_anchors = sampled_vertex_indices.numel()
        self.num_anchor_patches = num_patches
        self.msg_mgr.log_info(
            f"[Anchor] Loaded {self.num_anchors} sampled vertices grouped into "
            f"{self.num_anchor_patches} patches from {self.anchor_pt_path}"
        )

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_before_test2(self):
        self._freeze_module(self.HumanSpace_Conv)
        self._freeze_module(self.Mask_Branch)
        for gait_net in self.Gait_Nets:
            self._freeze_module(gait_net.temb_proj)
            for single_net in (gait_net.Gait_Net_1, gait_net.Gait_Net_2):
                self._freeze_module(single_net.pre_rgb)
                self._freeze_module(single_net.post_backbone)

    def _set_frozen_modules_eval(self):
        self.HumanSpace_Conv.eval()
        self.Mask_Branch.eval()
        for gait_net in self.Gait_Nets:
            gait_net.temb_proj.eval()
            gait_net.Gait_Net_1.pre_rgb.eval()
            gait_net.Gait_Net_1.post_backbone.eval()
            gait_net.Gait_Net_2.pre_rgb.eval()
            gait_net.Gait_Net_2.post_backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "SAM_Engine"):
            self.SAM_Engine.eval()
        if mode and getattr(self, "freeze_pre_test2", False):
            self._set_frozen_modules_eval()
        return self

    def _load_ckpt(self, save_name):
        checkpoint = torch.load(save_name, map_location=torch.device("cuda", self.device))
        raw_state_dict = checkpoint["model"]
        current_state_dict = self.state_dict()

        filtered_state_dict = {}
        skipped_keys = []
        for key, value in raw_state_dict.items():
            if key not in current_state_dict:
                skipped_keys.append((key, "missing_in_current"))
                continue
            if ".FCs." in key or ".BNNecks." in key:
                skipped_keys.append((key, "skip_head_reinit"))
                continue
            if current_state_dict[key].shape != value.shape:
                skipped_keys.append((key, "shape_mismatch"))
                continue
            filtered_state_dict[key] = value

        incompatible = self.load_state_dict(filtered_state_dict, strict=False)
        self.msg_mgr.log_info(
            f"[Restore] Loaded {len(filtered_state_dict)} tensors from {save_name}"
        )
        if skipped_keys:
            self.msg_mgr.log_info(
                f"[Restore] Skipped {len(skipped_keys)} tensors "
                f"(new head / shape mismatch / absent in current model)."
            )
        if incompatible.missing_keys:
            self.msg_mgr.log_info(
                f"[Restore] Missing keys after filtered load: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            self.msg_mgr.log_info(
                f"[Restore] Unexpected keys after filtered load: {incompatible.unexpected_keys}"
            )

        if self.training:
            self.msg_mgr.log_warning(
                f"Restore NO Optimizer/Scheduler from {save_name} for OT-anchor finetuning."
            )
            if self.reset_iteration_after_restore:
                self.iteration = 0
                self.msg_mgr.log_info(
                    "[Restore] Reset iteration to 0 for head-only finetuning."
                )

    def extract_anchor_features(
        self,
        feat_map,
        anchor_vertices,
        cam_t,
        cam_int,
        dense_depth_map,
        h_feat,
        w_feat,
        target_h,
        target_w,
        return_debug=False,
    ):
        del dense_depth_map
        bsz, channels, _, _ = feat_map.shape
        device = feat_map.device

        v_cam = anchor_vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx = cam_int[:, 0, 0].unsqueeze(1)
        fy = cam_int[:, 1, 1].unsqueeze(1)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy

        u_cont = (u / target_w * w_feat).clamp(0, w_feat - 1e-4)
        v_cont = (v / target_h * h_feat).clamp(0, h_feat - 1e-4)
        u_feat = u_cont.long()
        v_feat = v_cont.long()
        flat_pixel_indices = v_feat * w_feat + u_feat

        front_valid = torch.ones_like(z, dtype=torch.bool)
        if self.anchor_visibility_mode == "all":
            is_visible = front_valid
        else:
            sparse_depth_flat = torch.full((bsz, h_feat * w_feat), 1e6, device=device)
            masked_z = torch.where(front_valid, z, torch.full_like(z, 1e6))
            sparse_depth_flat.scatter_reduce_(
                1, flat_pixel_indices, masked_z, reduce="amin", include_self=False
            )
            sparse_min_depth = torch.gather(sparse_depth_flat, 1, flat_pixel_indices)
            is_visible = front_valid & (z <= (sparse_min_depth + 1e-4))

        if self.anchor_sampling_mode == "bilinear":
            if w_feat > 1:
                grid_x = (u_cont / (w_feat - 1)) * 2.0 - 1.0
            else:
                grid_x = torch.zeros_like(u_cont)
            if h_feat > 1:
                grid_y = (v_cont / (h_feat - 1)) * 2.0 - 1.0
            else:
                grid_y = torch.zeros_like(v_cont)
            sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)
            sampled_feat = F.grid_sample(
                feat_map,
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            anchor_feats = rearrange(sampled_feat, "b c 1 k -> b k c").contiguous()
            anchor_valid = is_visible
            anchor_feats = anchor_feats * anchor_valid.unsqueeze(-1).to(anchor_feats.dtype)
        else:
            feat_flat = rearrange(feat_map, "b c h w -> b (h w) c").contiguous()
            anchor_feats = feat_map.new_zeros(bsz, self.num_anchors, channels)
            anchor_valid = torch.zeros(
                bsz, self.num_anchors, dtype=torch.bool, device=device
            )
            for b_idx in range(bsz):
                visible_idx = is_visible[b_idx].nonzero(as_tuple=False).squeeze(1)
                if visible_idx.numel() == 0:
                    continue
                pixel_idx = flat_pixel_indices[b_idx, visible_idx]
                anchor_feats[b_idx, visible_idx] = feat_flat[b_idx, pixel_idx]
                anchor_valid[b_idx, visible_idx] = True

        if return_debug:
            return anchor_feats, anchor_valid, {
                "u_cont": u_cont,
                "v_cont": v_cont,
                "is_visible": is_visible,
            }
        return anchor_feats, anchor_valid

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        branch_pca_after_cnn = [None] * self.num_branches
        branch_layer2_norm = [None] * self.num_branches
        all_anchor_feats = [[] for _ in range(self.num_branches)]
        all_anchor_valid = [[] for _ in range(self.num_branches)]
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

            cam_int_tgt, cam_t_tgt = self.build_target_camera(
                curr_bs, rgb.device, target_h, target_w
            )
            branch_warped_feats = []
            branch_target_depth = [None] * self.num_branches
            branch_rotated_verts = [None] * self.num_branches
            for b_idx, branch_cfg in enumerate(self.branch_configs):
                branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
                rotated_verts, _, _ = self.rotate_branch_geometry(
                    branch_geo["verts"],
                    branch_geo["keypoints"],
                    global_rot,
                    branch_geo["yaw"],
                    branch_geo["apply_global_rot_alignment"],
                )
                warp_feat, _, tgt_depth_map = self.warp_features_with_ot(
                    human_feat,
                    human_mask.float(),
                    pred_verts,
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
                branch_rotated_verts[b_idx] = rotated_verts

            debug_test_1 = should_log_pca_vis and (chunk_idx == num_rgb_chunks - 1)
            for b_idx, warp_feat in enumerate(branch_warped_feats):
                warp_feat_5d = rearrange(
                    warp_feat, "(n s) c h w -> n c s h w", n=n, s=s
                ).contiguous()
                if debug_test_1:
                    outs, gait_debug = self.Gait_Nets[b_idx].test_1(
                        warp_feat_5d, return_debug=True
                    )
                    layer2_feat = rearrange(
                        gait_debug["layer2_feat"], "n c s h w -> (n s) c h w"
                    ).contiguous()
                    branch_layer2_norm[b_idx] = self._build_feature_norm_on_depth_vis_batch(
                        branch_target_depth[b_idx][:5], layer2_feat[:5]
                    )
                    branch_pca_after_cnn[b_idx] = self._build_pca_vis_batch(
                        rearrange(outs, "n c s h w -> (n s) c h w").contiguous()[:5]
                    )
                elif self.training:
                    outs = torch.utils.checkpoint.checkpoint(
                        self.Gait_Nets[b_idx].test_1,
                        warp_feat_5d,
                        use_reentrant=False,
                    )
                else:
                    outs = self.Gait_Nets[b_idx].test_1(warp_feat_5d)

                out_h, out_w = outs.shape[-2:]
                anchor_feat_map = rearrange(outs, "n c s h w -> (n s) c h w").contiguous()
                anchor_vertices = branch_rotated_verts[b_idx][:, self.sampled_vertex_indices, :]
                anchor_feats_flat, anchor_valid_flat = self.extract_anchor_features(
                    anchor_feat_map,
                    anchor_vertices,
                    cam_t_tgt,
                    cam_int_tgt,
                    branch_target_depth[b_idx],
                    out_h,
                    out_w,
                    target_h,
                    target_w,
                )

                anchor_feats = rearrange(
                    anchor_feats_flat, "(n s) k c -> n c s k", n=n, s=s
                ).contiguous()
                anchor_valid = rearrange(
                    anchor_valid_flat, "(n s) k -> n s k", n=n, s=s
                ).contiguous()
                all_anchor_feats[b_idx].append(anchor_feats)
                all_anchor_valid[b_idx].append(anchor_valid)

        embed_grouped = [[] for _ in range(self.num_FPN)]
        log_grouped = [[] for _ in range(self.num_FPN)]

        for b_idx in range(self.num_branches):
            branch_anchor_feat = torch.cat(all_anchor_feats[b_idx], dim=2)
            branch_anchor_valid = torch.cat(all_anchor_valid[b_idx], dim=1)
            e_list, l_list = self.Gait_Nets[b_idx].test_2(
                branch_anchor_feat, seqL, branch_anchor_valid, self.anchor_patch_labels
            )
            for i in range(self.num_FPN):
                embed_grouped[i].append(e_list[i])
                log_grouped[i].append(l_list[i])

        embed_list = [torch.cat(feats, dim=-1) for feats in embed_grouped]
        log_list = [torch.cat(logits, dim=-1) for logits in log_grouped]
        pca_after_cnn_summary = self._stack_branch_vis(branch_pca_after_cnn)
        cnn_layer2_norm_summary = self._stack_branch_vis(branch_layer2_norm)

        if self.training:
            retval = {
                "training_feat": {
                    "triplet": {
                        "embeddings": torch.cat(embed_list, dim=-1),
                        "labels": labs,
                    },
                    "softmax": {
                        "logits": torch.cat(log_list, dim=-1),
                        "labels": labs,
                    },
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
                retval["visual_summary"]["image/pca_after_cnn"] = (
                    pca_after_cnn_summary.float()
                )
            if cnn_layer2_norm_summary is not None:
                retval["visual_summary"]["image/cnn_layer2_l2norm"] = (
                    cnn_layer2_norm_summary.float()
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
