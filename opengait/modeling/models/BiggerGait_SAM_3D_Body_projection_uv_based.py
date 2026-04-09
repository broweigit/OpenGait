import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

from .BiggerGait_SAM_3D_Body_projection_mask_based import (
    BiggerGait__SAM3DBody__Projection_Mask_Based_Gaitbase_Share,
)
from .BigGait_utils.save_img import pca_image


class BiggerGait__SAM3DBody__Projection_UV_Based_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_Based_Gaitbase_Share
):
    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        uv_cfg = model_cfg.get("uv_cfg", {})
        atlas_size = uv_cfg.get("atlas_size", [64, 64])
        if len(atlas_size) != 2:
            raise ValueError(f"uv_cfg.atlas_size must be [H, W], got {atlas_size}")
        self.uv_h = int(atlas_size[0])
        self.uv_w = int(atlas_size[1])
        self.uv_pool_rows = int(uv_cfg.get("pool_rows", 8))
        self.uv_pool_cols = int(uv_cfg.get("pool_cols", 4))
        expected_parts = self.uv_pool_rows * self.uv_pool_cols
        if expected_parts != self.Gait_Net.Gait_List[0].FCs.p:
            raise ValueError(
                f"UV pooling produces {expected_parts} parts, but SeparateFCs expects "
                f"{self.Gait_Net.Gait_List[0].FCs.p}."
            )

    def generate_mhr_apose_vertices(self, pose_out):
        required_keys = ["shape", "scale", "face", "global_rot", "hand", "body_pose"]
        if not all(key in pose_out for key in required_keys):
            verts = pose_out["pred_vertices"].float().clone()
            verts = verts - verts.mean(dim=1, keepdim=True)
            return verts

        device = pose_out["pred_vertices"].device
        batch_size = pose_out["pred_vertices"].shape[0]

        pred_shape = pose_out["shape"].float()
        pred_scale = pose_out["scale"].float()
        pred_face = pose_out["face"].float()

        zero_global_trans = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        zero_global_rot = torch.zeros_like(pose_out["global_rot"], dtype=torch.float32)
        zero_hand_pose = torch.zeros_like(pose_out["hand"], dtype=torch.float32)
        apose_body = torch.zeros_like(pose_out["body_pose"], dtype=torch.float32)

        angle_rad = math.radians(-20.0)
        apose_body[:, 25] = angle_rad
        apose_body[:, 35] = angle_rad

        with torch.no_grad(), torch.amp.autocast(enabled=False, device_type="cuda"):
            apose_outputs = self.SAM_Engine.head_pose.mhr_forward(
                global_trans=zero_global_trans,
                global_rot=zero_global_rot,
                body_pose_params=apose_body,
                hand_pose_params=zero_hand_pose,
                scale_params=pred_scale,
                shape_params=pred_shape,
                expr_params=pred_face,
                return_keypoints=True,
            )

        apose_verts = apose_outputs[0]
        apose_verts[..., [1, 2]] *= -1
        return apose_verts

    def build_cylindrical_pseudo_uv(self, canonical_verts):
        x, y, z = canonical_verts.unbind(-1)
        u = (torch.atan2(x, z) + math.pi) / (2.0 * math.pi)
        y_min = y.min(dim=1, keepdim=True)[0]
        y_max = y.max(dim=1, keepdim=True)[0]
        v = 1.0 - (y - y_min) / (y_max - y_min + 1e-6)
        return torch.stack([u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)], dim=-1)

    def get_source_vertex_index_map(
        self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w
    ):
        bsz, num_verts, _ = vertices.shape
        device = vertices.device

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx = cam_int[:, 0, 0].unsqueeze(1)
        fy = cam_int[:, 1, 1].unsqueeze(1)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy

        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)
        flat_pixel_indices = v_feat * w_feat + u_feat

        depth_map_flat = torch.full((bsz, h_feat * w_feat), 1e6, device=device)
        depth_map_flat.scatter_reduce_(
            1, flat_pixel_indices, z, reduce="amin", include_self=False
        )

        min_depth_per_vertex = torch.gather(depth_map_flat, 1, flat_pixel_indices)
        is_visible = z < (min_depth_per_vertex + 1e-4)

        index_map_flat = torch.full(
            (bsz, h_feat * w_feat), -1, dtype=torch.long, device=device
        )
        vertex_indices = torch.arange(num_verts, device=device).unsqueeze(0).expand(bsz, -1)

        mask_flat = is_visible.reshape(-1)
        batch_offsets = torch.arange(bsz, device=device).unsqueeze(1) * (h_feat * w_feat)
        global_pixel_indices = (flat_pixel_indices + batch_offsets).reshape(-1)

        valid_pixel_indices = global_pixel_indices[mask_flat]
        valid_vertex_indices = vertex_indices.reshape(-1)[mask_flat]

        index_map_global = index_map_flat.reshape(-1)
        index_map_global[valid_pixel_indices] = valid_vertex_indices

        return (
            index_map_global.reshape(bsz, h_feat, w_feat),
            depth_map_flat.reshape(bsz, 1, h_feat, w_feat),
        )

    def accumulate_pixel_features_to_vertices(self, feat_map, vertex_index_map, num_verts):
        bsz, channels, _, _ = feat_map.shape
        flat_feat = rearrange(feat_map, "b c h w -> b (h w) c")
        flat_idx = vertex_index_map.view(bsz, -1)
        valid_mask = flat_idx >= 0

        vertex_feat = feat_map.new_zeros((bsz * num_verts, channels))
        vertex_count = feat_map.new_zeros((bsz * num_verts, 1))

        if valid_mask.any():
            batch_offsets = (
                torch.arange(bsz, device=feat_map.device).unsqueeze(1) * num_verts
            )
            global_ids = (flat_idx + batch_offsets).reshape(-1)
            valid_flat = valid_mask.reshape(-1)
            global_ids = global_ids[valid_flat]
            valid_feat = flat_feat.reshape(-1, channels)[valid_flat]

            vertex_feat.index_add_(0, global_ids, valid_feat)
            vertex_count.index_add_(
                0, global_ids, vertex_count.new_ones((global_ids.numel(), 1))
            )

        vertex_feat = vertex_feat.view(bsz, num_verts, channels)
        vertex_count = vertex_count.view(bsz, num_verts, 1)
        vertex_feat = vertex_feat / vertex_count.clamp_min(1.0)
        vertex_valid = vertex_count.squeeze(-1) > 0
        return vertex_feat, vertex_valid

    def scatter_vertices_to_uv(self, vertex_feat, vertex_valid, uv_coords):
        bsz, num_verts, channels = vertex_feat.shape
        uv_x = (uv_coords[..., 0] * (self.uv_w - 1)).round().long().clamp(0, self.uv_w - 1)
        uv_y = (uv_coords[..., 1] * (self.uv_h - 1)).round().long().clamp(0, self.uv_h - 1)
        uv_idx = uv_y * self.uv_w + uv_x

        flat_uv_feat = vertex_feat.new_zeros((bsz * self.uv_h * self.uv_w, channels))
        flat_uv_count = vertex_feat.new_zeros((bsz * self.uv_h * self.uv_w, 1))

        if vertex_valid.any():
            batch_offsets = (
                torch.arange(bsz, device=vertex_feat.device).unsqueeze(1) * (self.uv_h * self.uv_w)
            )
            global_ids = (uv_idx + batch_offsets).reshape(-1)
            valid_flat = vertex_valid.reshape(-1)
            global_ids = global_ids[valid_flat]
            valid_feat = vertex_feat.reshape(-1, channels)[valid_flat]

            flat_uv_feat.index_add_(0, global_ids, valid_feat)
            flat_uv_count.index_add_(
                0, global_ids, flat_uv_count.new_ones((global_ids.numel(), 1))
            )

        flat_uv_feat = flat_uv_feat.view(bsz, self.uv_h * self.uv_w, channels)
        flat_uv_count = flat_uv_count.view(bsz, self.uv_h * self.uv_w, 1)
        uv_feat = flat_uv_feat / flat_uv_count.clamp_min(1.0)
        uv_valid = flat_uv_count > 0

        uv_feat = rearrange(uv_feat, "b (h w) c -> b c h w", h=self.uv_h, w=self.uv_w)
        uv_valid = rearrange(
            uv_valid.float(), "b (h w) c -> b c h w", h=self.uv_h, w=self.uv_w
        )
        return uv_feat, uv_valid

    def build_uv_atlas(self, human_feat, pose_out, pred_verts, pred_cam_t, cam_int, target_h, target_w):
        feat_h, feat_w = human_feat.shape[-2:]
        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int, feat_h, feat_w, target_h, target_w
        )
        vertex_feat, vertex_valid = self.accumulate_pixel_features_to_vertices(
            human_feat, src_idx_map, pred_verts.shape[1]
        )
        canonical_verts = self.generate_mhr_apose_vertices(pose_out)
        uv_coords = self.build_cylindrical_pseudo_uv(canonical_verts)
        return self.scatter_vertices_to_uv(vertex_feat, vertex_valid, uv_coords)

    def uv_grid_pool(self, x):
        n, c, h, w = x.shape
        if h % self.uv_pool_rows != 0 or w % self.uv_pool_cols != 0:
            raise ValueError(
                f"Feature map size {(h, w)} is not divisible by UV grid "
                f"{(self.uv_pool_rows, self.uv_pool_cols)}."
            )
        x = x.view(
            n,
            c,
            self.uv_pool_rows,
            h // self.uv_pool_rows,
            self.uv_pool_cols,
            w // self.uv_pool_cols,
        )
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        pooled_mean = x.mean(dim=(-1, -2))
        pooled_max = x.amax(dim=(-1, -2))
        pooled = pooled_mean + pooled_max
        return pooled.view(n, c, self.uv_pool_rows * self.uv_pool_cols)

    def uv_single_test_2(self, gait_single, outs, seqL):
        outs = gait_single.TP(outs, seqL, options={"dim": 2})[0]
        pooled = self.uv_grid_pool(outs)
        embed_1 = gait_single.FCs(pooled)
        _, logits = gait_single.BNNecks(embed_1)
        return embed_1, logits

    def uv_test_2(self, gait_net, x, seqL):
        x_list = torch.chunk(x, gait_net.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(gait_net.num_FPN):
            embed_1, logits = self.uv_single_test_2(gait_net.Gait_List[i], x_list[i], seqL)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list

    @staticmethod
    def _stack_fpn_vis(fpn_vis_list):
        valid_vis = [vis for vis in fpn_vis_list if vis is not None]
        if not valid_vis:
            return None
        num_frames = min(vis.shape[0] for vis in valid_vis)
        return torch.cat([vis[:num_frames] for vis in valid_vis], dim=3)

    @staticmethod
    def _tile_fpn_frame_vis(fpn_vis_list, max_frames=5):
        valid_vis = [vis for vis in fpn_vis_list if vis is not None]
        if not valid_vis:
            return None
        num_frames = min(max_frames, *(vis.shape[0] for vis in valid_vis))
        row_tiles = []
        for vis in valid_vis:
            row_tiles.append(torch.cat([vis[i] for i in range(num_frames)], dim=2))
        tiled = torch.cat(row_tiles, dim=1)
        return tiled.unsqueeze(0)

    def _build_uv_pca_vis_batch(self, feat_map, valid_mask=None, max_frames=5):
        feat_map = feat_map.detach().float().cpu()
        if valid_mask is not None:
            valid_mask = valid_mask.detach().float().cpu()
            num_frames = min(max_frames, feat_map.shape[0], valid_mask.shape[0])
        else:
            num_frames = min(max_frames, feat_map.shape[0])
        vis_frames = []

        for idx in range(num_frames):
            curr_feat = feat_map[idx]
            _, height, width = curr_feat.shape
            feat_np = rearrange(curr_feat.numpy(), "c h w -> 1 (h w) c")
            if valid_mask is not None:
                curr_mask = valid_mask[idx]
                if curr_mask.dim() == 3:
                    curr_mask = curr_mask[0]
                if curr_mask.shape != (height, width):
                    curr_mask = F.interpolate(
                        curr_mask.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)
                mask_np = (curr_mask.reshape(1, height * width).numpy() > 0.5).astype(np.uint8)
            else:
                mask_np = np.ones((1, height * width), dtype=np.uint8)
            if mask_np.sum() == 0:
                vis_frames.append(torch.zeros(3, height, width))
                continue
            pca_img = pca_image(
                data={"embeddings": feat_np, "h": height, "w": width},
                mask=mask_np,
                root=None,
                model_name=None,
                dataset=None,
                n_components=3,
                is_return=True,
            )[0, 0]
            vis_frames.append(torch.from_numpy(pca_img).float() / 255.0)

        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _build_feature_norm_on_uv_vis_batch(self, valid_masks, feat_map, max_frames=5):
        if valid_masks is not None:
            valid_masks = valid_masks.detach().float().cpu()
        feat_map = feat_map.detach().float().cpu()
        if valid_masks is not None:
            num_frames = min(max_frames, feat_map.shape[0], valid_masks.shape[0])
        else:
            num_frames = min(max_frames, feat_map.shape[0])
        vis_frames = []
        invalid_gray = 0.2
        valid_gray = 0.55

        for idx in range(num_frames):
            curr_feat = feat_map[idx]
            if valid_masks is not None:
                valid_mask = valid_masks[idx]
                if valid_mask.dim() == 3:
                    valid_mask = valid_mask[0]
                base_mask = F.interpolate(
                    valid_mask.unsqueeze(0).unsqueeze(0),
                    size=curr_feat.shape[-2:],
                    mode="nearest",
                ).squeeze(0).squeeze(0)
                base_frame = torch.full((3, curr_feat.shape[-2], curr_feat.shape[-1]), invalid_gray)
                base_frame[:, base_mask > 0.5] = valid_gray
            else:
                base_frame = torch.full((3, curr_feat.shape[-2], curr_feat.shape[-1]), valid_gray)

            norm_map = torch.linalg.vector_norm(curr_feat, ord=2, dim=0, keepdim=True)
            min_val = norm_map.min()
            max_val = norm_map.max()
            if (max_val - min_val) > 1e-6:
                norm_map = (norm_map - min_val) / (max_val - min_val)
            else:
                norm_map = torch.zeros_like(norm_map)

            x = norm_map[0].clamp(0, 1)
            heat_r = (1.5 - torch.abs(4.0 * x - 3.0)).clamp(0.0, 1.0)
            heat_g = (1.5 - torch.abs(4.0 * x - 2.0)).clamp(0.0, 1.0)
            heat_b = (1.5 - torch.abs(4.0 * x - 1.0)).clamp(0.0, 1.0)
            heat_map = torch.stack([heat_r, heat_g, heat_b], dim=0)
            alpha = 0.7 * norm_map.pow(0.85)
            vis_frame = base_frame * (1.0 - alpha) + heat_map * alpha
            vis_frames.append(vis_frame.clamp(0, 1))

        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        pca_uv_before_cnn_summary = None
        pca_uv_after_cnn_summary = None
        uv_valid_mask_summary = None
        uv_layer1_norm_summary = None
        uv_layer2_norm_summary = None
        uv_layer3_norm_summary = None
        uv_layer4_norm_summary = None
        all_outs = []
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
                pred_verts = pose_out["pred_vertices"]
                pred_cam_t = pose_out["pred_cam_t"]
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
                sub_app = rearrange(sub_app, "b (h w) c -> b c h w", h=h_feat).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_feat = torch.concat(processed_feat_list, dim=1)
            human_mask = self.preprocess(
                generated_mask, self.sils_size * 2, self.sils_size
            ).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)

            uv_feat, uv_valid_mask = self.build_uv_atlas(
                human_feat, pose_out, pred_verts, pred_cam_t, cam_int, target_h, target_w
            )
            uv_feat = uv_feat * uv_valid_mask.to(dtype=uv_feat.dtype)
            uv_feat_5d = rearrange(
                uv_feat.view(n, s, -1, self.uv_h, self.uv_w),
                "n s c h w -> n c s h w",
            ).contiguous()
            uv_valid_mask_5d = rearrange(
                uv_valid_mask.view(n, s, 1, self.uv_h, self.uv_w),
                "n s c h w -> n c s h w",
            ).contiguous()

            debug_test_1 = should_log_pca_vis and (chunk_idx == num_rgb_chunks - 1)
            test_1_outputs = self.Gait_Net.test_1(uv_feat_5d, return_debug=debug_test_1)
            if debug_test_1:
                outs, gait_debug = test_1_outputs
                pca_before_vis = []
                pca_after_vis = []
                layer1_vis = []
                layer2_vis = []
                layer3_vis = []
                layer4_vis = []
                uv_valid_flat = rearrange(
                    uv_valid_mask_5d, "n c s h w -> (n s) c h w"
                ).contiguous()

                for i in range(self.num_FPN):
                    in_chunk = torch.chunk(uv_feat_5d, self.num_FPN, dim=1)[i]
                    out_chunk = torch.chunk(outs, self.num_FPN, dim=1)[i]
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
                    pca_before_vis.append(
                        self._build_uv_pca_vis_batch(
                            rearrange(in_chunk, "n c s h w -> (n s) c h w").contiguous()[:5],
                            uv_valid_flat[:5],
                        )
                    )
                    pca_after_vis.append(
                        self._build_uv_pca_vis_batch(
                            rearrange(out_chunk, "n c s h w -> (n s) c h w").contiguous()[:5]
                        )
                    )
                    layer1_vis.append(
                        self._build_feature_norm_on_uv_vis_batch(None, layer1_feat[:5])
                    )
                    layer2_vis.append(
                        self._build_feature_norm_on_uv_vis_batch(None, layer2_feat[:5])
                    )
                    layer3_vis.append(
                        self._build_feature_norm_on_uv_vis_batch(None, layer3_feat[:5])
                    )
                    layer4_vis.append(
                        self._build_feature_norm_on_uv_vis_batch(None, layer4_feat[:5])
                    )

                pca_uv_before_cnn_summary = self._tile_fpn_frame_vis(pca_before_vis)
                pca_uv_after_cnn_summary = self._tile_fpn_frame_vis(pca_after_vis)
                uv_layer1_norm_summary = self._tile_fpn_frame_vis(layer1_vis)
                uv_layer2_norm_summary = self._tile_fpn_frame_vis(layer2_vis)
                uv_layer3_norm_summary = self._tile_fpn_frame_vis(layer3_vis)
                uv_layer4_norm_summary = self._tile_fpn_frame_vis(layer4_vis)
                uv_valid_mask_summary = self._tile_fpn_frame_vis(
                    [uv_valid_flat[:5].repeat(1, 3, 1, 1) for _ in range(self.num_FPN)]
                )
            else:
                outs = test_1_outputs

            all_outs.append(outs)

        embed_list, log_list = self.uv_test_2(
            self.Gait_Net, torch.cat(all_outs, dim=2), seqL
        )

        if self.training:
            retval = {
                "training_feat": {
                    "triplet": {"embeddings": torch.concat(embed_list, dim=-1), "labels": labs},
                    "softmax": {"logits": torch.concat(log_list, dim=-1), "labels": labs},
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
                    "embeddings": torch.concat(embed_list, dim=-1),
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
            }
            if uv_valid_mask_summary is not None:
                retval["visual_summary"]["image/uv_valid_mask"] = uv_valid_mask_summary.float()
            if pca_uv_before_cnn_summary is not None:
                retval["visual_summary"]["image/pca_uv_before_cnn"] = (
                    pca_uv_before_cnn_summary.float()
                )
            if pca_uv_after_cnn_summary is not None:
                retval["visual_summary"]["image/pca_uv_after_cnn"] = (
                    pca_uv_after_cnn_summary.float()
                )
            if uv_layer1_norm_summary is not None:
                retval["visual_summary"]["image/uv_cnn_layer1_l2norm"] = (
                    uv_layer1_norm_summary.float()
                )
            if uv_layer2_norm_summary is not None:
                retval["visual_summary"]["image/uv_cnn_layer2_l2norm"] = (
                    uv_layer2_norm_summary.float()
                )
            if uv_layer3_norm_summary is not None:
                retval["visual_summary"]["image/uv_cnn_layer3_l2norm"] = (
                    uv_layer3_norm_summary.float()
                )
            if uv_layer4_norm_summary is not None:
                retval["visual_summary"]["image/uv_cnn_layer4_l2norm"] = (
                    uv_layer4_norm_summary.float()
                )
        else:
            retval = {
                "training_feat": {},
                "visual_summary": {},
                "inference_feat": {
                    "embeddings": torch.concat(embed_list, dim=-1),
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                },
            }
        return retval
