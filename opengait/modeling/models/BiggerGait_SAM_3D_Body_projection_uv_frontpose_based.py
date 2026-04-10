import math
import os

import roma
import torch
from einops import rearrange
from PIL import Image, ImageDraw

from .BiggerGait_SAM_3D_Body_projection_uv_based import (
    BiggerGait__SAM3DBody__Projection_UV_Based_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_UV_FrontPose_Based_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_UV_Based_Gaitbase_Share
):
    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        front_view_cfg = model_cfg.get("front_view_cfg", {})
        self.front_view_yaw = float(front_view_cfg.get("yaw", 0.0))
        self.debug_uv_max_frames = int(front_view_cfg.get("debug_max_frames", 5))

    def build_front_view_vertices(self, pose_out):
        required_keys = ["pred_vertices", "pred_keypoints_3d", "global_rot"]
        if not all(key in pose_out for key in required_keys):
            verts = pose_out["pred_vertices"].float().clone()
            return verts - verts.mean(dim=1, keepdim=True)

        verts = pose_out["pred_vertices"].float()
        keypoints = pose_out["pred_keypoints_3d"].float()
        global_rot = pose_out["global_rot"].float()
        batch_size = verts.shape[0]
        device = verts.device

        midhip = (keypoints[:, 9] + keypoints[:, 10]) / 2.0
        centered_verts = verts - midhip.unsqueeze(1)

        cy = math.cos(math.radians(self.front_view_yaw))
        sy = math.sin(math.radians(self.front_view_yaw))
        r_yaw = torch.tensor(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 3).expand(batch_size, 3, 3)

        rot_fix = global_rot.clone()
        rot_fix[..., [0, 1, 2]] *= -1
        r_canon = roma.euler_to_rotmat("XYZ", rot_fix)
        r_comp = torch.matmul(r_canon.transpose(1, 2), r_yaw.transpose(1, 2))

        verts_tmp = centered_verts.clone()
        verts_tmp[..., [1, 2]] *= -1
        rotated_smpl = torch.bmm(verts_tmp, r_comp)
        rotated_cv = rotated_smpl.clone()
        rotated_cv[..., [1, 2]] *= -1
        return rotated_cv

    def build_uv_atlas(
        self,
        human_feat,
        pose_out,
        pred_verts,
        pred_cam_t,
        cam_int,
        target_h,
        target_w,
        src_idx_map=None,
        return_aux=False,
    ):
        feat_h, feat_w = human_feat.shape[-2:]
        if src_idx_map is None:
            src_idx_map, _ = self.get_source_vertex_index_map(
                pred_verts, pred_cam_t, cam_int, feat_h, feat_w, target_h, target_w
            )
        vertex_feat, vertex_valid, vertex_count = (
            self.accumulate_pixel_features_to_vertices_with_counts(
                human_feat, src_idx_map, pred_verts.shape[1]
            )
        )
        front_view_verts = self.build_front_view_vertices(pose_out)
        uv_coords = self.build_cylindrical_pseudo_uv(front_view_verts)
        uv_feat, uv_valid = self.scatter_vertices_to_uv(vertex_feat, vertex_valid, uv_coords)
        if not return_aux:
            return uv_feat, uv_valid
        return uv_feat, uv_valid, {
            "src_idx_map": src_idx_map,
            "vertex_feat": vertex_feat,
            "vertex_valid": vertex_valid,
            "vertex_count": vertex_count,
            "front_view_verts": front_view_verts,
            "uv_coords": uv_coords,
        }

    def _render_uv_reference_canvas(
        self, uv_coords, colors, valid_mask=None, canvas_hw=(768, 768)
    ):
        uv_coords = uv_coords.detach().float().cpu()
        colors = colors.detach().float().cpu()
        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool().cpu()
        else:
            valid_mask = torch.ones(uv_coords.shape[0], dtype=torch.bool)

        canvas_h, canvas_w = canvas_hw
        img = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
        draw = ImageDraw.Draw(img)

        grid_color = (55, 55, 55)
        hpp_bin_num = getattr(self.Gait_Net.Gait_List[0].HPP, "bin_num", [32])
        vis_rows = max(int(b) for b in hpp_bin_num) if len(hpp_bin_num) > 0 else 1
        for row in range(vis_rows + 1):
            y = int(round(row * (canvas_h - 1) / vis_rows))
            draw.line((0, y, canvas_w - 1, y), fill=grid_color, width=1)

        draw.rectangle((0, 0, canvas_w - 1, canvas_h - 1), outline=(180, 180, 180), width=2)

        for idx in range(uv_coords.shape[0]):
            if not bool(valid_mask[idx]):
                continue
            u, v = uv_coords[idx].tolist()
            x = int(round(u * (canvas_w - 1)))
            y = int(round(v * (canvas_h - 1)))
            color = tuple(int(255 * c) for c in colors[idx].tolist())
            radius = 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        return img

    def _save_uv_reference_png_sequence(
        self,
        uv_coords_seq,
        color_seq,
        save_path,
        valid_mask_seq=None,
        max_frames=5,
        canvas_hw=(768, 768),
    ):
        uv_coords_seq = uv_coords_seq.detach().float().cpu()
        color_seq = color_seq.detach().float().cpu()
        if valid_mask_seq is not None:
            valid_mask_seq = valid_mask_seq.detach().bool().cpu()

        num_frames = min(max_frames, uv_coords_seq.shape[0], color_seq.shape[0])
        if num_frames <= 0:
            return

        frame_imgs = []
        for idx in range(num_frames):
            frame_valid = None if valid_mask_seq is None else valid_mask_seq[idx]
            frame_imgs.append(
                self._render_uv_reference_canvas(
                    uv_coords_seq[idx],
                    color_seq[idx],
                    valid_mask=frame_valid,
                    canvas_hw=canvas_hw,
                )
            )

        canvas_h, canvas_w = canvas_hw
        tiled = Image.new("RGB", (canvas_w * num_frames, canvas_h), (10, 10, 10))
        for idx, frame_img in enumerate(frame_imgs):
            tiled.paste(frame_img, (idx * canvas_w, 0))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tiled.save(save_path)

    def _export_uv_debug_artifacts(
        self,
        label_value,
        first_frame_verts,
        seq0_front_verts,
        seq0_uv_coords,
        per_fpn_frame_vertex_feat,
        per_fpn_frame_vertex_valid,
    ):
        if not self.training or not torch.distributed.is_initialized():
            return
        if torch.distributed.get_rank() != 0:
            return

        save_root = self._get_uv_debug_export_root()
        os.makedirs(save_root, exist_ok=True)
        label_str = str(int(label_value)) if torch.is_tensor(label_value) else str(label_value)

        first_frame_verts = first_frame_verts.detach().float().cpu()
        seq0_front_verts = seq0_front_verts.detach().float().cpu()
        seq0_uv_coords = seq0_uv_coords.detach().float().cpu()

        geom_colors = self._height_angle_colors(first_frame_verts)
        geom_ply_path = os.path.join(
            save_root, f"seq0_label{label_str}_frame0_frontpose_geom_height_angle.ply"
        )
        self._save_ascii_ply(first_frame_verts, geom_colors, geom_ply_path)

        uv_color_seq = torch.stack(
            [self._height_angle_colors(frame_verts) for frame_verts in seq0_front_verts], dim=0
        )
        uv_png_path = os.path.join(
            save_root, f"seq0_label{label_str}_frontpose_geom_height_angle_uv_seq.png"
        )
        self._save_uv_reference_png_sequence(
            seq0_uv_coords,
            uv_color_seq,
            uv_png_path,
            max_frames=self.debug_uv_max_frames,
        )

        body_span = seq0_front_verts[..., 0].max() - seq0_front_verts[..., 0].min()
        spacing = max(float(body_span.item()) * 2.0, 0.8)
        pca_verts_all = []
        pca_colors_all = []

        for fpn_idx, (frame_feats, frame_valids) in enumerate(
            zip(per_fpn_frame_vertex_feat, per_fpn_frame_vertex_valid)
        ):
            frame_feats = frame_feats.detach().float().cpu()
            frame_valids = frame_valids.detach().bool().cpu()
            flat_valid = frame_valids.reshape(-1)
            if int(flat_valid.sum().item()) == 0:
                continue

            flat_feats = frame_feats.reshape(-1, frame_feats.shape[-1])[flat_valid]
            flat_verts = seq0_front_verts.reshape(-1, 3)[flat_valid]
            shifted_verts = flat_verts.clone()
            shifted_verts[:, 0] += fpn_idx * spacing
            pca_colors = self._feature_pca_colors(
                flat_feats, torch.ones(flat_feats.shape[0], dtype=torch.bool)
            )

            pca_verts_all.append(shifted_verts)
            pca_colors_all.append(pca_colors)

        if pca_verts_all:
            pca_ply_path = os.path.join(
                save_root, f"seq0_label{label_str}_fpn_feature_pca_multiframe.ply"
            )
            self._save_ascii_ply(
                torch.cat(pca_verts_all, dim=0),
                torch.cat(pca_colors_all, dim=0),
                pca_ply_path,
            )
            self.msg_mgr.log_info(f"[UV FrontPose] Saved temporary UV debug exports to {save_root}")
        else:
            self.msg_mgr.log_warning(
                f"[UV FrontPose] No valid FPN vertex features found for temporary export at {save_root}"
            )

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        should_export_uv_debug = self.debug_uv_export and self._should_log_visual_summary()
        pca_uv_before_cnn_summary = None
        pca_uv_after_cnn_summary = None
        uv_valid_mask_summary = None
        uv_layer1_norm_summary = None
        uv_layer2_norm_summary = None
        uv_layer3_norm_summary = None
        uv_layer4_norm_summary = None
        debug_seq0_front_verts = []
        debug_seq0_uv_coords = []
        debug_seq0_vertex_feat = [[] for _ in range(self.num_FPN)]
        debug_seq0_vertex_valid = [[] for _ in range(self.num_FPN)]
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
                sub_app = torch.nn.LayerNorm(curr_dim, eps=1e-6, elementwise_affine=False)(
                    sub_app
                )
                sub_app = rearrange(sub_app, "b (h w) c -> b c h w", h=h_feat).contiguous()
                reduced_feat = self.HumanSpace_Conv[i](sub_app)
                processed_feat_list.append(reduced_feat)

            human_mask = self.preprocess(
                generated_mask, self.sils_size * 2, self.sils_size
            ).detach().clone()
            human_mask_bool = (human_mask > 0.5).to(dtype=processed_feat_list[0].dtype)
            processed_feat_list = [feat * human_mask_bool for feat in processed_feat_list]
            human_feat = torch.concat(processed_feat_list, dim=1)

            src_idx_map, _ = self.get_source_vertex_index_map(
                pred_verts,
                pred_cam_t,
                cam_int,
                human_feat.shape[-2],
                human_feat.shape[-1],
                target_h,
                target_w,
            )
            if should_log_pca_vis or should_export_uv_debug:
                uv_feat, uv_valid_mask, uv_aux = self.build_uv_atlas(
                    human_feat,
                    pose_out,
                    pred_verts,
                    pred_cam_t,
                    cam_int,
                    target_h,
                    target_w,
                    src_idx_map=src_idx_map,
                    return_aux=True,
                )
            else:
                uv_feat, uv_valid_mask = self.build_uv_atlas(
                    human_feat,
                    pose_out,
                    pred_verts,
                    pred_cam_t,
                    cam_int,
                    target_h,
                    target_w,
                    src_idx_map=src_idx_map,
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

            if should_export_uv_debug:
                num_verts = pred_verts.shape[1]
                seq0_src_idx = src_idx_map.view(
                    n, s, human_feat.shape[-2], human_feat.shape[-1]
                )[0]
                seq0_front = uv_aux["front_view_verts"].view(n, s, num_verts, 3)[0]
                seq0_uv = uv_aux["uv_coords"].view(n, s, num_verts, 2)[0]
                debug_seq0_front_verts.append(seq0_front.detach().float().cpu())
                debug_seq0_uv_coords.append(seq0_uv.detach().float().cpu())

                for i in range(self.num_FPN):
                    seq0_feat_map = processed_feat_list[i].view(
                        n,
                        s,
                        processed_feat_list[i].shape[1],
                        human_feat.shape[-2],
                        human_feat.shape[-1],
                    )[0]
                    vertex_feat, vertex_valid = self.accumulate_pixel_features_to_vertices(
                        seq0_feat_map, seq0_src_idx, num_verts
                    )
                    debug_seq0_vertex_feat[i].append(vertex_feat.detach().float().cpu())
                    debug_seq0_vertex_valid[i].append(vertex_valid.detach().bool().cpu())

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
        embed_list, log_list = self.uv_test_2(self.Gait_Net, torch.cat(all_outs, dim=2), seqL)

        if should_export_uv_debug and debug_seq0_front_verts:
            seq0_front_verts = torch.cat(debug_seq0_front_verts, dim=0)
            seq0_uv_coords = torch.cat(debug_seq0_uv_coords, dim=0)
            per_fpn_frame_vertex_feat = [
                torch.cat(frame_feat_list, dim=0) for frame_feat_list in debug_seq0_vertex_feat
            ]
            per_fpn_frame_vertex_valid = [
                torch.cat(frame_valid_list, dim=0)
                for frame_valid_list in debug_seq0_vertex_valid
            ]
            self._export_uv_debug_artifacts(
                labs[0],
                seq0_front_verts[0],
                seq0_front_verts,
                seq0_uv_coords,
                per_fpn_frame_vertex_feat,
                per_fpn_frame_vertex_valid,
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
