import copy
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

from utils import ddp_all_gather

from ..modules import SeparateBNNecks, SeparateFCs
from PIL import Image, ImageDraw
from .BigGait_utils.save_img import pca_image
from .BiggerGait_DINOv2_Projection_Mask_OT_based import (
    BiggerGait__DINOv2__Projection_Mask_OT_Based,
)


class BiggerGait__DINOv2__Projection_Mask_OT_PairWeight_Based(
    BiggerGait__DINOv2__Projection_Mask_OT_Based
):
    def build_network(self, model_cfg):
        super().build_network(model_cfg)

        pairmask_cfg = model_cfg.get("pairmask_cfg", {})
        self.pairmask_parts_num = int(model_cfg["SeparateFCs"]["parts_num"])
        bin_num = model_cfg.get("bin_num", [])
        if sum(bin_num) != self.pairmask_parts_num:
            raise ValueError(
                "Pair-weight branch expects sum(bin_num) == SeparateFCs.parts_num, "
                f"got {sum(bin_num)} vs {self.pairmask_parts_num}."
            )

        self.pairmask_in_channels = (
            int(model_cfg["SeparateFCs"]["in_channels"]) * self.num_FPN * self.num_branches
        )
        self.pairmask_proj_dim = int(pairmask_cfg.get("proj_dim", 128))
        self.pairmask_logit_scale = float(pairmask_cfg.get("logit_scale", 4.0))
        self.pairmask_partner_mode = str(pairmask_cfg.get("partner_mode", "hard_positive")).lower()
        self.pairmask_warmup_start = int(pairmask_cfg.get("warmup_start_iter", 0))
        self.pairmask_warmup_end = int(pairmask_cfg.get("warmup_end_iter", self.pairmask_warmup_start))
        self.pairmask_vis_num_pairs = int(pairmask_cfg.get("vis_num_pairs", 5))
        self.pairmask_vis_pos_count = int(pairmask_cfg.get("vis_pos_count", 3))
        self.pairmask_vis_neg_count = int(pairmask_cfg.get("vis_neg_count", 2))
        self.pairmask_vis_rgb_frames = int(pairmask_cfg.get("vis_rgb_frames", 5))
        self.pairmask_vis_rgb_height = int(pairmask_cfg.get("vis_rgb_height", 64))
        self.pairmask_vis_tp_head_height = int(pairmask_cfg.get("vis_tp_head_height", 24))
        self.pairmask_vis_separator = int(pairmask_cfg.get("vis_separator", 6))

        aux_out_channels = int(
            pairmask_cfg.get("aux_out_channels", model_cfg["SeparateFCs"]["out_channels"])
        )

        aux_fc_cfg = copy.deepcopy(model_cfg["SeparateFCs"])
        aux_fc_cfg["in_channels"] = self.pairmask_in_channels
        aux_fc_cfg["out_channels"] = aux_out_channels
        aux_fc_cfg["parts_num"] = self.pairmask_parts_num
        self.PairMaskFC = SeparateFCs(**aux_fc_cfg)

        aux_bn_cfg = copy.deepcopy(model_cfg["SeparateBNNecks"])
        aux_bn_cfg["in_channels"] = aux_out_channels
        aux_bn_cfg["parts_num"] = self.pairmask_parts_num
        self.PairMaskBNNecks = SeparateBNNecks(**aux_bn_cfg)

        self.PairMaskInputNorm = nn.BatchNorm1d(self.pairmask_in_channels, affine=False)
        self.PairMaskQ = nn.Conv1d(self.pairmask_in_channels, self.pairmask_proj_dim, kernel_size=1, bias=False)
        self.PairMaskK = nn.Conv1d(self.pairmask_in_channels, self.pairmask_proj_dim, kernel_size=1, bias=False)

        self.msg_mgr.log_info(
            "[PairWeight] Enabled shared part weights with "
            f"in_channels={self.pairmask_in_channels}, parts={self.pairmask_parts_num}, "
            f"proj_dim={self.pairmask_proj_dim}, warmup=({self.pairmask_warmup_start}, {self.pairmask_warmup_end})."
        )

    def _pairmask_alpha(self):
        if not self.training:
            return 1.0

        current_iter = int(getattr(self, "iteration", 0))
        if current_iter < self.pairmask_warmup_start:
            return 0.0
        if self.pairmask_warmup_end <= self.pairmask_warmup_start:
            return 1.0
        if current_iter >= self.pairmask_warmup_end:
            return 1.0
        return float(current_iter - self.pairmask_warmup_start) / float(
            self.pairmask_warmup_end - self.pairmask_warmup_start
        )

    def _gather_tensor(self, tensor, requires_grad=False):
        if dist.is_available() and dist.is_initialized():
            return ddp_all_gather(tensor, requires_grad=requires_grad)
        return tensor

    def _mine_positive_indices(self, local_embeddings, local_labels):
        if self.pairmask_partner_mode != "hard_positive":
            raise ValueError(f"Unsupported pairmask partner mode: {self.pairmask_partner_mode}")

        global_embeddings = self._gather_tensor(local_embeddings, requires_grad=False)
        global_labels = self._gather_tensor(local_labels, requires_grad=False)

        local_flat = local_embeddings.reshape(local_embeddings.size(0), -1).float()
        global_flat = global_embeddings.reshape(global_embeddings.size(0), -1).float()
        dist_mat = torch.cdist(local_flat, global_flat, p=2)

        same_mask = local_labels.unsqueeze(1) == global_labels.unsqueeze(0)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        local_count = local_embeddings.size(0)
        self_idx = torch.arange(local_count, device=local_embeddings.device) + rank * local_count
        same_mask[torch.arange(local_count, device=local_embeddings.device), self_idx] = False

        partner_idx = self_idx.clone()
        has_pos = same_mask.any(dim=1)
        if has_pos.any():
            pos_dist = dist_mat.masked_fill(~same_mask, -1.0)
            partner_idx[has_pos] = pos_dist[has_pos].argmax(dim=1)
        return partner_idx

    def _encode_pairmask(self, pairmask_parts):
        normed = self.PairMaskInputNorm(pairmask_parts)
        q = F.normalize(self.PairMaskQ(normed), dim=1)
        k = F.normalize(self.PairMaskK(normed), dim=1)
        return q, k

    def _compute_shared_part_weights(self, q_local, k_local, q_partner, k_partner):
        score_left = (q_local * k_partner).sum(dim=1)
        score_right = (q_partner * k_local).sum(dim=1)
        logits = 0.5 * (score_left + score_right) * self.pairmask_logit_scale
        pred = torch.sigmoid(logits)
        alpha = self._pairmask_alpha()
        return ((1.0 - alpha) + alpha * pred).clamp(0.0, 1.0)

    def _should_log_visual_summary(self):
        if not self.training:
            return False
        log_iter = self.engine_cfg.get("log_iter", None)
        if not log_iter:
            return False
        return ((self.iteration + 1) % log_iter) == 0

    def _prepare_rgb_for_vis(self, rgb, flip_flags):
        vis_frames = min(self.pairmask_vis_rgb_frames, rgb.size(1))
        rgb_vis = rgb[:, :vis_frames].detach().float().contiguous()
        flat_rgb = rearrange(rgb_vis, "n s c h w -> (n s) c h w")
        flat_rgb = self._apply_sequence_hflip(flat_rgb, flip_flags, vis_frames)

        if flat_rgb.shape[-1] == flat_rgb.shape[-2]:
            cutting = flat_rgb.shape[-1] // 4
            if cutting != 0:
                flat_rgb = flat_rgb[..., cutting:-cutting]

        if flat_rgb.max() > 1.0:
            flat_rgb = flat_rgb / 255.0
        flat_rgb = flat_rgb.clamp(0.0, 1.0)
        return rearrange(flat_rgb, "(n s) c h w -> n s c h w", n=rgb.size(0), s=vis_frames)

    def _resize_panel(self, panel, target_h, target_w):
        if panel.shape[-2] == target_h and panel.shape[-1] == target_w:
            return panel
        return F.interpolate(
            panel.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _pairmask_tile_hw(self):
        tile_h = self.pairmask_vis_rgb_height
        tile_w = max(1, int(round(float(tile_h) / 2.0)))
        return tile_h, tile_w

    def _fit_panel_to_tile(self, panel, target_h, target_w, fill_value=0.96):
        src_h, src_w = panel.shape[-2:]
        if src_h <= 0 or src_w <= 0:
            return torch.ones(
                3, target_h, target_w, dtype=panel.dtype, device=panel.device
            ) * fill_value

        scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))
        resized = self._resize_panel(panel, new_h, new_w)

        canvas = torch.ones(
            3,
            target_h,
            target_w,
            dtype=panel.dtype,
            device=panel.device,
        ) * fill_value
        top = max(0, (target_h - new_h) // 2)
        left = max(0, (target_w - new_w) // 2)
        canvas[:, top : top + new_h, left : left + new_w] = resized
        return canvas

    def _build_rgb_tiles(self, rgb_frames):
        tile_h, tile_w = self._pairmask_tile_hw()
        tiles = []
        for frame in rgb_frames:
            tiles.append(self._fit_panel_to_tile(frame, tile_h, tile_w, fill_value=0.96))
        return tiles

    def _annotate_tile(self, tile, label_id):
        tile_cpu = tile.detach().float().clamp(0.0, 1.0).cpu()
        tile_np = (tile_cpu.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        image = Image.fromarray(tile_np)
        draw = ImageDraw.Draw(image)
        text = f"ID {int(label_id)}"

        shadow_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in shadow_offsets:
            draw.text((6 + dx, 6 + dy), text, fill=(0, 0, 0))
        draw.text((5, 5), text, fill=(255, 255, 0))

        annotated = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        return annotated.to(tile.device)

    def _build_rgb_strip(self, rgb_frames):
        strips = []
        for frame in rgb_frames:
            frame_h, frame_w = frame.shape[-2:]
            target_h = self.pairmask_vis_rgb_height
            target_w = max(1, int(round(frame_w * float(target_h) / float(frame_h))))
            strips.append(self._resize_panel(frame, target_h, target_w))
        if not strips:
            return None
        return self._concat_panels_with_gap(strips, gap_px=2, fill_value=0.96)

    def _concat_panels_with_gap(self, panels, gap_px=2, fill_value=0.96):
        if not panels:
            return None
        if len(panels) == 1:
            return panels[0]

        gap = torch.ones(
            3,
            panels[0].shape[-2],
            gap_px,
            dtype=panels[0].dtype,
            device=panels[0].device,
        ) * fill_value
        merged = []
        for idx, panel in enumerate(panels):
            merged.append(panel)
            if idx != len(panels) - 1:
                merged.append(gap)
        return torch.cat(merged, dim=-1)

    def _add_panel_padding(self, panel, pad_px=1, fill_value=0.96):
        if pad_px <= 0:
            return panel
        return F.pad(
            panel,
            (pad_px, pad_px, pad_px, pad_px),
            mode="constant",
            value=fill_value,
        )

    def _pad_row_width(self, row_tensor, target_width, fill_value=0.95):
        curr_width = row_tensor.shape[-1]
        if curr_width == target_width:
            return row_tensor
        if curr_width > target_width:
            return row_tensor[..., :target_width]
        pad_width = target_width - curr_width
        return F.pad(
            row_tensor,
            (0, pad_width, 0, 0),
            mode="constant",
            value=fill_value,
        )

    def _annotate_rgb_strip(self, rgb_strip, label_id):
        strip = rgb_strip.detach().float().clamp(0.0, 1.0).cpu()
        strip_np = (strip.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        image = Image.fromarray(strip_np)
        draw = ImageDraw.Draw(image)
        text = f"ID {int(label_id)}"

        shadow_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in shadow_offsets:
            draw.text((6 + dx, 6 + dy), text, fill=(0, 0, 0))
        draw.text((5, 5), text, fill=(255, 255, 0))

        annotated = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        return annotated.to(rgb_strip.device)

    def _fallback_pca_vis(self, feat_map):
        feat_map = feat_map.detach().float()
        if feat_map.size(0) >= 3:
            vis = feat_map[:3]
        else:
            vis = feat_map.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        vis = vis - vis.amin(dim=(-2, -1), keepdim=True)
        vis = vis / (vis.amax(dim=(-2, -1), keepdim=True) + 1.0e-6)
        return vis.clamp(0.0, 1.0)

    def _build_single_map_pca(self, feat_map):
        feat_cpu = feat_map.detach().float().cpu()
        _, height, width = feat_cpu.shape
        feat_np = rearrange(feat_cpu.numpy(), "c h w -> 1 (h w) c")
        mask_np = np.ones((1, height * width), dtype=np.uint8)
        try:
            pca_img = pca_image(
                data={"embeddings": feat_np, "h": height, "w": width},
                mask=mask_np,
                root=None,
                model_name=None,
                dataset=None,
                n_components=3,
                is_return=True,
            )[0, 0]
            return torch.from_numpy(pca_img).float().to(feat_map.device) / 255.0
        except Exception:
            return self._fallback_pca_vis(feat_map)

    def _weight_to_heatmap(self, weights, height, width):
        weight_map = F.interpolate(
            weights.view(1, 1, -1, 1),
            size=(height, width),
            mode="nearest",
        ).squeeze(0)
        x = weight_map[0].clamp(0.0, 1.0)
        heat_r = (1.5 - torch.abs(4.0 * x - 3.0)).clamp(0.0, 1.0)
        heat_g = (1.5 - torch.abs(4.0 * x - 2.0)).clamp(0.0, 1.0)
        heat_b = (1.5 - torch.abs(4.0 * x - 1.0)).clamp(0.0, 1.0)
        return torch.stack([heat_r, heat_g, heat_b], dim=0)

    def _build_aggregated_tp_pca(self, tp_maps, sample_idx, weights, target_h, target_w):
        masked_maps = []
        for tp_map in tp_maps:
            curr_map = tp_map[sample_idx].detach()
            curr_h, curr_w = curr_map.shape[-2:]
            weight_map = F.interpolate(
                weights.view(1, 1, -1, 1),
                size=(curr_h, curr_w),
                mode="nearest",
            ).squeeze(0)
            masked_maps.append(curr_map * weight_map.sqrt())

        if not masked_maps:
            return None

        merged_map = torch.cat(masked_maps, dim=0)
        pca_panel = self._build_single_map_pca(merged_map)
        return self._fit_panel_to_tile(pca_panel, target_h, target_w, fill_value=0.96)

    def _build_pair_mask_panel(self, weights, target_h, target_w):
        mask_panel = self._weight_to_heatmap(weights, max(32, target_h), 24)
        return self._fit_panel_to_tile(mask_panel, target_h, target_w, fill_value=0.96)

    def _build_sample_vis_row(self, rgb_frames, label_id, pca_panel, mask_panel):
        rgb_tiles = self._build_rgb_tiles(rgb_frames)
        if not rgb_tiles:
            return None
        rgb_tiles[0] = self._annotate_tile(rgb_tiles[0], label_id)
        row_tiles = rgb_tiles + [pca_panel, mask_panel]
        return self._concat_panels_with_gap(
            row_tiles,
            gap_px=2,
            fill_value=0.96,
        )

    def _compose_pair_row(self, anchor_row, partner_row, border_color):
        pair_gap = self.pairmask_vis_separator
        target_width = max(anchor_row.shape[-1], partner_row.shape[-1])
        anchor_row = self._pad_row_width(anchor_row, target_width, fill_value=0.96)
        partner_row = self._pad_row_width(partner_row, target_width, fill_value=0.96)

        center_gap = torch.ones(
            3,
            anchor_row.shape[-2],
            pair_gap,
            dtype=anchor_row.dtype,
            device=anchor_row.device,
        ) * 0.94
        center_gap[:, :, : max(1, pair_gap // 2)] = torch.tensor(
            border_color,
            dtype=anchor_row.dtype,
            device=anchor_row.device,
        ).view(3, 1, 1)
        return torch.cat([anchor_row, center_gap, partner_row], dim=-1)

    def _compose_pair_triptych(
        self,
        anchor_rgb_strip,
        partner_rgb_strip,
        anchor_pca,
        partner_pca,
        anchor_mask,
        partner_mask,
        border_color,
    ):
        gap = self.pairmask_vis_separator
        border = torch.tensor(
            border_color, dtype=anchor_rgb_strip.dtype, device=anchor_rgb_strip.device
        ).view(3, 1, 1)

        def _make_cell_gap(row_tensor):
            return torch.ones(
                3,
                row_tensor.shape[-2],
                gap,
                dtype=row_tensor.dtype,
                device=row_tensor.device,
            ) * 0.92

        rgb_row = torch.cat(
            [anchor_rgb_strip, _make_cell_gap(anchor_rgb_strip), partner_rgb_strip], dim=-1
        )
        pca_row = torch.cat(
            [anchor_pca, _make_cell_gap(anchor_pca), partner_pca], dim=-1
        )
        mask_row = torch.cat(
            [anchor_mask, _make_cell_gap(anchor_mask), partner_mask], dim=-1
        )

        target_width = max(rgb_row.shape[-1], pca_row.shape[-1], mask_row.shape[-1])
        rgb_row = self._pad_row_width(rgb_row, target_width, fill_value=0.95)
        pca_row = self._pad_row_width(pca_row, target_width, fill_value=0.95)
        mask_row = self._pad_row_width(mask_row, target_width, fill_value=0.95)
        border = border.expand(3, gap, target_width)
        row_gap = torch.ones(
            3,
            gap,
            target_width,
            dtype=anchor_rgb_strip.dtype,
            device=anchor_rgb_strip.device,
        ) * 0.95

        return torch.cat([border, rgb_row, row_gap, pca_row, row_gap, mask_row], dim=1)

    def _select_visual_anchor_and_partners(self, labels, embeddings):
        batch_size = labels.size(0)
        if batch_size <= 1:
            return None, []

        anchor_idx = None
        best_pos = -1
        best_neg = -1
        for idx in range(batch_size):
            pos_count = int((labels == labels[idx]).sum().item()) - 1
            neg_count = batch_size - pos_count - 1
            if pos_count <= 0:
                continue
            if pos_count > best_pos or (pos_count == best_pos and neg_count > best_neg):
                anchor_idx = idx
                best_pos = pos_count
                best_neg = neg_count

        if anchor_idx is None:
            return None, []

        flat_embeddings = embeddings.detach().reshape(batch_size, -1).float()
        dist_to_anchor = torch.cdist(
            flat_embeddings[anchor_idx : anchor_idx + 1],
            flat_embeddings,
            p=2,
        )[0]

        same_mask = labels == labels[anchor_idx]
        pos_idx = torch.nonzero(same_mask, as_tuple=False).flatten()
        pos_idx = pos_idx[pos_idx != anchor_idx]
        neg_idx = torch.nonzero(~same_mask, as_tuple=False).flatten()

        if pos_idx.numel() > 0:
            pos_order = dist_to_anchor.index_select(0, pos_idx).argsort(descending=True)
            pos_idx = pos_idx.index_select(0, pos_order)
        if neg_idx.numel() > 0:
            neg_order = dist_to_anchor.index_select(0, neg_idx).argsort(descending=False)
            neg_idx = neg_idx.index_select(0, neg_order)

        selected = []
        selected.extend(pos_idx[: self.pairmask_vis_pos_count].tolist())
        selected.extend(neg_idx[: self.pairmask_vis_neg_count].tolist())

        if len(selected) < self.pairmask_vis_num_pairs:
            remaining = torch.cat(
                [pos_idx[self.pairmask_vis_pos_count :], neg_idx[self.pairmask_vis_neg_count :]],
                dim=0,
            )
            selected.extend(remaining[: self.pairmask_vis_num_pairs - len(selected)].tolist())

        deduped = []
        for idx in selected:
            if idx not in deduped:
                deduped.append(idx)
        return anchor_idx, deduped[: self.pairmask_vis_num_pairs]

    def _build_pairmask_visual_grid(self, rgb_vis, tp_map_groups, pairmask_q, pairmask_k, labs, embeddings):
        if rgb_vis is None or labs.size(0) <= 1:
            return None

        anchor_idx, partner_indices = self._select_visual_anchor_and_partners(
            labs.detach(), embeddings.detach()
        )
        if anchor_idx is None or not partner_indices:
            return None

        tile_h, tile_w = self._pairmask_tile_hw()

        rows = []
        gap = self.pairmask_vis_separator
        for partner_idx in partner_indices:
            weights = self._compute_shared_part_weights(
                pairmask_q[anchor_idx : anchor_idx + 1],
                pairmask_k[anchor_idx : anchor_idx + 1],
                pairmask_q[partner_idx : partner_idx + 1],
                pairmask_k[partner_idx : partner_idx + 1],
            )[0].detach()

            anchor_pca = self._build_aggregated_tp_pca(
                tp_map_groups, anchor_idx, weights, tile_h, tile_w
            )
            partner_pca = self._build_aggregated_tp_pca(
                tp_map_groups, partner_idx, weights, tile_h, tile_w
            )
            if anchor_pca is None or partner_pca is None:
                continue
            anchor_mask = self._build_pair_mask_panel(weights, tile_h, tile_w)
            partner_mask = self._build_pair_mask_panel(weights, tile_h, tile_w)

            anchor_row = self._build_sample_vis_row(
                rgb_vis[anchor_idx],
                labs[anchor_idx].item(),
                anchor_pca,
                anchor_mask,
            )
            partner_row = self._build_sample_vis_row(
                rgb_vis[partner_idx],
                labs[partner_idx].item(),
                partner_pca,
                partner_mask,
            )
            if anchor_row is None or partner_row is None:
                continue

            is_positive = bool(labs[partner_idx].item() == labs[0].item())
            pair_color = (0.15, 0.75, 0.20) if is_positive else (0.90, 0.20, 0.20)

            rows.append(self._compose_pair_row(anchor_row, partner_row, pair_color))

        if not rows:
            return None

        row_gap = torch.ones(
            3,
            gap,
            rows[0].shape[-1],
            dtype=rows[0].dtype,
            device=rows[0].device,
        ) * 0.96
        canvas_rows = []
        for idx, row in enumerate(rows):
            canvas_rows.append(row)
            if idx != len(rows) - 1:
                canvas_rows.append(row_gap)
        return torch.cat(canvas_rows, dim=1).unsqueeze(0)

    def _run_branch_heads(self, branch_model, branch_seq_feat, seqL):
        feat_list = torch.chunk(branch_seq_feat, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        raw_part_list = []
        tp_map_list = []
        for i in range(self.num_FPN):
            gait_single = branch_model.Gait_List[i]
            pooled = gait_single.TP(feat_list[i], seqL, options={"dim": 2})[0]
            raw_parts = gait_single.HPP(pooled)
            embed_1 = gait_single.FCs(raw_parts)
            _, logits = gait_single.BNNecks(embed_1)
            embed_list.append(embed_1)
            log_list.append(logits)
            raw_part_list.append(raw_parts)
            tp_map_list.append(pooled)
        return embed_list, log_list, raw_part_list, tp_map_list

    def forward(self, inputs):
        timing_info = {
            "model_hflip": 0.0,
            "model_rgb_preprocess": 0.0,
            "model_dino": 0.0,
            "model_sam_unpack": 0.0,
            "model_project_mask": 0.0,
            "model_humanspace": 0.0,
            "model_ot": 0.0,
            "model_gait_head": 0.0,
            "model_pairmask": 0.0,
        }

        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        sam_decoder = ipts[1]
        del ipts
        model_start = self._perf_now(rgb.device)
        should_log_pairmask_vis = self._should_log_visual_summary()

        if self.training and self.sync_hflip_prob > 0:
            flip_flags = torch.rand(rgb.size(0), device=rgb.device) < self.sync_hflip_prob
        else:
            flip_flags = torch.zeros(rgb.size(0), device=rgb.device, dtype=torch.bool)

        rgb_vis = self._prepare_rgb_for_vis(rgb, flip_flags) if should_log_pairmask_vis else None

        num_chunks = (rgb.size(1) // self.chunk_size) + 1
        rgb_chunks = torch.chunk(rgb, num_chunks, dim=1)
        chunk_lengths = [chunk.size(1) for chunk in rgb_chunks]

        all_outs = [[] for _ in range(self.num_branches)]
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = self.image_size // 7, self.image_size // 14
        last_rgb_img = None
        last_generated_mask = None
        last_human_mask = None
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
                intermediates = partial(nn.LayerNorm, eps=1e-6)(
                    self.f4_dim * len(outs), elementwise_affine=False
                )(torch.concat(outs, dim=-1))[:, 1:]
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

            stage_start = self._perf_now(rgb.device)
            intermediates = [
                torch.cat(intermediates[i : i + self.group_layer_num], dim=1).contiguous()
                for i in range(0, self.total_layer_num, self.group_layer_num)
            ]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            human_feat = torch.concat(intermediates, dim=1)
            human_mask = self.preprocess(generated_mask, self.sils_size)
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            timing_info["model_humanspace"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            self._apose_cache = None
            self._apose_cache_key = None
            cam_int_tgt, cam_t_tgt = self.build_target_camera(rgb_img.shape[0], rgb.device, target_h, target_w)
            branch_warped_feats = []
            branch_target_masks = []
            for branch_cfg in self.branch_configs:
                branch_geo = self.build_branch_geometry(branch_cfg, pose_out)
                warp_feat, tgt_valid_mask, _ = self.warp_features_with_ot(
                    human_feat,
                    human_mask.float(),
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
            timing_info["model_ot"] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            for b_idx, warp_feat in enumerate(branch_warped_feats):
                warp_feat_5d = rearrange(warp_feat, "(n s) c h w -> n c s h w", n=n, s=s).contiguous()
                outs = self.Gait_Nets[b_idx].test_1(warp_feat_5d)
                all_outs[b_idx].append(outs)
            timing_info["model_gait_head"] += self._perf_now(rgb.device) - stage_start

            last_rgb_img = rgb_img
            last_generated_mask = generated_mask
            last_human_mask = human_mask
            last_target_masks = branch_target_masks

        stage_start = self._perf_now(rgb.device)
        embed_grouped = [[] for _ in range(self.num_FPN)]
        log_grouped = [[] for _ in range(self.num_FPN)]
        pairmask_raw_parts = []
        pairmask_tp_maps = []
        for b_idx in range(self.num_branches):
            branch_seq_feat = torch.cat(all_outs[b_idx], dim=2)
            e_list, l_list, raw_part_list, tp_map_list = self._run_branch_heads(
                self.Gait_Nets[b_idx], branch_seq_feat, seqL
            )
            for i in range(self.num_FPN):
                embed_grouped[i].append(e_list[i])
                log_grouped[i].append(l_list[i])
                pairmask_raw_parts.append(raw_part_list[i])
                pairmask_tp_maps.append(tp_map_list[i])
        embed_list = [torch.cat(feats, dim=-1) for feats in embed_grouped]
        log_list = [torch.cat(logits, dim=-1) for logits in log_grouped]
        embeddings = torch.cat(embed_list, dim=-1)
        logits = torch.cat(log_list, dim=-1)
        timing_info["model_gait_head"] += self._perf_now(rgb.device) - stage_start

        stage_start = self._perf_now(rgb.device)
        pairmask_parts = torch.cat(pairmask_raw_parts, dim=1)
        pairmask_parts_detached = pairmask_parts.detach()
        pairmask_aux_embeddings = self.PairMaskFC(pairmask_parts_detached)
        pairmask_q, pairmask_k = self._encode_pairmask(pairmask_parts_detached)

        pairmask_weights = None
        masked_embeddings = None
        masked_logits = None
        if self.training:
            partner_idx = self._mine_positive_indices(embeddings.detach(), labs.detach())
            gathered_q = self._gather_tensor(pairmask_q, requires_grad=True)
            gathered_k = self._gather_tensor(pairmask_k, requires_grad=True)
            partner_q = gathered_q.index_select(0, partner_idx)
            partner_k = gathered_k.index_select(0, partner_idx)
            pairmask_weights = self._compute_shared_part_weights(
                pairmask_q, pairmask_k, partner_q, partner_k
            )
            masked_parts = pairmask_parts_detached * pairmask_weights.clamp_min(1.0e-12).sqrt().unsqueeze(1)
            masked_embeddings = self.PairMaskFC(masked_parts)
            _, masked_logits = self.PairMaskBNNecks(masked_embeddings)
        timing_info["model_pairmask"] += self._perf_now(rgb.device) - stage_start

        model_end = self._perf_now(rgb.device)
        timed_sum = sum(timing_info.values())
        timing_info["model_misc"] = max(model_end - model_start - timed_sum, 0.0)

        if self.training:
            visual_summary = {
                "image/rgb_img": last_rgb_img[:5].float(),
                "image/generated_3d_mask_lowres": last_generated_mask[:5].float(),
                "image/generated_3d_mask_interpolated": last_human_mask[:5].float(),
            }
            if last_target_masks:
                visual_summary["image/ot_target_mask_branch0"] = last_target_masks[0][:5].float()
            pairmask_pair_grid = None
            if should_log_pairmask_vis:
                pairmask_pair_grid = self._build_pairmask_visual_grid(
                    rgb_vis,
                    pairmask_tp_maps,
                    pairmask_q,
                    pairmask_k,
                    labs,
                    embeddings,
                )
            if pairmask_pair_grid is not None:
                visual_summary["image/pairmask_pair_grid"] = pairmask_pair_grid.float()

            retval = {
                "training_feat": {
                    "triplet": {"embeddings": embeddings, "labels": labs},
                    "softmax": {"logits": logits, "labels": labs},
                    "mask_triplet": {"embeddings": masked_embeddings, "labels": labs},
                    "mask_softmax": {"logits": masked_logits, "labels": labs},
                },
                "visual_summary": visual_summary,
                "inference_feat": {
                    "embeddings": embeddings,
                    **{f"embeddings_{i}": embed_list[i] for i in range(self.num_FPN)},
                    "pairmask_aux_embeddings": pairmask_aux_embeddings,
                    "pairmask_q": pairmask_q,
                    "pairmask_k": pairmask_k,
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
                    "pairmask_aux_embeddings": pairmask_aux_embeddings,
                    "pairmask_q": pairmask_q,
                    "pairmask_k": pairmask_k,
                },
                "timing_info": timing_info,
            }
        return retval
