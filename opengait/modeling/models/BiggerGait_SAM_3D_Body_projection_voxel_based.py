import math
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from torch.nn import functional as F

from ..modules import PackSequenceWrapper, SeparateBNNecks, SeparateFCs
from .BigGait_utils.BigGait_GaitBase import get_timestep_embedding
from .BiggerGait_SAM_3D_Body_projection_mask_OT_based import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share,
    ResizeToHW,
    infoDistillation,
)


class SetBlockWrapper3D(nn.Module):
    def __init__(self, forward_block):
        super().__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        n, c, s, d, h, w = x.size()
        x = x.transpose(1, 2).reshape(-1, c, d, h, w)
        x = self.forward_block(x, *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class FlexibleSequential3D(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            try:
                input = module(input, *args, **kwargs)
            except TypeError:
                input = module(input)
        return input


class BasicBlock3D_Time(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.temb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, planes),
        )

    def forward(self, x, temb=None, use_Time=True):
        identity = x
        out = self.conv1(x)
        if temb is not None and use_Time:
            out = out + self.temb_proj(temb)[:, :, None, None, None]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class VoxelResNetPre(nn.Module):
    def __init__(self, in_channel, channels, strides):
        super().__init__()
        self.inplanes = channels[0]
        self.conv1 = nn.Conv3d(
            in_channel,
            self.inplanes,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], blocks=1, stride=self._to_3tuple(strides[0]))

    @staticmethod
    def _to_3tuple(stride):
        if isinstance(stride, int):
            return (stride, stride, stride)
        return tuple(stride)

    def _make_layer(self, planes, blocks, stride):
        if max(stride) > 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            downsample = None
        layers = [BasicBlock3D_Time(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D_Time(self.inplanes, planes))
        return FlexibleSequential3D(*layers)

    def forward(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x, *args, **kwargs)
        return x


class VoxelResNetPost(nn.Module):
    def __init__(self, inplanes, channels, layers, strides):
        super().__init__()
        self.inplanes = inplanes
        self.layer2 = self._make_layer(channels[1], layers[1], self._to_3tuple(strides[1]))
        self.layer3 = self._make_layer(channels[2], layers[2], self._to_3tuple(strides[2]))
        self.layer4 = self._make_layer(channels[3], layers[3], self._to_3tuple(strides[3]))

    @staticmethod
    def _to_3tuple(stride):
        if isinstance(stride, int):
            return (stride, stride, stride)
        return tuple(stride)

    def _make_layer(self, planes, blocks, stride):
        if max(stride) > 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            downsample = None
        layers = [BasicBlock3D_Time(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D_Time(self.inplanes, planes))
        return FlexibleSequential3D(*layers)

    def forward(self, x, *args, **kwargs):
        x = self.layer2(x, *args, **kwargs)
        x = self.layer3(x, *args, **kwargs)
        x = self.layer4(x, *args, **kwargs)
        return x


class VoxelHorizontalPoolingPyramid(nn.Module):
    def __init__(self, bin_num=None):
        super().__init__()
        self.bin_num = [32] if bin_num is None else bin_num

    def forward(self, x):
        n, c, d, h, w = x.size()
        features = []
        for b in self.bin_num:
            if d % b != 0:
                raise ValueError(f"Voxel depth/height dimension {d} must be divisible by bin size {b}.")
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, dim=-1)


class VoxelBaselineSingle(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        backbone_cfg = model_cfg["backbone_cfg"]
        channels = backbone_cfg["channels"]
        layers = backbone_cfg["layers"]
        strides = backbone_cfg["strides"]
        in_channel = backbone_cfg["in_channel"]

        self.pre_rgb = SetBlockWrapper3D(VoxelResNetPre(in_channel, channels, strides))
        self.post_backbone = SetBlockWrapper3D(VoxelResNetPost(channels[0], channels, layers, strides))
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = VoxelHorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

    def test_1(self, appearance, return_debug=False, *args, **kwargs):
        outs = self.pre_rgb(appearance, *args, **kwargs)
        if not return_debug:
            return self.post_backbone(outs, *args, **kwargs)

        n, c, s, d, h, w = outs.shape
        layer1_feat = outs.contiguous()
        x = outs.transpose(1, 2).reshape(-1, c, d, h, w)
        post_block = self.post_backbone.forward_block
        layer2_feat = post_block.layer2(x, *args, **kwargs)
        layer3_feat = post_block.layer3(layer2_feat, *args, **kwargs)
        layer4_feat = post_block.layer4(layer3_feat, *args, **kwargs)

        final_outs = layer4_feat.reshape(n, s, *layer4_feat.shape[1:]).transpose(1, 2).contiguous()
        layer2_feat = layer2_feat.reshape(n, s, *layer2_feat.shape[1:]).transpose(1, 2).contiguous()
        layer3_feat = layer3_feat.reshape(n, s, *layer3_feat.shape[1:]).transpose(1, 2).contiguous()
        return final_outs, {
            "layer1_feat": layer1_feat,
            "layer2_feat": layer2_feat,
            "layer3_feat": layer3_feat,
            "layer4_feat": final_outs,
        }

    def test_2(self, outs, seqL):
        outs = self.TP(outs, seqL, options={"dim": 2})[0]
        outs = self.HPP(outs)
        embed_1 = self.FCs(outs)
        _, logits = self.BNNecks(embed_1)
        return embed_1, logits


class VoxelBaselineShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.num_FPN = model_cfg["num_FPN"]
        self.Gait_Net_1 = VoxelBaselineSingle(model_cfg)
        self.Gait_Net_2 = VoxelBaselineSingle(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1 for _ in range(self.num_FPN - self.num_FPN // 2)] +
            [self.Gait_Net_2 for _ in range(self.num_FPN // 2)]
        )
        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def test_1(self, x, return_debug=False, *args, **kwargs):
        x_list = list(torch.chunk(x, self.num_FPN, dim=1))
        n, _, s = x.shape[:3]
        t = torch.arange(self.num_FPN, device=x.device).view(1, -1).repeat(n * s, 1)
        debug_layer_feats = {
            "layer1_feat": [],
            "layer2_feat": [],
            "layer3_feat": [],
            "layer4_feat": [],
        }
        for i in range(self.num_FPN):
            temb = get_timestep_embedding(
                t[:, i],
                self.t_channel,
                max_timesteps=self.num_FPN,
            ).to(x)
            temb = self.temb_proj(temb)
            outputs = self.Gait_List[i].test_1(
                x_list[i], return_debug=return_debug, temb=temb, *args, **kwargs
            )
            if return_debug:
                x_list[i], debug_info = outputs
                for key in debug_layer_feats:
                    debug_layer_feats[key].append(debug_info[key])
            else:
                x_list[i] = outputs
        x = torch.cat(x_list, dim=1)
        if return_debug:
            debug_info = {
                key: torch.cat(feats, dim=1)
                for key, feats in debug_layer_feats.items()
            }
            for key, feats in debug_layer_feats.items():
                debug_info[f"{key}_list"] = feats
            return x, debug_info
        return x

    def test_2(self, x, seqL):
        x_list = torch.chunk(x, self.num_FPN, dim=1)
        embed_list = []
        log_list = []
        for i in range(self.num_FPN):
            embed_1, logits = self.Gait_List[i].test_2(x_list[i], seqL)
            embed_list.append(embed_1)
            log_list.append(logits)
        return embed_list, log_list


class BiggerGait__SAM3DBody__Projection_Voxel_Based_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_Gaitbase_Share
):
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

    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.pretrained_mask_branch = model_cfg["pretrained_mask_branch"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)
        self.debug_pca_vis = model_cfg.get("debug_pca_vis", False)
        self.debug_voxel_export = model_cfg.get("debug_voxel_export", False)

        layer_cfg = model_cfg.get("layer_config", {})
        self.hook_mask = layer_cfg.get("hook_mask", [False] * 16 + [True] * 16)
        if len(self.hook_mask) != 32:
            raise ValueError(f"hook_mask length must be 32, got {len(self.hook_mask)}")
        self.hook_sample_type = layer_cfg.get("hook_sample_type", "chunk")

        self.total_hooked_layers = sum(self.hook_mask)
        if self.total_hooked_layers == 0:
            raise ValueError("hook_mask selects no layers.")
        if self.total_hooked_layers % self.num_FPN != 0:
            raise ValueError(
                f"Hook layers ({self.total_hooked_layers}) must be divisible by FPN heads ({self.num_FPN})"
            )

        self.layers_per_head = self.total_hooked_layers // self.num_FPN
        input_dim = self.f4_dim * self.layers_per_head

        self.branch_configs = model_cfg.get("branch_configs", [{"use_apose": True, "yaw": 0.0}])
        self.voxel_branch_cfg = self.branch_configs[0]

        voxel_cfg = model_cfg.get("voxel_cfg", {})
        grid_size = voxel_cfg.get("grid_size", [64, 32, 32])
        if len(grid_size) != 3:
            raise ValueError(f"voxel_cfg.grid_size must be [64, 32, 32]-style, got {grid_size}")
        self.grid_d = int(grid_size[0])
        self.grid_h = int(grid_size[1])
        self.grid_w = int(grid_size[2])
        self.depth_center_mode = voxel_cfg.get("depth_center_mode", "bbox_mid")
        self.depth_center_offset = float(voxel_cfg.get("depth_center_offset", 0.0))

        self.Gait_Net = VoxelBaselineShareTime_2B(model_cfg)
        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(input_dim, affine=False),
                nn.Conv2d(input_dim, self.f4_dim // 2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim // 2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim // 2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size * 2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid(),
            ) for _ in range(self.num_FPN)
        ])
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])
        self.init_SAM_Backbone()

    def accumulate_pixel_features_to_vertices_with_counts(
        self, feat_map, vertex_index_map, num_verts, valid_pixel_mask=None
    ):
        bsz, channels, _, _ = feat_map.shape
        flat_feat = rearrange(feat_map, "b c h w -> b (h w) c")
        flat_idx = vertex_index_map.view(bsz, -1)
        valid_mask = flat_idx >= 0
        if valid_pixel_mask is not None:
            valid_mask = valid_mask & valid_pixel_mask.view(bsz, -1)

        vertex_feat = feat_map.new_zeros((bsz * num_verts, channels))
        vertex_count = feat_map.new_zeros((bsz * num_verts, 1))

        if valid_mask.any():
            batch_offsets = torch.arange(bsz, device=feat_map.device).unsqueeze(1) * num_verts
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
        return vertex_feat, vertex_valid, vertex_count.squeeze(-1)

    def scatter_vertices_to_volume(self, vertex_feat, vertex_valid, v_idx, u_idx, z_idx):
        bsz, num_verts, channels = vertex_feat.shape
        flat_size = self.grid_d * self.grid_h * self.grid_w
        flat_ids = (v_idx * self.grid_h + u_idx) * self.grid_w + z_idx

        flat_volume = vertex_feat.new_zeros((bsz * flat_size, channels))
        flat_count = vertex_feat.new_zeros((bsz * flat_size, 1))

        if vertex_valid.any():
            batch_offsets = torch.arange(bsz, device=vertex_feat.device).unsqueeze(1) * flat_size
            global_ids = (flat_ids + batch_offsets).reshape(-1)
            valid_flat = vertex_valid.reshape(-1)
            global_ids = global_ids[valid_flat]
            valid_feat = vertex_feat.reshape(-1, channels)[valid_flat]
            flat_volume.index_add_(0, global_ids, valid_feat)
            flat_count.index_add_(
                0, global_ids, flat_count.new_ones((global_ids.numel(), 1))
            )

        flat_volume = flat_volume.view(bsz, self.grid_d * self.grid_h * self.grid_w, channels)
        flat_count = flat_count.view(bsz, self.grid_d * self.grid_h * self.grid_w, 1)
        volume_feat = flat_volume / flat_count.clamp_min(1.0)
        volume_occ = flat_count > 0

        volume_feat = rearrange(
            volume_feat,
            "b (d h w) c -> b c d h w",
            d=self.grid_d,
            h=self.grid_h,
            w=self.grid_w,
        )
        volume_occ = rearrange(
            volume_occ.float(),
            "b (d h w) c -> b c d h w",
            d=self.grid_d,
            h=self.grid_h,
            w=self.grid_w,
        )
        return volume_feat, volume_occ

    def _generate_color_grid(self, batch_size, height, width, device):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing="ij",
        )
        color_grid = torch.stack([grid_x, grid_y, 1 - grid_x], dim=0)
        return color_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _paint_points_to_canvas(self, v_idx, u_idx, colors):
        batch_size = v_idx.shape[0]
        canvas = torch.full(
            (batch_size, 3, self.grid_d, self.grid_h),
            0.08,
            device=v_idx.device,
            dtype=colors.dtype,
        )
        for b in range(batch_size):
            canvas[b, :, v_idx[b], u_idx[b]] = colors[b].transpose(0, 1)
        return canvas

    def _make_alignment_triptych(self, target_mask_2d, point_projection, voxel_front_mask):
        color_grid = self._generate_color_grid(target_mask_2d.shape[0], self.grid_d, self.grid_h, target_mask_2d.device)
        target_grid = color_grid * target_mask_2d
        voxel_grid = color_grid * voxel_front_mask
        return torch.cat([target_grid, point_projection, voxel_grid], dim=-1)

    def _should_export_debug(self):
        if not self.debug_voxel_export or not self.training:
            return False
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
            return False
        return self._should_log_visual_summary()

    def _get_voxel_debug_export_root(self):
        return os.path.join(self.save_path, "voxel_debug", f"iter_{self.iteration + 1:05d}")

    def _save_ascii_ply(self, verts, colors, save_path):
        verts_np = verts.detach().float().cpu().numpy()
        colors_np = (colors.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {verts_np.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(verts_np, colors_np):
                f.write(f"{x:.6f} {-y:.6f} {-z:.6f} {int(r)} {int(g)} {int(b)}\n")

    def _feature_pca_colors(self, feats, valid_mask=None):
        feats = feats.detach().float().cpu()
        if valid_mask is None:
            valid_mask = torch.ones(feats.shape[0], dtype=torch.bool)
        else:
            valid_mask = valid_mask.detach().bool().cpu()

        colors = torch.full((feats.shape[0], 3), 0.18, dtype=torch.float32)
        if int(valid_mask.sum().item()) < 3:
            return colors

        valid_feats = feats[valid_mask]
        valid_feats = valid_feats - valid_feats.mean(dim=0, keepdim=True)
        q = min(3, valid_feats.shape[0], valid_feats.shape[1])
        _, _, v = torch.pca_lowrank(valid_feats, q=q)
        proj = valid_feats @ v[:, :q]
        if q < 3:
            proj = F.pad(proj, (0, 3 - q))

        proj_min = proj.min(dim=0, keepdim=True)[0]
        proj_max = proj.max(dim=0, keepdim=True)[0]
        proj = (proj - proj_min) / (proj_max - proj_min + 1e-6)
        colors[valid_mask] = 0.1 + 0.9 * proj
        return colors

    def _feature_norm_colors(self, feats, valid_mask=None):
        feats = feats.detach().float().cpu()
        if valid_mask is None:
            valid_mask = torch.ones(feats.shape[0], dtype=torch.bool)
        else:
            valid_mask = valid_mask.detach().bool().cpu()

        colors = torch.full((feats.shape[0], 3), 0.18, dtype=torch.float32)
        if int(valid_mask.sum().item()) == 0:
            return colors

        norms = torch.linalg.vector_norm(feats, ord=2, dim=-1)
        valid_norms = norms[valid_mask]
        norm_min = valid_norms.min()
        norm_max = valid_norms.max()
        if (norm_max - norm_min) > 1e-6:
            x = ((norms - norm_min) / (norm_max - norm_min + 1e-6)).clamp(0, 1)
        else:
            x = torch.zeros_like(norms)

        heat_r = (1.5 - torch.abs(4.0 * x - 3.0)).clamp(0.0, 1.0)
        heat_g = (1.5 - torch.abs(4.0 * x - 2.0)).clamp(0.0, 1.0)
        heat_b = (1.5 - torch.abs(4.0 * x - 1.0)).clamp(0.0, 1.0)
        heat = torch.stack([heat_r, heat_g, heat_b], dim=-1)
        colors[valid_mask] = heat[valid_mask]
        return colors

    def _figure_to_tensor(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
        fig.clear()
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def _render_matplotlib_pointcloud(
        self,
        points,
        colors,
        valid_mask=None,
        canvas_hw=(256, 256),
        max_points=12000,
        point_size=4.0,
        view_mode="front",
    ):
        points = points.detach().float().cpu()
        colors = colors.detach().float().cpu().clamp(0.0, 1.0)
        if valid_mask is None:
            valid_mask = torch.ones(points.shape[0], dtype=torch.bool)
        else:
            valid_mask = valid_mask.detach().bool().cpu()

        if int(valid_mask.sum().item()) == 0:
            return torch.full((3, canvas_hw[0], canvas_hw[1]), 0.06)

        points = points[valid_mask]
        colors = colors[valid_mask]
        if points.shape[0] > max_points:
            keep_idx = torch.linspace(0, points.shape[0] - 1, max_points).long()
            points = points[keep_idx]
            colors = colors[keep_idx]

        plot_points = torch.stack([points[:, 0], points[:, 2], -points[:, 1]], dim=-1)
        mins = plot_points.min(dim=0)[0]
        maxs = plot_points.max(dim=0)[0]
        center = 0.5 * (mins + maxs)
        ranges = (maxs - mins).clamp_min(1e-3)
        radius = 0.55 * float(ranges.max())

        canvas_h, canvas_w = canvas_hw
        fig = Figure(figsize=(canvas_w / 100.0, canvas_h / 100.0), dpi=100)
        fig.patch.set_facecolor((0.02, 0.02, 0.02))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor((0.02, 0.02, 0.02))
        ax.scatter(
            plot_points[:, 0].numpy(),
            plot_points[:, 1].numpy(),
            plot_points[:, 2].numpy(),
            c=colors.numpy(),
            s=point_size,
            depthshade=False,
            linewidths=0.0,
        )
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        if view_mode == "front":
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
            ax.set_box_aspect((1.0, 0.35, 2.0))
            ax.view_init(elev=6.0, azim=-90.0)
        elif view_mode == "iso":
            try:
                ax.set_proj_type("persp")
            except Exception:
                pass
            ax.set_box_aspect((1.0, 1.0, 2.0))
            ax.view_init(elev=20.0, azim=35.0)
        else:
            raise ValueError(f"Unsupported pointcloud view_mode: {view_mode}")
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        return self._figure_to_tensor(fig)

    def _render_matplotlib_voxel_points(
        self,
        centers,
        feat_map,
        valid_mask,
        color_mode="pca",
        canvas_hw=(256, 256),
        point_size=14.0,
        max_points=16000,
    ):
        centers = centers.detach().float().cpu()
        feat_map = feat_map.detach().float().cpu()
        valid_mask = (valid_mask.detach().float().cpu() > 0.5)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask[0]

        if int(valid_mask.sum().item()) == 0:
            return torch.full((3, canvas_hw[0], canvas_hw[1]), 0.06)

        flat_feat = rearrange(feat_map, "c d h w -> (d h w) c")
        flat_centers = rearrange(centers, "d h w c -> (d h w) c")
        flat_valid = valid_mask.reshape(-1)
        if color_mode == "pca":
            flat_colors = self._feature_pca_colors(flat_feat, flat_valid)
        elif color_mode == "norm":
            flat_colors = self._feature_norm_colors(flat_feat, flat_valid)
        else:
            raise ValueError(f"Unsupported voxel color_mode: {color_mode}")

        return self._render_matplotlib_pointcloud(
            flat_centers,
            flat_colors,
            valid_mask=flat_valid,
            canvas_hw=canvas_hw,
            max_points=max_points,
            point_size=point_size,
            view_mode="front",
        )

    def _build_pointcloud_pca_3d_vis_batch(
        self,
        points_batch,
        feat_batch,
        valid_mask_batch=None,
        max_frames=5,
    ):
        points_batch = points_batch.detach()
        feat_batch = feat_batch.detach()
        if valid_mask_batch is not None:
            valid_mask_batch = valid_mask_batch.detach()
            num_frames = min(max_frames, points_batch.shape[0], valid_mask_batch.shape[0])
        else:
            num_frames = min(max_frames, points_batch.shape[0])

        vis_frames = []
        for idx in range(num_frames):
            points = points_batch[idx]
            feats = feat_batch[idx]
            valid_mask = valid_mask_batch[idx] if valid_mask_batch is not None else None
            colors = self._feature_pca_colors(feats, valid_mask)
            vis_frames.append(
                self._render_matplotlib_pointcloud(
                    points,
                    colors,
                    valid_mask=valid_mask,
                    point_size=5.0,
                    view_mode="front",
                )
            )
        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _build_voxel_3d_vis_batch(
        self,
        centers_batch,
        feat_batch,
        valid_mask_batch,
        color_mode="pca",
        max_frames=5,
    ):
        centers_batch = centers_batch.detach()
        feat_batch = feat_batch.detach()
        valid_mask_batch = valid_mask_batch.detach()
        num_frames = min(max_frames, centers_batch.shape[0], feat_batch.shape[0], valid_mask_batch.shape[0])

        vis_frames = []
        for idx in range(num_frames):
            vis_frames.append(
                self._render_matplotlib_voxel_points(
                    centers_batch[idx],
                    feat_batch[idx],
                    valid_mask_batch[idx],
                    color_mode=color_mode,
                    point_size=14.0,
                )
            )
        if not vis_frames:
            return None
        return torch.stack(vis_frames, dim=0)

    def _build_uniform_voxel_centers(self, x_min, y_min, z_min, voxel_size, d, h, w):
        device = x_min.device
        x_step = voxel_size * (self.grid_h / float(h))
        y_step = voxel_size * (self.grid_d / float(d))
        z_step = voxel_size * (self.grid_w / float(w))

        d_centers = (torch.arange(d, device=device, dtype=torch.float32) + 0.5).view(1, d, 1, 1)
        h_centers = (torch.arange(h, device=device, dtype=torch.float32) + 0.5).view(1, 1, h, 1)
        w_centers = (torch.arange(w, device=device, dtype=torch.float32) + 0.5).view(1, 1, 1, w)

        voxel_x = x_min.view(-1, 1, 1, 1) + h_centers * x_step.view(-1, 1, 1, 1)
        voxel_y = y_min.view(-1, 1, 1, 1) + d_centers * y_step.view(-1, 1, 1, 1)
        voxel_z = z_min.view(-1, 1, 1, 1) + w_centers * z_step.view(-1, 1, 1, 1)
        voxel_x = voxel_x.expand(-1, d, h, w)
        voxel_y = voxel_y.expand(-1, d, h, w)
        voxel_z = voxel_z.expand(-1, d, h, w)
        return torch.stack([voxel_x, voxel_y, voxel_z], dim=-1)

    def _resize_voxel_valid_mask(self, valid_mask, output_size):
        return F.adaptive_max_pool3d(valid_mask.float(), output_size=output_size)

    def _export_alignment_debug(self, label_value, aux_dict):
        save_root = self._get_voxel_debug_export_root()
        os.makedirs(save_root, exist_ok=True)
        label_str = str(int(label_value)) if torch.is_tensor(label_value) else str(label_value)

        triptych = aux_dict["alignment_triptych"][0].detach().float().cpu().clamp(0, 1)
        triptych_np = (triptych.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(triptych_np).save(
            os.path.join(save_root, f"seq0_label{label_str}_alignment_triptych.png")
        )

        self._save_ascii_ply(
            aux_dict["canonical_points"][0],
            aux_dict["point_colors"][0],
            os.path.join(save_root, f"seq0_label{label_str}_canonical_points.ply"),
        )

        occ_mask = aux_dict["voxel_occ"][0, 0] > 0.5
        voxel_centers = aux_dict["voxel_centers"][0][occ_mask]
        voxel_colors = aux_dict["voxel_colors"][0][occ_mask]
        if voxel_centers.numel() > 0:
            self._save_ascii_ply(
                voxel_centers,
                voxel_colors,
                os.path.join(save_root, f"seq0_label{label_str}_occupied_voxels.ply"),
            )

    def build_voxel_volume(
        self,
        human_feat,
        human_mask,
        pose_out,
        pred_verts,
        pred_cam_t,
        global_rot,
        cam_int_src,
        target_h,
        target_w,
    ):
        src_idx_map, _ = self.get_source_vertex_index_map(
            pred_verts,
            pred_cam_t,
            cam_int_src,
            self.sils_size * 2,
            self.sils_size,
            target_h,
            target_w,
        )
        vertex_feat, vertex_valid, _ = self.accumulate_pixel_features_to_vertices_with_counts(
            human_feat,
            src_idx_map,
            pred_verts.shape[1],
            valid_pixel_mask=human_mask.squeeze(1) > 0.5,
        )

        branch_geo = self.build_branch_geometry(self.voxel_branch_cfg, pose_out)
        canonical_points, _, _ = self.rotate_branch_geometry(
            branch_geo["verts"],
            branch_geo["keypoints"],
            global_rot,
            branch_geo["yaw"],
            branch_geo["apply_global_rot_alignment"],
        )

        cam_int_tgt, cam_t_tgt = self.build_target_camera(
            human_feat.shape[0],
            human_feat.device,
            target_h,
            target_w,
        )
        target_mask_2d = self.project_vertices_to_mask(
            canonical_points,
            cam_t_tgt,
            cam_int_tgt,
            self.grid_d,
            self.grid_h,
            target_h,
            target_w,
        )

        v_cam = canonical_points + cam_t_tgt.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z = z.clamp(min=1e-3)

        fx = cam_int_tgt[:, 0, 0].unsqueeze(1)
        fy = cam_int_tgt[:, 1, 1].unsqueeze(1)
        cx = cam_int_tgt[:, 0, 2].unsqueeze(1)
        cy = cam_int_tgt[:, 1, 2].unsqueeze(1)

        z_ref = cam_t_tgt[:, 2:3]
        x_min = ((0.0 - cx) / fx) * z_ref
        x_max = ((float(target_w) - cx) / fx) * z_ref
        y_min = ((0.0 - cy) / fy) * z_ref
        y_max = ((float(target_h) - cy) / fy) * z_ref

        voxel_size_x = (x_max - x_min) / float(self.grid_h)
        voxel_size_y = (y_max - y_min) / float(self.grid_d)
        voxel_size = 0.5 * (voxel_size_x + voxel_size_y)

        if self.depth_center_mode == "bbox_mid":
            z_center = 0.5 * (z.amin(dim=1, keepdim=True) + z.amax(dim=1, keepdim=True))
        elif self.depth_center_mode == "z_mean":
            z_center = z.mean(dim=1, keepdim=True)
        elif self.depth_center_mode == "camera_ref":
            z_center = z_ref
        else:
            raise ValueError(f"Unsupported voxel depth_center_mode: {self.depth_center_mode}")
        z_center = z_center + self.depth_center_offset
        z_half_extent = voxel_size * (self.grid_w / 2.0)
        z_min = z_center - z_half_extent
        z_max = z_center + z_half_extent

        u_idx = torch.floor((x - x_min) / voxel_size).long().clamp(0, self.grid_h - 1)
        v_idx = torch.floor((y - y_min) / voxel_size).long().clamp(0, self.grid_d - 1)
        z_idx = torch.floor((z - z_min) / voxel_size).long().clamp(0, self.grid_w - 1)

        volume_feat, voxel_occ = self.scatter_vertices_to_volume(
            vertex_feat, vertex_valid, v_idx, u_idx, z_idx
        )
        volume_feat = volume_feat * voxel_occ.to(dtype=volume_feat.dtype)

        color_grid = self._generate_color_grid(human_feat.shape[0], self.grid_d, self.grid_h, human_feat.device)
        point_colors = color_grid.permute(0, 2, 3, 1)[
            torch.arange(human_feat.shape[0], device=human_feat.device).unsqueeze(1),
            v_idx,
            u_idx,
        ]
        point_projection = self._paint_points_to_canvas(v_idx, u_idx, point_colors)
        voxel_front_mask = voxel_occ.amax(dim=-1)
        alignment_triptych = self._make_alignment_triptych(
            target_mask_2d,
            point_projection,
            voxel_front_mask,
        )

        d_centers = (
            torch.arange(self.grid_d, device=human_feat.device, dtype=torch.float32) + 0.5
        ).view(1, self.grid_d, 1, 1)
        h_centers = (
            torch.arange(self.grid_h, device=human_feat.device, dtype=torch.float32) + 0.5
        ).view(1, 1, self.grid_h, 1)
        w_centers = (
            torch.arange(self.grid_w, device=human_feat.device, dtype=torch.float32) + 0.5
        ).view(1, 1, 1, self.grid_w)

        voxel_x = x_min.view(-1, 1, 1, 1) + h_centers * voxel_size.view(-1, 1, 1, 1)
        voxel_y = y_min.view(-1, 1, 1, 1) + d_centers * voxel_size.view(-1, 1, 1, 1)
        voxel_z = z_min.view(-1, 1, 1, 1) + w_centers * voxel_size.view(-1, 1, 1, 1)
        voxel_x = voxel_x.expand(-1, self.grid_d, self.grid_h, self.grid_w)
        voxel_y = voxel_y.expand(-1, self.grid_d, self.grid_h, self.grid_w)
        voxel_z = voxel_z.expand(-1, self.grid_d, self.grid_h, self.grid_w)
        voxel_centers = torch.stack([voxel_x, voxel_y, voxel_z], dim=-1) - cam_t_tgt[:, None, None, None, :]

        voxel_colors = color_grid.permute(0, 2, 3, 1).unsqueeze(3).expand(-1, -1, -1, self.grid_w, -1)

        aux_dict = {
            "target_mask_2d": target_mask_2d,
            "voxel_front_mask": voxel_front_mask,
            "point_projection": point_projection,
            "alignment_triptych": alignment_triptych,
            "canonical_points": canonical_points,
            "point_colors": point_colors,
            "vertex_feat": vertex_feat,
            "vertex_valid": vertex_valid,
            "voxel_centers": voxel_centers,
            "voxel_colors": voxel_colors,
            "voxel_occ": voxel_occ,
            "x_min": x_min,
            "y_min": y_min,
            "z_min": z_min,
            "voxel_size": voxel_size,
        }
        return volume_feat, voxel_occ, aux_dict

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        should_log_pca_vis = self.debug_pca_vis and self._should_log_visual_summary()
        all_outs = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        target_mask_2d_summary = None
        pointcloud_pca_before_summary = None
        voxel_pca_before_summary = None
        voxel_pca_after_summary = None
        voxel_norm_before_summary = None
        voxel_norm_after_summary = None
        export_aux = None

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
                    pred_verts,
                    pred_cam_t,
                    cam_int_src,
                    h_feat,
                    w_feat,
                    target_h,
                    target_w,
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

                sub_app = torch.cat(sub_feats, dim=-1)
                curr_dim = self.f4_dim * len(sub_feats)
                sub_app = partial(nn.LayerNorm, eps=1e-6)(
                    curr_dim,
                    elementwise_affine=False,
                )(sub_app)
                sub_app = rearrange(sub_app, "b (h w) c -> b c h w", h=h_feat).contiguous()
                processed_feat_list.append(self.HumanSpace_Conv[i](sub_app))

            human_feat = torch.cat(processed_feat_list, dim=1)
            human_mask = self.preprocess(
                generated_mask,
                self.sils_size * 2,
                self.sils_size,
            ).detach().clone()
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)

            voxel_feat, _, aux_dict = self.build_voxel_volume(
                human_feat,
                human_mask.float(),
                pose_out,
                pred_verts,
                pred_cam_t,
                global_rot,
                cam_int_src,
                target_h,
                target_w,
            )

            voxel_feat_6d = rearrange(
                voxel_feat.view(n, s, *voxel_feat.shape[1:]),
                "n s c d h w -> n c s d h w",
            ).contiguous()

            debug_test_1 = should_log_pca_vis and (chunk_idx == num_rgb_chunks - 1)
            if debug_test_1:
                outs, gait_debug = self.Gait_Net.test_1(voxel_feat_6d, return_debug=True)
            elif self.training:
                outs = torch.utils.checkpoint.checkpoint(
                    self.Gait_Net.test_1,
                    voxel_feat_6d,
                    use_reentrant=False,
                )
            else:
                outs = self.Gait_Net.test_1(voxel_feat_6d)
            all_outs.append(outs)

            if chunk_idx == num_rgb_chunks - 1:
                target_mask_2d_summary = aux_dict["target_mask_2d"][:5].float()
                export_aux = aux_dict
                if debug_test_1:
                    pointcloud_pca_vis = []
                    voxel_pca_before_vis = []
                    voxel_pca_after_vis = []
                    voxel_norm_before_vis = []
                    voxel_norm_after_vis = []

                    pre_voxel_occ = rearrange(aux_dict["voxel_occ"], "(n s) c d h w -> n c s d h w", n=n, s=s)
                    pre_voxel_centers = aux_dict["voxel_centers"]
                    pre_vertex_feat = rearrange(aux_dict["vertex_feat"], "(n s) v c -> n s v c", n=n, s=s)
                    pre_vertex_valid = rearrange(aux_dict["vertex_valid"], "(n s) v -> n s v", n=n, s=s)
                    canonical_points = rearrange(aux_dict["canonical_points"], "(n s) v c -> n s v c", n=n, s=s)

                    for i in range(self.num_FPN):
                        point_feat_chunk = torch.chunk(pre_vertex_feat, self.num_FPN, dim=-1)[i]
                        pointcloud_pca_vis.append(
                            self._build_pointcloud_pca_3d_vis_batch(
                                rearrange(canonical_points, "n s v c -> (n s) v c").contiguous()[:5],
                                rearrange(point_feat_chunk, "n s v c -> (n s) v c").contiguous()[:5],
                                rearrange(pre_vertex_valid, "n s v -> (n s) v").contiguous()[:5],
                            )
                        )

                        pre_chunk = torch.chunk(voxel_feat_6d, self.num_FPN, dim=1)[i]
                        pre_chunk_flat = rearrange(pre_chunk, "n c s d h w -> (n s) c d h w").contiguous()
                        pre_occ_flat = rearrange(pre_voxel_occ, "n c s d h w -> (n s) c d h w").contiguous()
                        voxel_pca_before_vis.append(
                            self._build_voxel_3d_vis_batch(
                                pre_voxel_centers[:5],
                                pre_chunk_flat[:5],
                                pre_occ_flat[:5],
                                color_mode="pca",
                            )
                        )
                        voxel_norm_before_vis.append(
                            self._build_voxel_3d_vis_batch(
                                pre_voxel_centers[:5],
                                pre_chunk_flat[:5],
                                pre_occ_flat[:5],
                                color_mode="norm",
                            )
                        )

                        out_chunk = torch.chunk(outs, self.num_FPN, dim=1)[i]
                        out_chunk_flat = rearrange(out_chunk, "n c s d h w -> (n s) c d h w").contiguous()
                        out_d, out_h, out_w = out_chunk_flat.shape[-3:]
                        post_centers = self._build_uniform_voxel_centers(
                            aux_dict["x_min"],
                            aux_dict["y_min"],
                            aux_dict["z_min"],
                            aux_dict["voxel_size"],
                            out_d,
                            out_h,
                            out_w,
                        )
                        post_occ = self._resize_voxel_valid_mask(
                            aux_dict["voxel_occ"],
                            (out_d, out_h, out_w),
                        )
                        voxel_pca_after_vis.append(
                            self._build_voxel_3d_vis_batch(
                                post_centers[:5],
                                out_chunk_flat[:5],
                                post_occ[:5],
                                color_mode="pca",
                            )
                        )
                        voxel_norm_after_vis.append(
                            self._build_voxel_3d_vis_batch(
                                post_centers[:5],
                                out_chunk_flat[:5],
                                post_occ[:5],
                                color_mode="norm",
                            )
                        )

                    pointcloud_pca_before_summary = self._tile_fpn_frame_vis(pointcloud_pca_vis)
                    voxel_pca_before_summary = self._tile_fpn_frame_vis(voxel_pca_before_vis)
                    voxel_pca_after_summary = self._tile_fpn_frame_vis(voxel_pca_after_vis)
                    voxel_norm_before_summary = self._tile_fpn_frame_vis(voxel_norm_before_vis)
                    voxel_norm_after_summary = self._tile_fpn_frame_vis(voxel_norm_after_vis)

        embed_list, log_list = self.Gait_Net.test_2(torch.cat(all_outs, dim=2), seqL)

        if self._should_export_debug() and export_aux is not None:
            self._export_alignment_debug(labs[0], export_aux)

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
            if target_mask_2d_summary is not None:
                retval["visual_summary"]["image/target_mask_2d"] = target_mask_2d_summary
            if pointcloud_pca_before_summary is not None:
                retval["visual_summary"]["image/pointcloud_pca_before_cnn_3d"] = pointcloud_pca_before_summary.float()
            if voxel_pca_before_summary is not None:
                retval["visual_summary"]["image/voxel_pca_before_cnn_3d"] = voxel_pca_before_summary.float()
            if voxel_pca_after_summary is not None:
                retval["visual_summary"]["image/voxel_pca_after_cnn_3d"] = voxel_pca_after_summary.float()
            if voxel_norm_before_summary is not None:
                retval["visual_summary"]["image/voxel_norm_before_cnn_3d"] = voxel_norm_before_summary.float()
            if voxel_norm_after_summary is not None:
                retval["visual_summary"]["image/voxel_norm_after_cnn_3d"] = voxel_norm_after_summary.float()
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
