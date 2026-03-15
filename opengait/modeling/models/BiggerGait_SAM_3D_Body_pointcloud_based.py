import sys

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from functools import partial

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks
from .lidargaitv2_utils import PointNetSetAbstraction, PPPooling, PPPooling_UDP, NetVLAD, farthest_point_sample


class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)


class BiggerGait__SAM3DBody__PointCloud_Based(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)

        self.point_input_num = model_cfg.get("point_input_num", 1024)
        self.point_sample_method = model_cfg.get("point_sample_method", "fps")

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

        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(input_dim, affine=False),
                nn.Conv2d(input_dim, self.f4_dim // 2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim // 2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim // 2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size * 2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])

        self.point_channel = model_cfg["channel"]
        self.point_scale_aware = model_cfg.get("scale_aware", False)
        self.point_normalize_dp = model_cfg.get("normalize_dp", True)
        self.point_sampling = model_cfg.get("sampling", "knn")
        self.point_npoints = model_cfg.get("npoints", [512, 256, 128])
        self.point_nsample = model_cfg.get("nsample", 32)
        self.point_pool = model_cfg.get("pool", "PPP_UDP")
        self.point_bins = model_cfg.get("scale", [32])

        xyz_dim = 4 if self.point_scale_aware else 3
        point_feat_dim = self.num_unknown * self.num_FPN
        c = self.point_channel
        c_out = model_cfg["SeparateFCs"]["in_channels"]

        self.sa1 = PointNetSetAbstraction(
            npoint=self.point_npoints[0], radius=0.1, nsample=self.point_nsample,
            in_channel=point_feat_dim + xyz_dim, mlp=[2 * c, 2 * c, 4 * c],
            group_all=False, sampling=self.point_sampling,
            scale_aware=self.point_scale_aware, normalize_dp=self.point_normalize_dp
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=self.point_npoints[1], radius=0.2, nsample=self.point_nsample,
            in_channel=4 * c + xyz_dim, mlp=[4 * c, 4 * c, 8 * c],
            group_all=False, sampling=self.point_sampling,
            scale_aware=self.point_scale_aware, normalize_dp=self.point_normalize_dp
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=self.point_npoints[2], radius=0.4, nsample=self.point_nsample,
            in_channel=8 * c + xyz_dim, mlp=[8 * c, 8 * c, 16 * c],
            group_all=False, sampling=self.point_sampling,
            scale_aware=self.point_scale_aware, normalize_dp=self.point_normalize_dp
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=16 * c + xyz_dim, mlp=[16 * c, 16 * c, c_out],
            group_all=True, sampling=self.point_sampling,
            scale_aware=self.point_scale_aware, normalize_dp=self.point_normalize_dp
        )

        if self.point_pool == 'VLAD':
            self.pool = NetVLAD(num_clusters=16, dim=c_out, alpha=1.0)
        elif self.point_pool == 'GMaxP':
            self.pool = PPPooling_UDP([1])
        elif self.point_pool == 'PPP_UDP':
            self.pool = PPPooling_UDP(self.point_bins)
        elif self.point_pool == 'PPP_UAP':
            self.pool = PPPooling(scale_aware=False, bin_num=self.point_bins)
        elif self.point_pool == 'PPP_HAP':
            self.pool = PPPooling(scale_aware=True, bin_num=self.point_bins)
        else:
            raise ValueError(f"Unsupported point pool type: {self.point_pool}")

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.init_SAM_Backbone()

    def init_SAM_Backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)

        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise ImportError(f"Cannot import setup_sam_3d_body. Error: {e}")

        self.msg_mgr.log_info("[SAM3D] Loading SAM 3D Body (Encoder + Decoder)...")
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device='cpu')

        self.SAM_Engine = estimator.model
        if hasattr(self.SAM_Engine, 'backbone'):
            raw_backbone = self.SAM_Engine.backbone
        elif hasattr(self.SAM_Engine, 'image_encoder'):
            raw_backbone = self.SAM_Engine.image_encoder
        else:
            raise RuntimeError("Cannot find backbone in SAM Engine")

        if hasattr(raw_backbone, 'encoder'):
            self.Backbone = raw_backbone.encoder
        else:
            self.Backbone = raw_backbone

        self.SAM_Engine.cpu()
        self.intermediate_features = {}
        self.hook_handles = []

        def get_activation(idx_in_list):
            def hook(model, input, output):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if isinstance(output, (list, tuple)):
                    output = output[0]
                self.intermediate_features[idx_in_list] = output
            return hook

        if hasattr(self.Backbone, 'blocks'):
            all_blocks = self.Backbone.blocks
        elif hasattr(self.Backbone, 'layers'):
            all_blocks = self.Backbone.layers
        else:
            raise RuntimeError("Cannot find blocks in Backbone")

        hook_count = 0
        for layer_idx, should_hook in enumerate(self.hook_mask):
            if should_hook:
                handle = all_blocks[layer_idx].register_forward_hook(get_activation(hook_count))
                self.hook_handles.append(handle)
                hook_count += 1

        self.msg_mgr.log_info(f"[SAM3D] Hooked {hook_count} layers.")
        self.SAM_Engine.eval()
        for param in self.SAM_Engine.parameters():
            param.requires_grad = False

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        self.init_SAM_Backbone()
        self.SAM_Engine.eval()
        self.SAM_Engine.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Model Count: {:.5f}M'.format(n_parameters / 1e6))

    def _prepare_dummy_batch(self, image_embeddings, target_h, target_w):
        bsz = image_embeddings.shape[0]
        device = image_embeddings.device

        estimated_focal_length = max(target_h, target_w) * 1.1
        cx, cy = target_w / 2.0, target_h / 2.0

        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(bsz, 3, 3).clone()
        cam_int[:, 0, 0] = estimated_focal_length
        cam_int[:, 1, 1] = estimated_focal_length
        cam_int[:, 0, 2] = cx
        cam_int[:, 1, 2] = cy

        y_grid, x_grid = torch.meshgrid(
            torch.arange(target_h, device=device),
            torch.arange(target_w, device=device),
            indexing='ij'
        )
        ray_x = (x_grid - cx) / estimated_focal_length
        ray_y = (y_grid - cy) / estimated_focal_length
        ray_cond = torch.stack([ray_x, ray_y], dim=0).unsqueeze(0).expand(bsz, 2, target_h, target_w)

        bbox_scale = torch.tensor([max(target_h, target_w)], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 1)
        bbox_center = torch.tensor([cx, cy], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 2)
        img_size = torch.tensor([float(target_w), float(target_h)], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 2)
        affine_trans = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, 2, 3)

        return {
            "img": torch.zeros(bsz, 1, 3, target_h, target_w, device=device),
            "ori_img_size": img_size,
            "img_size": img_size,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_int": cam_int,
            "affine_trans": affine_trans,
            "ray_cond": ray_cond,
        }

    def get_source_vertex_index_map(self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w):
        bsz, num_verts, _ = vertices.shape
        device = vertices.device

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam.unbind(-1)
        z_safe = z.clamp(min=1e-3)

        fx, fy = cam_int[:, 0, 0].unsqueeze(1), cam_int[:, 1, 1].unsqueeze(1)
        cx, cy = cam_int[:, 0, 2].unsqueeze(1), cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z_safe) * fx + cx
        v = (y / z_safe) * fy + cy

        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)
        flat_pixel_indices = v_feat * w_feat + u_feat

        depth_map_flat = torch.full((bsz, h_feat * w_feat), 1e6, device=device)
        depth_map_flat.scatter_reduce_(1, flat_pixel_indices, z, reduce='amin', include_self=False)

        min_depth_per_vertex = torch.gather(depth_map_flat, 1, flat_pixel_indices)
        is_visible = z < (min_depth_per_vertex + 1e-4)

        index_map_flat = torch.full((bsz, h_feat * w_feat), -1, dtype=torch.long, device=device)
        vertex_indices = torch.arange(num_verts, device=device).unsqueeze(0).expand(bsz, -1)

        mask_flat = is_visible.reshape(-1)
        batch_offsets = torch.arange(bsz, device=device).unsqueeze(1) * (h_feat * w_feat)
        global_pixel_indices = (flat_pixel_indices + batch_offsets).reshape(-1)

        valid_pixel_indices = global_pixel_indices[mask_flat]
        valid_vertex_indices = vertex_indices.reshape(-1)[mask_flat]

        index_map_global = index_map_flat.reshape(-1)
        index_map_global[valid_pixel_indices] = valid_vertex_indices
        return index_map_global.reshape(bsz, h_feat, w_feat), depth_map_flat.reshape(bsz, 1, h_feat, w_feat)

    def preprocess(self, sils, h, w, mode='bilinear'):
        return F.interpolate(sils, (h, w), mode=mode, align_corners=False)

    def sample_points_to_fixed_count(self, xyz, feats):
        num_points = xyz.shape[0]
        if num_points == self.point_input_num:
            return xyz, feats

        if num_points > self.point_input_num:
            if self.point_sample_method == "fps":
                sample_idx = farthest_point_sample(xyz[:, :3].unsqueeze(0), self.point_input_num)[0]
            elif self.point_sample_method == "random":
                sample_idx = torch.randperm(num_points, device=xyz.device)[:self.point_input_num]
            else:
                raise ValueError(f"Unsupported point_sample_method: {self.point_sample_method}")
            return xyz[sample_idx], feats[sample_idx]

        if num_points == 0:
            xyz = xyz.new_zeros((1, xyz.shape[-1]))
            feats = feats.new_zeros((1, feats.shape[-1]))
            num_points = 1

        pad_idx = torch.randint(num_points, (self.point_input_num - num_points,), device=xyz.device)
        xyz = torch.cat([xyz, xyz[pad_idx]], dim=0)
        feats = torch.cat([feats, feats[pad_idx]], dim=0)
        return xyz, feats

    def build_feature_point_cloud(self, human_feat, pred_verts, pred_keypoints, pred_cam_t, cam_int, target_h, target_w):
        bsz, feat_dim, feat_h, feat_w = human_feat.shape
        src_idx_map, depth_map = self.get_source_vertex_index_map(
            pred_verts, pred_cam_t, cam_int, feat_h, feat_w, target_h, target_w
        )

        flat_feat = rearrange(human_feat, 'b c h w -> b (h w) c')
        flat_idx = src_idx_map.view(bsz, -1)
        valid_mask = flat_idx >= 0

        safe_idx = flat_idx.clamp(min=0)
        flat_xyz = torch.gather(pred_verts, 1, safe_idx.unsqueeze(-1).expand(-1, -1, 3))
        midhip = (pred_keypoints[:, 9] + pred_keypoints[:, 10]) / 2.0
        flat_xyz = flat_xyz - midhip.unsqueeze(1)

        xyz_points, feat_points = [], []
        for b in range(bsz):
            valid_idx = valid_mask[b].nonzero(as_tuple=False).flatten()
            xyz_b = flat_xyz[b, valid_idx]
            feat_b = flat_feat[b, valid_idx]

            xyz_b, feat_b = self.sample_points_to_fixed_count(xyz_b, feat_b)

            if self.point_scale_aware:
                raw_z = xyz_b[:, 2:3] - xyz_b[:, 2:3].min(dim=0, keepdim=True)[0]
                xyz_b = torch.cat([xyz_b, raw_z], dim=-1)

            xyz_points.append(xyz_b)
            feat_points.append(feat_b)

        return torch.stack(xyz_points, dim=0), torch.stack(feat_points, dim=0), depth_map

    def encode_points(self, xyz_points, feat_points):
        xyz = rearrange(xyz_points, 'b n c -> b c n')
        points = rearrange(feat_points, 'b n c -> b c n')

        l1_xyz, l1_points = self.sa1(xyz, points)
        l1_points = torch.max(l1_points, dim=-2)[0]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = torch.max(l2_points, dim=-2)[0]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, dim=-2)[0]

        _, l4_points = self.sa4(l3_xyz, l3_points)
        return self.pool(l4_points, l3_xyz)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1)
        chunk_descs = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16

        for _, rgb_img in enumerate(rgb_chunks):
            n, s, c, h, w = rgb_img.size()
            rgb_img = rearrange(rgb_img, 'n s c h w -> (n s) c h w').contiguous()
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

                with torch.amp.autocast(enabled=False, device_type='cuda'):
                    _, pose_outs = self.SAM_Engine.forward_decoder(
                        image_embeddings=sam_emb,
                        keypoints=dummy_kp,
                        condition_info=cond_info,
                        batch=dummy_batch
                    )

                pose_out = pose_outs[-1]
                pred_verts = pose_out['pred_vertices']
                pred_cam_t = pose_out['pred_cam_t']
                pred_keypoints = pose_out['pred_keypoints_3d']
                cam_int_src = dummy_batch['cam_int']

                features_to_use = []
                for i in range(len(self.hook_handles)):
                    feat = self.intermediate_features[i]
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :]
                    features_to_use.append(feat)

            processed_feat_list = []
            step = len(features_to_use) // self.num_FPN
            for i in range(self.num_FPN):
                if self.hook_sample_type == 'interleave':
                    sub_feats = features_to_use[i::self.num_FPN]
                elif self.hook_sample_type == 'chunk':
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
                sub_app = rearrange(sub_app, 'b (h w) c -> b c h w', h=h_feat).contiguous()
                processed_feat_list.append(self.HumanSpace_Conv[i](sub_app))

            human_feat = torch.concat(processed_feat_list, dim=1)
            xyz_points, feat_points, depth_map = self.build_feature_point_cloud(
                human_feat, pred_verts, pred_keypoints, pred_cam_t, cam_int_src, target_h, target_w
            )
            point_desc = self.encode_points(xyz_points, feat_points)
            chunk_descs.append(rearrange(point_desc, '(n s) feat p -> n s feat p', n=n, s=s).contiguous())

        seq_desc = torch.cat(chunk_descs, dim=1)
        feat = seq_desc.max(1)[0]
        embed = self.FCs(feat)
        _, logits = self.BNNecks(embed)

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/rgb_img': rgb_img.view(n * s, c, h, w)[:5].float(),
                'image/generated_3d_mask_interpolated': (depth_map < 1e5).float().view(n * s, 1, self.sils_size * 2, self.sils_size)[:5].float(),
            },
            'inference_feat': {
                'embeddings': embed,
            }
        }
        return retval
