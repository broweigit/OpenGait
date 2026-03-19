import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from functools import partial
from torch.nn import functional as F

from ..base_model import BaseModel
from .BigGait_utils.BigGait_GaitBase import *
from utils import list2var, np2var


class ResizeToHW(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)


class BiggerGait__DINOv2__Projection_Mask_Based(BaseModel):
    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.f4_dim = model_cfg['source_dim']
        self.num_unknown = model_cfg["num_unknown"]

        self.total_layer_num = model_cfg["total_layer_num"]
        self.group_layer_num = model_cfg["group_layer_num"]
        self.head_num = model_cfg["head_num"]
        assert self.total_layer_num % self.group_layer_num == 0
        assert (self.total_layer_num // self.group_layer_num) % self.head_num == 0
        self.num_FPN = self.total_layer_num // self.group_layer_num

        self.gradient_checkpointing = model_cfg.get("gradient_checkpointing", False)
        self.chunk_size = model_cfg.get("chunk_size", 96)
        self.sync_hflip_prob = model_cfg.get("sync_hflip_prob", 0.5)
        self.enable_timing = model_cfg.get("enable_timing", True)

        self.Gait_Net = Baseline_Share(model_cfg)
        self.HumanSpace_Conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.f4_dim * self.group_layer_num, affine=False),
                nn.Conv2d(self.f4_dim * self.group_layer_num, self.f4_dim // 2, kernel_size=1),
                nn.BatchNorm2d(self.f4_dim // 2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.f4_dim // 2, self.num_unknown, kernel_size=1),
                ResizeToHW((self.sils_size * 2, self.sils_size)),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid()
            ) for _ in range(self.num_FPN)
        ])

    def init_DINOv2(self):
        from transformers import Dinov2Config, Dinov2Model
        config = Dinov2Config.from_pretrained(self.pretrained_lvm + "/config.json")
        self.Backbone = Dinov2Model.from_pretrained(
            self.pretrained_lvm,
            config=config,
        )
        if self.training and self.gradient_checkpointing:
            self.Backbone.gradient_checkpointing_enable()
            self.msg_mgr.log_info("Gradient Checkpointing Enabled for DINOv2!")
        self.Backbone.cpu()
        self.msg_mgr.log_info(f'load model from: {self.pretrained_lvm}')

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('Expect Backbone Count: {:.5f}M'.format(n_parameters / 1e6))

        self.init_DINOv2()
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)

        n_parameters = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info('All Backbone Count: {:.5f}M'.format(n_parameters / 1e6))
        self.msg_mgr.log_info("=> init successfully")

    def inputs_pretreament(self, inputs):
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(
                    len(seqs_batch), len(seq_trfs)
                )
            )

        requires_grad = bool(self.training)
        rgb = np2var(
            np.asarray([seq_trfs[0](seq) for seq in seqs_batch[0]]),
            requires_grad=requires_grad,
        ).float()

        # Keep the offline SAM decoder frames as Python objects so we can unpack
        # pred_vertices / pred_cam_t / cam_int inside forward.
        sam_decoder = [list(seq_trfs[1](seq)) for seq in seqs_batch[1]]

        labs = list2var(labs_batch).long()
        seqL = np2var(seqL_batch).int() if seqL_batch is not None else None
        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            rgb = rgb[:, :seqL_sum]
            sam_decoder = [seq[:seqL_sum] for seq in sam_decoder]

        return [rgb, sam_decoder], labs, typs_batch, vies_batch, seqL

    def preprocess(self, sils, image_size, mode='bilinear'):
        return F.interpolate(sils, (image_size * 2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    def _stack_sam_frames(self, sam_frames, device):
        pred_vertices = torch.stack([
            torch.as_tensor(frame["pred_vertices"], dtype=torch.float32) for frame in sam_frames
        ], dim=0).to(device)
        pred_cam_t = torch.stack([
            torch.as_tensor(frame["pred_cam_t"], dtype=torch.float32) for frame in sam_frames
        ], dim=0).to(device)
        cam_int = torch.stack([
            torch.as_tensor(frame["cam_int"], dtype=torch.float32) for frame in sam_frames
        ], dim=0).to(device)
        return pred_vertices, pred_cam_t, cam_int

    def _resize_cam_int(self, cam_int, target_h, target_w):
        cam_int = cam_int.clone()
        src_w = (cam_int[:, 0, 2] * 2.0).clamp(min=1.0)
        src_h = (cam_int[:, 1, 2] * 2.0).clamp(min=1.0)
        scale_x = float(target_w) / src_w
        scale_y = float(target_h) / src_h

        cam_int[:, 0, 0] *= scale_x
        cam_int[:, 0, 2] *= scale_x
        cam_int[:, 1, 1] *= scale_y
        cam_int[:, 1, 2] *= scale_y
        return cam_int

    def project_vertices_to_mask(self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w):
        bsz = vertices.shape[0]
        device = vertices.device

        v_cam = vertices + cam_t.unsqueeze(1)
        x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
        z = z.clamp(min=1e-3)

        fx = cam_int[:, 0, 0].unsqueeze(1)
        fy = cam_int[:, 1, 1].unsqueeze(1)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
        u = (x / z) * fx + cx
        v = (y / z) * fy + cy

        u_feat = (u / target_w * w_feat).long().clamp(0, w_feat - 1)
        v_feat = (v / target_h * h_feat).long().clamp(0, h_feat - 1)

        mask = torch.zeros(bsz, 1, h_feat, w_feat, device=device)
        flat_indices = v_feat * w_feat + u_feat
        ones = torch.ones_like(flat_indices, dtype=mask.dtype)
        mask.view(bsz, -1).scatter_(1, flat_indices, ones)
        return mask

    def _apply_sequence_hflip(self, x, flip_flags, seq_len):
        if x is None or not torch.any(flip_flags):
            return x
        flat_flip = flip_flags.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        x = x.clone()
        x[flat_flip] = torch.flip(x[flat_flip], dims=[-1])
        return x

    def _perf_now(self, device):
        if self.enable_timing and isinstance(device, torch.device) and device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def forward(self, inputs):
        timing_info = {
            'model_hflip': 0.0,
            'model_dino': 0.0,
            'model_sam_unpack': 0.0,
            'model_project_mask': 0.0,
            'model_humanspace': 0.0,
            'model_gait_head': 0.0,
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

        all_outs = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = self.image_size // 7, self.image_size // 14

        seq_start = 0
        for rgb_chunk, chunk_len in zip(rgb_chunks, chunk_lengths):
            seq_end = seq_start + chunk_len
            sam_chunk = [seq[seq_start:seq_end] for seq in sam_decoder]
            seq_start = seq_end

            n, s, c, h, w = rgb_chunk.size()
            flat_sam_frames = [frame for seq in sam_chunk for frame in seq]
            if len(flat_sam_frames) != n * s:
                raise RuntimeError(f"SAM frame count mismatch: expected {n * s}, got {len(flat_sam_frames)}")

            with torch.no_grad():
                stage_start = self._perf_now(rgb.device)
                rgb_img = rearrange(rgb_chunk, 'n s c h w -> (n s) c h w').contiguous()
                rgb_img = self._apply_sequence_hflip(rgb_img, flip_flags, s)
                timing_info['model_hflip'] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                outs = self.preprocess(rgb_img, self.image_size)
                outs = self.Backbone(outs, output_hidden_states=True).hidden_states[1:]

                intermediates = partial(nn.LayerNorm, eps=1e-6)(
                    self.f4_dim * len(outs), elementwise_affine=False
                )(torch.concat(outs, dim=-1))[:, 1:]
                intermediates = rearrange(
                    intermediates.view(n, s, h_feat, w_feat, -1),
                    'n s h w c -> (n s) c h w'
                ).contiguous()
                intermediates = list(torch.chunk(intermediates, self.total_layer_num, dim=1))
                timing_info['model_dino'] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                pred_vertices, pred_cam_t, cam_int = self._stack_sam_frames(flat_sam_frames, rgb.device)
                cam_int = self._resize_cam_int(cam_int, target_h, target_w)
                timing_info['model_sam_unpack'] += self._perf_now(rgb.device) - stage_start

                stage_start = self._perf_now(rgb.device)
                generated_mask = self.project_vertices_to_mask(
                    pred_vertices, pred_cam_t, cam_int, h_feat, w_feat, target_h, target_w
                )
                generated_mask = self._apply_sequence_hflip(generated_mask, flip_flags, s)
                timing_info['model_project_mask'] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            intermediates = [
                torch.cat(intermediates[i:i + self.group_layer_num], dim=1).contiguous()
                for i in range(0, self.total_layer_num, self.group_layer_num)
            ]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            intermediates = torch.concat(intermediates, dim=1)

            human_mask = self.preprocess(generated_mask, self.sils_size).detach().clone()
            intermediates = intermediates * (human_mask > 0.5).to(intermediates)
            intermediates = rearrange(
                intermediates.view(n, s, -1, self.sils_size * 2, self.sils_size),
                'n s c h w -> n c s h w'
            ).contiguous()
            timing_info['model_humanspace'] += self._perf_now(rgb.device) - stage_start

            stage_start = self._perf_now(rgb.device)
            outs = self.Gait_Net.test_1(intermediates)
            all_outs.append(outs)
            timing_info['model_gait_head'] += self._perf_now(rgb.device) - stage_start

        stage_start = self._perf_now(rgb.device)
        embed_list, log_list = self.Gait_Net.test_2(
            torch.cat(all_outs, dim=2),
            seqL,
        )
        timing_info['model_gait_head'] += self._perf_now(rgb.device) - stage_start

        embeddings = torch.concat(embed_list, dim=-1)
        logits = torch.concat(log_list, dim=-1)
        model_end = self._perf_now(rgb.device)
        timed_sum = sum(timing_info.values())
        timing_info['model_misc'] = max(model_end - model_start - timed_sum, 0.0)

        if self.training:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embeddings, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/rgb_img': rgb_img.view(n * s, c, h, w)[:5].float(),
                    'image/generated_3d_mask_lowres': generated_mask.view(n * s, 1, h_feat, w_feat)[:5].float(),
                },
                'inference_feat': {
                    'embeddings': embeddings,
                },
                'timing_info': timing_info,
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {
                    'embeddings': embeddings,
                },
                'timing_info': timing_info,
            }
        return retval
