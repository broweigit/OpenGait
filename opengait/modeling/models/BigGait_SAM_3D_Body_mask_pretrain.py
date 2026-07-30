import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel


class infoDistillation(nn.Module):
    """The original BigGait two-class information bottleneck."""

    def __init__(self, source_dim, target_dim, p, softmax, Relu=False, Up=True):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim // 2),
                nn.BatchNorm1d(source_dim // 2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim // 2, target_dim),
            )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim // 2),
                    nn.BatchNorm1d(source_dim // 2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim // 2, source_dim),
                )
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
        else:
            d_x = torch.sigmoid(self.bn_t(d_x))
        if not self.Up:
            return d_x, None
        u_x = self.up_sampling(d_x)
        return d_x, self.mse(u_x, x)


class BigGait__SAM3DBody_MaskBranch_Pretrain(BaseModel):
    """Train only BigGait's mask branch on frozen SAM3D DINOv3 tokens."""

    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.image_size = int(model_cfg.get("image_size", 256))
        self.sils_size = int(model_cfg.get("sils_size", 32))
        self.chunk_size = int(model_cfg.get("chunk_size", 8))
        self.f4_dim = int(model_cfg["Mask_Branch"]["source_dim"])
        self.mask_dim = int(model_cfg["Mask_Branch"]["target_dim"])
        self.hook_layer = int(model_cfg.get("hook_layer", 31))
        self.mask_branch_export_path = model_cfg.get(
            "mask_branch_export_path",
            "pretrained_LVMs/MaskBranch_sam3dbody_ccpg_iter200.pt",
        )
        self.Mask_Branch = infoDistillation(**model_cfg["Mask_Branch"])

    def init_SAM_backbone(self):
        if self.pretrained_lvm not in sys.path:
            sys.path.insert(0, self.pretrained_lvm)
        from notebook.utils import setup_sam_3d_body

        estimator = setup_sam_3d_body(
            hf_repo_id="facebook/sam-3d-body-dinov3", device="cpu"
        )
        full_model = estimator.model
        if hasattr(full_model, "backbone"):
            raw_backbone = full_model.backbone
        elif hasattr(full_model, "image_encoder"):
            raw_backbone = full_model.image_encoder
        else:
            raise RuntimeError("Cannot find the SAM 3D Body image backbone.")
        self.Backbone = (
            raw_backbone.encoder if hasattr(raw_backbone, "encoder") else raw_backbone
        )

        blocks = getattr(self.Backbone, "blocks", None)
        if blocks is None:
            blocks = getattr(self.Backbone, "layers", None)
        if blocks is None:
            raise RuntimeError("Cannot find transformer blocks in SAM3D backbone.")
        if not 0 <= self.hook_layer < len(blocks):
            raise ValueError(
                f"hook_layer={self.hook_layer} is outside [0, {len(blocks) - 1}]."
            )

        self._last_tokens = None

        def save_tokens(_module, _inputs, output):
            while isinstance(output, (list, tuple)):
                output = output[0]
            self._last_tokens = output

        self._hook_handle = blocks[self.hook_layer].register_forward_hook(save_tokens)

        # Keep only the image encoder. The MHR decoder and its heads are not
        # registered in this OpenGait model and therefore consume no parameters,
        # FLOPs, or checkpoint space.
        del full_model
        del estimator
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.msg_mgr.log_info(
            f"[MaskPretrain] SAM3D encoder only; hooked layer {self.hook_layer}."
        )

    def init_parameters(self):
        for module in self.Mask_Branch.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.affine:
                    nn.init.normal_(module.weight, 1.0, 0.02)
                    nn.init.constant_(module.bias, 0.0)

        self.init_SAM_backbone()
        self.Mask_Branch.train()
        self.Mask_Branch.requires_grad_(True)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.msg_mgr.log_info(
            f"[MaskPretrain] Trainable Mask_Branch parameters: {trainable / 1e6:.6f}M"
        )

    def preprocess(self, images):
        return F.interpolate(
            images,
            (self.image_size * 2, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def connect_loss(self, masks):
        channels = masks.shape[1]
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=masks.dtype,
            device=masks.device,
        ).view(1, 1, 3, 3).repeat(1, channels, 1, 1)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=masks.dtype,
            device=masks.device,
        ).view(1, 1, 3, 3).repeat(1, channels, 1, 1)
        grad_x = F.conv2d(masks, sobel_x, padding=1)
        grad_y = F.conv2d(masks, sobel_y, padding=1)
        return (grad_x.abs().mean() + grad_y.abs().mean())

    def forward(self, inputs):
        ipts, _labs, _ty, _vi, _seqL = inputs
        rgb = ipts[0]
        n, total_s, c, h, w = rgb.shape

        self.Backbone.eval()
        self.Mask_Branch.train(self.training)
        chunk_count = (total_s // self.chunk_size) + 1
        rgb_chunks = torch.chunk(rgb, chunk_count, dim=1)
        mse_losses = []
        connect_losses = []
        last_probs = None
        last_rgb = None

        for rgb_chunk in rgb_chunks:
            _, s, _, _, _ = rgb_chunk.shape
            flat_rgb = rearrange(
                rgb_chunk, "n s c h w -> (n s) c h w"
            ).contiguous()
            with torch.no_grad():
                self._last_tokens = None
                _ = self.Backbone(self.preprocess(flat_rgb))
                tokens = self._last_tokens
                if tokens is None:
                    raise RuntimeError("SAM3D hook did not capture image tokens.")
                target_h = (self.image_size * 2) // 16
                target_w = self.image_size // 16
                target_tokens = target_h * target_w
                if tokens.shape[1] > target_tokens:
                    tokens = tokens[:, -target_tokens:, :]
                tokens = F.layer_norm(tokens, (self.f4_dim,))
                tokens = rearrange(
                    tokens, "b (h w) c -> b c h w", h=target_h, w=target_w
                )
                tokens = F.interpolate(
                    tokens,
                    (self.sils_size * 2, self.sils_size),
                    mode="bilinear",
                    align_corners=False,
                )
                tokens = rearrange(tokens, "b c h w -> (b h w) c").contiguous()

            probs, loss_mse = self.Mask_Branch(tokens)
            probs_2d = rearrange(
                probs,
                "(b h w) c -> b c h w",
                b=n * s,
                h=self.sils_size * 2,
                w=self.sils_size,
            )
            mse_losses.append(loss_mse)
            connect_losses.append(self.connect_loss(probs_2d) * 0.02)
            last_probs = probs_2d
            last_rgb = flat_rgb

        shape_mse = torch.stack(mse_losses).mean()
        shape_connect = torch.stack(connect_losses).mean()
        embeddings = last_probs.mean(dim=(2, 3))

        return {
            "training_feat": {
                "shape_connect": shape_connect,
                "shape_mse": shape_mse,
            },
            "visual_summary": {
                "image/input": last_rgb[:5].float(),
                "image/foreground_channel1": last_probs[:5, 1:2].float(),
            },
            "inference_feat": {"embeddings": embeddings},
        }

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() != 0:
            return
        checkpoint_dir = os.path.join(self.save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        payload = {
            "model": {
                key: value.detach().cpu()
                for key, value in self.Mask_Branch.state_dict().items()
            },
            "iteration": int(iteration),
            "source": "BigGait__SAM3DBody_MaskBranch_Pretrain",
        }
        save_name = self.engine_cfg["save_name"]
        standard_path = os.path.join(
            checkpoint_dir, f"{save_name}-{iteration:05d}.pt"
        )
        torch.save(payload, standard_path)

        export_path = self.mask_branch_export_path
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        torch.save(payload, export_path)
        self.msg_mgr.log_info(
            f"[MaskPretrain] Saved standalone Mask_Branch to {standard_path}"
        )
        self.msg_mgr.log_info(
            f"[MaskPretrain] Exported standalone Mask_Branch to {export_path}"
        )
