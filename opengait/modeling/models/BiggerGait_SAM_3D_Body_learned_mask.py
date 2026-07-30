import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

from .BiggerGait_SAM_3D_Body_official import (
    BiggerGait__SAM3DBody_Official_Gaitbase_Share,
)
from .BiggerGait_SAM_3D_Body_projection_mask import (
    BiggerGait__SAM3DBody__Projection_Mask_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Learned_Mask_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_Gaitbase_Share
):
    """Old-A3-compatible downstream with BigGait's learned human mask.

    The module names and tensor shapes of the original A3 downstream are kept,
    so its trained checkpoint can be restored non-strictly. The online MHR
    decoder/projected mask is removed. A separately pretrained Mask_Branch is
    reloaded after the old checkpoint so that its stale random Mask_Branch keys
    cannot overwrite the learned segmentation weights.
    """

    def build_network(self, model_cfg):
        self.mask_foreground_channel = model_cfg.get("mask_foreground_channel", 1)
        super().build_network(model_cfg)

    def init_SAM_Backbone(self):
        # This implementation retains only the DINOv3 image encoder and
        # explicitly deletes the MHR decoder and associated prediction heads.
        BiggerGait__SAM3DBody_Official_Gaitbase_Share.init_SAM_Backbone(self)

    def _mask_state_dict(self, payload):
        state = payload.get("model", payload)
        if not isinstance(state, dict):
            raise TypeError("Mask checkpoint must contain a state dict.")
        prefix = "Mask_Branch."
        if any(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
        return state

    def init_Mask_Branch(self):
        if not self.pretrained_mask_branch or self.pretrained_mask_branch == "BYPASS":
            raise ValueError(
                "Learned-mask A3 requires model_cfg.pretrained_mask_branch."
            )
        self.msg_mgr.log_info(
            f"[LearnedMask] Loading Mask_Branch from {self.pretrained_mask_branch}"
        )
        payload = torch.load(
            self.pretrained_mask_branch, map_location=torch.device("cpu")
        )
        state = self._mask_state_dict(payload)
        msg = self.Mask_Branch.load_state_dict(state, strict=True)
        self.msg_mgr.log_info(f"[LearnedMask] Missing keys: {msg.missing_keys}")
        self.msg_mgr.log_info(f"[LearnedMask] Unexpected keys: {msg.unexpected_keys}")
        self.Mask_Branch.eval()
        self.Mask_Branch.requires_grad_(False)

    def init_parameters(self):
        backbone_module_ids = {id(module) for module in self.Backbone.modules()}
        for module in self.modules():
            if id(module) in backbone_module_ids:
                continue
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.affine:
                    nn.init.normal_(module.weight, 1.0, 0.02)
                    nn.init.constant_(module.bias, 0.0)

        self.Backbone.eval()
        self.Backbone.requires_grad_(False)
        self.init_Mask_Branch()
        total = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info(
            f"[LearnedMask] Encoder-only model parameters: {total / 1e6:.5f}M"
        )

    def _load_ckpt(self, save_name):
        # Restore the old A3 downstream first. restore_ckpt_strict must be
        # false because the old checkpoint contains now-removed SAM_Engine keys.
        super()._load_ckpt(save_name)
        # The old A3 checkpoint also contains a random, unused Mask_Branch.
        # Reload the independently trained branch after checkpoint restoration.
        self.init_Mask_Branch()
        self.msg_mgr.log_info(
            "[LearnedMask] Re-applied pretrained Mask_Branch after old A3 restore."
        )

    def _foreground_probability(self, probs, batch, height, width):
        probs = probs.view(batch, height, width, 2)
        channel = self.mask_foreground_channel
        if channel == "auto":
            border0 = (
                probs[:, 0, :, 0].mean(1)
                + probs[:, -1, :, 0].mean(1)
                + probs[:, :, 0, 0].mean(1)
                + probs[:, :, -1, 0].mean(1)
            )
            border1 = (
                probs[:, 0, :, 1].mean(1)
                + probs[:, -1, :, 1].mean(1)
                + probs[:, :, 0, 1].mean(1)
                + probs[:, :, -1, 1].mean(1)
            )
            use_channel1 = border1 < border0
            fg0 = probs[..., 0]
            fg1 = probs[..., 1]
            return torch.where(use_channel1[:, None, None], fg1, fg0)
        channel = int(channel)
        if channel not in (0, 1):
            raise ValueError("mask_foreground_channel must be 0, 1, or 'auto'.")
        return probs[..., channel]

    def forward(self, inputs):
        ipts, labs, _ty, _vi, seqL = inputs
        rgb = ipts[0]
        total_s = rgb.size(1)
        rgb_chunks = torch.chunk(
            rgb, (total_s // self.chunk_size) + 1, dim=1
        )
        all_outs = []
        target_h, target_w = self.image_size * 2, self.image_size
        h_feat, w_feat = target_h // 16, target_w // 16
        target_tokens = h_feat * w_feat

        self.Backbone.eval()
        self.Mask_Branch.eval()

        for rgb_img in rgb_chunks:
            n, s, c, h, w = rgb_img.size()
            flat_rgb = rearrange(
                rgb_img, "n s c h w -> (n s) c h w"
            ).contiguous()
            batch = n * s

            with torch.no_grad():
                encoder_input = self.preprocess(
                    flat_rgb, target_h, target_w
                )
                self.intermediate_features = {}
                _ = self.Backbone(encoder_input)

                features_to_use = []
                for idx in range(len(self.hook_handles)):
                    feat = self.intermediate_features[idx]
                    if feat.shape[1] > target_tokens:
                        feat = feat[:, -target_tokens:, :]
                    features_to_use.append(feat)

                mask_tokens = partial(
                    nn.LayerNorm, eps=1e-6
                )(self.f4_dim, elementwise_affine=False)(features_to_use[-1])
                mask_probs, _ = self.Mask_Branch(
                    mask_tokens.reshape(-1, self.f4_dim)
                )
                foreground = self._foreground_probability(
                    mask_probs, batch, h_feat, w_feat
                )
                human_mask_low = (foreground > 0.5).to(mask_probs)
                human_mask = F.interpolate(
                    human_mask_low.unsqueeze(1),
                    (self.sils_size * 2, self.sils_size),
                    mode="bilinear",
                    align_corners=False,
                ).detach()

            processed_feat_list = []
            step = len(features_to_use) // self.num_FPN
            for idx in range(self.num_FPN):
                sub_feats = features_to_use[idx * step:(idx + 1) * step]
                sub_app = torch.concat(sub_feats, dim=-1)
                sub_app = rearrange(
                    sub_app, "b (h w) c -> b c h w", h=h_feat, w=w_feat
                ).contiguous()
                sub_app = self.Pre_Conv(sub_app)
                sub_app = rearrange(sub_app, "b c h w -> b (h w) c")
                curr_dim = self.f4_dim * len(sub_feats)
                sub_app = partial(
                    nn.LayerNorm, eps=1e-6
                )(curr_dim, elementwise_affine=False)(sub_app)
                sub_app = rearrange(
                    sub_app, "b (h w) c -> b c h w", h=h_feat, w=w_feat
                ).contiguous()
                processed_feat_list.append(self.HumanSpace_Conv[idx](sub_app))

            human_feat = torch.concat(processed_feat_list, dim=1)
            # Match BiggerGait_DINOv2.py: mask the reduced LVM feature map,
            # rather than masking raw tokens before the FPN projection.
            human_feat = human_feat * (human_mask > 0.5).to(human_feat)
            human_feat = rearrange(
                human_feat.view(
                    n, s, -1, self.sils_size * 2, self.sils_size
                ),
                "n s c h w -> n c s h w",
            ).contiguous()
            all_outs.append(self.Gait_Net.test_1(human_feat))

        embed_list, log_list = self.Gait_Net.test_2(
            torch.cat(all_outs, dim=2), seqL
        )
        embeddings = torch.concat(embed_list, dim=-1)
        if self.training:
            training_feat = {
                "triplet": {"embeddings": embeddings, "labels": labs},
                "softmax": {
                    "logits": torch.concat(log_list, dim=-1),
                    "labels": labs,
                },
            }
            visual_summary = {
                "image/rgb_img": flat_rgb[:5].float(),
                "image/human_mask": human_mask[:5].float(),
            }
        else:
            training_feat = {}
            visual_summary = {}

        return {
            "training_feat": training_feat,
            "visual_summary": visual_summary,
            "inference_feat": {
                "embeddings": embeddings,
                **{
                    f"embeddings_{idx}": embed
                    for idx, embed in enumerate(embed_list)
                },
            },
        }
