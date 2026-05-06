import os
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from .BiggerGait_DINOv2 import BiggerGait__DINOv2


class BiggerGait__DINOv2_Original_NoMask(BiggerGait__DINOv2):
    """Original BiggerGait DINOv2 branch that bypasses the mask branch.

    The provided CCGR-MINI DINOv2-L checkpoint was trained with a full-one
    foreground mask instead of the learned mask branch. We keep the backbone and
    gait head structure identical so the checkpoint can be restored, but skip
    loading/using the mask branch during inference.
    """

    def init_Mask_Branch(self):
        if not self.pretrained_mask_branch or not os.path.isfile(self.pretrained_mask_branch):
            self.msg_mgr.log_info(
                f"Mask_Branch checkpoint not found: {self.pretrained_mask_branch}, skip loading."
            )
            return
        super().init_Mask_Branch()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        rgb = ipts[0]
        del ipts

        chunk_size = max(1, int(getattr(self, "chunk_size", 20) or 20))
        rgb_chunks = torch.chunk(rgb, (rgb.size(1) // chunk_size) + 1, dim=1)

        all_outs = []
        for rgb_img in rgb_chunks:
            with torch.no_grad():
                n, s, c, h, w = rgb_img.size()
                rgb_img = rearrange(rgb_img, "n s c h w -> (n s) c h w").contiguous()
                outs = self.preprocess(rgb_img, self.image_size)
                outs = self.Backbone(outs, output_hidden_states=True).hidden_states[1:]

                intermediates = partial(nn.LayerNorm, eps=1e-6)(
                    self.f4_dim * len(outs), elementwise_affine=False
                )(torch.concat(outs, dim=-1))[:, 1:]
                intermediates = rearrange(
                    intermediates.view(n, s, self.image_size // 7, self.image_size // 14, -1),
                    "n s h w c -> (n s) c h w",
                ).contiguous()
                intermediates = list(torch.chunk(intermediates, self.total_layer_num, dim=1))

                human_mask = torch.ones(
                    n * s, 1, self.sils_size * 2, self.sils_size, device=rgb_img.device
                )

            intermediates = [
                torch.cat(intermediates[i:i + self.group_layer_num], dim=1).contiguous()
                for i in range(0, self.total_layer_num, self.group_layer_num)
            ]
            for i in range(self.num_FPN):
                intermediates[i] = self.HumanSpace_Conv[i](intermediates[i])
            intermediates = torch.concat(intermediates, dim=1)
            intermediates = intermediates * (human_mask > 0.5).to(intermediates)
            intermediates = rearrange(
                intermediates.view(n, s, -1, self.sils_size * 2, self.sils_size),
                "n s c h w -> n c s h w",
            ).contiguous()

            outs = self.Gait_Net.test_1(intermediates)
            all_outs.append(outs)

        embed_list, log_list = self.Gait_Net.test_2(torch.cat(all_outs, dim=2), seqL)

        if self.training:
            return {
                "training_feat": {
                    "triplet": {"embeddings": torch.concat(embed_list, dim=-1), "labels": labs},
                    "softmax": {"logits": torch.concat(log_list, dim=-1), "labels": labs},
                },
                "visual_summary": {
                    "image/rgb_img": rgb_img.view(n * s, c, h, w)[:5].float(),
                    "image/human_mask": self.min_max_norm(
                        human_mask.view(n * s, -1, self.sils_size * 2, self.sils_size)[:5].float()
                    ),
                },
                "inference_feat": {
                    "embeddings": torch.concat(embed_list, dim=-1),
                },
            }

        return {
            "training_feat": {},
            "visual_summary": {},
            "inference_feat": {
                "embeddings": torch.concat(embed_list, dim=-1),
            },
        }
