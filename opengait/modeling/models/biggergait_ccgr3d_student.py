"""RGB-only BiggerGait student with structured feature export for adapters."""
import torch

from .BiggerGait_DINOv2 import BiggerGait__DINOv2


class BiggerGaitCCGR3DStudent(BiggerGait__DINOv2):
    def build_network(self, model_cfg):
        self.export_cache_only = bool(model_cfg.get("export_cache_only", False))
        super().build_network(model_cfg)

    def forward(self, inputs):
        retval = super().forward(inputs)
        embedding = retval["inference_feat"]["embeddings"]
        if embedding.shape[-1] % self.num_FPN != 0:
            raise RuntimeError(f"Cannot split FPN feature shape {tuple(embedding.shape)}")
        retval["inference_feat"]["embeddings_structured"] = torch.stack(
            torch.chunk(embedding, self.num_FPN, dim=-1), dim=1
        )
        if self.export_cache_only:
            retval["inference_feat"] = {
                "embeddings_structured": retval["inference_feat"]["embeddings_structured"]
            }
        return retval
