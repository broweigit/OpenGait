import torch

from .BiggerGait_SAM_3D_Body_projection_mask import (
    BiggerGait__SAM3DBody__Projection_Mask_Gaitbase_Share,
)


class BiggerGait__SAM3DBody__Projection_Mask_OcclusionEval_Gaitbase_Share(
    BiggerGait__SAM3DBody__Projection_Mask_Gaitbase_Share
):
    """Evaluation-only A3 model for paired image-occlusion controls.

    The subclass adds no parameters and preserves the original A3 forward for
    every individual condition. It only repeats that forward with named input
    perturbations so one dataloader traversal can evaluate all conditions.
    """

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.occlusion_eval_cfg = model_cfg.get("occlusion_eval", {}) or {}
        self.occlusion_eval_enabled = bool(
            self.occlusion_eval_cfg.get("enabled", False)
        )
        self._active_occlusion_variant = {
            "type": "clean",
            "name": "clean",
        }
        if self.occlusion_eval_enabled:
            fractions = self.occlusion_eval_cfg.get(
                "occlusion_fractions", [0.4, 0.6, 0.8]
            )
            if not fractions:
                raise ValueError(
                    "occlusion_eval.occlusion_fractions cannot be empty."
                )
            for fraction in fractions:
                if not 0.0 < float(fraction) < 1.0:
                    raise ValueError(
                        f"Invalid occlusion fraction: {fraction}"
                    )
            self.msg_mgr.log_info(
                "[A3 Occlusion] Paired clean/occlusion evaluation enabled: "
                f"fractions={list(fractions)}"
            )

    def _occlusion_variants(self):
        variants = [{"type": "clean", "name": "clean"}]
        fractions = self.occlusion_eval_cfg.get(
            "occlusion_fractions", [0.4, 0.6, 0.8]
        )
        for fraction in fractions:
            fraction = float(fraction)
            percent = int(round(fraction * 100.0))
            variants.append({
                "type": "occlusion",
                "name": f"occlusion_{percent}pct",
                "fraction": fraction,
            })
        return variants

    def preprocess(self, sils, h, w, mode="bilinear"):
        backbone_input = super().preprocess(sils, h, w, mode=mode)
        variant = self._active_occlusion_variant
        if variant.get("type") != "occlusion":
            return backbone_input

        fraction = float(variant["fraction"])
        band_h = max(1, int(round(h * fraction)))
        band_top = (h - band_h) // 2
        band_bottom = band_top + band_h
        occluded = backbone_input.clone()

        # BaseRgbTransform applies ImageNet normalization before the model, so
        # normalized RGB(0, 0, 0) is used to create the same black band as A4.
        black = torch.tensor(
            [
                -0.485 / 0.229,
                -0.456 / 0.224,
                -0.406 / 0.225,
            ],
            device=occluded.device,
            dtype=occluded.dtype,
        ).view(1, 3, 1, 1)
        occluded[:, :, band_top:band_bottom, :] = black
        return occluded

    def forward(self, inputs):
        if not self.occlusion_eval_enabled:
            return super().forward(inputs)
        if self.training:
            raise RuntimeError("A3 occlusion variant evaluation is test-only.")

        inference_feat = {}
        try:
            for variant in self._occlusion_variants():
                self._active_occlusion_variant = variant
                retval = super().forward(inputs)
                embedding = retval["inference_feat"]["embeddings"]
                inference_feat[f"embeddings_{variant['name']}"] = embedding
                if variant["type"] == "clean":
                    inference_feat["embeddings"] = embedding
                del retval
        finally:
            self._active_occlusion_variant = {
                "type": "clean",
                "name": "clean",
            }

        return {
            "training_feat": {},
            "visual_summary": {},
            "inference_feat": inference_feat,
        }
