"""DINOv2-S PuppetGait teacher for CCGR-3D.

Unlike the exploratory model, this teacher consumes canonical A-pose geometry
precomputed with SAM 3D Body.  The heavy SAM3D network is therefore never a
registered submodule and is not present in training checkpoints.
"""
import torch

from .BiggerGait_DINOv2_Projection_Mask_OT_based import (
    BiggerGait__DINOv2__Projection_Mask_OT_Based,
)


class PuppetGaitCCGR3DTeacher(BiggerGait__DINOv2__Projection_Mask_OT_Based):
    def build_network(self, model_cfg):
        self.export_cache_only = bool(model_cfg.get("export_cache_only", False))
        super().build_network(model_cfg)

    def init_sam_pose_engine(self):
        self.SAM_Engine = None
        self.msg_mgr.log_info(
            "[CCGR3D teacher] SAM3D engine disabled; using cached canonical A-pose geometry."
        )

    def generate_mhr_apose(self, pose_out):
        missing = [
            key for key in ("canonical_vertices", "canonical_keypoints")
            if key not in pose_out
        ]
        if missing:
            raise KeyError(
                "Missing cached canonical geometry {}. Re-run CCGR-3D preprocessing "
                "with --save-canonical-apose.".format(missing)
            )
        return pose_out["canonical_vertices"].float(), pose_out["canonical_keypoints"].float()

    def _stack_full_pose_frames(self, sam_frames, device):
        pose_out, cam_int = super()._stack_full_pose_frames(sam_frames, device)
        final_frames = []
        for frame in sam_frames:
            value = frame["pose_outs"]
            final_frames.append(value[-1] if isinstance(value, (list, tuple)) else value)
        for key in ("canonical_vertices", "canonical_keypoints"):
            if key not in final_frames[0]:
                raise KeyError(
                    f"Missing `{key}` in offline SAM3D data. "
                    "Use preprocess_ccgr3d_sam3d_body_decoder_dataset.py."
                )
            pose_out[key] = torch.stack(
                [torch.as_tensor(frame[key], dtype=torch.float32) for frame in final_frames],
                dim=0,
            ).to(device)
        return pose_out, cam_int

    def forward(self, inputs):
        retval = super().forward(inputs)
        inference = retval["inference_feat"]
        # The parent returns one [N,C,P] tensor per FPN group, with branches
        # concatenated along P.  Export branch-aware tensors for distillation.
        original_groups = []
        canonical_groups = []
        for i in range(self.num_FPN):
            grouped = inference[f"embeddings_{i}"]
            if grouped.shape[-1] % self.num_branches != 0:
                raise RuntimeError(f"Cannot split branch feature shape {tuple(grouped.shape)}")
            branch_parts = torch.chunk(grouped, self.num_branches, dim=-1)
            original_groups.append(branch_parts[0])
            canonical_groups.append(branch_parts[1] if self.num_branches > 1 else branch_parts[0])
        inference["embeddings_original"] = torch.stack(original_groups, dim=1)
        inference["embeddings_canonical"] = torch.stack(canonical_groups, dim=1)
        if self.export_cache_only:
            retval["inference_feat"] = {
                "embeddings_canonical": inference["embeddings_canonical"]
            }
        return retval
