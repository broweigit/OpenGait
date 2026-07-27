"""DINOv2-S appearance with online ROMP/SMPL geometry and sparse OT."""

from pathlib import Path

import numpy as np
import torch

from utils import list2var, np2var
from .BiggerGait_DINOv2_Projection_Mask_OT_based import (
    BiggerGait__DINOv2__Projection_Mask_OT_Based,
)
from .BiggerGait_SAM_3D_Body_projection_mask_OT_based_SparseTopK4_ROMP import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share,
)


class BiggerGait__DINOv2__Projection_Mask_OT_Based_SparseTopK4_OnlineROMP_Gaitbase_Share(
    BiggerGait__DINOv2__Projection_Mask_OT_Based
):
    """End-to-end lightweight variant with DINOv2-S and online ROMP.

    The dataset supplies RGB only. ROMP executes inside every model forward,
    so no precomputed geometry is needed and ROMP latency is included in the
    end-to-end timing. ROMP is frozen and excluded from gait checkpoints.
    """

    _extract_pose_frame = staticmethod(
        BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share._extract_pose_frame
    )
    _frame_value = staticmethod(
        BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share._frame_value
    )
    _stack_romp_frames = (
        BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share._stack_romp_frames
    )
    get_romp_source_vertex_index_map = (
        BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share.get_romp_source_vertex_index_map
    )
    rotate_branch_geometry = (
        BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share.rotate_branch_geometry
    )
    warp_features_with_romp_ot = (
        BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_ROMP_Gaitbase_Share.warp_features_with_romp_ot
    )

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.romp_hip_indices = tuple(model_cfg.get("romp_hip_indices", [1, 2]))
        self.romp_require_verts_camed_org = bool(
            model_cfg.get("romp_require_verts_camed_org", True)
        )
        self.romp_target_h = int(model_cfg.get("romp_target_h", 512))
        self.romp_target_w = int(model_cfg.get("romp_target_w", 256))
        self.romp_focal_scale = float(model_cfg.get("romp_focal_scale", 443.4 / 512.0))
        self.romp_missing_policy = str(model_cfg.get("romp_missing_policy", "fail"))
        romp_home = model_cfg.get("romp_home")
        self.romp_home = Path(romp_home).expanduser().resolve() if romp_home else None
        self._online_romp_processor = None
        self._romp_source_camera_vertices = None

    def init_sam_pose_engine(self):
        """Override the parent hook: initialize online ROMP, never SAM3D."""
        try:
            from misc.preprocess_romp_smpl_geometry_dataset import ROMPGeometryPreprocessor
        except ImportError as exc:
            raise ImportError(
                "Online DINOv2-S+ROMP requires simple-romp and its dependencies in "
                "the OpenGait training environment."
            ) from exc

        local_device = int(torch.cuda.current_device())
        processor = ROMPGeometryPreprocessor(
            target_h=self.romp_target_h,
            target_w=self.romp_target_w,
            focal_scale=self.romp_focal_scale,
            gpu=local_device,
            vertices_dtype=np.float32,
            joints_dtype=np.float32,
            cam_t_dtype=np.float32,
            missing_policy=self.romp_missing_policy,
            romp_home=self.romp_home,
        )
        # simple-romp wraps its network in DataParallel by default. OpenGait
        # already uses one DDP process per GPU, so unwrap it to prevent every
        # rank from attempting to replicate ROMP across all visible devices.
        if isinstance(processor.model.model, torch.nn.DataParallel):
            processor.model.model = processor.model.model.module
        processor.model.eval()
        processor.model.requires_grad_(False)
        romp_parameter_count = sum(parameter.numel() for parameter in processor.model.parameters())

        # Keep ROMP outside the OpenGait module tree: it is already on this
        # process's local GPU, is frozen, and must not be converted to SyncBN
        # or serialized into every gait checkpoint.
        object.__setattr__(self, "_online_romp_processor", processor)
        self.SAM_Engine = None
        self.msg_mgr.log_info(
            f"[OnlineROMP] cuda:{local_device}, frozen parameters={romp_parameter_count / 1e6:.3f}M"
        )
        self.msg_mgr.log_info(
            "[OnlineROMP] Geometry is computed inside forward; no offline ROMP pkl is used."
        )

    def inputs_pretreament(self, inputs):
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != 1 or len(seq_trfs) != 1:
            raise ValueError(
                "Online DINOv2-S+ROMP expects one RGB modality and one transform; "
                f"got {len(seqs_batch)} modalities and {len(seq_trfs)} transforms."
            )

        raw_sequences = [np.asarray(sequence) for sequence in seqs_batch[0]]
        rgb = np2var(
            np.asarray(raw_sequences, dtype=np.float32),
            requires_grad=bool(self.training),
        ).float()
        labs = list2var(labs_batch).long()
        seqL = np2var(seqL_batch).int() if seqL_batch is not None else None
        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            rgb = rgb[:, :seqL_sum]
            raw_sequences = [sequence[:seqL_sum] for sequence in raw_sequences]
        return [rgb, raw_sequences], labs, typs_batch, vies_batch, seqL

    def _stack_full_pose_frames(self, rgb_frames, device):
        if not rgb_frames:
            raise ValueError("Online ROMP received an empty RGB frame list.")
        rgb_sequence = np.stack([np.asarray(frame) for frame in rgb_frames], axis=0)
        geometry_frames = self._online_romp_processor.process_sequence(rgb_sequence)
        pose_out = self._stack_romp_frames(geometry_frames, device)
        cam_int = pose_out["cam_int"]

        smpl_vertices = pose_out["pred_vertices"]
        source_camera_vertices = self._romp_projection_to_camera_vertices(
            pose_out["verts_camed_org"], cam_int
        )
        pose_out["romp_smpl_vertices"] = smpl_vertices
        pose_out["pred_vertices"] = source_camera_vertices
        pose_out["pred_cam_t"] = torch.zeros_like(pose_out["pred_cam_t"])
        self._romp_source_camera_vertices = source_camera_vertices
        return pose_out, cam_int

    @staticmethod
    def _romp_projection_to_camera_vertices(verts_camed_org, cam_int):
        u = verts_camed_org[..., 0]
        v = verts_camed_org[..., 1]
        depth = (-verts_camed_org[..., 2]).clamp(min=1e-4)
        fx = cam_int[:, 0, 0].unsqueeze(1).clamp(min=1e-4)
        fy = cam_int[:, 1, 1].unsqueeze(1).clamp(min=1e-4)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        return torch.stack([x, y, depth], dim=-1)

    @staticmethod
    def _camera_vertices_to_romp_projection(camera_vertices, cam_int):
        x, y, z = camera_vertices.unbind(-1)
        z_safe = z.clamp(min=1e-4)
        fx = cam_int[:, 0, 0].unsqueeze(1)
        fy = cam_int[:, 1, 1].unsqueeze(1)
        cx = cam_int[:, 0, 2].unsqueeze(1)
        cy = cam_int[:, 1, 2].unsqueeze(1)
        u = x / z_safe * fx + cx
        v = y / z_safe * fy + cy
        return torch.stack([u, v, -z_safe], dim=-1)

    def get_source_vertex_index_map(
        self, vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w
    ):
        if vertices is self._romp_source_camera_vertices:
            source_projection = self._camera_vertices_to_romp_projection(vertices, cam_int)
            return self.get_romp_source_vertex_index_map(
                source_projection, h_feat, w_feat, target_h, target_w
            )
        return super().get_source_vertex_index_map(
            vertices, cam_t, cam_int, h_feat, w_feat, target_h, target_w
        )

    def build_branch_geometry(self, branch_cfg, pose_out):
        if bool(branch_cfg.get("use_apose", False)):
            raise ValueError(
                "Online ROMP supplies SMPL walking-pose geometry; set use_apose: false."
            )
        branch_pose = dict(pose_out)
        branch_pose["pred_vertices"] = pose_out["romp_smpl_vertices"]
        return super().build_branch_geometry(branch_cfg, branch_pose)

    def warp_features_with_ot(
        self,
        human_feat,
        mask_src,
        pred_verts,
        branch_verts,
        branch_keypoints,
        pred_cam_t,
        global_rot,
        cam_int_src,
        cam_int_tgt,
        cam_t_tgt,
        h_feat,
        w_feat,
        target_h,
        target_w,
        yaw,
        apply_global_rot_alignment,
        flat_flip_flags=None,
    ):
        if flat_flip_flags is not None and torch.any(flat_flip_flags):
            raise ValueError("Set sync_hflip_prob: 0.0 when using online ROMP geometry.")
        source_projection = self._camera_vertices_to_romp_projection(pred_verts, cam_int_src)
        return self.warp_features_with_romp_ot(
            human_feat,
            mask_src,
            branch_verts,
            source_projection,
            branch_verts,
            branch_keypoints,
            pred_cam_t,
            global_rot,
            cam_int_src,
            cam_int_tgt,
            cam_t_tgt,
            h_feat,
            w_feat,
            target_h,
            target_w,
            yaw,
            apply_global_rot_alignment,
        )

    def forward(self, inputs):
        retval = super().forward(inputs)
        timing_info = retval.get("timing_info", {})
        if "model_sam_unpack" in timing_info:
            timing_info["model_online_romp"] = timing_info.pop("model_sam_unpack")
        return retval
