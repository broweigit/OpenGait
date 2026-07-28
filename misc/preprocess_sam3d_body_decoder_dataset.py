#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


def is_distributed_launch() -> bool:
    return "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1


def maybe_launch_distributed(args: argparse.Namespace) -> bool:
    if is_distributed_launch():
        return False
    if not torch.cuda.is_available():
        return False
    if str(args.device).startswith("cpu"):
        return False
    if args.gpus <= 1:
        return False

    nproc = min(args.gpus, torch.cuda.device_count())
    if nproc <= 1:
        return False

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "--master_port",
        str(args.master_port),
        str(Path(__file__).resolve()),
        "--dataset-type",
        args.dataset_type,
        "--input-root",
        args.input_root,
        "--output-root",
        args.output_root,
        "--pretrained-lvm-root",
        args.pretrained_lvm_root,
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--hook-layer",
        str(args.hook_layer),
        "--rgb-mode",
        args.rgb_mode,
        "--save-mode",
        args.save_mode,
        "--vertices-dtype",
        args.vertices_dtype,
        "--cam-t-dtype",
        args.cam_t_dtype,
        "--cam-int-dtype",
        args.cam_int_dtype,
        "--gpus",
        str(nproc),
    ]

    if args.overwrite:
        cmd.append("--overwrite")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.debug_vis:
        cmd.append("--debug-vis")
    if args.debug_max_frames is not None:
        cmd.extend(["--debug-max-frames", str(args.debug_max_frames)])
    if args.debug_point_size is not None:
        cmd.extend(["--debug-point-size", str(args.debug_point_size)])

    raise SystemExit(subprocess.call(cmd))


def init_distributed_from_env(local_rank_arg: int) -> tuple[int, int, int]:
    if not is_distributed_launch():
        return 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", local_rank_arg))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size, local_rank


def destroy_distributed() -> None:
    return None


def resize_with_padding(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    is_batch = img.dim() == 4
    if not is_batch:
        img = img.unsqueeze(0)

    _, _, h, w = img.shape
    scale = torch.min(torch.tensor([target_h / h, target_w / w], device=img.device))
    new_h, new_w = (h * scale).long(), (w * scale).long()
    img_resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    return img_padded.squeeze(0) if not is_batch else img_padded


def batch_remove_black_border(batch_imgs: torch.Tensor, threshold: int = 10, target_h: int = 256, target_w: int = 128) -> torch.Tensor:
    assert batch_imgs.dim() == 4, "Input must be a 4D tensor (B x C x H x W)"
    bsz, channels, _, _ = batch_imgs.shape
    batch_imgs = batch_imgs.float()

    if channels > 1:
        gray = batch_imgs.mean(dim=1)
    else:
        gray = batch_imgs.squeeze(1)

    mask = gray > threshold
    result = torch.zeros(bsz, channels, target_h, target_w, device=batch_imgs.device, dtype=batch_imgs.dtype)

    for i in range(bsz):
        if not mask[i].any():
            result[i] = resize_with_padding(batch_imgs[i], target_h, target_w)
            continue

        y_nonzero = torch.any(mask[i], dim=1)
        x_nonzero = torch.any(mask[i], dim=0)
        y_indices = torch.where(y_nonzero)[0]
        x_indices = torch.where(x_nonzero)[0]

        if y_indices.numel() == 0 or x_indices.numel() == 0:
            result[i] = resize_with_padding(batch_imgs[i], target_h, target_w)
            continue

        y_min, y_max = y_indices[[0, -1]]
        x_min, x_max = x_indices[[0, -1]]
        cropped = batch_imgs[i, :, y_min : y_max + 1, x_min : x_max + 1]
        result[i] = resize_with_padding(cropped, target_h, target_w)

    return result


class BaseRgbTransformLite:
    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.mean = np.array(mean, dtype=np.float32).reshape((1, 3, 1, 1))
        self.std = np.array(std, dtype=np.float32).reshape((1, 3, 1, 1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != 3:
            if len(x.shape) == 3:
                x = x[:, None, ...]
                x = np.repeat(x, repeats=3, axis=1)
            else:
                x = x.transpose(0, 3, 1, 2)
        return (x.astype(np.float32) - self.mean) / self.std


@dataclass
class SequenceRecord:
    pid: str
    seq_type: str
    view: str
    seq_dir: Path
    rgb_path: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def resize_frames_for_vis(frames_chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    frames = torch.from_numpy(frames_chw).float()
    frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return frames.cpu().numpy()


def save_projected_vertices_overlay(
    img_chw: np.ndarray,
    vertices: torch.Tensor,
    cam_t: torch.Tensor,
    cam_int: torch.Tensor,
    save_path: Path,
    point_size: float = 8.0,
) -> None:
    img_np = np.transpose(img_chw, (1, 2, 0))
    if img_np.dtype != np.uint8:
        img_min = float(img_np.min())
        img_max = float(img_np.max())
        img_np = (img_np - img_min) / (img_max - img_min + 1e-6)
    else:
        img_np = img_np.astype(np.float32) / 255.0

    vertices = vertices.detach().cpu().float()
    cam_t = cam_t.detach().cpu().float()
    cam_int = cam_int.detach().cpu().float()

    verts_cam = vertices + cam_t.unsqueeze(0)
    x = verts_cam[:, 0]
    y = verts_cam[:, 1]
    z = verts_cam[:, 2].clamp(min=1e-3)

    fx = cam_int[0, 0]
    fy = cam_int[1, 1]
    cx = cam_int[0, 2]
    cy = cam_int[1, 2]

    u = (x / z) * fx + cx
    v = (y / z) * fy + cy

    h, w = img_np.shape[:2]
    u_np = u.numpy()
    v_np = v.numpy()
    mask = (u_np >= 0) & (u_np < w) & (v_np >= 0) & (v_np < h)

    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(8, 8 * h / max(w, 1)))
    ax.imshow(img_np)
    ax.scatter(u_np[mask], v_np[mask], s=point_size, c="r", alpha=0.55)
    ax.set_title("SAM3D Projection Alignment")
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def load_rgb_pickle_sequence(path: Path) -> np.ndarray:
    seq = np.asarray(load_pickle(path))
    if seq.ndim != 4:
        raise ValueError(f"Unexpected RGB-pickle shape at {path}: {seq.shape}")
    if seq.shape[1] == 3:
        return seq
    if seq.shape[-1] == 3:
        return seq.transpose(0, 3, 1, 2)
    raise ValueError(f"Cannot interpret RGB-pickle shape at {path}: {seq.shape}")


def load_ccpg_rgb_sequence(path: Path) -> np.ndarray:
    return load_rgb_pickle_sequence(path)


def load_ccgr_video_sequence(path: Path) -> np.ndarray:
    try:
        from torchvision.io import read_video

        video, _, _ = read_video(str(path), output_format="TCHW", pts_unit="sec")
    except Exception:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.transpose(2, 0, 1))
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames found in video: {path}")

        video = torch.from_numpy(np.stack(frames, axis=0))

    video = batch_remove_black_border(video, target_h=256, target_w=128)
    return video.cpu().numpy().astype(np.float32)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported rgb mode: {mode}")


def iter_ccpg_sequences(dataset_root: Path) -> Iterable[SequenceRecord]:
    for pid in sorted(os.listdir(dataset_root)):
        pid_dir = dataset_root / pid
        if not pid_dir.is_dir():
            continue
        for seq_type in sorted(os.listdir(pid_dir)):
            seq_type_dir = pid_dir / seq_type
            if not seq_type_dir.is_dir():
                continue
            for view in sorted(os.listdir(seq_type_dir)):
                seq_dir = seq_type_dir / view
                if not seq_dir.is_dir():
                    continue
                rgb_candidates = sorted(seq_dir.glob("*aligned-rgbs.pkl"))
                if not rgb_candidates:
                    continue
                yield SequenceRecord(pid=pid, seq_type=seq_type, view=view, seq_dir=seq_dir, rgb_path=rgb_candidates[0])


def iter_rgb_pickle_sequences(
    dataset_root: Path,
    rgb_glob: str,
) -> Iterable[SequenceRecord]:
    """Iterate OpenGait's common pid/sequence/view RGB-pickle hierarchy."""
    for pid in sorted(os.listdir(dataset_root)):
        pid_dir = dataset_root / pid
        if not pid_dir.is_dir():
            continue
        for seq_type in sorted(os.listdir(pid_dir)):
            seq_type_dir = pid_dir / seq_type
            if not seq_type_dir.is_dir():
                continue
            for view in sorted(os.listdir(seq_type_dir)):
                seq_dir = seq_type_dir / view
                if not seq_dir.is_dir():
                    continue
                rgb_candidates = sorted(seq_dir.glob(rgb_glob))
                if not rgb_candidates:
                    continue
                yield SequenceRecord(
                    pid=pid,
                    seq_type=seq_type,
                    view=view,
                    seq_dir=seq_dir,
                    rgb_path=rgb_candidates[0],
                )


def iter_casiab_sequences(dataset_root: Path) -> Iterable[SequenceRecord]:
    return iter_rgb_pickle_sequences(dataset_root, "*-rgbs.pkl")


def iter_sustech1k_sequences(dataset_root: Path) -> Iterable[SequenceRecord]:
    return iter_rgb_pickle_sequences(dataset_root, "*-aligned-rgbs.pkl")


def iter_ccgr_mini_sequences(dataset_root: Path) -> Iterable[SequenceRecord]:
    for pid in sorted(os.listdir(dataset_root)):
        pid_dir = dataset_root / pid
        if not pid_dir.is_dir():
            continue
        for seq_type in sorted(os.listdir(pid_dir)):
            seq_type_dir = pid_dir / seq_type
            if not seq_type_dir.is_dir():
                continue
            for view in sorted(os.listdir(seq_type_dir)):
                seq_dir = seq_type_dir / view
                if not seq_dir.is_dir():
                    continue
                rgb_candidates = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() == ".avi"])
                if not rgb_candidates:
                    continue
                yield SequenceRecord(pid=pid, seq_type=seq_type, view=view, seq_dir=seq_dir, rgb_path=rgb_candidates[0])


def detach_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)
    return obj


def slice_by_batch(obj: Any, batch_idx: int, batch_size: int) -> Any:
    if torch.is_tensor(obj):
        cpu_obj = obj.detach().cpu()
        if cpu_obj.ndim > 0 and cpu_obj.shape[0] == batch_size:
            # Clone the per-sample slice so pickle does not serialize the full
            # backing storage of the original batched tensor view.
            return cpu_obj[batch_idx].clone()
        return cpu_obj
    if isinstance(obj, np.ndarray):
        if obj.ndim > 0 and obj.shape[0] == batch_size:
            return obj[batch_idx].copy()
        return obj
    if isinstance(obj, dict):
        return {k: slice_by_batch(v, batch_idx, batch_size) for k, v in obj.items()}
    if isinstance(obj, list):
        return [slice_by_batch(v, batch_idx, batch_size) for v in obj]
    if isinstance(obj, tuple):
        return tuple(slice_by_batch(v, batch_idx, batch_size) for v in obj)
    return obj


def to_numpy_contiguous(value: Any, dtype: np.dtype) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


OT_LITE_POSE_KEYS = [
    "pred_vertices",
    "pred_cam_t",
    "pred_keypoints_3d",
    "global_rot",
    "shape",
    "scale",
    "face",
    "body_pose",
    "hand",
]


class SAM3DDecoderPreprocessor:
    def __init__(
        self,
        pretrained_lvm_root: Path,
        device: str,
        target_h: int = 512,
        target_w: int = 256,
        hook_layer: int = -1,
        save_mode: str = "minimal",
        debug_vis: bool = False,
        debug_max_frames: int = 1,
        debug_point_size: float = 8.0,
        debug_vis_dir: Optional[Path] = None,
        vertices_dtype: np.dtype = np.float16,
        cam_t_dtype: np.dtype = np.float32,
        cam_int_dtype: np.dtype = np.float32,
    ):
        self.pretrained_lvm_root = Path(pretrained_lvm_root)
        self.device = torch.device(device)
        self.target_h = target_h
        self.target_w = target_w
        self.h_feat = target_h // 16
        self.w_feat = target_w // 16
        self.save_mode = save_mode
        self.rgb_transform = BaseRgbTransformLite()
        self.debug_vis = debug_vis
        self.debug_max_frames = debug_max_frames
        self.debug_point_size = debug_point_size
        self.debug_vis_dir = debug_vis_dir
        self._saved_debug_frames = 0
        self.vertices_dtype = np.dtype(vertices_dtype)
        self.cam_t_dtype = np.dtype(cam_t_dtype)
        self.cam_int_dtype = np.dtype(cam_int_dtype)

        if str(self.pretrained_lvm_root) not in sys.path:
            sys.path.insert(0, str(self.pretrained_lvm_root))
        from notebook.utils import setup_sam_3d_body

        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", device=str(self.device))
        self.model = estimator.model.eval().to(self.device)
        self.backbone_dtype = getattr(self.model, "backbone_dtype", None)

        if hasattr(self.model, "backbone"):
            raw_backbone = self.model.backbone
        elif hasattr(self.model, "image_encoder"):
            raw_backbone = self.model.image_encoder
        else:
            raise RuntimeError("Cannot find SAM-3D-body backbone.")

        self.backbone = raw_backbone.encoder if hasattr(raw_backbone, "encoder") else raw_backbone
        if self.backbone_dtype is None:
            self.backbone_dtype = next(self.backbone.parameters()).dtype
        all_blocks = self.backbone.blocks if hasattr(self.backbone, "blocks") else self.backbone.layers
        self.hook_layer = hook_layer if hook_layer >= 0 else len(all_blocks) + hook_layer
        if self.hook_layer < 0 or self.hook_layer >= len(all_blocks):
            raise ValueError(f"Invalid hook_layer={hook_layer}, total blocks={len(all_blocks)}")

        self._hook_output = None

        def capture_output(_, __, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(output, (list, tuple)):
                output = output[0]
            self._hook_output = output

        self._hook_handle = all_blocks[self.hook_layer].register_forward_hook(capture_output)

        for param in self.model.parameters():
            param.requires_grad = False

    def close(self) -> None:
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _prepare_dummy_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.device
        estimated_focal_length = max(self.target_h, self.target_w) * 1.1
        cx, cy = self.target_w / 2.0, self.target_h / 2.0

        cam_int = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3).clone()
        cam_int[:, 0, 0] = estimated_focal_length
        cam_int[:, 1, 1] = estimated_focal_length
        cam_int[:, 0, 2] = cx
        cam_int[:, 1, 2] = cy

        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.target_h, device=device),
            torch.arange(self.target_w, device=device),
            indexing="ij",
        )
        ray_x = (x_grid - cx) / estimated_focal_length
        ray_y = (y_grid - cy) / estimated_focal_length
        ray_cond = torch.stack([ray_x, ray_y], dim=0).unsqueeze(0).expand(batch_size, 2, self.target_h, self.target_w)

        bbox_scale = torch.tensor([max(self.target_h, self.target_w)], device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 1)
        bbox_center = torch.tensor([cx, cy], device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 2)
        img_size = torch.tensor([float(self.target_w), float(self.target_h)], device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 2)
        affine_trans = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 2, 3)

        return {
            "img": torch.zeros(batch_size, 1, 3, self.target_h, self.target_w, device=device),
            "ori_img_size": img_size,
            "img_size": img_size,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_int": cam_int,
            "affine_trans": affine_trans,
            "ray_cond": ray_cond,
        }

    def _frames_to_embeddings(self, frames_chw: np.ndarray) -> torch.Tensor:
        frames_norm = self.rgb_transform(frames_chw)
        frames_tensor = torch.from_numpy(frames_norm).to(self.device, non_blocking=True)
        frames_tensor = F.interpolate(frames_tensor, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        frames_tensor = frames_tensor.to(self.backbone_dtype)

        self._hook_output = None
        _ = self.backbone(frames_tensor)
        if self._hook_output is None:
            raise RuntimeError("Failed to capture backbone output for SAM-3D-body decoder.")

        sam_emb = self._hook_output
        target_tokens = self.h_feat * self.w_feat
        if sam_emb.shape[1] > target_tokens:
            sam_emb = sam_emb[:, -target_tokens:, :]
        sam_emb = sam_emb.transpose(1, 2).reshape(frames_tensor.shape[0], -1, self.h_feat, self.w_feat).contiguous()
        return sam_emb

    def process_sequence(self, rgb_sequence: np.ndarray, chunk_size: int, debug_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        total_frames = rgb_sequence.shape[0]

        with torch.no_grad():
            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                frame_chunk = rgb_sequence[start:end]
                vis_chunk = None
                if self.debug_vis and self._saved_debug_frames < self.debug_max_frames:
                    vis_chunk = resize_frames_for_vis(frame_chunk, self.target_h, self.target_w)
                sam_emb = self._frames_to_embeddings(frame_chunk)
                batch_size = sam_emb.shape[0]
                dummy_batch = self._prepare_dummy_batch(batch_size)

                self.model._batch_size = batch_size
                self.model._max_num_person = 1
                self.model.body_batch_idx = torch.arange(batch_size, device=self.device)
                self.model.hand_batch_idx = []

                condition_info = torch.zeros(batch_size, 3, device=self.device)
                condition_info[:, 2] = 1.1
                dummy_keypoints = torch.zeros(batch_size, 1, 3, device=self.device)
                dummy_keypoints[..., -1] = -2

                with torch.amp.autocast(enabled=False, device_type="cuda" if self.device.type == "cuda" else "cpu"):
                    decoder_tokens, pose_outs = self.model.forward_decoder(
                        image_embeddings=sam_emb,
                        keypoints=dummy_keypoints,
                        condition_info=condition_info,
                        batch=dummy_batch,
                    )

                final_pose_out = pose_outs[-1] if isinstance(pose_outs, (list, tuple)) else pose_outs
                if self.debug_vis and vis_chunk is not None and self.debug_vis_dir is not None:
                    pred_verts = final_pose_out["pred_vertices"]
                    pred_cam_t = final_pose_out["pred_cam_t"]
                    for local_idx in range(batch_size):
                        if self._saved_debug_frames >= self.debug_max_frames:
                            break
                        save_name = f"{debug_prefix or 'seq'}_frame{start + local_idx:03d}_overlay.png"
                        save_projected_vertices_overlay(
                            vis_chunk[local_idx],
                            pred_verts[local_idx],
                            pred_cam_t[local_idx],
                            dummy_batch["cam_int"][local_idx],
                            self.debug_vis_dir / save_name,
                            point_size=self.debug_point_size,
                        )
                        self._saved_debug_frames += 1

                if self.save_mode == "full":
                    decoder_tokens = detach_to_cpu(decoder_tokens)
                    pose_outs = detach_to_cpu(pose_outs)
                    dummy_batch = detach_to_cpu(dummy_batch)
                    condition_info = detach_to_cpu(condition_info)
                    dummy_keypoints = detach_to_cpu(dummy_keypoints)

                    for batch_idx in range(batch_size):
                        outputs.append(
                            {
                                "decoder_tokens": slice_by_batch(decoder_tokens, batch_idx, batch_size),
                                "pose_outs": slice_by_batch(pose_outs, batch_idx, batch_size),
                                "cam_int": slice_by_batch(dummy_batch["cam_int"], batch_idx, batch_size),
                                "ori_img_size": slice_by_batch(dummy_batch["ori_img_size"], batch_idx, batch_size),
                                "img_size": slice_by_batch(dummy_batch["img_size"], batch_idx, batch_size),
                                "bbox_center": slice_by_batch(dummy_batch["bbox_center"], batch_idx, batch_size),
                                "bbox_scale": slice_by_batch(dummy_batch["bbox_scale"], batch_idx, batch_size),
                                "affine_trans": slice_by_batch(dummy_batch["affine_trans"], batch_idx, batch_size),
                                "ray_cond": slice_by_batch(dummy_batch["ray_cond"], batch_idx, batch_size),
                                "condition_info": slice_by_batch(condition_info, batch_idx, batch_size),
                                "dummy_keypoints": slice_by_batch(dummy_keypoints, batch_idx, batch_size),
                            }
                        )
                else:
                    for batch_idx in range(batch_size):
                        pose_out = {}
                        for key in OT_LITE_POSE_KEYS:
                            dtype = self.cam_t_dtype if key == "pred_cam_t" else self.vertices_dtype
                            pose_out[key] = to_numpy_contiguous(
                                slice_by_batch(final_pose_out[key], batch_idx, batch_size),
                                dtype,
                            )

                        cam_int = to_numpy_contiguous(
                            slice_by_batch(dummy_batch["cam_int"], batch_idx, batch_size),
                            self.cam_int_dtype,
                        )
                        frame_dict = {
                            "pred_vertices": pose_out["pred_vertices"],
                            "pred_cam_t": pose_out["pred_cam_t"],
                            "cam_int": cam_int,
                            "pose_outs": pose_out,
                        }
                        outputs.append(frame_dict)

        if len(outputs) != total_frames:
            raise RuntimeError(f"Frame count mismatch: expected {total_frames}, got {len(outputs)}")
        return outputs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute SAM-3D-body decoder outputs for OpenGait RGB datasets.")
    parser.add_argument("--local_rank", type=int, default=0, help="Passed by torch.distributed.launch.")
    parser.add_argument("--local-rank", type=int, default=0, help="Passed by torchrun.")
    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["ccpg", "ccgr_mini", "casiab", "sustech1k"],
    )
    parser.add_argument("--input-root", required=True, help="Source RGB dataset root.")
    parser.add_argument("--output-root", required=True, help="Output dataset root.")
    parser.add_argument("--pretrained-lvm-root", default="pretrained_LVMs/sam-3d-body")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=32, help="Decoder batch size.")
    parser.add_argument("--hook-layer", type=int, default=-1, help="Backbone block index for the feature hook. Default -1 means the last block.")
    parser.add_argument("--rgb-mode", default="symlink", choices=["copy", "symlink", "hardlink"])
    parser.add_argument("--save-mode", default="minimal", choices=["minimal", "full"], help="minimal: OT-compatible lite dump with top-level pred_vertices/pred_cam_t/cam_int plus pose_outs. full: keep the old full decoder dump.")
    parser.add_argument("--vertices-dtype", default="float16", help="Numpy dtype used for pose tensors when save-mode=minimal.")
    parser.add_argument("--cam-t-dtype", default="float32", help="Numpy dtype used for pred_cam_t when save-mode=minimal.")
    parser.add_argument("--cam-int-dtype", default="float32", help="Numpy dtype used for cam_int when save-mode=minimal.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sequences.")
    parser.add_argument("--debug-vis", action="store_true", help="Save temporary 3D projection overlays for manual inspection.")
    parser.add_argument("--debug-max-frames", type=int, default=1, help="Number of frames to visualize across the whole run.")
    parser.add_argument("--debug-point-size", type=float, default=8.0)
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use for single-node distributed preprocessing.")
    parser.add_argument("--master-port", type=int, default=29600, help="Master port for auto-launched distributed preprocessing.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    maybe_launch_distributed(args)

    rank, world_size, local_rank = init_distributed_from_env(args.local_rank)
    is_main = rank == 0

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    pretrained_lvm_root = (REPO_ROOT / args.pretrained_lvm_root).resolve() if not osp.isabs(args.pretrained_lvm_root) else Path(args.pretrained_lvm_root)

    ensure_dir(output_root)

    if args.dataset_type == "ccpg":
        records = list(iter_ccpg_sequences(input_root))
    elif args.dataset_type == "ccgr_mini":
        records = list(iter_ccgr_mini_sequences(input_root))
    elif args.dataset_type == "casiab":
        records = list(iter_casiab_sequences(input_root))
    else:
        records = list(iter_sustech1k_sequences(input_root))

    if args.limit is not None:
        records = records[: args.limit]

    all_records = records
    records = records[rank::world_size]

    device = args.device
    if str(device).startswith("cuda") and torch.cuda.is_available():
        if world_size > 1:
            device = f"cuda:{local_rank}"
        elif device == "cuda":
            device = "cuda:0"

    if is_main:
        print(f"[Info] Dataset type     : {args.dataset_type}")
        print(f"[Info] Input root       : {input_root}")
        print(f"[Info] Output root      : {output_root}")
        print(f"[Info] Sequences found  : {len(all_records)}")
        print(f"[Info] Local shard      : {len(records)} / rank {rank} of {world_size}")
        print(f"[Info] RGB mode         : {args.rgb_mode}")
        print(f"[Info] Save mode        : {args.save_mode}")
        print(f"[Info] Vertices dtype   : {args.vertices_dtype}")
        print(f"[Info] Cam_t dtype      : {args.cam_t_dtype}")
        print(f"[Info] Cam_int dtype    : {args.cam_int_dtype}")
        print(f"[Info] Decoder device   : {device}")
        print(f"[Info] Chunk size       : {args.chunk_size}")
        print(f"[Info] Hook layer       : {args.hook_layer}")
        print(f"[Info] Debug vis        : {args.debug_vis}")
        print(f"[Info] GPUs requested   : {args.gpus}")

    debug_vis_dir = output_root / "_debug_projection_vis" if args.debug_vis and is_main else None
    processor = SAM3DDecoderPreprocessor(
        pretrained_lvm_root=pretrained_lvm_root,
        device=device,
        hook_layer=args.hook_layer,
        save_mode=args.save_mode,
        debug_vis=args.debug_vis and is_main,
        debug_max_frames=args.debug_max_frames,
        debug_point_size=args.debug_point_size,
        debug_vis_dir=debug_vis_dir,
        vertices_dtype=np.dtype(args.vertices_dtype),
        cam_t_dtype=np.dtype(args.cam_t_dtype),
        cam_int_dtype=np.dtype(args.cam_int_dtype),
    )

    try:
        progress = tqdm(records, desc=f"Preprocessing {args.dataset_type} [rank {rank}]") if is_main else records
        for record in progress:
            out_seq_dir = output_root / record.pid / record.seq_type / record.view
            ensure_dir(out_seq_dir)

            sam_out_path = out_seq_dir / "01-sam3d-body_decoder.pkl"
            need_sam = args.overwrite or not sam_out_path.exists()

            if args.dataset_type != "ccgr_mini":
                rgb_out_path = out_seq_dir / "00-rgb.pkl"
                need_rgb = args.overwrite or not rgb_out_path.exists()

                if not need_rgb and not need_sam:
                    continue

                if need_rgb:
                    copy_or_link(record.rgb_path, rgb_out_path, args.rgb_mode)

                if need_sam:
                    rgb_seq = load_rgb_pickle_sequence(record.rgb_path)
                    debug_prefix = f"{record.pid}_{record.seq_type}_{record.view}"
                    sam_sequence = processor.process_sequence(rgb_seq, chunk_size=args.chunk_size, debug_prefix=debug_prefix)
                    save_pickle(sam_out_path, sam_sequence)
            else:
                rgb_out_path = out_seq_dir / "00-rgb.avi"
                legacy_rgb_pkl_path = out_seq_dir / "00-rgb.pkl"
                if legacy_rgb_pkl_path.exists() or legacy_rgb_pkl_path.is_symlink():
                    legacy_rgb_pkl_path.unlink()
                need_rgb = args.overwrite or not rgb_out_path.exists()

                if not need_rgb and not need_sam:
                    continue

                if need_rgb:
                    copy_or_link(record.rgb_path, rgb_out_path, args.rgb_mode)

                if need_sam:
                    rgb_seq = load_ccgr_video_sequence(record.rgb_path)
                    debug_prefix = f"{record.pid}_{record.seq_type}_{record.view}"
                    sam_sequence = processor.process_sequence(rgb_seq, chunk_size=args.chunk_size, debug_prefix=debug_prefix)
                    save_pickle(sam_out_path, sam_sequence)

    finally:
        processor.close()
        destroy_distributed()


if __name__ == "__main__":
    main()
