#!/usr/bin/env python3
"""Precompute ROMP/SMPL geometry for OpenGait RGB sequences.

Run this script in the ROMP preprocessing environment. It writes per-frame
geometry pkl files that can later be consumed by an OpenGait model branch
without importing ROMP during DDP training/inference.
"""

import argparse
import copy
import os
import pickle
import subprocess
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


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
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def parse_gpu_ids(spec: str) -> List[int]:
    ids = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


def maybe_launch_multi_gpu(args: argparse.Namespace) -> None:
    gpu_ids = parse_gpu_ids(args.gpus)
    if args.single_rgb_pkl or args.world_size > 1 or len(gpu_ids) <= 1:
        return

    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset-type",
            args.dataset_type,
            "--input-root",
            args.input_root,
            "--output-root",
            args.output_root,
            "--target-h",
            str(args.target_h),
            "--target-w",
            str(args.target_w),
            "--focal-scale",
            str(args.focal_scale),
            "--gpu",
            "0",
            "--gpus",
            "0",
            "--rank",
            str(rank),
            "--world-size",
            str(len(gpu_ids)),
            "--rgb-mode",
            args.rgb_mode,
            "--output-name",
            args.output_name,
            "--missing-policy",
            args.missing_policy,
            "--vertices-dtype",
            args.vertices_dtype,
            "--joints-dtype",
            args.joints_dtype,
            "--cam-t-dtype",
            args.cam_t_dtype,
            "--debug-max-frames",
            str(args.debug_max_frames),
            "--debug-point-size",
            str(args.debug_point_size),
            "--romp-home",
            args.romp_home,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.debug_vis:
            cmd.append("--debug-vis")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(
            f"[Launch] rank={rank}/{len(gpu_ids)} physical_gpu={gpu_id} "
            f"CUDA_VISIBLE_DEVICES={gpu_id}: {' '.join(cmd)}"
        )
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(max(exit_codes))
    raise SystemExit(0)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if mode == "none":
        return
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
                yield SequenceRecord(
                    pid=pid,
                    seq_type=seq_type,
                    view=view,
                    seq_dir=seq_dir,
                    rgb_path=rgb_candidates[0],
                )


def iter_ccgr_sequences(dataset_root: Path) -> Iterable[SequenceRecord]:
    """Iterate the CCGR-MINI pid/sequence/view directories containing AVI RGB."""
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
                rgb_candidates = sorted(seq_dir.glob("*.avi"))
                if not rgb_candidates:
                    continue
                yield SequenceRecord(
                    pid=pid,
                    seq_type=seq_type,
                    view=view,
                    seq_dir=seq_dir,
                    rgb_path=rgb_candidates[0],
                )


def record_from_single_rgb_pkl(path: Path) -> SequenceRecord:
    path = path.resolve()
    parent = path.parent
    view = parent.name
    seq_type = parent.parent.name if parent.parent != parent else "single"
    pid = parent.parent.parent.name if parent.parent.parent != parent.parent else "single"
    return SequenceRecord(pid=pid, seq_type=seq_type, view=view, seq_dir=parent, rgb_path=path)


def load_rgb_sequence(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".avi":
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"Cannot decode any RGB frames from {path}")
        return np.stack(frames, axis=0).transpose(0, 3, 1, 2)

    seq = np.asarray(load_pickle(path))
    if seq.ndim != 4:
        raise ValueError(f"Unexpected RGB pkl shape at {path}: {seq.shape}")
    if seq.shape[1] == 3:
        return seq
    if seq.shape[-1] == 3:
        return seq.transpose(0, 3, 1, 2)
    raise ValueError(f"Cannot interpret RGB pkl shape at {path}: {seq.shape}")


def chw_sequence_to_uint8_rgb(seq_chw: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq_chw)
    if seq.ndim != 4 or seq.shape[1] != 3:
        raise ValueError(f"Expected [T, 3, H, W], got {seq.shape}")
    seq = seq.transpose(0, 2, 3, 1)
    if np.issubdtype(seq.dtype, np.floating) and float(np.nanmax(seq)) <= 1.5:
        seq = seq * 255.0
    return np.clip(seq, 0, 255).astype(np.uint8)


def resize_rgb_sequence(seq_rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in seq_rgb
    ]
    return np.stack(resized, axis=0)


def build_cam_int(target_h: int, target_w: int, focal_scale: float) -> np.ndarray:
    focal = max(target_h, target_w) * focal_scale
    cam_int = np.eye(3, dtype=np.float32)
    cam_int[0, 0] = focal
    cam_int[1, 1] = focal
    cam_int[0, 2] = target_w / 2.0
    cam_int[1, 2] = target_h / 2.0
    return cam_int


def to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def first_existing(outputs: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def infer_person_count(outputs: Dict[str, Any]) -> int:
    for key in ("verts", "vertices", "joints", "pj2d", "cam", "cam_trans"):
        value = outputs.get(key)
        if value is None:
            continue
        arr = to_numpy(value)
        if arr.ndim >= 2:
            return int(arr.shape[0])
    return 0


def select_person_index(outputs: Dict[str, Any]) -> int:
    person_count = infer_person_count(outputs)
    if person_count <= 0:
        raise ValueError("ROMP produced no people for this frame")

    pj2d = first_existing(outputs, ("pj2d_org", "pj2d", "pj2ds", "joints_2d"))
    if pj2d is not None:
        pts = to_numpy(pj2d, np.float32)
        if pts.ndim == 3 and pts.shape[0] == person_count:
            xy = pts[..., :2]
            finite = np.isfinite(xy).all(axis=-1)
            areas = np.zeros((person_count,), dtype=np.float32)
            for idx in range(person_count):
                if not finite[idx].any():
                    continue
                curr = xy[idx][finite[idx]]
                wh = curr.max(axis=0) - curr.min(axis=0)
                areas[idx] = max(float(wh[0] * wh[1]), 0.0)
            return int(np.argmax(areas))

    conf = first_existing(outputs, ("center_confs", "center_conf", "scores", "score"))
    if conf is not None:
        conf_np = to_numpy(conf, np.float32).reshape(-1)
        if conf_np.size >= person_count:
            return int(np.argmax(conf_np[:person_count]))

    return 0


def weak_perspective_to_cam_t(
    cam: np.ndarray,
    cam_int: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32).reshape(-1)
    if cam.size < 3:
        raise ValueError(f"Expected weak-perspective cam with 3 values, got {cam.shape}")
    scale, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])
    focal = float((cam_int[0, 0] + cam_int[1, 1]) * 0.5)
    img_extent = float(max(target_h, target_w))
    z = 2.0 * focal / max(scale * img_extent, 1e-6)
    return np.asarray([tx, ty, z], dtype=np.float32)


def estimate_perspective_translation(
    joints_3d: np.ndarray,
    joints_2d: np.ndarray,
    cam_int: np.ndarray,
    min_points: int = 6,
) -> Optional[np.ndarray]:
    joints_3d = np.asarray(joints_3d, dtype=np.float32)
    joints_2d = np.asarray(joints_2d, dtype=np.float32)
    if joints_3d.ndim != 2 or joints_2d.ndim != 2:
        return None
    if joints_3d.shape[0] == 0 or joints_2d.shape[0] == 0:
        return None

    count = min(joints_3d.shape[0], joints_2d.shape[0])
    xyz = joints_3d[:count, :3]
    uv = joints_2d[:count, :2]
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1)
    if int(valid.sum()) < min_points:
        return None

    xyz = xyz[valid]
    uv = uv[valid]
    fx, fy = float(cam_int[0, 0]), float(cam_int[1, 1])
    cx, cy = float(cam_int[0, 2]), float(cam_int[1, 2])

    rows = []
    rhs = []
    for (x, y, z), (u, v) in zip(xyz, uv):
        u_c = float(u - cx)
        v_c = float(v - cy)
        rows.append([fx, 0.0, -u_c])
        rhs.append(u_c * float(z) - fx * float(x))
        rows.append([0.0, fy, -v_c])
        rhs.append(v_c * float(z) - fy * float(y))

    try:
        trans, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float32), np.asarray(rhs, dtype=np.float32), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(trans).all() or trans[2] <= 1e-4:
        return None
    return trans.astype(np.float32)


def project_points(points_3d: np.ndarray, cam_t: np.ndarray, cam_int: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_3d, dtype=np.float32) + np.asarray(cam_t, dtype=np.float32).reshape(1, 3)
    z = np.clip(points_cam[:, 2], 1e-4, None)
    u = (points_cam[:, 0] / z) * float(cam_int[0, 0]) + float(cam_int[0, 2])
    v = (points_cam[:, 1] / z) * float(cam_int[1, 1]) + float(cam_int[1, 2])
    return np.stack([u, v], axis=-1).astype(np.float32)


def selected_output_array(outputs: Dict[str, Any], keys: Tuple[str, ...], person_idx: int) -> Optional[np.ndarray]:
    value = first_existing(outputs, keys)
    if value is None:
        return None
    arr = to_numpy(value, np.float32)
    if arr.ndim >= 2 and arr.shape[0] > person_idx:
        arr = arr[person_idx]
    return np.ascontiguousarray(arr)


def tensor_dict_to_numpy(outputs: Dict[str, Any]) -> Dict[str, Any]:
    converted = {}
    for key, value in outputs.items():
        if torch.is_tensor(value):
            converted[key] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            converted[key] = tensor_dict_to_numpy(value)
        else:
            converted[key] = value
    return converted


def extract_cam_t(
    outputs: Dict[str, Any],
    person_idx: int,
    joints_3d: np.ndarray,
    cam_int: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, str]:
    pj2d_org = first_existing(outputs, ("pj2d_org", "joints_2d_org", "keypoints_2d_org"))
    if pj2d_org is not None and joints_3d.size:
        pj2d_np = to_numpy(pj2d_org, np.float32)
        if pj2d_np.ndim == 3 and pj2d_np.shape[0] > person_idx:
            pj2d_np = pj2d_np[person_idx]
        cam_t = estimate_perspective_translation(joints_3d, pj2d_np, cam_int)
        if cam_t is not None:
            return cam_t, "fit_perspective_from_pj2d_org"

    trans = first_existing(
        outputs,
        ("cam_trans", "cam_t", "pred_cam_t", "trans", "translation", "cam_transes"),
    )
    if trans is not None:
        trans_np = to_numpy(trans, np.float32)
        if trans_np.ndim >= 2 and trans_np.shape[0] > person_idx:
            return np.asarray(trans_np[person_idx]).reshape(-1)[:3].astype(np.float32), "translation"
        return np.asarray(trans_np).reshape(-1)[:3].astype(np.float32), "translation"

    cam = first_existing(outputs, ("cam", "cams", "pred_cam"))
    if cam is None:
        raise ValueError("ROMP output has neither camera translation nor weak-perspective cam")
    cam_np = to_numpy(cam, np.float32)
    if cam_np.ndim >= 2 and cam_np.shape[0] > person_idx:
        cam_np = cam_np[person_idx]
    return weak_perspective_to_cam_t(cam_np, cam_int, target_h, target_w), "weak_perspective_fallback"


def extract_pose_fields(outputs: Dict[str, Any], person_idx: int) -> Dict[str, np.ndarray]:
    pose_fields: Dict[str, np.ndarray] = {}
    thetas = first_existing(outputs, ("smpl_thetas", "theta", "poses", "pose"))
    if thetas is not None:
        theta_np = to_numpy(thetas, np.float32)
        if theta_np.ndim >= 2 and theta_np.shape[0] > person_idx:
            theta_np = theta_np[person_idx]
        theta_np = theta_np.reshape(-1).astype(np.float32)
        pose_fields["smpl_thetas"] = theta_np
        if theta_np.size >= 3:
            pose_fields["global_rot"] = theta_np[:3]
        if theta_np.size > 3:
            pose_fields["body_pose"] = theta_np[3:]

    betas = first_existing(outputs, ("smpl_betas", "betas", "shape"))
    if betas is not None:
        beta_np = to_numpy(betas, np.float32)
        if beta_np.ndim >= 2 and beta_np.shape[0] > person_idx:
            beta_np = beta_np[person_idx]
        pose_fields["smpl_betas"] = beta_np.reshape(-1).astype(np.float32)
        pose_fields["shape"] = pose_fields["smpl_betas"]

    return pose_fields


def extract_frame_geometry(
    outputs: Dict[str, Any],
    cam_int: np.ndarray,
    target_h: int,
    target_w: int,
    vertices_dtype: np.dtype,
    joints_dtype: np.dtype,
    cam_t_dtype: np.dtype,
) -> Dict[str, Any]:
    person_idx = select_person_index(outputs)

    verts = first_existing(outputs, ("verts", "vertices", "pred_vertices"))
    if verts is None:
        raise ValueError("ROMP output does not contain verts/vertices")
    verts_np = to_numpy(verts, np.float32)
    if verts_np.ndim == 3:
        verts_np = verts_np[person_idx]
    verts_np = verts_np.reshape(-1, 3)

    joints = first_existing(
        outputs,
        ("joints", "pred_keypoints_3d", "joints_h36m17", "kp3d", "keypoints_3d"),
    )
    if joints is None:
        joints_np = np.zeros((0, 3), dtype=np.float32)
    else:
        joints_np = to_numpy(joints, np.float32)
        if joints_np.ndim == 3:
            joints_np = joints_np[person_idx]
        joints_np = joints_np.reshape(-1, joints_np.shape[-1])[..., :3]

    cam_t, cam_source = extract_cam_t(outputs, person_idx, joints_np, cam_int, target_h, target_w)
    pose_fields = extract_pose_fields(outputs, person_idx)
    pj2d_org = selected_output_array(outputs, ("pj2d_org", "joints_2d_org", "keypoints_2d_org"), person_idx)
    verts_camed_org = selected_output_array(outputs, ("verts_camed_org",), person_idx)
    raw_cam = selected_output_array(outputs, ("cam", "cams", "pred_cam"), person_idx)
    raw_cam_trans = selected_output_array(
        outputs,
        ("cam_trans", "cam_t", "pred_cam_t", "trans", "translation", "cam_transes"),
        person_idx,
    )

    reproj_error_mean = np.asarray(np.nan, dtype=np.float32)
    reproj_error_median = np.asarray(np.nan, dtype=np.float32)
    if pj2d_org is not None and joints_np.size:
        count = min(joints_np.shape[0], pj2d_org.shape[0])
        pred_2d = project_points(joints_np[:count], cam_t, cam_int)
        target_2d = pj2d_org[:count, :2]
        valid = np.isfinite(pred_2d).all(axis=1) & np.isfinite(target_2d).all(axis=1)
        if valid.any():
            error = np.linalg.norm(pred_2d[valid] - target_2d[valid], axis=1)
            reproj_error_mean = np.asarray(error.mean(), dtype=np.float32)
            reproj_error_median = np.asarray(np.median(error), dtype=np.float32)

    pose_out = {
        "pred_vertices": to_numpy(verts_np, vertices_dtype),
        "pred_keypoints_3d": to_numpy(joints_np, joints_dtype),
        "pred_cam_t": to_numpy(cam_t, cam_t_dtype),
        "global_rot": to_numpy(pose_fields.get("global_rot", np.zeros(3, dtype=np.float32)), np.float32),
        "global_rot_type": "axis_angle",
        "romp_person_index": int(person_idx),
        "romp_cam_source": cam_source,
        "romp_reproj_error_mean_px": reproj_error_mean,
        "romp_reproj_error_median_px": reproj_error_median,
    }
    if pj2d_org is not None:
        pose_out["pj2d_org"] = to_numpy(pj2d_org, np.float32)
    if verts_camed_org is not None:
        pose_out["verts_camed_org"] = to_numpy(verts_camed_org, np.float32)
    if raw_cam is not None:
        pose_out["romp_cam"] = to_numpy(raw_cam, np.float32)
    if raw_cam_trans is not None:
        pose_out["romp_raw_cam_trans"] = to_numpy(raw_cam_trans, np.float32)
    for key, value in pose_fields.items():
        pose_out[key] = to_numpy(value, np.float32)

    return {
        "pred_vertices": pose_out["pred_vertices"],
        "pred_cam_t": pose_out["pred_cam_t"],
        "pred_keypoints_3d": pose_out["pred_keypoints_3d"],
        "cam_int": to_numpy(cam_int, np.float32),
        "pose_outs": pose_out,
        "geometry_source": "simple_romp",
        "valid": True,
    }


def make_blank_invalid_geometry(
    cam_int: np.ndarray,
    vertices_dtype: np.dtype,
    joints_dtype: np.dtype,
    cam_t_dtype: np.dtype,
    reason: str,
    num_vertices: int = 6890,
    num_joints: int = 71,
) -> Dict[str, Any]:
    verts = np.zeros((num_vertices, 3), dtype=np.float32)
    joints = np.zeros((num_joints, 3), dtype=np.float32)
    cam_t = np.asarray([0.0, 0.0, 2.2], dtype=np.float32)
    verts_camed_org = np.full((num_vertices, 3), -1.0, dtype=np.float32)
    pose_out = {
        "pred_vertices": to_numpy(verts, vertices_dtype),
        "pred_keypoints_3d": to_numpy(joints, joints_dtype),
        "pred_cam_t": to_numpy(cam_t, cam_t_dtype),
        "global_rot": np.zeros((3,), dtype=np.float32),
        "global_rot_type": "axis_angle",
        "verts_camed_org": verts_camed_org,
        "romp_person_index": -1,
        "romp_cam_source": "missing_blank",
        "romp_reproj_error_mean_px": np.asarray(np.nan, dtype=np.float32),
        "romp_reproj_error_median_px": np.asarray(np.nan, dtype=np.float32),
        "romp_blank_geometry": True,
    }
    return {
        "pred_vertices": pose_out["pred_vertices"],
        "pred_cam_t": pose_out["pred_cam_t"],
        "pred_keypoints_3d": pose_out["pred_keypoints_3d"],
        "cam_int": to_numpy(cam_int, np.float32),
        "pose_outs": pose_out,
        "geometry_source": "simple_romp_missing_blank",
        "valid": False,
        "missing_reason": reason,
    }


def clone_invalid_from_previous(
    previous: Dict[str, Any],
    reason: str,
    reuse_source: str = "previous",
) -> Dict[str, Any]:
    cloned = copy.deepcopy(previous)
    cloned["valid"] = False
    cloned["missing_reason"] = reason
    if isinstance(cloned.get("pose_outs"), dict):
        cloned["pose_outs"][f"romp_reused_{reuse_source}"] = True
    return cloned


def save_projected_vertices_overlay(
    img_rgb: np.ndarray,
    vertices: np.ndarray,
    cam_t: np.ndarray,
    cam_int: np.ndarray,
    save_path: Path,
    point_size: float,
) -> None:
    img = img_rgb.astype(np.float32) / 255.0
    verts_cam = vertices.astype(np.float32) + cam_t.reshape(1, 3).astype(np.float32)
    z = np.clip(verts_cam[:, 2], 1e-3, None)
    u = (verts_cam[:, 0] / z) * cam_int[0, 0] + cam_int[0, 2]
    v = (verts_cam[:, 1] / z) * cam_int[1, 1] + cam_int[1, 2]
    h, w = img.shape[:2]
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(u) & np.isfinite(v)

    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(7, 7 * h / max(w, 1)))
    ax.imshow(img)
    ax.scatter(u[mask], v[mask], s=point_size, c="red", alpha=0.55)
    ax.set_title("ROMP mesh projection")
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_official_vertices_overlay(
    img_rgb: np.ndarray,
    vertices_camed_org: np.ndarray,
    save_path: Path,
    point_size: float,
) -> None:
    img = img_rgb.astype(np.float32) / 255.0
    verts = np.asarray(vertices_camed_org, dtype=np.float32)
    u = verts[:, 0]
    v = verts[:, 1]
    h, w = img.shape[:2]
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(u) & np.isfinite(v)

    ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(7, 7 * h / max(w, 1)))
    ax.imshow(img)
    ax.scatter(u[mask], v[mask], s=point_size, c="red", alpha=0.55)
    ax.set_title("ROMP official weak-perspective vertices")
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def depth_gray_colors(points: np.ndarray) -> np.ndarray:
    depth = points[:, 2].astype(np.float32)
    valid = np.isfinite(depth)
    colors = np.full((points.shape[0], 3), 160, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = (depth - lo) / max(hi - lo, 1e-6)
        gray = (80.0 + 120.0 * np.clip(norm, 0.0, 1.0)).astype(np.uint8)
        colors[:, :] = gray[:, None]
    return colors


def load_smpl_faces(romp_home: Optional[Path]) -> Optional[np.ndarray]:
    candidates = []
    if romp_home is not None:
        candidates.append(romp_home / "SMPL_NEUTRAL.pth")
        candidates.append(romp_home / "SMPLA_NEUTRAL.pth")
    candidates.append(Path.home() / ".romp" / "SMPL_NEUTRAL.pth")
    candidates.append(Path.home() / ".romp" / "SMPLA_NEUTRAL.pth")

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("f", "faces", "face"):
                if key not in data:
                    continue
                faces = to_numpy(data[key], np.int64)
                if faces.ndim == 2 and faces.shape[1] == 3:
                    return faces
    return None


def save_ascii_ply(
    points: np.ndarray,
    colors: np.ndarray,
    save_path: Path,
    faces: Optional[np.ndarray] = None,
) -> None:
    ensure_dir(save_path.parent)
    if faces is None:
        faces = np.zeros((0, 3), dtype=np.int64)
    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int label\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


class ROMPGeometryPreprocessor:
    def __init__(
        self,
        target_h: int,
        target_w: int,
        focal_scale: float,
        gpu: int,
        vertices_dtype: np.dtype,
        joints_dtype: np.dtype,
        cam_t_dtype: np.dtype,
        missing_policy: str,
        romp_home: Optional[Path] = None,
    ):
        self.target_h = target_h
        self.target_w = target_w
        self.cam_int = build_cam_int(target_h, target_w, focal_scale)
        self.vertices_dtype = np.dtype(vertices_dtype)
        self.joints_dtype = np.dtype(joints_dtype)
        self.cam_t_dtype = np.dtype(cam_t_dtype)
        self.missing_policy = missing_policy
        self.romp_home = Path(romp_home).expanduser().resolve() if romp_home else None
        self.model = self._build_romp(gpu)

    def _build_romp(self, gpu: int):
        import romp

        settings_ctor = getattr(romp.main, "default_settings", None)
        if settings_ctor is None:
            raise RuntimeError("Cannot find romp.main.default_settings in simple_romp.")
        settings = settings_ctor() if callable(settings_ctor) else copy.deepcopy(settings_ctor)

        if self.romp_home is not None:
            model_path = self.romp_home / "ROMP.pkl"
            smpl_path = self.romp_home / "SMPL_NEUTRAL.pth"
            if not model_path.is_file() or not smpl_path.is_file():
                raise FileNotFoundError(
                    f"romp_home must contain ROMP.pkl and SMPL_NEUTRAL.pth: {self.romp_home}"
                )
            settings.model_path = str(model_path)
            settings.smpl_path = str(smpl_path)

        for key, value in {
            "mode": "image",
            "GPU": int(gpu),
            "gpu": int(gpu),
            "show": False,
            "render_mesh": False,
            "save_video": False,
            "calc_smpl": True,
            "center_thresh": 0.12,
        }.items():
            try:
                setattr(settings, key, value)
            except Exception:
                pass

        if not hasattr(romp, "ROMP"):
            raise RuntimeError("Cannot find romp.ROMP. Please check simple_romp installation.")
        engine = romp.ROMP(settings)
        # simple-romp always wraps the network in DataParallel, including CPU
        # mode. This helper is already launched once per assigned device.
        if isinstance(engine.model, torch.nn.DataParallel):
            engine.model = engine.model.module
        return engine

    def _run_frame(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        # Official simple_romp examples use cv2.imread output, so pass BGR.
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            outputs, image_pad_info = self.model.single_image_forward(frame_bgr)
            if outputs is not None and self.model.settings.temporal_optimize:
                outputs = self.model.temporal_optimization(outputs, signal_ID=0)
            if outputs is not None:
                from romp.post_parser import body_mesh_projection2image
                from romp.utils import convert_cam_to_3d_trans

                outputs["cam_trans"] = convert_cam_to_3d_trans(outputs["cam"])
                if self.model.settings.calc_smpl:
                    outputs = self.model.smpl_parser(outputs, root_align=self.model.settings.root_align)
                    outputs.update(
                        body_mesh_projection2image(
                            outputs["joints"],
                            outputs["cam"],
                            vertices=outputs["verts"],
                            input2org_offsets=image_pad_info,
                        )
                    )
        if outputs is None:
            raise ValueError("ROMP returned None")
        if not isinstance(outputs, dict):
            raise ValueError(f"Expected ROMP output dict, got {type(outputs)!r}")
        return tensor_dict_to_numpy(outputs)

    def process_sequence(
        self,
        rgb_sequence_chw: np.ndarray,
        debug_vis_dir: Optional[Path] = None,
        debug_prefix: str = "seq",
        debug_max_frames: int = 0,
        debug_point_size: float = 8.0,
        smpl_faces: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        rgb_hwc = chw_sequence_to_uint8_rgb(rgb_sequence_chw)
        rgb_hwc = resize_rgb_sequence(rgb_hwc, self.target_h, self.target_w)

        frames: List[Dict[str, Any]] = []
        pending_initial_missing: List[Tuple[int, str]] = []
        saved_debug = 0
        for frame_idx, frame_rgb in enumerate(rgb_hwc):
            try:
                outputs = self._run_frame(frame_rgb)
                frame_dict = extract_frame_geometry(
                    outputs,
                    self.cam_int,
                    self.target_h,
                    self.target_w,
                    self.vertices_dtype,
                    self.joints_dtype,
                    self.cam_t_dtype,
                )
            except Exception as exc:
                if self.missing_policy == "fail":
                    raise RuntimeError(f"ROMP failed at frame {frame_idx}: {exc}") from exc
                if self.missing_policy == "blank":
                    frame_dict = make_blank_invalid_geometry(
                        self.cam_int,
                        self.vertices_dtype,
                        self.joints_dtype,
                        self.cam_t_dtype,
                        str(exc),
                    )
                    frames.append(frame_dict)
                    continue
                if self.missing_policy != "previous":
                    raise ValueError(f"Unsupported missing policy: {self.missing_policy}")
                if not frames:
                    pending_initial_missing.append((frame_idx, str(exc)))
                    continue
                frame_dict = clone_invalid_from_previous(frames[-1], str(exc), reuse_source="previous")
            else:
                if pending_initial_missing:
                    tqdm.write(
                        f"[Warn] {debug_prefix}: first {len(pending_initial_missing)} frame(s) "
                        "had no ROMP detection; backfilled from the first valid frame."
                    )
                    for _, reason in pending_initial_missing:
                        frames.append(clone_invalid_from_previous(frame_dict, reason, reuse_source="next"))
                    pending_initial_missing.clear()

            frames.append(frame_dict)

            if debug_vis_dir is not None and saved_debug < debug_max_frames:
                verts = to_numpy(frame_dict["pred_vertices"], np.float32)
                cam_t = to_numpy(frame_dict["pred_cam_t"], np.float32)
                cam_int = to_numpy(frame_dict["cam_int"], np.float32)
                pose_out = frame_dict.get("pose_outs", {})
                verts_camed_org = pose_out.get("verts_camed_org") if isinstance(pose_out, dict) else None
                if verts_camed_org is not None:
                    save_official_vertices_overlay(
                        frame_rgb,
                        to_numpy(verts_camed_org, np.float32),
                        debug_vis_dir / f"{debug_prefix}_frame{frame_idx:03d}_overlay.png",
                        debug_point_size,
                    )
                else:
                    save_projected_vertices_overlay(
                        frame_rgb,
                        verts,
                        cam_t,
                        cam_int,
                        debug_vis_dir / f"{debug_prefix}_frame{frame_idx:03d}_overlay.png",
                        debug_point_size,
                    )
                save_projected_vertices_overlay(
                    frame_rgb,
                    verts,
                    cam_t,
                    cam_int,
                    debug_vis_dir / f"{debug_prefix}_frame{frame_idx:03d}_perspective_overlay.png",
                    debug_point_size,
                )
                points = verts + cam_t.reshape(1, 3)
                colors = depth_gray_colors(points)
                faces = None
                if smpl_faces is not None and smpl_faces.size and int(smpl_faces.max()) < points.shape[0]:
                    faces = smpl_faces
                save_ascii_ply(
                    points,
                    colors,
                    debug_vis_dir / f"{debug_prefix}_frame{frame_idx:03d}_romp_mesh.ply",
                    faces=faces,
                )
                cv2.imwrite(
                    str(debug_vis_dir / f"{debug_prefix}_frame{frame_idx:03d}_input.png"),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                )
                saved_debug += 1

        if pending_initial_missing and not frames:
            num_vertices = 6890
            if smpl_faces is not None and smpl_faces.size:
                num_vertices = max(num_vertices, int(np.max(smpl_faces)) + 1)
            tqdm.write(
                f"[Warn] {debug_prefix}: all {len(pending_initial_missing)} frame(s) had no "
                "ROMP detection; wrote blank invalid geometry."
            )
            for _, reason in pending_initial_missing:
                frames.append(
                    make_blank_invalid_geometry(
                        self.cam_int,
                        self.vertices_dtype,
                        self.joints_dtype,
                        self.cam_t_dtype,
                        reason,
                        num_vertices=num_vertices,
                    )
                )

        return frames


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute ROMP/SMPL geometry for OpenGait RGB pkl datasets.")
    parser.add_argument("--dataset-type", default="ccpg", choices=["ccpg", "ccgr"])
    parser.add_argument("--input-root", help="Source RGB dataset root. Required unless --single-rgb-pkl is used.")
    parser.add_argument("--output-root", required=True, help="Output dataset root or debug output root.")
    parser.add_argument("--single-rgb-pkl", help="Process one RGB pkl directly for quick geometry visualization.")
    parser.add_argument("--target-h", type=int, default=512)
    parser.add_argument("--target-w", type=int, default=256)
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=443.4 / 512.0,
        help="Source camera focal/max(H,W). Default matches ROMP's internal focal length.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id passed to simple_romp. Use -1 for CPU.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for sharded preprocessing, e.g. 0,1,2,3.")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for internal multi-GPU launch.")
    parser.add_argument("--world-size", type=int, default=1, help="Total shards for internal multi-GPU launch.")
    parser.add_argument("--rgb-mode", default="symlink", choices=["copy", "symlink", "hardlink", "none"])
    parser.add_argument("--output-name", default="01-romp-smpl_geometry.pkl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sequences.")
    parser.add_argument("--missing-policy", default="previous", choices=["previous", "blank", "fail"])
    parser.add_argument("--vertices-dtype", default="float16")
    parser.add_argument("--joints-dtype", default="float32")
    parser.add_argument("--cam-t-dtype", default="float32")
    parser.add_argument("--debug-vis", action="store_true", help="Save overlay png and PLY for manual inspection.")
    parser.add_argument("--debug-max-frames", type=int, default=3)
    parser.add_argument("--debug-point-size", type=float, default=7.0)
    parser.add_argument("--romp-home", default=str(Path.home() / ".romp"), help="Used only to find SMPL faces for PLY mesh export.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    maybe_launch_multi_gpu(args)

    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    if args.single_rgb_pkl:
        records = [record_from_single_rgb_pkl(Path(args.single_rgb_pkl))]
    else:
        if not args.input_root:
            raise ValueError("--input-root is required unless --single-rgb-pkl is used")
        input_root = Path(args.input_root).resolve()
        if args.dataset_type == "ccpg":
            records = list(iter_ccpg_sequences(input_root))
        elif args.dataset_type == "ccgr":
            records = list(iter_ccgr_sequences(input_root))
        else:
            raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    if args.limit is not None:
        records = records[: args.limit]

    if args.world_size > 1:
        records = records[args.rank::args.world_size]

    print(f"[Info] Records found     : {len(records)}")
    print(f"[Info] Output root       : {output_root}")
    print(f"[Info] Target size       : {args.target_h}x{args.target_w}")
    print(f"[Info] GPU               : {args.gpu}")
    print(f"[Info] Shard             : rank {args.rank} / world_size {args.world_size}")
    print(f"[Info] Missing policy    : {args.missing_policy}")
    print(f"[Info] Debug vis         : {args.debug_vis}")

    smpl_faces = load_smpl_faces(Path(args.romp_home) if args.romp_home else None)
    if smpl_faces is not None:
        print(f"[Info] SMPL faces loaded : {smpl_faces.shape[0]}")
    else:
        print("[Info] SMPL faces loaded : no, PLY will be point-only")

    processor = ROMPGeometryPreprocessor(
        target_h=args.target_h,
        target_w=args.target_w,
        focal_scale=args.focal_scale,
        gpu=args.gpu,
        vertices_dtype=np.dtype(args.vertices_dtype),
        joints_dtype=np.dtype(args.joints_dtype),
        cam_t_dtype=np.dtype(args.cam_t_dtype),
        missing_policy=args.missing_policy,
        romp_home=Path(args.romp_home) if args.romp_home else None,
    )

    for record in tqdm(records, desc="Preprocessing ROMP geometry"):
        out_seq_dir = output_root / record.pid / record.seq_type / record.view
        ensure_dir(out_seq_dir)

        geom_out_path = out_seq_dir / args.output_name
        rgb_suffix = record.rgb_path.suffix.lower()
        rgb_out_path = out_seq_dir / f"00-rgb{rgb_suffix}"
        if geom_out_path.exists() and not args.overwrite:
            continue

        if args.rgb_mode != "none":
            copy_or_link(record.rgb_path, rgb_out_path, args.rgb_mode)

        rgb_seq = load_rgb_sequence(record.rgb_path)
        debug_dir = None
        if args.debug_vis:
            debug_dir = output_root / "_debug_romp_vis" / record.pid / record.seq_type / record.view
        debug_prefix = f"{record.pid}_{record.seq_type}_{record.view}"
        geom_sequence = processor.process_sequence(
            rgb_seq,
            debug_vis_dir=debug_dir,
            debug_prefix=debug_prefix,
            debug_max_frames=args.debug_max_frames if args.debug_vis else 0,
            debug_point_size=args.debug_point_size,
            smpl_faces=smpl_faces,
        )
        save_pickle(geom_out_path, geom_sequence)

    print("[Info] Done.")


if __name__ == "__main__":
    main()
