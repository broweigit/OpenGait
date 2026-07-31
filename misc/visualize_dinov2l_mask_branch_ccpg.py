#!/usr/bin/env python3
"""Visualize BiggerGait's official DINOv2-L (ViT-L/14) mask branch."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.io import read_video

from visualize_sam3dbody_mask_branch_ccpg import (
    MaskBranch,
    chw_uint8_to_pil,
    collect_samples,
    evenly_spaced_indices,
    labeled_panel,
    make_overlay,
    mask_to_pil,
    prepare_rgb,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-kind", choices=("ccpg", "ccgr"), default="ccpg"
    )
    parser.add_argument(
        "--dataset-root", default="/data/CCPG/Released/CCPG-ratio-pkl"
    )
    parser.add_argument("--partition", default="datasets/CCPG/CCPG.json")
    parser.add_argument(
        "--mask-checkpoint",
        default="pretrained_LVMs/MaskBranch_vitl14.pt",
    )
    parser.add_argument(
        "--dinov2-root", default="pretrained_LVMs/dinov2-large"
    )
    parser.add_argument(
        "--output-dir",
        default="visual_assets/mask_branch_vitl14_ccpg_check",
    )
    parser.add_argument("--num-identities", type=int, default=4)
    parser.add_argument("--sequences-per-identity", type=int, default=2)
    parser.add_argument("--frames-per-sequence", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resize_with_padding(image, target_h=256, target_w=128):
    """Match OpenGait's CCGR RGB preprocessing for one CHW image."""
    _, height, width = image.shape
    scale = min(target_h / height, target_w / width)
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    return F.pad(
        resized,
        (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        ),
        mode="constant",
        value=0,
    )


def remove_black_border(images, threshold=10):
    """Match DataSet.__loader__ for CCGR videos, restricted to chosen frames."""
    processed = []
    for image in images.float():
        gray = image.mean(dim=0)
        mask = gray > threshold
        if mask.any():
            y_indices = torch.where(mask.any(dim=1))[0]
            x_indices = torch.where(mask.any(dim=0))[0]
            image = image[
                :,
                y_indices[0]:y_indices[-1] + 1,
                x_indices[0]:x_indices[-1] + 1,
            ]
        processed.append(resize_with_padding(image))
    return torch.stack(processed)


def collect_ccgr_samples(args):
    root = Path(args.dataset_root)
    partition = json.loads(Path(args.partition).read_text())
    candidates = [
        identity
        for identity in partition["TRAIN_SET"]
        if (root / identity).is_dir()
    ]
    if not candidates:
        raise RuntimeError("No CCGR training identities were found.")

    rng = random.Random(args.seed)
    identities = sorted(
        rng.sample(candidates, min(args.num_identities, len(candidates)))
    )
    samples = []
    for identity in identities:
        sequence_files = sorted((root / identity).glob("*/*.avi/*.avi"))
        if not sequence_files:
            continue
        sequence_indices = evenly_spaced_indices(
            len(sequence_files), args.sequences_per_identity
        )
        for sequence_idx in sequence_indices:
            video_path = sequence_files[sequence_idx]
            video, _, _ = read_video(
                str(video_path), output_format="TCHW", pts_unit="sec"
            )
            frame_indices = evenly_spaced_indices(
                len(video), args.frames_per_sequence
            )
            frames = remove_black_border(video[frame_indices]).cpu().numpy()
            for frame_idx, frame in zip(frame_indices, frames):
                samples.append(
                    {
                        "identity": identity,
                        "sequence": str(video_path.relative_to(root)),
                        "frame_index": int(frame_idx),
                        "rgb": frame,
                    }
                )
    if not samples:
        raise RuntimeError("Balanced CCGR sampling produced no frames.")
    return samples


def load_models(args, device):
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config.from_pretrained(
        str(Path(args.dinov2_root) / "config.json")
    )
    encoder = Dinov2Model.from_pretrained(args.dinov2_root, config=config)
    encoder.eval().requires_grad_(False).to(device)

    payload = torch.load(args.mask_checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    if any(key.startswith("Mask_Branch.") for key in state):
        state = {
            key[len("Mask_Branch."):]: value
            for key, value in state.items()
            if key.startswith("Mask_Branch.")
        }
    branch = MaskBranch(source_dim=1024, target_dim=2)
    message = branch.load_state_dict(state, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(str(message))
    branch.eval().requires_grad_(False).to(device)
    return encoder, branch


def infer_masks(encoder, branch, images, device):
    images = images.to(device, non_blocking=True)
    images = F.interpolate(
        images, size=(448, 224), mode="bilinear", align_corners=False
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = encoder(images, output_hidden_states=True)
        tokens = output.hidden_states[-1][:, 1:].contiguous()
        tokens = F.layer_norm(tokens, (1024,), eps=1e-6)
        probabilities = branch(tokens.reshape(-1, 1024))
        probabilities = probabilities.view(len(images), 32, 16, 2)
        masks = []
        for channel in (0, 1):
            low = (probabilities[..., channel] > 0.5).float().unsqueeze(1)
            high = F.interpolate(
                low, size=(448, 224), mode="nearest"
            )
            masks.append(high)
    return torch.cat(masks, dim=1).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = (
        collect_ccgr_samples(args)
        if args.dataset_kind == "ccgr"
        else collect_samples(args)
    )
    encoder, branch = load_models(args, device)

    rows = []
    panels = []
    sample_number = 0
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        raw, normalized = prepare_rgb(batch)
        masks = infer_masks(encoder, branch, normalized, device)
        for local_idx, sample in enumerate(batch):
            rgb = chw_uint8_to_pil(raw[local_idx]).resize((224, 448))
            mask0 = mask_to_pil(masks[local_idx, 0])
            mask1 = mask_to_pil(masks[local_idx, 1])
            overlay0 = make_overlay(rgb, mask0)
            overlay1 = make_overlay(rgb, mask1)
            stem = (
                f"{sample_number:03d}_{sample['identity']}_"
                f"f{sample['frame_index']:03d}"
            )
            rgb.save(output_dir / f"{stem}_rgb.png")
            mask0.save(output_dir / f"{stem}_mask_ch0.png")
            mask1.save(output_dir / f"{stem}_mask_ch1.png")
            overlay0.save(output_dir / f"{stem}_overlay_ch0.png")
            overlay1.save(output_dir / f"{stem}_overlay_ch1.png")
            panel = labeled_panel(
                [rgb, mask0, overlay0, mask1, overlay1],
                [
                    "RGB",
                    "mask ch0",
                    "overlay ch0",
                    "mask ch1 (USED)",
                    "overlay ch1 (USED)",
                ],
            )
            panel.save(output_dir / f"{stem}_pair.png")
            panels.append(panel.resize((560, 242)))
            rows.append(
                {
                    "sample": stem,
                    "identity": sample["identity"],
                    "sequence": sample["sequence"],
                    "frame_index": sample["frame_index"],
                    "channel0_white_ratio": f"{masks[local_idx, 0].mean():.6f}",
                    "channel1_white_ratio": f"{masks[local_idx, 1].mean():.6f}",
                }
            )
            sample_number += 1

    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    sheet = Image.new(
        "RGB",
        (max(panel.width for panel in panels), sum(panel.height for panel in panels)),
        "white",
    )
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(output_dir / "contact_sheet.png")
    mean0 = np.mean([float(row["channel0_white_ratio"]) for row in rows])
    mean1 = np.mean([float(row["channel1_white_ratio"]) for row in rows])
    print(f"Saved {len(rows)} samples to {output_dir}")
    print(f"Mean white ratio: channel 0={mean0:.4f}, channel 1={mean1:.4f}")
    print("BiggerGait uses channel 1; verify that its white region covers the person.")
    print(f"Open {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
