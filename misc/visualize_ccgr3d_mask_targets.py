#!/usr/bin/env python3
"""Visualize RGB, official silhouette, SAM3D pseudo-mask and overlay."""
import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", default="32")
    parser.add_argument("--condition", default="NM1")
    parser.add_argument("--view", default="90_0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--rgb-root", default="/data4/CCGR_3D/RGB")
    parser.add_argument("--official-root", default="/data4/CCGR_3D/2D")
    parser.add_argument("--sam-root", default="/data4/SAM_3D_BODY/CCGR3D/CCGR3D-RGB-SAM3D")
    parser.add_argument("--output", default="output/CCGR3D/mask_debug.png")
    args = parser.parse_args()

    seq = f"{args.view}.avi"
    rgb_path = next((Path(args.rgb_root) / args.pid / args.condition / seq).glob("*-rgbs.pkl"))
    sil_path = next((Path(args.official_root) / args.pid / args.condition / seq).glob("*-sil.pkl"))
    sam_path = Path(args.sam_root) / args.pid / args.condition / seq / "02-sam3d-projected-mask.pkl"
    rgb, sil, sam = load(rgb_path), load(sil_path), load(sam_path)
    index = min(args.frame, len(rgb) - 1, len(sil) - 1, len(sam) - 1)
    image = rgb[index].transpose(1, 2, 0) if rgb[index].shape[0] == 3 else rgb[index]
    official = cv2.resize(sil[index], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    pseudo = cv2.resize(sam[index], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.astype(np.float32).copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], pseudo * 255)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for axis, data, title, cmap in zip(
        axes,
        [image, official, pseudo, overlay],
        ["RGB", "Official silhouette", "SAM3D projection mask", "SAM mask overlay"],
        [None, "gray", "gray", None],
    ):
        axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Saved {output.resolve()}\nRGB: {rgb_path}\nSilhouette: {sil_path}\nSAM3D mask: {sam_path}")


if __name__ == "__main__":
    main()
