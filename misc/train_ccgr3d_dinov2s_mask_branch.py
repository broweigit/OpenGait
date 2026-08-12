#!/usr/bin/env python3
"""Train BigGait's tiny DINOv2-S mask branch from SAM3D projection masks."""
import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class MaskBranch(nn.Module):
    """State-dict compatible with BiggerGait.infoDistillation."""
    def __init__(self, source_dim=384, target_dim=2):
        super().__init__()
        self.dropout = nn.Dropout(0.0)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        self.down_sampling = nn.Linear(source_dim, target_dim)
        self.up_sampling = nn.Linear(target_dim, source_dim)

    def forward(self, tokens):
        return F.softmax(self.down_sampling(self.bn_s(self.dropout(tokens))), dim=-1)


def sequence_paths(root: Path, train_ids):
    paths = []
    for pid in train_ids:
        for mask in root.glob(f"{pid}/*/*/02-sam3d-projected-mask.pkl"):
            rgb = mask.with_name("00-rgb.pkl")
            if rgb.exists():
                paths.append((rgb, mask))
    if not paths:
        raise FileNotFoundError(f"No aligned RGB/mask sequences below {root}")
    return paths


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_batch(paths, batch_size, rng):
    images, masks = [], []
    sequences = rng.sample(paths, k=min(8, batch_size, len(paths)))
    while len(images) < batch_size:
        for rgb_path, mask_path in sequences:
            rgb = np.asarray(load_pickle(rgb_path))
            mask = np.asarray(load_pickle(mask_path))
            length = min(len(rgb), len(mask))
            samples_here = min(max(1, batch_size // len(sequences)), batch_size - len(images), length)
            for index in rng.sample(range(length), k=samples_here):
                images.append(rgb[index])
                masks.append(mask[index])
            if len(images) >= batch_size:
                break
    return np.stack(images), np.stack(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data4/SAM_3D_BODY/CCGR3D/CCGR3D-RGB-SAM3D")
    parser.add_argument("--partition", default="datasets/CCGR3D/CCGR3D.json")
    parser.add_argument("--dinov2", default="pretrained_LVMs/dinov2-small")
    parser.add_argument("--output", default="pretrained_LVMs/CCGR3D_MaskBranch_vits14.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rng = random.Random(args.seed + rank)

    partition = json.loads(Path(args.partition).read_text())
    paths = sequence_paths(Path(args.root), partition["TRAIN_SET"])
    from transformers import Dinov2Model
    backbone = Dinov2Model.from_pretrained(args.dinov2).eval().to(device)
    backbone.requires_grad_(False)
    branch = MaskBranch().to(device)
    if distributed:
        branch = nn.parallel.DistributedDataParallel(branch, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=1e-4)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for step in range(1, args.steps + 1):
        rgb_np, mask_np = sample_batch(paths, args.batch_size, rng)
        rgb = torch.from_numpy(rgb_np).float().to(device) / 255.0
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate((rgb - mean) / std, size=(448, 224), mode="bilinear", align_corners=False)
        target = torch.from_numpy(mask_np).long().to(device).reshape(-1)
        with torch.no_grad():
            tokens = backbone(rgb, output_hidden_states=True).hidden_states[-1][:, 1:]
            tokens = F.layer_norm(tokens, (tokens.shape[-1],)).reshape(-1, tokens.shape[-1])
        prob = branch(tokens)
        foreground = prob[:, 1]
        ce = F.nll_loss(prob.clamp_min(1e-6).log(), target)
        target_f = target.float()
        dice = 1.0 - (2 * (foreground * target_f).sum() + 1) / (foreground.sum() + target_f.sum() + 1)
        loss = ce + dice
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if rank == 0 and (step == 1 or step % 50 == 0):
            pred = foreground > 0.5
            inter = (pred & target.bool()).sum().float()
            union = (pred | target.bool()).sum().float().clamp_min(1)
            print(f"step={step:05d} loss={loss.item():.4f} ce={ce.item():.4f} dice={dice.item():.4f} IoU={(inter/union).item():.4f}", flush=True)

    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        module = branch.module if hasattr(branch, "module") else branch
        torch.save({"model": module.state_dict(), "steps": args.steps}, output)
        print(f"Saved {output.resolve()}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
