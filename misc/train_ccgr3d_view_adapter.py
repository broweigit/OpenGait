#!/usr/bin/env python3
"""Train a lightweight RGB-only latent view adapter from cached features.

No DINOv2 or SAM3D model is instantiated by this program.  Real target-view
features supervise camera transfer, while PuppetGait's canonical features are
used through a coordinate-invariant relational distillation loss.
"""
import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


VIEW_ANGLES = {
    "0_75": (0.0, 75.0),
    "90_0": (90.0, 0.0),
    "120_25": (120.0, 25.0),
    "150_60": (150.0, 60.0),
    "270_40": (270.0, 40.0),
}


def normalize_view(view):
    return str(view).removesuffix(".avi")


def view_vector(view, canonical=False):
    if canonical:
        azimuth, elevation = 0.0, 0.0
    else:
        azimuth, elevation = VIEW_ANGLES[normalize_view(view)]
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.asarray([math.sin(a), math.cos(a), math.sin(e), math.cos(e)], np.float32)


def condition_code(source_view, target_view=None, canonical=False):
    source = view_vector(source_view)
    target = view_vector(target_view, canonical=canonical)
    # Circular relative azimuth is represented without a discontinuity.
    source_angle = math.atan2(source[0], source[1])
    target_angle = math.atan2(target[0], target[1])
    delta_a = target_angle - source_angle
    delta_e = math.atan2(target[2], target[3]) - math.atan2(source[2], source[3])
    delta = np.asarray(
        [math.sin(delta_a), math.cos(delta_a), math.sin(delta_e), math.cos(delta_e)],
        np.float32,
    )
    return np.concatenate([source, target, delta, np.asarray([float(canonical)], np.float32)])


def load_cache(path, preferred_key):
    cache = np.load(path, allow_pickle=False)
    if preferred_key not in cache:
        choices = sorted(key for key in cache.files if key.startswith("embeddings"))
        raise KeyError(f"{preferred_key} not in {path}; available embedding keys: {choices}")
    features = np.asarray(cache[preferred_key], dtype=np.float32)
    if features.ndim == 3:
        # Legacy flattened [N,C,F*P] representation.
        if features.shape[-1] % 6:
            raise ValueError(f"Cannot infer six FPN groups from {features.shape}")
        features = np.stack(np.split(features, 6, axis=-1), axis=1)
    if features.ndim != 4:
        raise ValueError(f"Expected [N,F,C,P], got {features.shape} from {path}")
    metadata = list(zip(cache["labels"].astype(str), cache["conditions"].astype(str), cache["views"].astype(str)))
    return features, metadata


class PairedFeatureDataset(Dataset):
    def __init__(self, student_cache, teacher_cache):
        student, student_meta = load_cache(student_cache, "embeddings_structured")
        teacher, teacher_meta = load_cache(teacher_cache, "embeddings_canonical")
        student_index = {meta: index for index, meta in enumerate(student_meta)}
        teacher_index = {meta: index for index, meta in enumerate(teacher_meta)}
        shared = sorted(set(student_index) & set(teacher_index))
        if not shared:
            raise ValueError("Student and teacher caches contain no matching sequence metadata.")
        self.student = student
        self.teacher = teacher
        self.student_index = student_index
        self.teacher_index = teacher_index
        grouped = defaultdict(list)
        for meta in shared:
            grouped[meta[:2]].append(meta)
        self.groups = [values for values in grouped.values() if len({v[2] for v in values}) >= 2]
        self.epoch = 0
        print(f"Loaded {len(shared)} aligned sequences in {len(self.groups)} multi-view ID-condition groups")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        # Deterministic per epoch/index, while changing the pair each epoch.
        rng = random.Random(self.epoch * 1000003 + index)
        source, target = rng.sample(group, 2)
        return {
            "source": torch.from_numpy(self.student[self.student_index[source]]),
            "target": torch.from_numpy(self.student[self.student_index[target]]),
            "canonical_teacher": torch.from_numpy(self.teacher[self.teacher_index[source]]),
            "view_code": torch.from_numpy(condition_code(source[2], target[2])),
            "canonical_code": torch.from_numpy(condition_code(source[2], canonical=True)),
            "source_view": source[2],
            "target_view": target[2],
            "pid": source[0],
        }


class ViewResidualBlock(nn.Module):
    def __init__(self, channels, fpn_groups, code_dim=13, bottleneck=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.film = nn.Sequential(nn.Linear(code_dim, 128), nn.GELU(), nn.Linear(128, channels * 2))
        self.part_mixer = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.fpn_mixer = nn.Linear(fpn_groups, fpn_groups, bias=False)
        self.ffn = nn.Sequential(nn.Linear(channels, bottleneck), nn.GELU(), nn.Linear(bottleneck, channels))

    def forward(self, x, code):
        # x: [B,F,C,P]
        b, f, c, p = x.shape
        residual = x
        z = self.norm(x.permute(0, 1, 3, 2))
        scale, bias = self.film(code).chunk(2, dim=-1)
        z = z * (1.0 + scale[:, None, None]) + bias[:, None, None]
        z = z.permute(0, 1, 3, 2).reshape(b * f, c, p)
        z = self.part_mixer(z).reshape(b, f, c, p)
        z = self.fpn_mixer(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        z = self.ffn(z.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return residual + z


class ViewAdapter(nn.Module):
    def __init__(self, channels=256, fpn_groups=6, blocks=2, bottleneck=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            ViewResidualBlock(channels, fpn_groups, bottleneck=bottleneck) for _ in range(blocks)
        ])
        self.output_scale = nn.Parameter(torch.tensor(-4.0))

    def forward(self, source, code):
        z = source
        for block in self.blocks:
            z = block(z, code)
        alpha = torch.sigmoid(self.output_scale)
        return source + alpha * (z - source)


def token_cosine_loss(prediction, target):
    return (1.0 - F.cosine_similarity(prediction, target, dim=2)).mean()


def flattened_normalized(feature):
    return F.normalize(feature.flatten(1), dim=-1)


def supervised_contrastive(prediction, target, pids, temperature=0.07):
    prediction = flattened_normalized(prediction)
    target = flattened_normalized(target)
    logits = prediction @ target.T / temperature
    pid_array = np.asarray(pids)
    positives = torch.as_tensor(pid_array[:, None] == pid_array[None, :], device=logits.device)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp_min(1)).mean()


def relational_kd(student, teacher):
    student = flattened_normalized(student)
    teacher = flattened_normalized(teacher)
    return F.smooth_l1_loss(student @ student.T, teacher @ teacher.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-cache", default="output/CCGR3D/feature_cache/student_train.npz")
    parser.add_argument("--teacher-cache", default="output/CCGR3D/feature_cache/teacher_train.npz")
    parser.add_argument("--output", default="output/CCGR3D/view_adapter/view_adapter.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--canonical-weight", type=float, default=1.0)
    parser.add_argument("--same-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    args = parser.parse_args()

    dataset = PairedFeatureDataset(args.student_cache, args.teacher_cache)
    sample = dataset[0]["source"]
    model = ViewAdapter(channels=sample.shape[1], fpn_groups=sample.shape[0]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    for epoch in range(args.epochs):
        dataset.epoch = epoch
        model.train()
        sums = defaultdict(float)
        steps = 0
        for batch in loader:
            source = batch["source"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            teacher = batch["canonical_teacher"].cuda(non_blocking=True)
            code = batch["view_code"].cuda(non_blocking=True)
            canonical_code = batch["canonical_code"].cuda(non_blocking=True)
            prediction = model(source, code)
            same_code = torch.stack([
                torch.from_numpy(condition_code(v, v)) for v in batch["source_view"]
            ]).to(source.device)
            same_prediction = model(source, same_code)
            canonical_prediction = model(source, canonical_code)
            view_loss = token_cosine_loss(prediction, target)
            same_loss = token_cosine_loss(same_prediction, source)
            contrastive = supervised_contrastive(prediction, target, batch["pid"])
            kd = relational_kd(canonical_prediction, teacher)
            loss = view_loss + args.same_weight * same_loss + args.contrastive_weight * contrastive + args.canonical_weight * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in {"loss": loss, "view": view_loss, "same": same_loss, "contrast": contrastive, "kd": kd}.items():
                sums[key] += value.item()
            steps += 1
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            summary = " ".join(f"{key}={value/max(steps,1):.4f}" for key, value in sums.items())
            print(f"epoch={epoch+1:03d} {summary} alpha={torch.sigmoid(model.output_scale).item():.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.cpu().state_dict(),
        "channels": sample.shape[1],
        "fpn_groups": sample.shape[0],
        "view_angles": VIEW_ANGLES,
        "args": vars(args),
    }, output)
    params = sum(p.numel() for p in model.parameters())
    print(f"Saved {output.resolve()} ({params/1e6:.3f}M parameters)")


if __name__ == "__main__":
    main()
