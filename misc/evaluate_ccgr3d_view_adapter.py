#!/usr/bin/env python3
"""Evaluate cached RGB features before/after target-view adaptation."""
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from train_ccgr3d_view_adapter import ViewAdapter, condition_code, load_cache


def nearest_rank1(probe, probe_pid, probe_view, gallery, gallery_pid, gallery_view, strict=True):
    probe = F.normalize(probe.flatten(1), dim=-1)
    gallery = F.normalize(gallery.flatten(1), dim=-1)
    similarity = probe @ gallery.T
    correct = []
    per_view = defaultdict(list)
    for i in range(len(probe)):
        valid = np.ones(len(gallery), dtype=bool)
        if strict:
            valid &= gallery_view != probe_view[i]
        candidates = np.flatnonzero(valid)
        predicted = candidates[similarity[i, candidates].argmax().item()]
        hit = gallery_pid[predicted] == probe_pid[i]
        correct.append(hit)
        per_view[probe_view[i]].append(hit)
    return np.mean(correct) * 100, {view: np.mean(values) * 100 for view, values in per_view.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-cache", default="output/CCGR3D/feature_cache/student_test.npz")
    parser.add_argument("--checkpoint", default="output/CCGR3D/view_adapter/view_adapter.pt")
    parser.add_argument("--reference-view", default="90_0", choices=sorted(__import__('train_ccgr3d_view_adapter').VIEW_ANGLES))
    parser.add_argument("--gallery-condition", default="NM1")
    args = parser.parse_args()

    features, metadata = load_cache(args.student_cache, "embeddings_structured")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ViewAdapter(channels=checkpoint["channels"], fpn_groups=checkpoint["fpn_groups"])
    model.load_state_dict(checkpoint["model"])
    model.eval().cuda()
    labels = np.asarray([m[0] for m in metadata])
    conditions = np.asarray([m[1] for m in metadata])
    views = np.asarray([m[2] for m in metadata])
    source = torch.from_numpy(features).cuda()
    codes = torch.from_numpy(np.stack([condition_code(view, args.reference_view) for view in views])).cuda()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(source), 512):
            outputs.append(model(source[start:start + 512], codes[start:start + 512]).cpu())
    adapted = torch.cat(outputs)
    raw = torch.from_numpy(features)
    gallery = conditions == args.gallery_condition
    probe = ~gallery
    for name, embedding in (("raw", raw), ("adapted", adapted)):
        rank1, per_view = nearest_rank1(
            embedding[probe], labels[probe], views[probe],
            embedding[gallery], labels[gallery], views[gallery], strict=True,
        )
        print(f"{name}: StrictCrossView Rank-1={rank1:.2f}% per-view={per_view}")


if __name__ == "__main__":
    main()
