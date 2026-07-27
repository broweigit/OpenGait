#!/usr/bin/env python3
"""Inventory OpenGait experiments with selected checkpoint iterations.

Examples:
    python misc/find_output_checkpoints.py
    python misc/find_output_checkpoints.py --dataset CCPG
    python misc/find_output_checkpoints.py /data/run_a/output /raid/run_b/output
    python misc/find_output_checkpoints.py --iterations 20000 30000 --paths-only
    python misc/find_output_checkpoints.py --dataset CCPG --contains Direct

The expected OpenGait layout is:
    output/<dataset>/<model>/<save_name>/checkpoints/<save_name>-<iter>.pt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
ITERATION_AT_END = re.compile(r"(?:^|[-_])(?P<iteration>\d+)$")


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint: Path
    experiment: Path
    dataset: str
    model: str
    save_name: str
    iteration: int
    size_bytes: int
    modified_time: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find OpenGait experiment directories containing checkpoints at "
            "the requested iterations (default: 20000 and 30000)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "roots",
        nargs="*",
        help=(
            "One or more output roots. If omitted, existing ./output and "
            "./Output directories are scanned."
        ),
    )
    parser.add_argument(
        "-i",
        "--iterations",
        nargs="+",
        type=int,
        default=[20000, 30000],
        metavar="ITER",
        help="Checkpoint iterations to match (default: 20000 30000).",
    )
    parser.add_argument(
        "--dataset",
        help="Only show one dataset directory, for example CCPG.",
    )
    parser.add_argument(
        "--contains",
        help=(
            "Case-insensitive substring filter over model, save_name, and "
            "experiment path, for example Direct or DINOv2."
        ),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--paths-only",
        action="store_true",
        help="Print only unique experiment directory paths.",
    )
    output_group.add_argument(
        "--tsv",
        action="store_true",
        help="Print one tab-separated row per matched checkpoint.",
    )
    return parser.parse_args(argv)


def default_roots() -> list[Path]:
    candidates = [Path("output"), Path("Output")]
    existing = [path for path in candidates if path.is_dir()]
    return existing if existing else [Path("output")]


def normalize_roots(raw_roots: Sequence[str]) -> list[Path]:
    roots = [Path(root).expanduser() for root in raw_roots] if raw_roots else default_roots()
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        normalized = root.resolve()
        if normalized not in seen:
            unique_roots.append(normalized)
            seen.add(normalized)
    return unique_roots


def checkpoint_iteration(path: Path) -> int | None:
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return None
    match = ITERATION_AT_END.search(path.stem)
    return int(match.group("iteration")) if match else None


def infer_layout(experiment: Path) -> tuple[str, str, str]:
    """Infer dataset/model/save_name from the standard OpenGait hierarchy."""
    save_name = experiment.name
    model = experiment.parent.name if experiment.parent != experiment else "-"
    dataset_parent = experiment.parent.parent
    dataset = dataset_parent.name if dataset_parent != experiment.parent else "-"
    return dataset, model, save_name


def walk_checkpoint_files(root: Path) -> Iterable[Path]:
    def report_walk_error(error: OSError) -> None:
        print(f"warning: cannot scan {error.filename}: {error}", file=sys.stderr)

    for current_dir, child_dirs, filenames in os.walk(
        root, topdown=True, onerror=report_walk_error, followlinks=False
    ):
        current_path = Path(current_dir)
        if current_path.name != "checkpoints":
            continue

        # OpenGait does not nest checkpoint directories; avoid needless descent.
        child_dirs.clear()
        for filename in filenames:
            yield current_path / filename


def collect_records(
    roots: Sequence[Path],
    iterations: set[int],
    dataset_filter: str | None,
    contains_filter: str | None,
) -> list[CheckpointRecord]:
    records: list[CheckpointRecord] = []
    seen_checkpoints: set[Path] = set()
    dataset_filter_folded = dataset_filter.casefold() if dataset_filter else None
    contains_filter_folded = contains_filter.casefold() if contains_filter else None

    for root in roots:
        if not root.is_dir():
            print(f"warning: output root does not exist: {root}", file=sys.stderr)
            continue

        for checkpoint in walk_checkpoint_files(root):
            iteration = checkpoint_iteration(checkpoint)
            if iteration not in iterations:
                continue

            resolved_checkpoint = checkpoint.resolve()
            if resolved_checkpoint in seen_checkpoints:
                continue
            seen_checkpoints.add(resolved_checkpoint)

            experiment = checkpoint.parent.parent.resolve()
            dataset, model, save_name = infer_layout(experiment)
            if (
                dataset_filter_folded is not None
                and dataset.casefold() != dataset_filter_folded
            ):
                continue

            searchable = f"{model}\n{save_name}\n{experiment}".casefold()
            if (
                contains_filter_folded is not None
                and contains_filter_folded not in searchable
            ):
                continue

            try:
                stat = checkpoint.stat()
            except OSError as error:
                print(f"warning: cannot stat {checkpoint}: {error}", file=sys.stderr)
                continue

            records.append(
                CheckpointRecord(
                    checkpoint=resolved_checkpoint,
                    experiment=experiment,
                    dataset=dataset,
                    model=model,
                    save_name=save_name,
                    iteration=iteration,
                    size_bytes=stat.st_size,
                    modified_time=stat.st_mtime,
                )
            )

    return sorted(
        records,
        key=lambda record: (
            record.dataset.casefold(),
            record.model.casefold(),
            record.save_name.casefold(),
            record.iteration,
            str(record.checkpoint),
        ),
    )


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size_bytes}B"


def latest_log_summary(experiment: Path) -> str:
    log_dir = experiment / "logs"
    if not log_dir.is_dir():
        return f"{log_dir} (missing)"

    try:
        log_files = [path for path in log_dir.iterdir() if path.is_file()]
    except OSError as error:
        return f"{log_dir} (unreadable: {error})"

    if not log_files:
        return f"{log_dir} (empty)"

    latest = max(log_files, key=lambda path: path.stat().st_mtime)
    return f"{log_dir} ({len(log_files)} files; latest: {latest.name})"


def print_detailed(records: Sequence[CheckpointRecord]) -> None:
    grouped: dict[Path, list[CheckpointRecord]] = {}
    for record in records:
        grouped.setdefault(record.experiment, []).append(record)

    print(
        f"Found {len(grouped)} experiment directories and "
        f"{len(records)} matching checkpoints."
    )
    for index, (experiment, experiment_records) in enumerate(grouped.items(), start=1):
        first = experiment_records[0]
        print()
        print(f"[{index}] dataset   : {first.dataset}")
        print(f"    model     : {first.model}")
        print(f"    save_name : {first.save_name}")
        print(f"    experiment: {experiment}")
        print(f"    logs      : {latest_log_summary(experiment)}")
        print("    checkpoints:")
        for record in experiment_records:
            modified = datetime.fromtimestamp(record.modified_time).isoformat(
                sep=" ", timespec="seconds"
            )
            print(
                f"      - iter={record.iteration:<6d} "
                f"size={human_size(record.size_bytes):>9s} "
                f"mtime={modified}  {record.checkpoint}"
            )


def print_tsv(records: Sequence[CheckpointRecord]) -> None:
    print(
        "iteration\tdataset\tmodel\tsave_name\texperiment\tcheckpoint"
        "\tsize_bytes\tmodified_time"
    )
    for record in records:
        modified = datetime.fromtimestamp(record.modified_time).isoformat(
            sep=" ", timespec="seconds"
        )
        print(
            "\t".join(
                [
                    str(record.iteration),
                    record.dataset,
                    record.model,
                    record.save_name,
                    str(record.experiment),
                    str(record.checkpoint),
                    str(record.size_bytes),
                    modified,
                ]
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    iterations = set(args.iterations)
    if not iterations or any(iteration < 0 for iteration in iterations):
        raise SystemExit("--iterations must contain non-negative integers")

    roots = normalize_roots(args.roots)
    records = collect_records(
        roots=roots,
        iterations=iterations,
        dataset_filter=args.dataset,
        contains_filter=args.contains,
    )

    if args.paths_only:
        for experiment in dict.fromkeys(record.experiment for record in records):
            print(experiment)
    elif args.tsv:
        print_tsv(records)
    else:
        print(f"Scanned roots: {', '.join(map(str, roots))}")
        print(f"Requested iterations: {', '.join(map(str, sorted(iterations)))}")
        if args.dataset:
            print(f"Dataset filter: {args.dataset}")
        if args.contains:
            print(f"Substring filter: {args.contains}")
        print_detailed(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
