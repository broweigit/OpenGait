#!/usr/bin/env python3
"""Convenience entry point for CCGR-3D privileged-geometry preprocessing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_sam3d_body_decoder_dataset import main


if __name__ == "__main__":
    if "--dataset-type" not in sys.argv:
        sys.argv[1:1] = ["--dataset-type", "ccgr3d"]
    if "--input-root" not in sys.argv:
        sys.argv[1:1] = ["--input-root", "/data4/CCGR_3D/RGB"]
    if "--output-root" not in sys.argv:
        sys.argv[1:1] = ["--output-root", "/data4/SAM_3D_BODY/CCGR3D/CCGR3D-RGB-SAM3D"]
    if "--save-canonical-apose" not in sys.argv:
        sys.argv.append("--save-canonical-apose")
    main()
