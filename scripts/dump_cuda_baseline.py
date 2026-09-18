#!/usr/bin/env python3
"""Dump official DECA encode/decode tensors for MLX comparison.

Run this on a machine with the official yfeng95/DECA repo and CUDA (or Colab).

Example (from a clone of yfeng95/DECA):

    python /path/to/deca-mlx/scripts/dump_cuda_baseline.py \\
        --deca-root /path/to/DECA \\
        --image /path/to/deca-mlx/tests/faces/IMG_0392_inputs.jpg \\
        --output /path/to/deca-mlx/tests/baselines/IMG_0392.npz

Colab: open the official notebook, then in a new cell:

    !pip install -q gdown
    # after DECA is cloned and data/deca_model.tar is present
    !python dump_cuda_baseline.py --deca-root /content/DECA \\
        --image /content/DECA/TestSamples/examples/IMG_0392_inputs.jpg \\
        --output /content/IMG_0392.npz --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump official DECA CUDA/PyTorch baseline NPZ")
    parser.add_argument("--deca-root", required=True, help="Path to a clone of yfeng95/DECA")
    parser.add_argument("--image", required=True, help="Input face image")
    parser.add_argument("--output", required=True, help="Destination .npz path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iscrop", action="store_true", help="Run official face crop (default: off for IMG_0392_inputs)")
    args = parser.parse_args()

    deca_root = Path(args.deca_root).resolve()
    sys.path.insert(0, str(deca_root))

    import torch
    from decalib.deca import DECA
    from decalib.datasets import datasets
    from decalib.utils.config import cfg as deca_cfg

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU. Encoder codes should still be comparable.")
        device = "cpu"

    testdata = datasets.TestData(args.image, iscrop=args.iscrop, face_detector="fan")
    sample = testdata[0]
    images = sample["image"].to(device)[None, ...]

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca = DECA(config=deca_cfg, device=device)
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, _ = deca.decode(codedict)

    payload = {
        "image": sample["image"].cpu().numpy(),
        "shape": codedict["shape"][0].cpu().numpy(),
        "tex": codedict["tex"][0].cpu().numpy(),
        "exp": codedict["exp"][0].cpu().numpy(),
        "pose": codedict["pose"][0].cpu().numpy(),
        "cam": codedict["cam"][0].cpu().numpy(),
        "light": codedict["light"][0].cpu().numpy(),
        "detail": codedict["detail"][0].cpu().numpy(),
        "verts": opdict["verts"][0].cpu().numpy(),
        "landmarks2d": opdict["landmarks2d"][0].cpu().numpy(),
        "landmarks3d": opdict["landmarks3d"][0].cpu().numpy(),
        "displacement_map": opdict["displacement_map"][0].cpu().numpy(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"wrote {out}")
    for key, value in payload.items():
        print(f"  {key:18s} {value.shape}")


if __name__ == "__main__":
    main()
