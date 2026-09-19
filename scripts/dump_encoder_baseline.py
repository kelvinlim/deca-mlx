#!/usr/bin/env python3
"""Dump official-checkpoint encoder outputs with CPU PyTorch.

This does not need CUDA or FLAME. It is the local numeric baseline used to
check the MLX ResNet encoders against ``deca_model.tar``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.config import default_config
from deca_mlx.preprocess import load_image
from deca_mlx.torch_backend.decoders import Generator
from deca_mlx.torch_backend.deca import decompose_code
from deca_mlx.torch_backend.encoders import ResnetEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=str(ROOT / "tests" / "faces" / "IMG_0392_inputs.jpg"))
    parser.add_argument("-o", "--output", default=str(ROOT / "tests" / "baselines" / "IMG_0392.npz"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = default_config()
    ckpt_path = Path(args.checkpoint or cfg.pretrained_modelpath)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    e_flame = ResnetEncoder(cfg.n_param)
    e_detail = ResnetEncoder(cfg.n_detail)
    e_flame.load_state_dict(checkpoint["E_flame"])
    e_detail.load_state_dict(checkpoint["E_detail"])
    e_flame.eval()
    e_detail.eval()

    sample = load_image(args.image, iscrop=False)
    images = torch.from_numpy(sample["image"][None, ...])
    with torch.no_grad():
        flame_code = e_flame(images)
        detail_code = e_detail(images)
    d_detail = Generator()
    d_detail.load_state_dict(checkpoint["D_detail"])
    d_detail.eval()
    payload = {key: value[0].detach().cpu().numpy() for key, value in decompose_code(flame_code, cfg.param_sizes).items()}
    payload["detail"] = detail_code[0].detach().cpu().numpy()
    cond = torch.cat(
        [torch.from_numpy(payload["pose"][3:])[None, ...], torch.from_numpy(payload["exp"])[None, ...], detail_code],
        dim=1,
    )
    with torch.no_grad():
        uv_z = d_detail(cond)
    payload["uv_z"] = uv_z[0].detach().cpu().numpy()
    if cfg.fixed_displacement_path.exists():
        fixed = np.load(cfg.fixed_displacement_path).astype(np.float32)
        payload["displacement_map"] = payload["uv_z"] + fixed[None, ...]
    payload["image"] = sample["image"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"wrote {out}")
    for key, value in payload.items():
        print(f"  {key:12s} {np.asarray(value).shape}")


if __name__ == "__main__":
    main()
