#!/usr/bin/env python3
"""Render the MLX mesh and compare it to the official CUDA vis + PyTorch FLAME verts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.backend import create_deca
from deca_mlx.compare import compare_arrays, format_report, load_baseline
from deca_mlx.render import Renderer, visualize_reconstruction


def _label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 22), (20, 20, 20))
    canvas.paste(image, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), text, fill=(255, 255, 255))
    return canvas


def main() -> None:
    baseline = load_baseline()
    deca = create_deca().load_pretrained()
    images = deca.asarray(baseline["image"][None, ...])
    codedict = deca.encode(images, use_detail=True)
    opdict = deca.decode(codedict, use_detail=True)
    pred = {
        "verts": deca.to_numpy(opdict["verts"])[0],
        "landmarks2d": deca.to_numpy(opdict["landmarks2d"])[0],
        "landmarks3d": deca.to_numpy(opdict["landmarks3d"])[0],
        "displacement_map": deca.to_numpy(opdict["displacement_map"])[0],
    }
    renderer = Renderer()
    vis = visualize_reconstruction(
        baseline["image"],
        pred["verts"],
        deca.to_numpy(opdict["trans_verts"])[0],
        pred["landmarks2d"],
        pred["landmarks3d"],
        lights=deca.to_numpy(codedict["light"])[0],
        tex_code=deca.to_numpy(codedict["tex"])[0],
        displacement=pred["displacement_map"],
        renderer=renderer,
    )
    Image.fromarray(vis).save(ROOT / "outputs" / "IMG_0392_mlx_vis.jpg", quality=95)

    report = compare_arrays(baseline, pred, keys=("verts", "landmarks2d", "landmarks3d", "displacement_map"))
    print(format_report(report))
    if "verts" in baseline:
        delta = np.abs(baseline["verts"] - pred["verts"])
        print(f"vertex mean abs {delta.mean():.4e}  max {delta.max():.4e}")

    official = Image.open(ROOT / "tests" / "faces" / "IMG_0392_inputs_vis.jpg").convert("RGB")
    official = official.resize((vis.shape[1], vis.shape[0]), Image.Resampling.BILINEAR)
    mlx_img = _label(Image.fromarray(vis), "MLX vis")
    official_img = _label(official, "official CUDA vis")
    collage = Image.new("RGB", (mlx_img.width, mlx_img.height + official_img.height), (12, 12, 12))
    collage.paste(mlx_img, (0, 0))
    collage.paste(official_img, (0, mlx_img.height))
    out = ROOT / "outputs" / "IMG_0392_mesh_compare.jpg"
    collage.save(out, quality=95)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
