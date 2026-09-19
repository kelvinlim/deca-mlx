"""Run DECA reconstruction on MLX (Apple) or PyTorch (ROCm / CUDA / CPU)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .backend import create_deca, resolve_backend
from .compare import compare_arrays, format_report, load_baseline
from .config import ROOT, default_config
from .preprocess import load_image
from .render import Renderer, visualize_reconstruction


def _to_uint8(image_nchw: np.ndarray) -> np.ndarray:
    image = np.clip(image_nchw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return image


def _uv_z_chw(deca, uv_z) -> np.ndarray:
    array = deca.to_numpy(uv_z)
    if array.ndim == 4 and array.shape[-1] == 1:
        return np.transpose(array[0], (2, 0, 1))
    if array.ndim == 4:
        return array[0]
    return array


def draw_landmarks(image_nchw: np.ndarray, landmarks: np.ndarray) -> Image.Image:
    canvas = Image.fromarray(_to_uint8(image_nchw))
    draw = ImageDraw.Draw(canvas)
    h, w = canvas.size[1], canvas.size[0]
    pts = landmarks.copy()
    pts[:, 0] = pts[:, 0] * w / 2 + w / 2
    pts[:, 1] = pts[:, 1] * h / 2 + h / 2
    for x, y in pts[:, :2]:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 0))
    return canvas


def build_comparison(input_vis: Path | None, recon_vis: Image.Image, official_vis: Path | None, out_path: Path, label: str) -> Path:
    rows = [(label, recon_vis)]
    if official_vis and official_vis.exists():
        official = Image.open(official_vis).convert("RGB")
        if official.size != recon_vis.size:
            official = official.resize(recon_vis.size, Image.Resampling.BILINEAR)
        rows.append(("official CUDA vis", official))
    width = max(panel.width for _, panel in rows)
    height = sum(panel.height + 22 for _, panel in rows)
    collage = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(collage)
    y = 0
    for row_label, image in rows:
        draw.text((6, y + 4), row_label, fill=(255, 255, 255))
        collage.paste(image, (0, y + 20))
        y += image.height + 22
    collage.save(out_path, quality=95)
    return out_path


def run(args: argparse.Namespace) -> None:
    cfg = default_config()
    backend = resolve_backend(args.backend)
    sample = load_image(args.input, iscrop=args.iscrop)
    deca = create_deca(backend, cfg, load_flame=not args.encode_only).load_pretrained(args.weights)
    print(f"backend {deca.backend_name}" + (f" device={deca.device}" if hasattr(deca, "device") else ""))
    images = deca.asarray(sample["image"][None, ...])
    codedict = deca.encode(images, use_detail=True)

    out_dir = Path(args.savefolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = sample["imagename"]
    pred = {
        key: deca.to_numpy(codedict[key])[0]
        for key in ("shape", "tex", "exp", "pose", "cam", "light")
        if key in codedict
    }
    if "detail" in codedict:
        pred["detail"] = deca.to_numpy(codedict["detail"])[0]
    pred["image"] = sample["image"]

    opdict = None
    if deca.flame is not None and not args.encode_only:
        opdict = deca.decode(codedict, use_detail=True)
        for key in ("verts", "landmarks2d", "landmarks3d", "displacement_map"):
            if key in opdict:
                pred[key] = deca.to_numpy(opdict[key])[0]
        pred["uv_z"] = _uv_z_chw(deca, opdict["uv_z"])
        obj_path = out_dir / f"{name}.obj"
        deca.save_obj(obj_path, opdict)
        print(f"wrote {obj_path}")
    else:
        pose = deca.to_numpy(codedict["pose"])
        exp = deca.to_numpy(codedict["exp"])
        detail = deca.to_numpy(codedict["detail"])
        cond = deca.asarray(np.concatenate([pose[:, 3:], exp, detail], axis=1))
        pred["uv_z"] = _uv_z_chw(deca, deca.D_detail(cond))
        if args.encode_only:
            print("encode-only: skipping FLAME mesh decode")
        else:
            print("FLAME generic_model.pkl not found; wrote encoder/detail outputs only.")

    np.savez(out_dir / f"{name}_{backend}.npz", **pred)

    official_vis = Path(args.official_vis) if args.official_vis else ROOT / "tests" / "faces" / "IMG_0392_inputs_vis.jpg"
    if opdict is not None:
        renderer = Renderer(cfg)
        vis_np = visualize_reconstruction(
            sample["image"],
            pred["verts"],
            deca.to_numpy(opdict["trans_verts"])[0],
            pred["landmarks2d"],
            pred["landmarks3d"],
            lights=pred.get("light"),
            tex_code=pred.get("tex"),
            displacement=pred.get("displacement_map"),
            renderer=renderer,
        )
        recon_vis = Image.fromarray(vis_np)
        vis_path = out_dir / f"{name}_vis.jpg"
        recon_vis.save(vis_path, quality=95)
        print(f"wrote {vis_path}")
    else:
        recon_vis = Image.fromarray(_to_uint8(sample["image"]))
    uv = pred["uv_z"][0]
    uv_vis = np.clip((uv - uv.min()) / max(float(uv.max() - uv.min()), 1e-8) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(uv_vis).save(out_dir / f"{name}_uv_z.png")
    comparison = build_comparison(
        None,
        recon_vis,
        official_vis if official_vis.exists() else None,
        out_dir / f"{name}_compare.jpg",
        f"{backend} vis",
    )
    print(f"wrote {comparison}")
    if official_vis.exists():
        shutil.copy2(official_vis, out_dir / official_vis.name)

    if args.baseline:
        baseline = load_baseline(args.baseline)
        report = compare_arrays(baseline, pred)
        print(format_report(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DECA on MLX or PyTorch/ROCm")
    parser.add_argument("-i", "--input", default=str(ROOT / "tests" / "faces" / "IMG_0392_inputs.jpg"))
    parser.add_argument("-s", "--savefolder", default=str(ROOT / "outputs"))
    parser.add_argument("--backend", default="auto", help="auto, mlx, or torch/rocm")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--iscrop", action="store_true", help="detect/crop the face (off for the already-cropped golden image)")
    parser.add_argument("--encode-only", action="store_true", help="skip FLAME decode / OBJ export")
    parser.add_argument("--baseline", default=str(ROOT / "tests" / "baselines" / "IMG_0392.npz"))
    parser.add_argument("--official-vis", default=str(ROOT / "tests" / "faces" / "IMG_0392_inputs_vis.jpg"))
    args = parser.parse_args()
    if args.baseline and not Path(args.baseline).exists():
        args.baseline = None
    run(args)


if __name__ == "__main__":
    main()
