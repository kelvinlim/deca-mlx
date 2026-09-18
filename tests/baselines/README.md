# Numeric baselines

Official DECA does not ship reconstruction `.obj` / `.mat` dumps. This folder holds tensors dumped from the official checkpoint so MLX can be compared numerically.

## Encoder baseline (no CUDA, no FLAME)

From the repo root, after `data/deca_model.tar` is present:

```bash
source .venv/bin/activate
python scripts/dump_encoder_baseline.py \
  -i tests/faces/IMG_0392_inputs.jpg \
  -o tests/baselines/IMG_0392.npz
```

This writes `shape`, `tex`, `exp`, `pose`, `cam`, `light`, `detail`, and the exact 224×224 `image` tensor.

## Full CUDA baseline (optional)

On a CUDA machine or in the [official Colab](https://colab.research.google.com/github/YadiraF/DECA/blob/master/Detailed_Expression_Capture_and_Animation.ipynb):

```bash
python scripts/dump_cuda_baseline.py \
  --deca-root /path/to/DECA \
  --image tests/faces/IMG_0392_inputs.jpg \
  --output tests/baselines/IMG_0392.npz \
  --device cuda
```

That dump also includes `verts`, `landmarks2d`, `landmarks3d`, and `displacement_map`.
