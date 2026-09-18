# DECA on MLX

Native Apple Silicon inference port of [DECA](https://github.com/yfeng95/DECA). Encode, FLAME decode, detail displacement, OBJ export, and the official 6-panel vis all run locally. See [CHANGELOG.md](CHANGELOG.md) for what landed in 0.1.0.

Official DECA is PyTorch + CUDA / PyTorch3D. This repo does **not** use that stack for inference: the networks and FLAME LBS are `mlx.core` / `mlx.nn` on Metal. The vis strip is a CPU software rasterizer that follows official `SRenderY` lighting, not PyTorch3D.

Training, expression transfer, and a differentiable rasterizer are out of scope.

## Status

| Piece | State |
|---|---|
| Weight convert (`deca_model.tar` → safetensors) | Done |
| `E_flame` / `E_detail` / `D_detail` on MLX | Done, matches official codes to ~`1e-7` |
| FLAME decode + landmarks | Done, verts vs official-equivalent PyTorch FLAME ~`3e-8` |
| Coarse / detail OBJ export | Done |
| 6-panel vis (input, 2D/3D lmk, coarse, detail, albedo) | Done; albedo uses mean texture unless BFM PCA is present |
| Numeric + visual check on `IMG_0392` | Done |
| Training / expression transfer | Not started |
| Differentiable MLX rasterizer | Not started |

Reconstruction of one cropped `224×224` frame is about **22 ms** on Metal (encode ~18 ms, decode ~4 ms). Writing the vis strip is ~0.8 s on CPU.

Licensed files stay local and gitignored: `data/deca_model.tar`, `data/deca_mlx.safetensors`, `data/generic_model.pkl`.

## Golden comparison face

Official DECA does not ship numeric CUDA mesh dumps. It does ship this already-processed pair:

- Input: `tests/faces/IMG_0392_inputs.jpg`
- Official CUDA visualization: `tests/faces/IMG_0392_inputs_vis.jpg`

The vis strip is input, 2D landmarks, 3D landmarks, coarse shape, detailed shape, and lit albedo. A second pair is `tests/faces/id04657_*`.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[convert]"
```

### Weights and FLAME

1. Official DECA checkpoint `data/deca_model.tar`  
   Google Drive id `1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje` (same as `fetch_data.sh`).
2. Convert to MLX:

   ```bash
   python -m deca_mlx.convert
   ```

3. FLAME 2020 `data/generic_model.pkl` from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) (registration + license). Required for mesh decode / OBJ export.

Public topology / UV assets from the official repo are already in `data/`. Optional `data/FLAME_albedo_from_BFM.npz` makes the 6th vis panel match official CUDA albedo; without it the panel uses `mean_texture.jpg`.

## Run

```bash
python -m deca_mlx.demo -i tests/faces/IMG_0392_inputs.jpg
```

Writes `outputs/IMG_0392_inputs.obj`, `_detail.obj`, the MLX 6-panel vis (`_vis.jpg`), and a stacked comparison against the official CUDA vis.

`--encode-only` skips FLAME if you only want to check `codedict`.

## Numeric baseline

Encoder-only, no CUDA:

```bash
python scripts/dump_encoder_baseline.py
```

Official-equivalent FLAME verts (PyTorch CPU, same codes):

```bash
python scripts/dump_official_flame_baseline.py
```

A true NVIDIA dump is optional: see [tests/baselines/README.md](tests/baselines/README.md).

```bash
python -m pytest
```

## Layout

```
deca_mlx/     encode, FLAME, detail, convert, export, software vis, demo
scripts/      weight convert, encoder / FLAME / CUDA baseline dumps
tests/faces/  official input + CUDA visualization
tests/baselines/  numeric NPZ for IMG_0392
```
