# Usage

After [install.md](install.md), reconstruction is the same command on Apple or ROCm.

## Demo

```bash
source .venv/bin/activate
python -m deca_mlx.demo --backend auto -i tests/faces/IMG_0392_inputs.jpg
```

`IMG_0392_inputs.jpg` is already cropped; leave `--iscrop` off for that file.

### Backend

| Value | Meaning |
|---|---|
| `auto` (default) | MLX if importable, else PyTorch |
| `mlx`, `apple`, `metal` | Apple MLX |
| `torch`, `pytorch`, `rocm`, `cuda`, `cpu` | PyTorch (`cuda` when ROCm/NVIDIA is present) |

Environment override: `DECA_BACKEND=torch`.

```bash
python -m deca_mlx.demo --backend mlx -i tests/faces/IMG_0392_inputs.jpg
python -m deca_mlx.demo --backend rocm -i tests/faces/IMG_0392_inputs.jpg
DECA_BACKEND=torch python -m deca_mlx.demo --encode-only
```

`--weights` is backend-specific:

- MLX: `data/deca_mlx.safetensors` (default after convert)
- Torch / ROCm: `data/deca_model.tar` (default)

`--encode-only` skips FLAME, OBJ, and the shape vis.

### Outputs

Under `outputs/` (gitignored):

| File | What |
|---|---|
| `{name}.obj` | Coarse FLAME mesh |
| `{name}_detail.obj` | Dense displacement mesh |
| `{name}_vis.jpg` | 6-panel vis (input, 2D/3D landmarks, coarse, detail, lit albedo) |
| `{name}_compare.jpg` | That vis stacked on the official CUDA strip when the golden face is used |
| `{name}_{backend}.npz` | Codes, verts, landmarks |
| `{name}_uv_z.png` | Displacement preview |

The last vis panel uses `mean_texture.jpg` unless you add `data/FLAME_albedo_from_BFM.npz`.

## Python API

```python
from deca_mlx import create_deca
from deca_mlx.preprocess import load_image

sample = load_image("tests/faces/IMG_0392_inputs.jpg", iscrop=False)
deca = create_deca("auto").load_pretrained()
codedict = deca.encode(deca.asarray(sample["image"][None, ...]))
opdict = deca.decode(codedict)
verts = deca.to_numpy(opdict["verts"])[0]
```

`create_deca("torch", device="cuda")` forces the ROCm/NVIDIA device. On Apple, `create_deca("mlx")` uses Metal.

`from deca_mlx import DECA` is the auto-selected class.

## Tests

```bash
python -m pytest
```

MLX tests skip without `mlx`. Torch tests skip without `torch` or `deca_model.tar`. Regenerating the numeric baseline:

```bash
python scripts/dump_encoder_baseline.py
python scripts/dump_official_flame_baseline.py
```

A real NVIDIA CUDA dump is optional: [tests/baselines/README.md](../tests/baselines/README.md).

## Troubleshooting

**`No inference backend found`**  
Install `.[apple]` on a Mac, or AMD/CUDA/CPU PyTorch on Linux.

**`MLX is not installed`** with `--backend mlx`  
You are on a non-Apple machine, or forgot the apple extra.

**`torch.cuda.is_available()` is False on a 395**  
ROCm / kernel / wheels are wrong. Reinstall AMD gfx1151 wheels; do not use PyPI torch. Check `rocminfo`.

**`Official DECA checkpoint not found`**  
`data/deca_model.tar` is missing. The torch backend does not use safetensors.

**`MLX weights not found`**  
On Apple, run `python -m deca_mlx.convert` after placing the tar.

**`FLAME model is not loaded`**  
`data/generic_model.pkl` is missing. Encode-only still works.

**Vis is slow (~0.8 s)**  
The 6-panel strip is a CPU software rasterizer. Encode + FLAME decode is the fast path (~22 ms on Apple Metal).

**Sixth vis panel is skin-colored, official is white**  
Missing BFM albedo PCA. Optional: `data/FLAME_albedo_from_BFM.npz`.
