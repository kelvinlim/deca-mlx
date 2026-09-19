# DECA on MLX / ROCm

Inference port of [DECA](https://github.com/yfeng95/DECA) with two array backends:

- **MLX** on Apple Silicon (Metal)
- **PyTorch** on AMD ROCm, NVIDIA CUDA, or CPU (`torch.device("cuda")` is the ROCm GPU)

The 6-panel vis, OBJ export, and preprocess are shared numpy. Training, expression transfer, and a differentiable rasterizer are out of scope. See [CHANGELOG.md](CHANGELOG.md).

## Status

| Piece | State |
|---|---|
| MLX encode / FLAME / detail | Done, matches official codes to ~`1e-7`, verts ~`3e-8` |
| PyTorch / ROCm encode / FLAME / detail | Done; loads `deca_model.tar` directly |
| Coarse / detail OBJ + 6-panel vis | Done (CPU software rasterizer) |
| Training / expression transfer | Not started |

`--backend auto` uses MLX if it is installed, otherwise PyTorch. Reconstruction of one cropped `224×224` frame is about **22 ms** on Apple Metal. On a Ryzen AI 395, the same path uses the iGPU through ROCm PyTorch; vis stays on CPU.

Licensed files stay local and gitignored: `data/deca_model.tar`, `data/deca_mlx.safetensors`, `data/generic_model.pkl`.

## Golden comparison face

- Input: `tests/faces/IMG_0392_inputs.jpg`
- Official CUDA visualization: `tests/faces/IMG_0392_inputs_vis.jpg`

## Docs

- [Install (Apple MLX and AMD ROCm)](docs/install.md)
- [Run, API, tests, troubleshooting](docs/usage.md)
- [Model assets](data/README.md)

## Setup (short)

Apple Silicon:

```bash
uv pip install -e ".[apple,convert]"
python -m deca_mlx.convert
```

Ryzen AI 395 / ROCm — do not install PyPI `torch`:

```bash
pip install -e .
pip install torch torchvision --index-url https://repo.amd.com/rocm/whl/gfx1151/
```

Both need `data/deca_model.tar` and FLAME `data/generic_model.pkl`. Full steps: [docs/install.md](docs/install.md).

## Run

```bash
python -m deca_mlx.demo --backend auto -i tests/faces/IMG_0392_inputs.jpg
```

`--backend mlx` or `--backend torch` (aliases: `rocm`, `cuda`) force a backend. `DECA_BACKEND` does the same. Details: [docs/usage.md](docs/usage.md).

## Layout

```
docs/                   install and usage
deca_mlx/               shared config, vis, export, demo, backend factory
deca_mlx/torch_backend/ PyTorch / ROCm encode + FLAME
scripts/                weight convert and baseline dumps
tests/faces/            official input + CUDA visualization
```
