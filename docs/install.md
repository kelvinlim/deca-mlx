# Install

Pick one path. Both need the licensed DECA / FLAME files in `data/` (not in git).

| Machine | Backend | Extra / wheels |
|---|---|---|
| Apple Silicon | MLX (Metal) | `.[apple,convert]` then convert weights |
| AMD ROCm (Ryzen AI 395, gfx1151) | PyTorch (`cuda` device) | package only, then AMD wheels |
| NVIDIA CUDA | PyTorch | `.[torch]` or a CUDA torch wheel |
| CPU only | PyTorch CPU | `.[torch]` |

`mlx` is not a hard dependency. A ROCm box should never install the Apple extra.

## Licensed files (both backends)

Place these under `data/` (gitignored):

1. `deca_model.tar` — official DECA checkpoint, Google Drive id `1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje` (same as DECA `fetch_data.sh`).
2. `generic_model.pkl` — FLAME 2020 from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) (registration + license). Required for mesh decode and OBJ export.

Public topology / UV assets are already in `data/`. See [data/README.md](../data/README.md).

## Apple Silicon (MLX)

```bash
git clone https://github.com/kelvinlim/deca-mlx.git
cd deca-mlx
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[apple,convert]"
```

Copy `deca_model.tar` and `generic_model.pkl` into `data/`, then convert weights for MLX:

```bash
python -m deca_mlx.convert
```

That writes `data/deca_mlx.safetensors`. Confirm Metal:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
```

You should see `Device(gpu, 0)`.

## AMD ROCm (Ryzen AI Max+ 395)

The 395 iGPU is `gfx1151` (Radeon 8060S). Official `mlx` will not run here. Use the PyTorch backend. `torch.device("cuda")` is the ROCm GPU.

Do **not** `pip install torch` from PyPI — that is NVIDIA or CPU.

```bash
git clone https://github.com/kelvinlim/deca-mlx.git
cd deca-mlx
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install AMD’s wheels (index changes by gfx). For gfx1151:

```bash
pip install torch torchvision --index-url https://repo.amd.com/rocm/whl/gfx1151/
```

Follow [AMD’s Ryzen PyTorch install](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html) if the index URL or ROCm version has moved.

Copy `deca_model.tar` and `generic_model.pkl` into `data/`. You do **not** run `python -m deca_mlx.convert`. The torch backend loads the tar directly.

Verify the iGPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expect `True` and a Radeon / Ryzen AI device name.

Linux on Ryzen APUs typically needs a recent OEM kernel (AMD documents 6.14-1018+). Ubuntu 24.04 is the tested ROCm desktop OS.

## NVIDIA CUDA

```bash
pip install -e ".[torch]"
```

Or install a CUDA PyTorch wheel from pytorch.org, then `pip install -e .`. Same `deca_model.tar` + FLAME files. `--backend torch` or `--backend cuda`.

## Next

See [usage.md](usage.md) for backend flags, the demo, and troubleshooting.
