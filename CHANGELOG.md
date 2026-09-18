# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-09-18

First working native MLX inference port of [yfeng95/DECA](https://github.com/yfeng95/DECA). Reconstruction runs on Apple Silicon via `mlx.core` / `mlx.nn` (Metal). Training and expression transfer are still out of scope.

### Added

- MLX ResNet50 encoders (`E_flame`, `E_detail`) and detail generator (`D_detail`), converted from official `deca_model.tar` to `data/deca_mlx.safetensors`.
- FLAME 2020 decoder (LBS, static/dynamic landmarks) with a chumpy-free pickle loader.
- Coarse and dense-detail OBJ export.
- Software rasterizer matching official DECA `SRenderY` lighting, producing the 6-panel vis (input, 2D/3D landmarks, coarse, detail, lit albedo).
- Golden-face assets and numeric baseline for `tests/faces/IMG_0392_inputs.jpg`, plus encoder / official-FLAME / CUDA dump scripts.

### Accuracy (`IMG_0392`)

MLX vs official checkpoint codes and official-equivalent PyTorch FLAME vertices (no NVIDIA GPU on this machine):

- encoder / detail / `uv_z`: max abs ~`1e-7`–`1e-8`
- verts: max abs `2.98e-8`
- landmarks 2D/3D: max abs `1.79e-7`
- displacement map: max abs `3.83e-8`

Shape panels match the official CUDA vis. The 6th (lit albedo) panel uses `mean_texture.jpg` unless `data/FLAME_albedo_from_BFM.npz` is present.

### Performance

After warmup, one already-cropped `224×224` frame on this machine:

- encode (MLX): ~18 ms
- decode FLAME + detail (MLX): ~4 ms
- 6-panel vis (CPU numpy): ~780 ms
- reconstruction only (no vis): ~22 ms
