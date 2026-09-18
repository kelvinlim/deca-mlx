**CUDA Support: Full / Native**

* The official DECA codebase (`yfeng95/DECA`) is built on **PyTorch**, **PyTorch3D**, and custom CUDA/C++ extensions (such as `rasterizer_type` or `kornia`).
* It runs out-of-the-box on Linux or Windows systems using NVIDIA GPUs with CUDA.

**MLX Support: No Native Integration**

* DECA has **no official MLX port or native Apple MLX support**.
* Because DECA depends heavily on the PyTorch ecosystem, PyTorch3D/rasterization libraries, and specific CUDA kernels for mesh rendering, running it directly in MLX would require re-implementing the FLAME mesh decoder, custom loss functions, and a 3D rasterizer using `mlx.core` and `mlx.nn`.

---

**Workaround for Apple Silicon (Mac)**

If you want to run DECA on a Mac with Apple Silicon (M1/M2/M3/M4/M5), you can still execute the PyTorch implementation using PyTorch's **MPS (Metal Performance Shaders)** device or standard CPU execution, though custom PyTorch3D CUDA rasterizers will fallback to software rasterization or standard PyTorch3D CPU rendering.

```python
import torch

# DECA can run on Apple Silicon via PyTorch MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

```