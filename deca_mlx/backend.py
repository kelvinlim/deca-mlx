"""Select the MLX or PyTorch/ROCm inference backend."""

from __future__ import annotations

import os

from .config import DECAConfig


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def torch_gpu_available() -> bool:
    if not torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def resolve_backend(name: str | None = None) -> str:
    requested = (name or os.environ.get("DECA_BACKEND") or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "mlx": "mlx",
        "apple": "mlx",
        "metal": "mlx",
        "torch": "torch",
        "pytorch": "torch",
        "rocm": "torch",
        "cuda": "torch",
        "cpu": "torch",
    }
    if requested not in aliases:
        raise ValueError(f"Unknown backend {name!r}. Use auto, mlx, or torch/rocm.")
    backend = aliases[requested]
    if backend == "auto":
        if mlx_available():
            return "mlx"
        if torch_available():
            return "torch"
        raise RuntimeError(
            "No inference backend found. On Apple Silicon install the apple extra (`pip install -e '.[apple]'`). "
            "On AMD ROCm install this package then AMD's gfx1151 PyTorch wheels."
        )
    if backend == "mlx" and not mlx_available():
        raise RuntimeError("MLX is not installed. `pip install -e '.[apple]'` on Apple Silicon.")
    if backend == "torch" and not torch_available():
        raise RuntimeError(
            "PyTorch is not installed. On ROCm use AMD's wheels "
            "(e.g. https://repo.amd.com/rocm/whl/gfx1151/), not `pip install torch` from PyPI."
        )
    return backend


def get_deca_class(backend: str | None = None):
    chosen = resolve_backend(backend)
    if chosen == "mlx":
        from .deca import DECA

        return DECA
    from .torch_backend.deca import DECA

    return DECA


def create_deca(
    backend: str | None = None,
    config: DECAConfig | None = None,
    load_flame: bool | None = None,
    device=None,
):
    chosen = resolve_backend(backend)
    cls = get_deca_class(chosen)
    if chosen == "torch":
        return cls(config, load_flame=load_flame, device=device)
    return cls(config, load_flame=load_flame)
