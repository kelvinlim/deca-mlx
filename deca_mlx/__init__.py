"""Native MLX / PyTorch inference port of DECA."""

from .backend import create_deca, get_deca_class, resolve_backend
from .config import DECAConfig, default_config

__all__ = ["DECA", "DECAConfig", "create_deca", "default_config", "get_deca_class", "resolve_backend"]
__version__ = "0.2.0"


def __getattr__(name: str):
    if name == "DECA":
        return get_deca_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
