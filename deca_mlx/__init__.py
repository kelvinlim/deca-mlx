"""Native MLX inference port of DECA (Detailed Expression Capture and Animation)."""

from .config import DECAConfig, default_config

__all__ = ["DECA", "DECAConfig", "default_config"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "DECA":
        from .deca import DECA

        return DECA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
