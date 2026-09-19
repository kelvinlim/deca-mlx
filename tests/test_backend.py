from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.backend import mlx_available, resolve_backend, torch_available


def test_resolve_auto_picks_an_installed_backend():
    backend = resolve_backend("auto")
    assert backend in {"mlx", "torch"}


@pytest.mark.skipif(not mlx_available(), reason="mlx is not installed")
def test_resolve_mlx():
    assert resolve_backend("mlx") == "mlx"


@pytest.mark.skipif(not torch_available(), reason="torch is not installed")
def test_resolve_torch_aliases():
    assert resolve_backend("rocm") == "torch"
    assert resolve_backend("cuda") == "torch"
