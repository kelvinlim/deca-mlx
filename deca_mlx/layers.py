"""Helpers that keep Sequential-style module lists aligned with the official checkpoint."""

from __future__ import annotations

import mlx.core as mx


def apply_seq(modules, x: mx.array) -> mx.array:
    for module in modules:
        x = module(x)
    return x
