"""Compare MLX outputs against a dumped official CUDA / PyTorch baseline NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ROOT

DEFAULT_BASELINE = ROOT / "tests" / "baselines" / "IMG_0392.npz"
COMPARE_KEYS = (
    "shape",
    "tex",
    "exp",
    "pose",
    "cam",
    "light",
    "detail",
    "uv_z",
    "verts",
    "landmarks2d",
    "landmarks3d",
    "displacement_map",
)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def compare_arrays(baseline: dict, pred: dict, keys: tuple[str, ...] = COMPARE_KEYS) -> dict[str, dict]:
    report = {}
    for key in keys:
        if key not in baseline or key not in pred:
            continue
        left = np.asarray(baseline[key])
        right = np.asarray(pred[key])
        if left.shape != right.shape:
            report[key] = {"status": "shape_mismatch", "baseline": left.shape, "pred": right.shape}
            continue
        err = max_abs(left, right)
        report[key] = {
            "status": "ok" if err < 1e-4 else "high",
            "max_abs": err,
            "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
        }
    return report


def format_report(report: dict[str, dict]) -> str:
    if not report:
        return "no overlapping keys to compare"
    lines = ["key                  max_abs       rmse   status"]
    for key, stats in report.items():
        if stats.get("status") == "shape_mismatch":
            lines.append(f"{key:20s} shape {stats['baseline']} vs {stats['pred']}")
        else:
            lines.append(f"{key:20s} {stats['max_abs']:.4e} {stats['rmse']:.4e} {stats['status']}")
    return "\n".join(lines)


def load_baseline(path: str | Path = DEFAULT_BASELINE) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline NPZ not found at {path}. Run scripts/dump_encoder_baseline.py "
            "or the official CUDA dump script first."
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
