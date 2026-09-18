from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.compare import compare_arrays, load_baseline
from deca_mlx.config import default_config
from deca_mlx.deca import DECA, to_numpy
from deca_mlx.preprocess import load_image

BASELINE = ROOT / "tests" / "baselines" / "IMG_0392.npz"
FACE = ROOT / "tests" / "faces" / "IMG_0392_inputs.jpg"


@pytest.mark.skipif(not BASELINE.exists(), reason="encoder baseline NPZ is not present")
@pytest.mark.skipif(not default_config().mlx_weights_path.exists(), reason="MLX weights are not converted")
def test_encoder_matches_official_checkpoint():
    import mlx.core as mx

    baseline = load_baseline(BASELINE)
    sample = load_image(FACE, iscrop=False)
    if "image" in baseline:
        images = mx.array(baseline["image"][None, ...])
    else:
        images = mx.array(sample["image"][None, ...])
    deca = DECA(load_flame=False).load_pretrained()
    codedict = deca.encode(images, use_detail=True)
    pred = {key: to_numpy(codedict[key])[0] for key in ("shape", "tex", "exp", "pose", "cam", "light", "detail")}
    cond = mx.concatenate(
        [codedict["pose"][:, 3:], codedict["exp"], codedict["detail"]],
        axis=1,
    )
    uv_z = to_numpy(deca.D_detail(cond))
    pred["uv_z"] = np.transpose(uv_z[0], (2, 0, 1))
    report = compare_arrays(
        baseline,
        pred,
        keys=("shape", "tex", "exp", "pose", "cam", "light", "detail", "uv_z"),
    )
    assert report, "baseline did not contain encoder keys"
    for key, stats in report.items():
        assert stats["status"] != "shape_mismatch", key
        assert stats["max_abs"] < 1e-4, f"{key} max abs {stats['max_abs']}"


def test_golden_face_exists():
    assert FACE.exists()
    vis = ROOT / "tests" / "faces" / "IMG_0392_inputs_vis.jpg"
    assert vis.exists()
