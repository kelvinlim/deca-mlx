from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.backend import create_deca, torch_available
from deca_mlx.compare import compare_arrays, load_baseline
from deca_mlx.config import default_config
from deca_mlx.preprocess import load_image

BASELINE = ROOT / "tests" / "baselines" / "IMG_0392.npz"
FACE = ROOT / "tests" / "faces" / "IMG_0392_inputs.jpg"


@pytest.mark.skipif(not torch_available(), reason="torch is not installed")
@pytest.mark.skipif(not default_config().pretrained_modelpath.exists(), reason="deca_model.tar is missing")
@pytest.mark.skipif(not BASELINE.exists(), reason="encoder baseline NPZ is not present")
def test_torch_encoder_matches_official_checkpoint():
    baseline = load_baseline(BASELINE)
    sample = load_image(FACE, iscrop=False)
    image = baseline["image"] if "image" in baseline else sample["image"]
    deca = create_deca("torch", load_flame=False, device="cpu").load_pretrained()
    codedict = deca.encode(image[None, ...], use_detail=True)
    pred = {key: deca.to_numpy(codedict[key])[0] for key in ("shape", "tex", "exp", "pose", "cam", "light", "detail")}
    pose = deca.to_numpy(codedict["pose"])
    exp = deca.to_numpy(codedict["exp"])
    detail = deca.to_numpy(codedict["detail"])
    cond = deca.asarray(np.concatenate([pose[:, 3:], exp, detail], axis=1))
    pred["uv_z"] = deca.to_numpy(deca.D_detail(cond))[0]
    report = compare_arrays(
        baseline,
        pred,
        keys=("shape", "tex", "exp", "pose", "cam", "light", "detail", "uv_z"),
    )
    assert report
    for key, stats in report.items():
        assert stats["status"] != "shape_mismatch", key
        assert stats["max_abs"] < 1e-4, f"{key} max abs {stats['max_abs']}"


@pytest.mark.skipif(not torch_available(), reason="torch is not installed")
@pytest.mark.skipif(not default_config().pretrained_modelpath.exists(), reason="deca_model.tar is missing")
@pytest.mark.skipif(not default_config().flame_model_path.exists(), reason="FLAME pickle is missing")
@pytest.mark.skipif(not BASELINE.exists() or "verts" not in np.load(BASELINE), reason="vertex baseline missing")
def test_torch_flame_matches_vertex_baseline():
    baseline = load_baseline(BASELINE)
    deca = create_deca("torch", device="cpu").load_pretrained()
    codedict = deca.encode(baseline["image"][None, ...], use_detail=True)
    opdict = deca.decode(codedict, use_detail=True)
    pred = {
        "verts": deca.to_numpy(opdict["verts"])[0],
        "landmarks2d": deca.to_numpy(opdict["landmarks2d"])[0],
        "landmarks3d": deca.to_numpy(opdict["landmarks3d"])[0],
        "displacement_map": deca.to_numpy(opdict["displacement_map"])[0],
    }
    report = compare_arrays(baseline, pred, keys=("verts", "landmarks2d", "landmarks3d", "displacement_map"))
    for key, stats in report.items():
        assert stats["max_abs"] < 1e-4, f"{key} max abs {stats['max_abs']}"
