from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.config import default_config
from deca_mlx.render import Renderer, add_directionlight, rasterize, vis_landmarks


def test_rasterize_covers_triangle_center():
    verts = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    attrs = np.ones((1, 3, 3), dtype=np.float32)
    image, alpha = rasterize(verts, faces, attrs, 32, 32)
    assert alpha[16, 16] == 1.0
    assert image[16, 16, 0] == pytest.approx(1.0)
    assert alpha[0, 0] == 0.0


def test_direction_light_front_is_bright():
    front = add_directionlight(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    back = add_directionlight(np.array([[0.0, 0.0, -1.0]], dtype=np.float32))
    assert front[0] > back[0]


def test_vis_landmarks_keeps_image_size():
    image = np.full((224, 224, 3), 0.4, dtype=np.float32)
    kpts = np.zeros((68, 2), dtype=np.float32)
    vis = vis_landmarks(image, kpts)
    assert vis.shape == (224, 224, 3)


@pytest.mark.skipif(not default_config().topology_path.exists(), reason="head_template.obj missing")
def test_renderer_visualize_is_six_panels():
    renderer = Renderer()
    verts = renderer.template_verts
    trans = verts - verts.mean(axis=0, keepdims=True)
    trans = trans * (1.6 / np.max(np.abs(trans)))
    trans[:, 1:] = -trans[:, 1:]
    image = np.full((224, 224, 3), 0.3, dtype=np.float32)
    lmk = np.zeros((68, 2), dtype=np.float32)
    vis = renderer.visualize(image, verts, trans, lmk, np.zeros((68, 3), dtype=np.float32), size=64)
    assert vis.shape == (64, 64 * 6, 3)
    # Coarse shape panel should not be empty.
    coarse = vis[:, 64 * 3 : 64 * 4]
    assert coarse.max() > 10
