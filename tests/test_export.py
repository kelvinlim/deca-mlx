from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.config import default_config
from deca_mlx.export import load_obj, vertex_normals, write_obj


def test_write_obj_roundtrip(tmp_path):
    cfg = default_config()
    verts, _, faces, _ = load_obj(cfg.topology_path)
    out = write_obj(tmp_path / "template.obj", verts, faces)
    assert out.exists()
    loaded_verts, _, loaded_faces, _ = load_obj(out)
    assert loaded_verts.shape == verts.shape
    assert loaded_faces.shape == faces.shape
    normals = vertex_normals(verts, faces)
    assert normals.shape == verts.shape
