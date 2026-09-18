"""DECA encode / decode on MLX."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DECAConfig, default_config
from .decoders import Generator
from .encoders import ResnetEncoder
from .export import load_obj, upsample_mesh, vertex_normals, write_obj
from .flame import FLAME


def decompose_code(code: mx.array, sizes: dict[str, int]) -> dict[str, mx.array]:
    code_dict = {}
    start = 0
    for key, size in sizes.items():
        end = start + size
        value = code[:, start:end]
        if key == "light":
            value = value.reshape((value.shape[0], 9, 3))
        code_dict[key] = value
        start = end
    return code_dict


def batch_orth_proj(points: mx.array, camera: mx.array) -> mx.array:
    camera = camera.reshape((-1, 1, 3))
    translated = mx.concatenate([points[:, :, :2] + camera[:, :, 1:], points[:, :, 2:]], axis=2)
    return camera[:, :, 0:1] * translated


def to_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


class DECA(nn.Module):
    def __init__(self, config: DECAConfig | None = None, load_flame: bool | None = None):
        super().__init__()
        self.cfg = config or default_config()
        self.E_flame = ResnetEncoder(outsize=self.cfg.n_param)
        self.E_detail = ResnetEncoder(outsize=self.cfg.n_detail)
        self.D_detail = Generator(
            latent_dim=self.cfg.n_detail + self.cfg.n_exp + 3,
            out_channels=1,
            out_scale=self.cfg.max_z,
        )
        self.flame = None
        should_load_flame = self.cfg.flame_model_path.exists() if load_flame is None else load_flame
        if should_load_flame:
            self.flame = FLAME(self.cfg)
        self.faces = None
        if self.cfg.topology_path.exists():
            _, _, faces, _ = load_obj(self.cfg.topology_path)
            self.faces = faces
        self.fixed_uv_dis = None
        if self.cfg.fixed_displacement_path.exists():
            self.fixed_uv_dis = np.load(self.cfg.fixed_displacement_path).astype(np.float32)
        self.dense_template = None
        if self.cfg.dense_template_path.exists():
            self.dense_template = np.load(self.cfg.dense_template_path, allow_pickle=True, encoding="latin1").item()
        self.eval()

    def load_pretrained(self, path: str | Path | None = None) -> "DECA":
        weights_path = Path(path or self.cfg.mlx_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"MLX weights not found at {weights_path}. Run `python -m deca_mlx.convert` first."
            )
        self.load_weights(str(weights_path), strict=False)
        self.eval()
        mx.eval(self.parameters())
        return self

    def encode(self, images: mx.array, use_detail: bool = True) -> dict[str, mx.array]:
        parameters = self.E_flame(images)
        codedict = decompose_code(parameters, self.cfg.param_sizes)
        codedict["images"] = images
        if use_detail:
            codedict["detail"] = self.E_detail(images)
        return codedict

    def decode(self, codedict: dict[str, mx.array], use_detail: bool = True) -> dict[str, mx.array]:
        if self.flame is None:
            raise RuntimeError("FLAME model is not loaded; cannot decode vertices.")
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict["shape"],
            expression_params=codedict["exp"],
            pose_params=codedict["pose"],
        )
        landmarks3d_world = landmarks3d
        landmarks2d = batch_orth_proj(landmarks2d, codedict["cam"])[:, :, :2]
        landmarks2d = mx.concatenate([landmarks2d[:, :, :1], -landmarks2d[:, :, 1:]], axis=2)
        landmarks3d = batch_orth_proj(landmarks3d, codedict["cam"])
        landmarks3d = mx.concatenate([landmarks3d[:, :, :1], -landmarks3d[:, :, 1:]], axis=2)
        trans_verts = batch_orth_proj(verts, codedict["cam"])
        trans_verts = mx.concatenate([trans_verts[:, :, :1], -trans_verts[:, :, 1:]], axis=2)
        opdict = {
            "verts": verts,
            "trans_verts": trans_verts,
            "landmarks2d": landmarks2d,
            "landmarks3d": landmarks3d,
            "landmarks3d_world": landmarks3d_world,
        }
        if use_detail:
            cond = mx.concatenate([codedict["pose"][:, 3:], codedict["exp"], codedict["detail"]], axis=1)
            uv_z = self.D_detail(cond)
            # Generator emits NHWC; keep a CHW map for the official baseline layout.
            displacement = uv_z.transpose(0, 3, 1, 2)
            if self.fixed_uv_dis is not None:
                displacement = displacement + mx.array(self.fixed_uv_dis)[None, None, :, :]
            opdict["uv_z"] = uv_z
            opdict["displacement_map"] = displacement
        return opdict

    def save_obj(self, filename: str | Path, opdict: dict[str, mx.array]) -> list[Path]:
        if self.faces is None:
            raise RuntimeError("head_template.obj is required to export meshes.")
        verts = to_numpy(opdict["verts"][0])
        written = [write_obj(filename, verts, self.faces)]
        if "displacement_map" in opdict and self.dense_template is not None:
            displacement = to_numpy(opdict["displacement_map"][0, 0])
            normals = vertex_normals(verts, self.faces)
            dense_vertices, dense_colors, dense_faces = upsample_mesh(
                verts, normals, displacement, None, self.dense_template
            )
            detail_path = Path(filename)
            detail_path = detail_path.with_name(detail_path.stem + "_detail.obj")
            written.append(
                write_obj(detail_path, dense_vertices, dense_faces, colors=dense_colors, inverse_face_order=True)
            )
        return written
