"""DECA encode / decode on PyTorch (CUDA, ROCm via ``cuda``, or CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import DECAConfig, default_config
from ..export import load_obj, upsample_mesh, vertex_normals, write_obj
from .decoders import Generator
from .encoders import ResnetEncoder
from .flame import FLAME


def decompose_code(code: torch.Tensor, sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    code_dict = {}
    start = 0
    for key, size in sizes.items():
        end = start + size
        value = code[:, start:end]
        if key == "light":
            value = value.reshape(value.shape[0], 9, 3)
        code_dict[key] = value
        start = end
    return code_dict


def batch_orth_proj(points: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    camera = camera.reshape(-1, 1, 3)
    translated = torch.cat([points[:, :, :2] + camera[:, :, 1:], points[:, :, 2:]], dim=2)
    return camera[:, :, :1] * translated


def default_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DECA(nn.Module):
    backend_name = "torch"

    def __init__(
        self,
        config: DECAConfig | None = None,
        load_flame: bool | None = None,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.cfg = config or default_config()
        self.device = torch.device(device) if device is not None else default_torch_device()
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
        self.to(self.device)
        self.eval()

    def asarray(self, array) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array.to(self.device)
        return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(self.device)

    def to_numpy(self, array) -> np.ndarray:
        if isinstance(array, np.ndarray):
            return array
        return array.detach().cpu().numpy()

    def load_pretrained(self, path: str | Path | None = None) -> "DECA":
        weights_path = Path(path or self.cfg.pretrained_modelpath)
        if weights_path.suffix in {".safetensors", ".mlx"}:
            raise ValueError(
                f"The torch/ROCm backend loads official deca_model.tar, not {weights_path.name}. "
                "Pass the tar path or omit --weights."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Official DECA checkpoint not found at {weights_path}. "
                "Download deca_model.tar into data/."
            )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.E_flame.load_state_dict(checkpoint["E_flame"])
        self.E_detail.load_state_dict(checkpoint["E_detail"])
        self.D_detail.load_state_dict(checkpoint["D_detail"])
        self.to(self.device)
        self.eval()
        return self

    def encode(self, images, use_detail: bool = True) -> dict[str, torch.Tensor]:
        images = self.asarray(images)
        parameters = self.E_flame(images)
        codedict = decompose_code(parameters, self.cfg.param_sizes)
        codedict["images"] = images
        if use_detail:
            codedict["detail"] = self.E_detail(images)
        return codedict

    def decode(self, codedict: dict[str, torch.Tensor], use_detail: bool = True) -> dict[str, torch.Tensor]:
        if self.flame is None:
            raise RuntimeError("FLAME model is not loaded; cannot decode vertices.")
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict["shape"],
            expression_params=codedict["exp"],
            pose_params=codedict["pose"],
        )
        landmarks3d_world = landmarks3d
        landmarks2d = batch_orth_proj(landmarks2d, codedict["cam"])[:, :, :2]
        landmarks2d = torch.cat([landmarks2d[:, :, :1], -landmarks2d[:, :, 1:]], dim=2)
        landmarks3d = batch_orth_proj(landmarks3d, codedict["cam"])
        landmarks3d = torch.cat([landmarks3d[:, :, :1], -landmarks3d[:, :, 1:]], dim=2)
        trans_verts = batch_orth_proj(verts, codedict["cam"])
        trans_verts = torch.cat([trans_verts[:, :, :1], -trans_verts[:, :, 1:]], dim=2)
        opdict = {
            "verts": verts,
            "trans_verts": trans_verts,
            "landmarks2d": landmarks2d,
            "landmarks3d": landmarks3d,
            "landmarks3d_world": landmarks3d_world,
        }
        if use_detail:
            cond = torch.cat([codedict["pose"][:, 3:], codedict["exp"], codedict["detail"]], dim=1)
            uv_z = self.D_detail(cond)
            displacement = uv_z
            if self.fixed_uv_dis is not None:
                displacement = displacement + self.asarray(self.fixed_uv_dis)[None, None, :, :]
            opdict["uv_z"] = uv_z
            opdict["displacement_map"] = displacement
        return opdict

    def save_obj(self, filename: str | Path, opdict: dict) -> list[Path]:
        if self.faces is None:
            raise RuntimeError("head_template.obj is required to export meshes.")
        verts = self.to_numpy(opdict["verts"][0])
        written = [write_obj(filename, verts, self.faces)]
        if "displacement_map" in opdict and self.dense_template is not None:
            displacement = self.to_numpy(opdict["displacement_map"][0, 0])
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
