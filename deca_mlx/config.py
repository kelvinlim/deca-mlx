from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass
class DECAConfig:
    data_dir: Path = DATA_DIR
    image_size: int = 224
    uv_size: int = 256
    n_shape: int = 100
    n_tex: int = 50
    n_exp: int = 50
    n_pose: int = 6
    n_cam: int = 3
    n_light: int = 27
    n_detail: int = 128
    max_z: float = 0.01
    param_list: tuple[str, ...] = ("shape", "tex", "exp", "pose", "cam", "light")
    pretrained_modelpath: Path = field(default_factory=lambda: DATA_DIR / "deca_model.tar")
    mlx_weights_path: Path = field(default_factory=lambda: DATA_DIR / "deca_mlx.safetensors")
    flame_model_path: Path = field(default_factory=lambda: DATA_DIR / "generic_model.pkl")
    flame_lmk_embedding_path: Path = field(default_factory=lambda: DATA_DIR / "landmark_embedding.npy")
    topology_path: Path = field(default_factory=lambda: DATA_DIR / "head_template.obj")
    dense_template_path: Path = field(default_factory=lambda: DATA_DIR / "texture_data_256.npy")
    fixed_displacement_path: Path = field(default_factory=lambda: DATA_DIR / "fixed_displacement_256.npy")
    face_mask_path: Path = field(default_factory=lambda: DATA_DIR / "uv_face_mask.png")
    face_eye_mask_path: Path = field(default_factory=lambda: DATA_DIR / "uv_face_eye_mask.png")
    mean_tex_path: Path = field(default_factory=lambda: DATA_DIR / "mean_texture.jpg")
    tex_path: Path = field(default_factory=lambda: DATA_DIR / "FLAME_albedo_from_BFM.npz")

    @property
    def n_param(self) -> int:
        return self.n_shape + self.n_tex + self.n_exp + self.n_pose + self.n_cam + self.n_light

    @property
    def param_sizes(self) -> dict[str, int]:
        return {
            "shape": self.n_shape,
            "tex": self.n_tex,
            "exp": self.n_exp,
            "pose": self.n_pose,
            "cam": self.n_cam,
            "light": self.n_light,
        }


def default_config() -> DECAConfig:
    return DECAConfig()
