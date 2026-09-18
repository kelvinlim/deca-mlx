"""Convert official ``deca_model.tar`` weights into an MLX safetensors checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DECAConfig, default_config


def _is_conv_weight(name: str, array: np.ndarray) -> bool:
    return array.ndim == 4 and name.endswith(".weight")


def convert_state_dict(state_dict: dict) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if key.endswith("num_batches_tracked"):
            continue
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if _is_conv_weight(key, array):
            array = np.transpose(array, (0, 2, 3, 1))
        converted[key] = np.ascontiguousarray(array)
    return converted


def convert_checkpoint(src: Path | None = None, dst: Path | None = None, config: DECAConfig | None = None) -> Path:
    config = config or default_config()
    src = Path(src or config.pretrained_modelpath)
    dst = Path(dst or config.mlx_weights_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Official DECA checkpoint not found at {src}. "
            "Download deca_model.tar (Google Drive id 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje) into data/."
        )

    import torch

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    weights: dict[str, np.ndarray] = {}
    for prefix in ("E_flame", "E_detail", "D_detail"):
        if prefix not in checkpoint:
            raise KeyError(f"checkpoint is missing {prefix}")
        for key, array in convert_state_dict(checkpoint[prefix]).items():
            weights[f"{prefix}.{key}"] = array

    dst.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    save_file(weights, str(dst))
    return dst


def main() -> None:
    path = convert_checkpoint()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
