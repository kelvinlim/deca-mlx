#!/usr/bin/env python3
"""Dump official-checkpoint encoder outputs with CPU PyTorch.

This does not need CUDA or FLAME. It is the local numeric baseline used to
check the MLX ResNet encoders against ``deca_model.tar``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.config import default_config
from deca_mlx.preprocess import load_image


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class ResnetEncoder(nn.Module):
    def __init__(self, outsize):
        super().__init__()
        self.encoder = ResNet()
        self.layers = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, outsize))

    def forward(self, x):
        return self.layers(self.encoder(x))


def decompose(code: torch.Tensor, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    out = {}
    start = 0
    for key, size in sizes.items():
        value = code[:, start : start + size]
        if key == "light":
            value = value.reshape(value.shape[0], 9, 3)
        out[key] = value[0].detach().cpu().numpy()
        start += size
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=str(ROOT / "tests" / "faces" / "IMG_0392_inputs.jpg"))
    parser.add_argument("-o", "--output", default=str(ROOT / "tests" / "baselines" / "IMG_0392.npz"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = default_config()
    ckpt_path = Path(args.checkpoint or cfg.pretrained_modelpath)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    e_flame = ResnetEncoder(cfg.n_param)
    e_detail = ResnetEncoder(cfg.n_detail)
    e_flame.load_state_dict(checkpoint["E_flame"])
    e_detail.load_state_dict(checkpoint["E_detail"])
    e_flame.eval()
    e_detail.eval()

    sample = load_image(args.image, iscrop=False)
    images = torch.from_numpy(sample["image"][None, ...])
    with torch.no_grad():
        flame_code = e_flame(images)
        detail_code = e_detail(images)
    class Generator(nn.Module):
        def __init__(self, latent_dim=181, out_channels=1, out_scale=0.01):
            super().__init__()
            self.out_scale = out_scale
            self.init_size = 8
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise):
            out = self.l1(noise)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            return self.conv_blocks(out) * self.out_scale

    d_detail = Generator()
    d_detail.load_state_dict(checkpoint["D_detail"])
    d_detail.eval()
    payload = decompose(flame_code, cfg.param_sizes)
    payload["detail"] = detail_code[0].detach().cpu().numpy()
    cond = torch.cat(
        [torch.from_numpy(payload["pose"][3:])[None, ...], torch.from_numpy(payload["exp"])[None, ...], detail_code],
        dim=1,
    )
    with torch.no_grad():
        uv_z = d_detail(cond)
    payload["uv_z"] = uv_z[0].detach().cpu().numpy()
    if cfg.fixed_displacement_path.exists():
        fixed = np.load(cfg.fixed_displacement_path).astype(np.float32)
        payload["displacement_map"] = payload["uv_z"] + fixed[None, ...]
    payload["image"] = sample["image"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f"wrote {out}")
    for key, value in payload.items():
        print(f"  {key:12s} {np.asarray(value).shape}")


if __name__ == "__main__":
    main()
