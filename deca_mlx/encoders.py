"""ResNet-50 encoders matching official DECA ``decalib/models/{resnet,encoders}.py``."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .layers import apply_seq


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: list | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        if downsample is not None:
            self.downsample = downsample
        self.stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if "downsample" in self:
            residual = apply_seq(self.downsample, x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, layers: tuple[int, int, int, int] = (3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> list:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = [
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * Bottleneck.expansion),
            ]
        modules = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            modules.append(Bottleneck(self.inplanes, planes))
        return modules

    def __call__(self, x: mx.array) -> mx.array:
        # Official DECA uses NCHW; MLX conv/pool expect NHWC.
        if x.ndim != 4:
            raise ValueError(f"expected NCHW image tensor, got shape {x.shape}")
        if x.shape[1] == 3:
            x = x.transpose(0, 2, 3, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = apply_seq(self.layer1, x)
        x = apply_seq(self.layer2, x)
        x = apply_seq(self.layer3, x)
        x = apply_seq(self.layer4, x)
        x = self.avgpool(x)
        return x.reshape((x.shape[0], -1))


class ResnetEncoder(nn.Module):
    def __init__(self, outsize: int):
        super().__init__()
        self.encoder = ResNet()
        self.layers = [
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize),
        ]

    def __call__(self, images: mx.array) -> mx.array:
        return apply_seq(self.layers, self.encoder(images))
