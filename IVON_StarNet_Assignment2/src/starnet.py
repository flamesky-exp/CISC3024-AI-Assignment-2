"""CIFAR-10 adaptation of StarNet.

Adapted from the Apache-2.0 official implementation:
https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py

The original ImageNet model uses a stride-2 stem and four stride-2 stage
transitions. For 32x32 inputs, this adaptation uses a stride-1 stem and a
stride-1 first stage. The remaining stage transitions use stride 2, producing
a 4x4 feature map before global average pooling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        with_bn: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=not with_bn,
            )
        ]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_probability: float = 0.0) -> None:
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.drop_probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x * random_tensor / keep_probability


class StarBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: int = 3,
        drop_path: float = 0.0,
        operation: str = "multiply",
    ) -> None:
        super().__init__()
        if operation not in {"multiply", "add"}:
            raise ValueError(f"Unsupported operation: {operation}")
        hidden_channels = mlp_ratio * channels
        self.operation = operation
        self.dwconv = ConvBN(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.f1 = ConvBN(channels, hidden_channels, with_bn=False)
        self.f2 = ConvBN(channels, hidden_channels, with_bn=False)
        self.activation = nn.ReLU6()
        self.projection = ConvBN(hidden_channels, channels, with_bn=True)
        self.dwconv2 = ConvBN(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            with_bn=False,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        branch_1 = self.activation(self.f1(x))
        branch_2 = self.f2(x)
        if self.operation == "multiply":
            x = branch_1 * branch_2
        else:
            x = branch_1 + branch_2
        x = self.dwconv2(self.projection(x))
        return residual + self.drop_path(x)


@dataclass(frozen=True)
class ArchitectureMetadata:
    name: str
    operation: str
    base_dim: int
    depths: tuple[int, ...]
    mlp_ratio: int
    stem_stride: int
    stage_strides: tuple[int, ...]
    num_classes: int


class CifarStarFamily(nn.Module):
    def __init__(
        self,
        name: str,
        operation: str,
        base_dim: int = 16,
        depths: tuple[int, ...] = (1, 1, 3, 1),
        mlp_ratio: int = 3,
        drop_path_rate: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.metadata = ArchitectureMetadata(
            name=name,
            operation=operation,
            base_dim=base_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            stem_stride=1,
            stage_strides=(1, 2, 2, 2),
            num_classes=num_classes,
        )
        in_channels = 32
        self.stem = nn.Sequential(
            ConvBN(3, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU6(),
        )

        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stages: list[nn.Module] = []
        block_index = 0
        for stage_index, depth in enumerate(depths):
            out_channels = base_dim * (2**stage_index)
            stride = self.metadata.stage_strides[stage_index]
            blocks: list[nn.Module] = [
                ConvBN(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
            ]
            in_channels = out_channels
            for _ in range(depth):
                blocks.append(
                    StarBlock(
                        in_channels,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_rates[block_index],
                        operation=operation,
                    )
                )
                block_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        shapes: list[tuple[int, ...]] = []
        x = self.stem(x)
        shapes.append(tuple(x.shape))
        for stage in self.stages:
            x = stage(x)
            shapes.append(tuple(x.shape))
        return x, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.forward_features(x)
        x = self.pool(self.norm(x)).flatten(1)
        return self.head(x)


def create_model(
    name: str, num_classes: int = 10, drop_path_rate: float = 0.0
) -> CifarStarFamily:
    operations = {
        "starnet_s050": "multiply",
        "sumnet_s050": "add",
    }
    if name not in operations:
        raise ValueError(f"Unknown model {name!r}. Choose from {sorted(operations)}")
    return CifarStarFamily(
        name=name,
        operation=operations[name],
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


def model_metadata(model: CifarStarFamily) -> dict[str, object]:
    return asdict(model.metadata)

