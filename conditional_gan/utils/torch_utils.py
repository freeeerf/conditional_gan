# Copyright (c) AlphaBetter. All rights reserved.
from typing import Iterator

from torch import nn

__all__ = [
    "initialize_weights",
]


def initialize_weights(modules: Iterator[nn.Module]) -> None:
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
