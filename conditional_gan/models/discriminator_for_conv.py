# Copyright (c) AlphaBetter. All rights reserved.
from typing import List

import torch
from torch import nn

from conditional_gan.utils.torch_utils import initialize_weights

__all__ = [
    "DiscriminatorForConv",
]


class DiscriminatorForConv(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, dropout: float = 0.5, num_classes: int = 10) -> None:
        """Discriminator model architecture.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28 (e.g., for MNIST).
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            dropout (float, optional): Dropout rate. Default is 0.5.
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        # Embedding layer for the labels.
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, int(channels * image_size * image_size)),
            nn.LeakyReLU(0.2, True),
        )

        self.backbone = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 3, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Conv2d(256, 512, 3, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Conv2d(512, channels, 3, bias=True),

            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Vanilla GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (torch.Tensor, optional): List of labels for conditional generation. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), self.channels, self.image_size, self.image_size)
        x = torch.concat([x, label_embedding], 1)
        x = self.backbone(x)
        return torch.flatten(x, 1)
