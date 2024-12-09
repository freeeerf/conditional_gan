# Copyright (c) AlphaBetter. All rights reserved.
from typing import List

import torch
from torch import nn

from conditional_gan.utils.torch_utils import initialize_weights

__all__ = [
    "ConvNet",
]


class ConvNet(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10, latent_dim: int = 100) -> None:
        """Implementation of the Vanilla GAN model.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28 (e.g., for MNIST).
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
            latent_dim (int, optional): Dimension of the latent noise vector. Default is 100.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        # Generate a random matrix of size (num_classes, num_classes)
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, channels, (4, 4), (2, 2), (1, 1), bias=True),

            # state size. channels x 64 x 64
            nn.Tanh()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: List = None) -> torch.Tensor:
        """Forward pass of the Vanilla GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (List, optional): List of labels for conditional generation. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        conditional_inputs = torch.cat([x, self.label_embedding(labels)], -1)
        x = self.backbone(conditional_inputs)
        return x.reshape(x.size(0), self.channels, self.image_size, self.image_size)
