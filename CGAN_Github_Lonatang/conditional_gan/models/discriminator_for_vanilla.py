# Copyright (c) AlphaBetter. All rights reserved.
from typing import List

import torch
from torch import nn

from conditional_gan.utils.torch_utils import initialize_weights

__all__ = [
    "DiscriminatorForVanilla",
]


class DiscriminatorForVanilla(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, dropout: float = 0.5, num_classes: int = 10) -> None:
        """Discriminator model architecture.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28 (e.g., for MNIST).
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            dropout (float, optional): Dropout rate. Default is 0.5.
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
        """
        super().__init__()
        # Embedding layer for the labels.
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.backbone = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: List) -> torch.Tensor:
        """Forward pass of the Vanilla GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (List): List of labels for conditional generation.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        x = torch.flatten(x, 1)
        label_embedding = self.label_embedding(labels)
        x = torch.cat([x, label_embedding], dim=-1)
        return self.backbone(x)
