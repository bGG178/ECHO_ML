# Copyright (c) AlphaBetter. All rights reserved.
import math
import torch
from torch import nn

from conditional_gan.utils.torch_utils import initialize_weights

__all__ = [
    "DiscriminatorForConv",
]


class DiscriminatorForConv(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        """Discriminator model architecture.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes

        # Embedding layer for the labels.
        self.label_embedding = nn.Sequential(
            nn.Linear(self.num_classes, int(self.channels * self.image_size * self.image_size)),
            nn.LeakyReLU(0.2, True),
        )

        self.backbone = nn.Sequential(
            nn.Conv2d(self.channels + 1, 64, 3),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, self.channels, 3),
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (torch.Tensor): List of labels for conditional generation.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor 'x' to have 4 dimensions, but got {x.dim()}.")

        if labels is None:
            raise ValueError("Labels must be provided for conditional generation.")

        label_embedding = self.label_embedding(labels).reshape(-1, self.channels, self.image_size, self.image_size)
        x = torch.cat([x, label_embedding], 1)
        x = self.backbone(x)
        return torch.flatten(x, 1)
