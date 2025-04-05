# Copyright (c) AlphaBetter. All rights reserved.
from typing import List

import torch
from torch import nn

from conditional_gan.utils.torch_utils import initialize_weights

__all__ = [
    "VanillaNet",
]


class VanillaNet(nn.Module):
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
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, int(channels * image_size * image_size)),
            nn.Tanh()
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
        conditional_inputs = torch.cat([x, self.label_embedding(labels)], -1)
        x = self.backbone(conditional_inputs)
        return x.reshape(x.size(0), self.channels, self.image_size, self.image_size)
