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
        """Implementation of the Conditional GAN model using Convolutional Neural Networks.

        Args:
            image_size (int, optional): Size of the generated square image (height = width). Default is 28.
            channels (int, optional): Number of channels in the generated image. Default is 1 (grayscale image).
            num_classes (int, optional): Number of classes for conditional generation. Default is 10.
            latent_dim (int, optional): Dimension of the latent noise vector. Default is 100.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Embedding layer for the labels.
        self.label_embedding = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 7 * 7 * self.latent_dim),
            nn.LeakyReLU(0.2, True),
        )

        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, self.channels, 4),

            nn.Tanh()
        )

        # Initializing all neural network weights.
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor, labels: List = None) -> torch.Tensor:
        """Forward pass of the Deep Conditional GAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent).
            labels (List, optional): List of labels for conditional generation. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, image_size, image_size).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input tensor 'x' to have 2 dimensions, but got {x.dim()}.")

        if labels is None:
            raise ValueError("Labels must be provided for conditional generation.")

        x = torch.cat([x, labels], 1)
        x = self.label_embedding(x).reshape(-1, self.latent_dim, 7, 7)
        return self.backbone(x)
