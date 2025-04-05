# Copyright (c) AlphaBetter. All rights reserved.
from pathlib import Path
from typing import Union

import torch
from torchvision import utils as vutils

from conditional_gan.utils.checkpoint import load_checkpoint
from conditional_gan.utils.envs import select_device


class Inferencer:
    def __init__(
            self,
            model_weights_path: Union[Path, str],
            device: str,
    ) -> None:
        """初始化推理器

        Args:
            model_weights_path (Union[Path, str]): 模型权重路径
            device (str): 设备
        """
        self.model_weights_path = Path(model_weights_path)
        self.device = select_device(device)

        self.model = load_checkpoint(model_weights_path, device=self.device)

    def __call__(self, save_path: Union[Path, str], conditional_index: int = 0, latent_dim: int = 100) -> None:
        """执行推理

        Args:
            save_path (Union[Path, str]): 保存路径
            conditional_index (int, optional): 条件索引. Defaults to 0.
            latent_dim (int, optional): 潜在维度. Defaults to 100.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        noise = torch.randn([1, latent_dim], device=self.device)
        conditional = torch.randint(conditional_index, conditional_index + 1, (1,), device=self.device)

        with torch.no_grad():
            generated_images = self.model(noise, conditional)

        vutils.save_image(generated_images, save_path, normalize=True)
