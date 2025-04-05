# Copyright (c) AlphaBetter. All rights reserved.
from typing import Iterator

import torch
from torch import nn

__all__ = [
    "get_gpu_info", "get_model_info", "initialize_weights",
]


def get_gpu_info(index: int) -> str:
    """Gets information about the specified index GPU

    Args:
        index: Device index

    Returns:
        str: A string that describes the name of the GPU and the total memory size (in MIBs).
    """
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def get_model_info(model: nn.Module, image_size: int = 64, device: torch.device = torch.device("cpu")) -> str:
    """Get model Params and GFlops.

    Args:
        model (nn.Module): The model whose information is to be retrieved.
        image_size (int, optional): The size of the image. Defaults to 64.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").

    Returns:
        str: The information about the model.
    """
    tensor = torch.rand([1, 3, image_size, image_size], device=device)

    params = sum([param.nelement() for param in model.parameters()])
    params /= 1e6
    return f"Params: {params:.2f} M"


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
