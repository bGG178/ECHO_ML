# Copyright (c) AlphaBetter. All rights reserved.
import os
import random
from typing import Union

import cv2
import numpy as np
import torch
import torch.backends.mps

from .events import LOGGER
from .torch_utils import get_gpu_info

__all__ = [
    "RANK", "LOCAL_RANK", "NUM_THREADS",
    "set_seed_everything", "select_device",
]

# Get the values of environment variables "RANK" and "LOCAL_RANK". Defaults to -1 if not found.
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# Set and limit the number of threads to use, with a minimum of 1 and a maximum of 8, not exceeding (CPU cores - 1).
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

# Configure the maximum number of threads for NUMEXPR
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)

# Configure CUBLAS workspace size to optimize GPU computation performance
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set PyTorch print options, adjusting line width, precision, and profile configuration
torch.set_printoptions(linewidth=320, precision=4, profile="default")

# Set NumPy print options, adjusting line width and float formatting
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})

# Set the number of threads used by OpenCV
cv2.setNumThreads(NUM_THREADS)


def set_seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set the random seed for Python, Numpy, and PyTorch to ensure the reproducibility of experiments.

    Args:
        seed (int): Random seed.
        deterministic (bool, optional): Whether to set the CuDNN backend to deterministic. Defaults to False.
    """
    # Set the random seed for Python.
    random.seed(seed)

    # Set the random seed for Numpy.
    np.random.seed(seed)

    # Set the random seed for PyTorch.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure the reproducibility of experiments by setting the CuDNN backend to deterministic.
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def select_device(device: Union[str, torch.device] = "", batch_size: int = 0, verbose: bool = True) -> torch.device:
    """
    Select a device for computation.

    Args:
        device (str): The device index, e.g., "cpu", "0", "0,1", etc.
        batch_size (int, optional): The batch size. Defaults to 0.
        verbose (bool, optional): Whether to print device information. Defaults to True.

    Returns:
        torch.device: The selected device.
    """
    # If the input is already a torch.device object, return it directly
    if isinstance(device, torch.device):
        return device

    # Clean up the input string by removing certain characters
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")

    # Determine the device type
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:  # Use CUDA devices
        if device == "cuda":
            device = "0"
        # Remove empty devices if multiple devices are specified
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])
        # Get the currently visible CUDA devices
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # Set CUDA_VISIBLE_DEVICES to the selected devices
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # Verify CUDA availability and device count
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            raise ValueError(f"Incompatible device: {device}, CUDA_VISIBLE_DEVICES: {visible}")

    # Initialize the message string
    message = ""
    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        num_device = len(devices)
        # Ensure batch size is divisible by the number of GPUs if multiple GPUs are used
        if num_device > 1:
            if batch_size < 1:
                raise ValueError("Please specify a valid batch size, i.e. batch=16.")
            if batch_size >= 0 and batch_size % num_device != 0:
                raise ValueError(
                    f"'batch_size={batch_size}' must be a multiple of GPU count {num_device}. Try 'batch_size={batch_size // num_device * num_device}' or "
                    f"'batch_size={batch_size // num_device * num_device + num_device}', the nearest batch sizes evenly divisible by {num_device}."
                )
        space = " " * (len(message) + 1)
        # Append detailed information about each GPU to the message
        for index, device_index in enumerate(devices):
            message += f"{'' if index == 0 else space}CUDA:{device_index} ({get_gpu_info(index)})\n"
        device_type = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        message += f"Using MPS\n"
        device_type = "mps"
    else:
        message += f"Using CPU\n"
        device_type = "cpu"

    if device_type in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)

    if verbose:
        LOGGER.info(message.rstrip())

    return torch.device(device_type)
