# Copyright (c) AlphaBetter. All rights reserved.
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn

from .events import LOGGER

__all__ = [
    "load_state_dict", "load_checkpoint", "save_checkpoint", "strip_optimizer",
]


def load_state_dict(weights_path: Union[Path, str], model: nn.Module, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Load weights from checkpoint file, only assign weights those layers name and shape are match.

    Args:
        weights_path (Union[Path, str]): path to weights file.
        model (nn.Module): model to load weights.
        device (torch.device, optional): device to load model. Defaults to torch.device("cpu").

    Returns:
        nn.Module: model with weights loaded.
    """
    # Define compilation status keywords
    compile_state = "_orig_mod"

    checkpoint = torch.load(str(weights_path), map_location=torch.device("cpu"))
    state_dict = checkpoint["model"].float().state_dict()
    new_state_dict = OrderedDict()

    # Check if the model has been compiled
    for k, v in state_dict.items():
        current_compile_state = k.split(".")[0]
        # load the model
        if current_compile_state != compile_state:
            name = compile_state + "." + k
        elif current_compile_state == compile_state:
            name = k[10:]
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # filter out unnecessary keys
    model_state_dict = model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}

    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    del checkpoint, state_dict, new_state_dict, model_state_dict
    return model


def load_checkpoint(weights_path: Union[Path, str], device: torch.device = torch.device("cpu")) -> nn.Module:
    """Load model from a checkpoint file.

    Args:
        weights_path (Union[Path, str]): Path to the weights file.
        device (torch.device, optional): Device to load the model. Defaults to torch.device("cpu").

    Returns:
        torch.nn.Module: The model with the weights loaded.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        LOGGER.error(f"No weights file found at `{weights_path}`")

    LOGGER.info(f"Loading checkpoint from `{weights_path}`")
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False)
    model = checkpoint["ema" if checkpoint.get("ema") else "model"].float()
    model = model.to(device)
    model = model.eval()
    return model


def save_checkpoint(
        checkpoint: Dict,
        save_dir: Union[Path, str],
        is_best: bool,
        current_model_name: Union[Path, str],
        best_model_name: Union[Path, str],
        last_model_name: Union[Path, str],
) -> None:
    """Save checkpoint to the disk.

    Args:
        checkpoint (Dict): The checkpoint to be saved.
        save_dir (Union[Path, str]): The directory where to save the checkpoint.
        is_best (bool): Whether this checkpoint is the best so far.
        current_model_name (Union[Path, str], optional): The name of the current model.
        best_model_name (Union[Path, str], optional): The name of the best model.
        last_model_name (Union[Path, str], optional): The name of the model.
    """
    save_dir = Path(save_dir)
    current_model_name = Path(current_model_name)
    best_model_name = Path(best_model_name)
    last_model_name = Path(last_model_name)

    save_dir.mkdir(parents=True, exist_ok=True)

    current_checkpoint_path = save_dir.joinpath(current_model_name)
    last_checkpoint_path = save_dir.joinpath(last_model_name)

    torch.save(checkpoint, current_checkpoint_path)
    torch.save(checkpoint, last_checkpoint_path)

    if is_best:
        best_checkpoint_path = Path(save_dir).joinpath(best_model_name)
        shutil.copyfile(str(current_checkpoint_path), str(best_checkpoint_path))


def strip_optimizer(file_path: Union[Path, str], updates: Dict = None) -> Dict:
    """Remove optimizer and other training-related information from a checkpoint file to reduce its size.

    Args:
        file_path (Union[Path, str]): Path to the checkpoint file to be loaded.
        updates (Dict): Additional information to update in the checkpoint, default is None.

    Returns:
        combined (Dict): A checkpoint dictionary with optimizer and training information removed.
    """
    try:
        # Load the checkpoint file
        checkpoint = torch.load(file_path, weights_only=False, map_location=torch.device("cpu"))
        assert isinstance(checkpoint, dict), "Checkpoint is not a Python dictionary"
        assert "model" in checkpoint, "'model' is missing from the checkpoint"
    except Exception as e:
        # Log the error and return an empty dictionary if loading fails
        LOGGER.error(f"Error loading {file_path}: {e}")
        return {}

    # Update model information
    if checkpoint.get("ema"):
        # Replace the original model with the Exponential Moving Average (EMA) model if available
        checkpoint["model"] = checkpoint["ema"]
    if hasattr(checkpoint["model"], "criterion"):
        # Remove the criterion attribute if present
        checkpoint["model"].criterion = None
    # Convert model parameters to half precision
    checkpoint["model"].half()
    for p in checkpoint["model"].parameters():
        # Disable gradient updates
        p.requires_grad = False

    # Remove unnecessary information from the checkpoint
    for k in "optimizer", "best_fitness", "ema", "updates":
        checkpoint[k] = None
    # Set "epoch" to -1 to indicate no training cycle information is retained
    checkpoint["epoch"] = -1

    # Combine the modified checkpoint with any updates provided
    combined = {**checkpoint, **(updates or {})}
    # Save the stripped checkpoint back to the file
    torch.save(combined, file_path)
    return combined
