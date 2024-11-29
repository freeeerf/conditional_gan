# Copyright (c) AlphaBetter. All rights reserved.
from logging import lastResort

import torch
from torch import nn
from typing import Dict, Union
from pathlib import Path
from .events import LOGGER
import shutil

__all__ = [
    "load_state_dict", "save_checkpoint", "strip_optimizer",
]


def load_state_dict(weights_path: str, model: nn.Module, map_location: torch.device) -> nn.Module:
    """Load weights from checkpoint file, only assign weights those layers" name and shape are match.

    Args:
        weights_path (str): path to weights file.
        model (nn.Module): model to load weights.
        map_location (torch.device): device to load weights.

    Returns:
        nn.Module: model with weights loaded.
    """
    checkpoint = torch.load(weights_path, map_location=map_location)
    state_dict = checkpoint["model"].float().state_dict()

    # filter out unnecessary keys
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}

    model.load_state_dict(state_dict, strict=False)
    del checkpoint, state_dict, model_state_dict
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
