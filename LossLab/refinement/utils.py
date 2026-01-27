"""Utility functions for refinement."""

from pathlib import Path

import torch
from loguru import logger


def number_to_letter(n: int) -> str:
    """Convert number to letter identifier (0->A, 1->B, etc.).

    Args:
        n: Number to convert

    Returns:
        Letter identifier
    """
    if n < 26:
        return chr(65 + n)
    else:
        # For n >= 26, use AA, AB, etc.
        first = chr(65 + (n // 26) - 1)
        second = chr(65 + (n % 26))
        return first + second


def initialize_bias(
    feature_dict: dict[str, torch.Tensor],
    device: str | torch.device,
    lr_additive: float = 1e-3,
    lr_multiplicative: float = 1e-3,
    weight_decay: float | None = None,
    starting_bias_path: Path | str | None = None,
    starting_weights_path: Path | str | None = None,
) -> tuple[dict, torch.optim.Optimizer, list[str]]:
    """Initialize MSA bias and optimizer.

    Args:
        feature_dict: Feature dictionary to add bias to
        device: PyTorch device
        lr_additive: Learning rate for additive bias
        lr_multiplicative: Learning rate for multiplicative weights
        weight_decay: Optional weight decay
        starting_bias_path: Path to starting bias tensor
        starting_weights_path: Path to starting weights tensor

    Returns:
        Tuple of (updated_features, optimizer, bias_names)
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Initialize bias tensors
    msa_shape = feature_dict["msa_feat"].shape

    if starting_bias_path and Path(starting_bias_path).exists():
        msa_bias = torch.load(starting_bias_path).to(device)
        logger.info(f"Loaded starting bias from {starting_bias_path}")
    else:
        msa_bias = torch.zeros(msa_shape, device=device, dtype=torch.float32)

    if starting_weights_path and Path(starting_weights_path).exists():
        feat_weights = torch.load(starting_weights_path).to(device)
        logger.info(f"Loaded starting weights from {starting_weights_path}")
    else:
        feat_weights = torch.ones(msa_shape, device=device, dtype=torch.float32)

    # Make learnable
    msa_bias.requires_grad = True
    feat_weights.requires_grad = True

    # Add to feature dict
    feature_dict["msa_feat_bias"] = msa_bias
    feature_dict["msa_feat_weights"] = feat_weights

    # Create optimizer
    param_groups = [
        {"params": [msa_bias], "lr": lr_additive},
        {"params": [feat_weights], "lr": lr_multiplicative},
    ]

    if weight_decay is not None:
        for pg in param_groups:
            pg["weight_decay"] = weight_decay

    optimizer = torch.optim.Adam(param_groups)

    bias_names = ["msa_feat_bias", "msa_feat_weights"]

    return feature_dict, optimizer, bias_names
