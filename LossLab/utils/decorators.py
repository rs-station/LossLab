"""Decorators for common patterns in coordinate refinement."""

import functools
import time
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger


def gpu_memory_tracked(func: Callable) -> Callable:
    """Track GPU memory usage before and after function call.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs memory usage
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**3

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            logger.debug(
                f"{func.__name__}: Memory before={mem_before:.2f}GB, "
                f"after={mem_after:.2f}GB, peak={mem_peak:.2f}GB"
            )

        return result

    return wrapper


def timed(func: Callable) -> Callable:
    """Time function execution.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__}: {elapsed:.3f}s")
        return result

    return wrapper


def validate_shapes(*expected_shapes):
    """Validate tensor shapes match expected dimensions.

    Args:
        expected_shapes: Tuples of (arg_index, expected_shape) or
                        (arg_name, expected_shape) for kwargs

    Example:
        @validate_shapes((0, (None, 3)), (1, (None, 3)))
        def align_coords(coords1, coords2):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for spec in expected_shapes:
                if isinstance(spec[0], int):
                    # Positional argument
                    tensor = args[spec[0]]
                    expected = spec[1]
                else:
                    # Keyword argument
                    tensor = kwargs.get(spec[0])
                    expected = spec[1]

                if tensor is not None and isinstance(tensor, torch.Tensor):
                    actual_shape = tensor.shape
                    for i, exp_dim in enumerate(expected):
                        if exp_dim is not None and actual_shape[i] != exp_dim:
                            raise ValueError(
                                f"Shape mismatch in {func.__name__}: "
                                f"expected {expected}, got {actual_shape}"
                            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


