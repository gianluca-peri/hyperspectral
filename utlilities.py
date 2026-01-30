import os

import matplotlib.pyplot as plt
import torch
import json

def choose_function(function_list, func_name):
    """Selects and returns a function from a list based on its name."""
    for func_expr, name in function_list:
        if name == func_name:
            return func_expr, name
    raise ValueError(f"Function '{func_name}' not found in the provided function list.")

def setup_torch_device(use_gpu=True, gpu_index=0, verbose=False):
    """
    Sets up the PyTorch device (GPU or CPU).
    Args:
        use_gpu (bool): If True, try to use GPU. If False, force CPU.
        gpu_index (int): CUDA device index to use if GPU is enabled.
        verbose (bool): Whether to print device info.
    Returns:
        torch.device
    """
    if use_gpu and torch.cuda.is_available():
        if gpu_index < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_index}")
            torch.cuda.set_device(device)
            if verbose:
                print(f"Using GPU: cuda:{gpu_index} ({torch.cuda.get_device_name(gpu_index)})")
        else:
            device = torch.device("cpu")
            if verbose:
                print(f"GPU index {gpu_index} not available. Using CPU.")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU.")
    return device


