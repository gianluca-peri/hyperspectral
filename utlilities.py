import os
import torch
import json

def save_plot(plot, save_path):
    """Saves a matplotlib plot to the specified path."""
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot.savefig(save_path)

def save_json(data, save_path):
    """Saves data as a JSON file to the specified path."""
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_dir_function(dir_path):
    ...

def save_results_regression(dir_to_save, func_name, h_dim, results):
    """Creates a directory to save results if it doesn't exist.
    Then, saves the training results there."""
    path = os.path.join(dir_to_save, func_name, f"hidden_{h_dim}", "triadic_mlp")
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    ...

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


