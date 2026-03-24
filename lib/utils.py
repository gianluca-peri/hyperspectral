import os
import json
import torch

def get_run_dir(base_dir):
    """
    This script is made to handle multiple training runs.
    If not present it creates the base directory (e.g. "mnist"), and then
    creates the subdirectory "run_0", in which to save the results.
    If the base directory already exists, it looks for the next available "run_i" subdirectory and creates it.
    Returns the path to the new run directory.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    i = 0
    while True:
        run_dir = os.path.join(base_dir, f"run_{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        i += 1

def save_history(run_dir, model_name, history):
    """
    Save the training history to a JSON file in the specified run directory.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    with open(os.path.join(run_dir, model_name, "history.json"), "w") as f:
        json.dump(history, f)

def save_final_test_evaluation(run_dir, model_name, test_evaluation):
    """
    Save the final test evaluation results to a JSON file in the specified run directory.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    with open(os.path.join(run_dir, model_name, "final_test_evaluation.json"), "w") as f:
        json.dump(test_evaluation, f)

def save_model(run_dir, model_name, model):
    """
    Save the model's state dictionary to a file in the specified run directory.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, model_name, "model_final.pth"))

def load_all_histories_of_a_model(base_dir, model_name):
    """
    Load all history.json files for the specified model across all runs in the base directory.
    Returns a list of history dictionaries.
    """

    histories = []
    for entry in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, entry)
        # Ignore files such as summary plots
        if not os.path.isdir(run_dir):
            continue
        history_path = os.path.join(run_dir, model_name, "history.json")
        with open(history_path, "r") as f:
            history = json.load(f)
            histories.append(history)
    return histories
    