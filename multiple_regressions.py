import matplotlib.pyplot as plt
import torch
import numpy as np
from training_classes import MLP, TriadicMLP, DeepTriadicMLP
from regression_functions import generate_dataloaders, train_and_evaluate
from utilities import setup_torch_device, choose_function
from list_of_functions_file import *

#%%
dim = 1  # Input dimension
function_list = list_of_2D_functions if dim==1 else list_of_3D_functions
device = setup_torch_device(use_gpu=True, gpu_index=1)
dir_to_save = "./Results/Regression"
func_expr, func_name = choose_function(function_list, func_name='sinusoidal')
func_kwargs = {'min_interval': -1.5, 'max_interval': 1.5,}
train, test = generate_dataloaders(func_expr, dim=dim, **func_kwargs)

h_dim = 2
n_layers_list = [3, 4, 5, 6]
figs = []
for n_layers in n_layers_list:
    if n_layers == 3:
        triadic_model = TriadicMLP(hidden_dim=h_dim, input_dim=1, output_dim=1).to(device)
    elif n_layers >= 3:
        triadic_model = DeepTriadicMLP(hidden_dim=h_dim, input_dim=dim, output_dim=1, num_layers=n_layers).to(device)
    else:
        raise ValueError("n_layers must be at least 3")
    #%%
    training_kwargs = {"epochs": 50, "learning_rate":1e-3,
                       "model_name": f"TriadicMLP_h{h_dim}_{func_name}", "save_dir": dir_to_save, "device": device}
    triadic_model = train_and_evaluate(triadic_model, train, test, **training_kwargs)
    #%%
    ## Evaluating the model
    model = triadic_model.to(device)
    state_dict = torch.load(f"{dir_to_save}/TriadicMLP_h{h_dim}_{func_name}.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    x_all = []
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x_batch, y_batch in test:
            x_batch = x_batch.to(device)
            y_pred_batch = triadic_model(x_batch)
            x_all.append(x_batch.cpu())
            y_true_all.append(y_batch.cpu())
            y_pred_all.append(y_pred_batch.cpu())
    x_test = torch.cat(x_all, dim=0).cpu().numpy()
    y_test = torch.cat(y_true_all, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred_all, dim=0).cpu().numpy()

    #%%
    ## Plotting the results
    ## sort for better visualization
    if dim==1:
        sorted_indices = x_test[:, 0].argsort()
        x_test = x_test[sorted_indices]
        y_test = y_test[sorted_indices]
        y_pred = y_pred[sorted_indices]

    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, 'o-', markersize=5, label="True Values")
    ax.plot(x_test, y_pred, 'x-', markersize=3, label="Predictions", alpha=0.7)
    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$f(x)$", fontsize=16)
    ax.set_title(f"{func_name}, h_dim = {h_dim}, n_layers= {n_layers}", fontsize=15)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    figs.append(fig)
    plt.show()
    #%%
    # fig.savefig(f'Plots/{func_name}_hdim={h_dim}_nlayers={n_layers}.png')

    # #%%
    # plt.plot(x_test, np.abs(y_test-y_pred), 'x-', markersize=5, label="True Values")
    # plt.yscale('log')
    # plt.grid()
    # plt.show()

