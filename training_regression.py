import matplotlib.pyplot as plt
import torch
from training_classes import MLP, TriadicMLP
from regression_functions import generate_dataloaders, train_and_evaluate
from utlilities import setup_torch_device, save_plot, save_results_regression
from list_of_functions import *

#%%
device = setup_torch_device(use_gpu=True, gpu_index=0)
h_dim = 1
dir_to_save = "./Results/Regression"
func_expr, func_name = list_of_2D_functions[1]
train, test = generate_dataloaders(func_expr)

triadic_model = TriadicMLP(hidden_dim=h_dim, input_dim=1).to(device)
triadic_save_dir = save_results_regression(dir_to_save, func_name, h_dim, "triadic_mlp")
kwargs = {"model_name": f"TriadicMLP_h{h_dim}_{func_name}", "save_dir": triadic_save_dir, "device": device}
triadic_model = train_and_evaluate(triadic_model, train, test, **kwargs)
#%%
## Evaluating the model
triadic_model.eval()
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
n_test = len(x_test)
subset = slice(0, n_test//2)  # Plot only a subset for clarity
plt.plot(x_test[subset], y_test[subset], 'x', markersize=5, label="True Values")
plt.plot(x_test[subset], y_pred[subset], 'o', markersize=3, label="Predictions", alpha=0.7)
plt.grid()
plt.legend()
plt.show()

