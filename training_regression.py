import matplotlib.pyplot as plt
import torch
from training_classes import MLP, TriadicMLP
from regression_functions import generate_dataloaders, train_and_evaluate
from utlilities import setup_torch_device, choose_function
from list_of_functions_file import *

#%%
dim = 1  # Input dimension
function_list = list_of_2D_functions if dim==1 else list_of_3D_functions
device = setup_torch_device(use_gpu=True, gpu_index=0)
h_dim = 2
dir_to_save = "./Results/Regression"
func_expr, func_name = choose_function(function_list, func_name='quadratic')
func_kwargs = {'min_interval': -3.0, 'max_interval': 3.0,}
train, test = generate_dataloaders(func_expr, dim=dim, **func_kwargs)

triadic_model = TriadicMLP(hidden_dim=h_dim, input_dim=1).to(device)
training_kwargs = {"epochs": 100, "learning_rate":1e-3,
                   "model_name": f"TriadicMLP_h{h_dim}_{func_name}", "save_dir": dir_to_save, "device": device}
triadic_model = train_and_evaluate(triadic_model, train, test, **training_kwargs)
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
## sort for better visualization
if dim==1:
    sorted_indices = x_test[:, 0].argsort()
    x_test = x_test[sorted_indices]
    y_test = y_test[sorted_indices]
    y_pred = y_pred[sorted_indices]
plt.plot(x_test, y_test, 'o-', markersize=5, label="True Values")
plt.plot(x_test, y_pred, 'x-', markersize=3, label="Predictions", alpha=0.7)
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$f(x)$", fontsize=16)
plt.title(f"{func_name}, h_dim = {h_dim}", fontsize=18)
plt.grid()
plt.legend()
plt.show()

