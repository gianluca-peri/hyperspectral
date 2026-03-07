import numpy as np
import matplotlib.pyplot as plt
import torch
from training_classes import MLP, TriadicMLP, DeepTriadicMLP, DeepTriadicMLP_Funnel
from regression_functions import generate_dataloaders, train_and_evaluate, train_and_evaluate_weighted
from utilities import setup_torch_device, choose_function

#%%
## extract the data from "Dati_UJT.txt"
t, x, y = np.loadtxt("Dati_UJT.txt", delimiter="\t", unpack=True)
## rescale the data
x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))
# y = y/2
t = t * 2000
half_t = t[len(t)//2]
t = t - half_t
# t = t[150:]
# x = x[150:]
# y = y[150:]

#%%
## plot the data
plt.plot(t, x)
plt.plot(t, y)
plt.grid(True)
plt.show()

#%%
## create a dataloader for the data, using t as input and x and y as output
device = setup_torch_device(use_gpu=True, gpu_index=1)
t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

## Stack outputs: target is [x, y]
xy_tensor = torch.cat([x_tensor, y_tensor], dim=1)  # shape: [N, 2]

## Split the data into train and test sets (80% train, 20% test)
num_samples = t_tensor.shape[0]
indices = torch.randperm(num_samples)

split_idx = int(0.9 * num_samples)
train_idx = indices[:split_idx]
test_idx  = indices[split_idx:]

t_train  = t_tensor[train_idx]
t_test   = t_tensor[test_idx]
xy_train = xy_tensor[train_idx]
xy_test  = xy_tensor[test_idx]

train_dataset = torch.utils.data.TensorDataset(t_train, xy_train)
test_dataset = torch.utils.data.TensorDataset(t_test, xy_test)

train = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

#%%
dir_to_save = "./Results/Regression"
func_name="UJT"
dim = 1
h_dim = 20
n_layers = 20 # 3 means 1 hidden layer
output_dim = 2

if n_layers == 3:
    triadic_model = TriadicMLP(hidden_dim=h_dim, input_dim=dim, output_dim=output_dim).to(device)
elif n_layers >= 3:
    triadic_model = DeepTriadicMLP(hidden_dim=h_dim, input_dim=dim, output_dim=output_dim, num_layers=n_layers).to(device)
else:
    raise ValueError("n_layers must be at least 3")

# summary(triadic_model, input_size=(1, dim))

#%%
training_kwargs = {"epochs": 300, "learning_rate":1*1e-5,
                   "model_name": f"TriadicMLP_h{h_dim}_{func_name}", "save_dir": dir_to_save, "device": device}
training_kwargs.update({
    "wx": 8.0,
    "wy": 1.0,
})
triadic_model = train_and_evaluate_weighted(triadic_model, train, test, **training_kwargs)

#%%
## Evaluating the model
model = triadic_model.to(device)
state_dict = torch.load(f"{dir_to_save}/TriadicMLP_h{h_dim}_{func_name}.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

t_all = []
xy_true_all = []
xy_pred_all = []

with torch.no_grad():
    for t_batch, xy_batch in test:
        t_batch = t_batch.to(device)
        xy_pred_batch = model(t_batch)

        t_all.append(t_batch.cpu())
        xy_true_all.append(xy_batch.cpu())
        xy_pred_all.append(xy_pred_batch.cpu())

t_test = torch.cat(t_all, dim=0).cpu().numpy()              # shape: [N, 1]
xy_test = torch.cat(xy_true_all, dim=0).cpu().numpy()       # shape: [N, 2]
xy_pred = torch.cat(xy_pred_all, dim=0).cpu().numpy()       # shape: [N, 2]
#%%
t_test = t_test + half_t
#%%
## Plotting the results
## sort for better visualization
if dim == 1:
    sorted_indices = t_test[:, 0].argsort()
    t_test = t_test[sorted_indices]
    xy_test = xy_test[sorted_indices]
    xy_pred = xy_pred[sorted_indices]

# Figure for y(t)

fig1, ax1 = plt.subplots()
ax1.plot(t_test, xy_test[:, 0], 'o-', markersize=4, alpha=0.5, label="Test y(t)")
ax1.plot(t_test, xy_pred[:, 0], 'x-', markersize=3, alpha=0.7, label="Pred. y(t)")
# ax1.plot(t_test, xy_test[:, 1], 'o-', markersize=4, alpha=0.5, label="Test x(t)")
# ax1.plot(t_test, xy_pred[:, 1], 'x-', markersize=3, alpha=0.7, label="Pred. x(t)")
ax1.set_xlabel(r"$t$", fontsize=16)
ax1.set_ylabel(r"$y(t)$", fontsize=16)
# ax1.set_title(f"{func_name}, h_dim = {h_dim}, n_layers= {n_layers}", fontsize=15)
ax1.grid()
ax1.legend(fontsize=14, loc="upper right")
fig1.tight_layout()
fig1.savefig(f"Plots/TriadicMLP_h{h_dim}_{func_name}_y.png")
plt.show()
#%%
# Figure for x(t)
fig2, ax2 = plt.subplots()
ax2.plot(t_test, xy_test[:, 1], 'o-', markersize=4, alpha=0.5, label="Test x(t)")
ax2.plot(t_test, xy_pred[:, 1], 'x-', markersize=3, alpha=0.7, label="Pred. x(t)")
ax2.set_xlabel(r"$t$", fontsize=16)
ax2.set_ylabel(r"$x(t)$", fontsize=16)
ax2.grid()
ax2.legend(fontsize=14)
fig2.tight_layout()
fig2.savefig(f"Plots/TriadicMLP_h{h_dim}_{func_name}_x.png")
plt.show()
#%%
## Compute the MAE for both outputs on the test set
mae_y = np.mean(np.abs(xy_test[:, 0] - xy_pred[:, 0]))
mae_x = np.mean(np.abs(xy_test[:, 1] - xy_pred[:, 1]))
total_mae = (mae_x + mae_y) / 2
print(f"MAE: {total_mae}")

