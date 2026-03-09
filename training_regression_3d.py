import matplotlib.pyplot as plt
import torch
import numpy as np
from torchinfo import summary
from training_classes import MLP, TriadicMLP, DeepTriadicMLP, DeepTriadicMLP_Funnel
from regression_functions import generate_dataloaders, train_and_evaluate
from utilities import setup_torch_device, choose_function
from list_of_functions_file import *

#%%
dim = 2  # Input dimension
function_list = list_of_2D_functions if dim==1 else list_of_3D_functions
device = setup_torch_device(use_gpu=True, gpu_index=1)
dir_to_save = "./Results/Regression"
func_expr, func_name = choose_function(function_list, func_name='double-well')  # Change function name as needed
func_kwargs = {'min_interval': -1.5, 'max_interval': 1.5, 'noise_std': 0.0, 'batch_size': 256, 'num_samples': 10000}
train, test = generate_dataloaders(func_expr, dim=dim, **func_kwargs)
#%%
h_dim = 20 # 10 gaussian
n_layers = 7 # 3 means 1 hidden layer #10 gaussian
if n_layers == 3:
    triadic_model = TriadicMLP(hidden_dim=h_dim, input_dim=dim, output_dim=1).to(device)
elif n_layers >= 3:
    triadic_model = DeepTriadicMLP(hidden_dim=h_dim, input_dim=dim, output_dim=1, num_layers=n_layers).to(device)
else:
    raise ValueError("n_layers must be at least 3")
summary(triadic_model, input_size=(1, dim))
#%%
training_kwargs = {"epochs": 1000, "learning_rate":1*1e-4,
                   "model_name": f"TriadicMLP_h{h_dim}_{func_name}", "save_dir": dir_to_save, "device": device}
# triadic_model = train_and_evaluate(triadic_model, train, test, **training_kwargs)
#%%
## Evaluating the model
model = triadic_model.to(device)
state_dict = torch.load(f"{dir_to_save}/TriadicMLP_h{h_dim}_{func_name}.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
x_all = []
z_true_all = []
z_pred_all = []
with torch.no_grad():
    for x_batch, z_batch in test:
        x_batch = x_batch.to(device)
        z_pred_batch = triadic_model(x_batch).squeeze(-1)
        x_all.append(x_batch.cpu())
        z_true_all.append(z_batch.cpu())
        z_pred_all.append(z_pred_batch.cpu())
x_test = torch.cat(x_all, dim=0).cpu().numpy()
z_test = torch.cat(z_true_all, dim=0).cpu().numpy()
z_pred = torch.cat(z_pred_all, dim=0).cpu().numpy()
x = x_test[:, 0]
y = x_test[:, 1]

#%%
## Plotting the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# s1 = ax.plot_trisurf(x, y, z_test.squeeze(), color = 'b', alpha=0.85, linewidth=0.2, label='True')
s2 = ax.plot_trisurf(x, y, z_pred.squeeze(), color = 'orange', alpha=0.45, linewidth=0.2, label='Predicted')
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$y$", fontsize=16)
ax.set_zlabel(r"$f(x,y)$", fontsize=16)
# ax.set_title(f"{func_name}, h_dim = {h_dim}, n_layers= {n_layers}", fontsize=15)
ax.grid()
ax.legend()
ax.set_xlim(-1., 1.)
ax.set_ylim(-1., 1.)
ax.set_zlim(-1.2, 1.5)
ax.view_init(elev=45, azim=75)
fig.tight_layout()
plt.show()
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s1 = ax.plot_trisurf(x, y, z_test.squeeze() - z_pred.squeeze(), cmap='viridis', alpha=0.85, linewidth=0.2, label='error')
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$y$", fontsize=16)
ax.set_zlabel(r"$f(x,y)$", fontsize=16)
ax.set_title(f"Error {func_name}, h_dim = {h_dim}, n_layers= {n_layers}", fontsize=15)
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
#%%
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

tri = mtri.Triangulation(x, y)

z_test_arr = z_test.squeeze()
z_pred_arr = z_pred.squeeze()

# range comune (se vuoi forzarlo al range noto della funzione: vmin=-1, vmax=14)
vmin = float(min(z_test_arr.min(), z_pred_arr.min()))
vmax = float(max(z_test_arr.max(), z_pred_arr.max()))
# vmin, vmax = -1.0, 14.0

# --- NORMALIZZAZIONE "asinh": buche ben visibili senza appiattire l'esterno ---
scale = 1.5  # più piccolo = più enfasi sulle buche
try:
    norm = mcolors.FuncNorm(
        (lambda z: np.arcsinh(z / scale), lambda u: scale * np.sinh(u)),
        vmin=vmin, vmax=vmax
    )
    z_test_plot, z_pred_plot = z_test_arr, z_pred_arr
except AttributeError:
    # fallback per matplotlib vecchi
    z_test_plot = np.arcsinh(z_test_arr / scale)
    z_pred_plot = np.arcsinh(z_pred_arr / scale)
    norm = Normalize(
        vmin=float(min(z_test_plot.min(), z_pred_plot.min())),
        vmax=float(max(z_test_plot.max(), z_pred_plot.max()))
    )

# --- Calcolo livelli di contour comuni (nell'unità dei dati mostrati) ---
cmin = float(min(np.nanmin(z_test_plot), np.nanmin(z_pred_plot)))
cmax = float(max(np.nanmax(z_test_plot), np.nanmax(z_pred_plot)))
# uso spacing non-lineare per concentrare i livelli vicino al minimo ("buche")
power = 3.0  # >1 => più densità vicino a cmin; aumentare per enfatizzare ancora di più
n_contours = 20
u = np.linspace(0.0, 1.0, n_contours)
contour_levels = cmin + (u ** power) * (cmax - cmin)
# split per due set di linee: dense (nelle buche) e sparse (fuori)
dense_frac = 0.65
n_dense = max(3, int(n_contours * dense_frac))
low_levels = contour_levels[:n_dense]
high_levels = contour_levels[n_dense:]
# prendi solo una linea su 'subsample' (es. 2 => una ogni 2)
subsample = 2
if len(low_levels) > 1:
    low_levels = low_levels[::subsample]
if len(high_levels) > 1:
    high_levels = high_levels[::subsample]

# figura: pannelli uguali + spazio riservato alla colorbar (a destra)
fig, axs = plt.subplots(
    1, 2,
    figsize=(13.5, 6.2),
    gridspec_kw={"wspace": 0.28}  # più spazio tra i pannelli
)

# spazio globale regolato meglio
fig.subplots_adjust(left=0.08, right=0.86, bottom=0.12, top=0.92)

# pannelli quadrati
for ax in axs:
    ax.set_aspect("equal", adjustable="box")

# -------- PRIMO PANNELLO --------
tcf0 = axs[0].tricontourf(
    tri, z_test_plot,
    levels=100, cmap="viridis", norm=norm
)
# sovrapponi linee di equipotenziale chiare per evidenziare le "buche"
# linee dense e leggermente più spesse vicino al minimo
if len(low_levels) > 0:
    cs0_low = axs[0].tricontour(tri, z_test_plot, levels=low_levels, colors='white', linewidths=1.0, alpha=0.95)
if len(high_levels) > 0:
    cs0_high = axs[0].tricontour(tri, z_test_plot, levels=high_levels, colors='white', linewidths=0.45, alpha=0.6)
# opzionale: etichette sui contour (commentare se ingombrano)
#axs[0].clabel(cs0_low, fmt='%1.2f', fontsize=8, colors='white')

axs[0].set_xlabel(r"$x$", fontsize=25)
axs[0].set_ylabel(r"$y$", fontsize=25, labelpad=8)
axs[0].tick_params(labelsize=16)
axs[0].set_title("Test", fontsize=25, pad=12)

# -------- SECONDO PANNELLO --------
tcf1 = axs[1].tricontourf(
    tri, z_pred_plot,
    levels=100, cmap="viridis", norm=norm
)
# sovrapponi le stesse linee di equipotenziale (stesso insieme di livelli)
if len(low_levels) > 0:
    cs1_low = axs[1].tricontour(tri, z_pred_plot, levels=low_levels, colors='white', linewidths=1.0, alpha=0.95)
if len(high_levels) > 0:
    cs1_high = axs[1].tricontour(tri, z_pred_plot, levels=high_levels, colors='white', linewidths=0.45, alpha=0.6)
#axs[1].clabel(cs1_low, fmt='%1.2f', fontsize=8, colors='white')

axs[1].set_xlabel(r"$x$", fontsize=25)
axs[1].set_ylabel(r"$y$", fontsize=25, labelpad=4)  # meno padding
axs[1].tick_params(labelsize=16)
axs[1].set_title("Recon.", fontsize=25, pad=12)

# -------- COLORBAR --------
pos1 = axs[1].get_position()

cbar_pad = 0.012
cbar_w   = 0.022

cax = fig.add_axes([pos1.x1 + cbar_pad, pos1.y0, cbar_w, pos1.height])
cbar = fig.colorbar(tcf1, cax=cax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r"$x^4 + y^4 + 2x^2y^2 - 2x^2$", fontsize=20)
# cbar.set_label(r"$\\frac{1}{\\sqrt{2\\pi}\\, \\sigma} \\exp\\!\\left(-\\frac{1}{2}\\left(\\frac{mu}{\\sigma}\\right)^2\\right)$", fontsize=18)

plt.show()


#%%
z_err = np.abs(z_test.squeeze() - z_pred.squeeze())
tri = mtri.Triangulation(x, y)
fig, ax = plt.subplots()
tcf = ax.tricontourf(tri, z_err, levels=100, cmap='viridis')
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$y$", fontsize=16)
ax.set_title(f"Error {func_name}, h_dim = {h_dim}, n_layers= {n_layers}", fontsize=15)
fig.colorbar(tcf, ax=ax)
fig.tight_layout()
plt.show()

#%%
fig.savefig(f'Plots/{func_name}_hdim={h_dim}_nlayers={n_layers}.png')

# #%%
# plt.plot(x_test, np.abs(y_test-y_pred), 'x-', markersize=5, label="True Values")
# plt.yscale('log')
# plt.grid()
# plt.show()
#%%
## Compute the MSE on the test set
mse = np.mean((z_test.squeeze() - z_pred.squeeze())**2)
print(f"Test MSE: {mse:.6f}")
