import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------------------------------------------------
# GLOBAL STYLE SETTINGS (publication-ready)
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.figsize": (8, 6),
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.autolayout": True
})

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def normalise_field(u):
    u_min, u_max = np.min(u), np.max(u)
    if u_max - u_min == 0:
        return np.zeros_like(u)
    return (u - u_min) / (u_max - u_min)

def L2_error(u_true, u_pred):
    squared_error = (u_pred - u_true) ** 2
    mse = np.mean(squared_error)
    return np.sqrt(mse)

def save_results_file(data_dict, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, filename + ".npz"), **data_dict)

# ---------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_parameters(parameters, true_values, save_path):
    os.makedirs(save_path, exist_ok=True)
    param_names = ["ε", "δ", "γ"]
    param_files = ["epsilon", "delta", "gamma"]

    for i, true_val in enumerate(true_values):
        param_data = parameters[i]
        iterations = np.arange(len(param_data)) * 50

        y_label = param_names[i]

        plt.figure(figsize=(6, 4))
        plt.plot(iterations, param_data, 'b-', linewidth=2, label="PINN Prediction")
        plt.axhline(true_val, linestyle="--", color="red", linewidth=2,
                    label=f"True = {true_val:.3f}")
        plt.ylim(0, 1)

        plt.xlabel("Training Iterations", fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')

        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.savefig(f"{save_path}/{param_files[i]}_convergence.png", dpi=300)
        plt.savefig(f"{save_path}/{param_files[i]}_convergence.pdf")
        plt.close()

def plot_losses(loss_u, loss_pde1, loss_pde2, save_path):
    os.makedirs(save_path, exist_ok=True)
    iterations = np.arange(len(loss_u)) * 50
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Log scale
    ax1.plot(iterations/16, loss_u, color=colors[0], label="Data Loss")
    ax1.plot(iterations/16, loss_pde1, color=colors[1], label="Auxiliary PDE")
    ax1.plot(iterations/16, loss_pde2, color=colors[2], label="Main PDE")
    ax1.set_yscale("log")

    ax1.set_xlabel("Training Epochs", fontweight='bold')
    ax1.set_ylabel("Loss (Log Scale)", fontweight='bold')

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Linear scale
    ax2.plot(iterations/16, loss_u, color=colors[0], label="Data Loss")
    ax2.plot(iterations/16, loss_pde1, color=colors[1], label="Auxiliary PDE")
    ax2.plot(iterations/16, loss_pde2, color=colors[2], label="Main PDE")

    ax2.set_xlabel("Training Epochs", fontweight='bold')
    ax2.set_ylabel("Loss (Linear Scale)", fontweight='bold')

    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_convergence.png", dpi=300)
    plt.savefig(f"{save_path}/loss_convergence.pdf")
    plt.close()

def plot_field(u_field, Lx, Ly, title="Field", cmap="viridis", save_path=None, filename=None):
    extent = [0, Lx, 0, Ly]

    plt.figure(figsize=(5, 4))
    im = plt.imshow(u_field, extent=extent, origin="lower", cmap=cmap,
                    interpolation="bilinear", aspect='equal')

    plt.xlabel("x", fontweight='bold')
    plt.ylabel("y", fontweight='bold')

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("u", rotation=270, labelpad=15)

    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}.png", dpi=300)
        plt.savefig(f"{save_path}/{filename}.pdf")
    plt.close()

def plot_pattern_comparison(u_true, u_pred, Lx, Ly, save_path, filename="pattern_comparison"):
    u_true_norm = normalise_field(u_true)
    u_pred_norm = normalise_field(u_pred)
    abs_error = np.abs(u_pred - u_true)

    mean_error = np.mean(abs_error)
    std_error = np.std(abs_error)
    max_error = np.max(abs_error)

    print(f"Absolute Error Statistics:")
    print(f"Mean ± Std: {mean_error:.6f} ± {std_error:.6f}")
    print(f"Max: {max_error:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    extent = [0, Lx, 0, Ly]

    # True
    im1 = axes[0].imshow(u_true_norm, extent=extent, origin="lower",
                         cmap="viridis", vmin=0, vmax=1, aspect='equal')
    axes[0].set_xlabel("x", fontweight='bold')
    axes[0].set_ylabel("y", fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Normalised u")

    # Pred
    im2 = axes[1].imshow(u_pred_norm, extent=extent, origin="lower",
                         cmap="viridis", vmin=0, vmax=1, aspect='equal')
    axes[1].set_xlabel("x", fontweight='bold')
    axes[1].set_ylabel("y", fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Normalised u")

    # Error
    im3 = axes[2].imshow(abs_error, extent=extent, origin="lower",
                         cmap="plasma", vmin=0, vmax=np.max(abs_error), aspect='equal')
    axes[2].set_xlabel("x", fontweight='bold')
    axes[2].set_ylabel("y", fontweight='bold')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04,
                 label="|u_pred - u_true|")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{filename}.png", dpi=300)
    plt.savefig(f"{save_path}/{filename}.pdf")
    plt.close()

# ---------------------------------------------------------
# MAIN POST-PROCESSOR CLASS (FIXED)
# ---------------------------------------------------------
class PINNPostProcessor:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config
        self.model = None
        self.l2_value_pred = None

        if 'run_path' not in config:
            print("WARNING: 'run_path' not found in config. Defaulting to 'results/temp'.")
            self.path_name = "results/temp"
        else:
            self.path_name = config['run_path']

        os.makedirs(self.path_name, exist_ok=True)

    def train(self):
        self.model = self.model_class(
            self.config['nodes'],
            self.config['xrange'],
            self.config['yrange'],
            self.config['learning_rates'][0],
            self.config['learning_rates'][1],
            self.config['u_n'],
            self.config['dx'],
            self.config['tol'],
            self.config['length_app'],
            self.config['batchsize'],
            self.config['sigma2']
        )
        self.model.train(self.config['length_total'], self.config['step_size'])

    def get_final_prediction(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        X, Y = np.meshgrid(self.config['xrange'], self.config['yrange'])
        XY = tf.Variable(np.column_stack([X.flatten(), Y.flatten()]), dtype=tf.float32)

        output = self.model.model_up(XY, training=False)
        u_flat, _ = tf.split(output, 2, axis=1)
        return u_flat.numpy().reshape((self.config['n'], self.config['n']))

    def plot_results(self, true_params):
        plot_parameters(self.model.parameters, true_params, self.path_name)
        plot_losses(self.model.loss_u_array,
                    self.model.loss_pde_1_array,
                    self.model.loss_pde_2_array,
                    self.path_name)

        u_pred = self.get_final_prediction()
        u_true = self.config['u_n']

        plot_pattern_comparison(u_true, u_pred,
                                self.config['Lx'], self.config['Ly'],
                                self.path_name)

        plot_field(u_pred,
                   self.config['Lx'], self.config['Ly'],
                   "PINN Prediction",
                   self.config.get('cmap', 'viridis'),
                   self.path_name, 'u_prediction')

        self.l2_value_pred = L2_error(u_true, u_pred)
        print(f"L2 Value (RMSE): {self.l2_value_pred:.6e}")

    def save_results(self, u_true, true_params):
        u_pred = self.get_final_prediction()
        abs_error = np.abs(u_pred - u_true)
        mse = np.mean((u_pred - u_true) ** 2)
        mae = np.mean(abs_error)

        std_error = np.std(abs_error)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        print("Absolute Error Statistics:")
        print(f"Mean ± Std: {mean_error:.6f} ± {std_error:.6f}")
        print(f"Max: {max_error:.6f}")

        pred_params = np.array(self.model.final_parameters[:3])
        true_params_arr = np.array(true_params)

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.where(true_params_arr != 0,
                                 (pred_params - true_params_arr) / true_params_arr,
                                 np.inf)

        data = {
            "final_parameters": pred_params.astype(float),
            "true_parameters": true_params_arr,
            "mse_pred_vs_true": mse,
            "mae_pred_vs_true": mae,
            "l2_value_rmse": getattr(self, 'l2_value_pred', np.sqrt(mse)),
            "relative_parameter_error": rel_error,
            "u_pred": u_pred.astype(float),
            "u_true": u_true.astype(float),
            "std_error": std_error,
            "max_error": max_error
        }

        save_results_file(data, self.path_name, "results")

        print("\n=== RESULTS SUMMARY (Prediction vs True Data) ===")
        print(f"MSE: {mse:.2e}")
        print(f"MAE: {mae:.2e}")
        print(f"L2 (RMSE): {self.l2_value_pred:.6e}")
        for i, name in enumerate(["ε", "δ", "γ"]):
            print(f"Parameter {name}: {pred_params[i]:.4f} (true: {true_params_arr[i]:.4f})")
