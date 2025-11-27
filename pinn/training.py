import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.integrate import odeint

# ---------------------------------------------------------
# GLOBAL STYLE SETTINGS (publication-ready)
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.figsize": (8, 6),  # Reasonable default size
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
# PARAMETER CONVERGENCE PLOTS
# ---------------------------------------------------------
def plot_parameters(parameters, true_values, save_path):
    """Plot parameter convergence with professional styling."""
    os.makedirs(save_path, exist_ok=True)
    param_names = ["ε", "δ", "γ"]  # Greek letters for report
    param_files = ["epsilon", "delta", "gamma"]  # File names

    for i, true_val in enumerate(true_values):
        param_data = parameters[i]
        iterations = np.arange(len(param_data)) * 50

        plt.figure(figsize=(6, 4))
        plt.plot(iterations, param_data, 'b-', linewidth=2, label="PINN Prediction")
        plt.axhline(true_val, linestyle="--", color="red", linewidth=2, 
                   label=f"True = {true_val:.3f}")

        # Dynamic y-limits
        if param_names[i] == "ε":
            plt.ylim(0, 1)
        else:
            dmin, dmax = np.min(param_data), np.max(param_data)
            if dmin == dmax:
                pad = max(0.1 * abs(true_val), 0.1)
                plt.ylim(true_val - pad, true_val + pad)
            else:
                pad = 0.1 * (dmax - dmin)
                plt.ylim(dmin - pad, dmax + pad)

        plt.xlabel("Training Iterations")
        plt.ylabel(f"Parameter {param_names[i]}")
        plt.title(f"Convergence of {param_names[i]}")
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        plt.savefig(f"{save_path}/{param_files[i]}_convergence.png", dpi=300)
        plt.savefig(f"{save_path}/{param_files[i]}_convergence.pdf")
        plt.close()

# ---------------------------------------------------------
# LOSS CURVES
# ---------------------------------------------------------
def plot_losses(loss_u, loss_pde1, loss_pde2, save_path):
    """Plot loss convergence on both log and linear scales."""
    os.makedirs(save_path, exist_ok=True)
    iterations = np.arange(len(loss_u)) * 50

    # Combined plot with both scales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional color scheme
    
    # Log scale
    ax1.plot(iterations, loss_u, color=colors[0], label="Data Loss")
    ax1.plot(iterations, loss_pde1, color=colors[1], label="Auxiliary PDE")
    ax1.plot(iterations, loss_pde2, color=colors[2], label="Main PDE")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Iterations")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.set_title("Loss Convergence (Log)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Linear scale
    ax2.plot(iterations, loss_u, color=colors[0], label="Data Loss")
    ax2.plot(iterations, loss_pde1, color=colors[1], label="Auxiliary PDE")
    ax2.plot(iterations, loss_pde2, color=colors[2], label="Main PDE")
    ax2.set_xlabel("Training Iterations")
    ax2.set_ylabel("Loss (Linear Scale)")
    ax2.set_title("Loss Convergence (Linear)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_convergence.png", dpi=300)
    plt.savefig(f"{save_path}/loss_convergence.pdf")
    plt.close()

# ---------------------------------------------------------
# COMBINED PATTERN COMPARISON (Main Report Figure)
# ---------------------------------------------------------
def plot_pattern_comparison(u_true, u_pred, Lx, Ly, save_path, filename="pattern_comparison"):
    """Create the main 3-panel comparison figure for reports."""
    
    def normalize_field(u):
        """Normalize field to [0, 1] range."""
        u_min, u_max = np.min(u), np.max(u)
        if u_max - u_min == 0:
            return np.zeros_like(u)
        return (u - u_min) / (u_max - u_min)

    # Normalize both fields
    u_true_norm = normalize_field(u_true)
    u_pred_norm = normalize_field(u_pred)
    abs_error = np.abs(u_pred - u_true)
    
    # Create figure with optimal size for reports
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    extent = [0, Lx, 0, Ly]
    
    # Consistent normalization limits for comparison
    vmin, vmax = 0, 1
    
    # (a) True/Initial pattern
    im1 = axes[0].imshow(u_true_norm, extent=extent, origin="lower", 
                        cmap="viridis", vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title("(a) True Pattern")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Normalized u", rotation=270, labelpad=15)

    # (b) Predicted pattern  
    im2 = axes[1].imshow(u_pred_norm, extent=extent, origin="lower", 
                        cmap="viridis", vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title("(b) PINN Prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Normalized u", rotation=270, labelpad=15)

    # (c) Absolute error
    im3 = axes[2].imshow(abs_error, extent=extent, origin="lower", 
                        cmap="plasma", vmin=0, vmax=np.max(abs_error), aspect='equal')
    axes[2].set_title("(c) Absolute Error")
    axes[2].set_xlabel("x") 
    axes[2].set_ylabel("y")
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label("|u_pred - u_true|", rotation=270, labelpad=15)

    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{filename}.png", dpi=300)
    plt.savefig(f"{save_path}/{filename}.pdf")
    plt.close()

# ---------------------------------------------------------
# INDIVIDUAL FIELD PLOTS (for supplementary material)
# ---------------------------------------------------------
def plot_field(u_field, Lx, Ly, title="Field", cmap="viridis", save_path=None, filename=None):
    """Plot individual field with professional styling."""
    extent = [0, Lx, 0, Ly]
    
    plt.figure(figsize=(5, 4))
    im = plt.imshow(u_field, extent=extent, origin="lower", cmap=cmap, 
                   interpolation="bilinear", aspect='equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("u", rotation=270, labelpad=15)

    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}.png", dpi=300)
        plt.savefig(f"{save_path}/{filename}.pdf")
    plt.close()

# ---------------------------------------------------------
# ERROR ANALYSIS PLOTS
# ---------------------------------------------------------
def plot_error_analysis(u_pred, u_true, Lx, Ly, save_path):
    """Create comprehensive error analysis plots."""
    os.makedirs(save_path, exist_ok=True)
    
    # Absolute error
    abs_error = np.abs(u_pred - u_true)
    plot_field(abs_error, Lx, Ly, "Absolute Error", "plasma", save_path, "absolute_error")
    
    # Relative error (with robust handling)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = (u_pred - u_true) / u_true
    finite_mask = np.isfinite(rel_error)
    if np.any(finite_mask):
        vmax = np.nanpercentile(np.abs(rel_error[finite_mask]), 95)
        rel_error = np.clip(rel_error, -vmax, vmax)
        rel_error[~finite_mask] = 0
        plot_field(rel_error, Lx, Ly, "Relative Error", "RdBu_r", save_path, "relative_error")

# ---------------------------------------------------------
# FILE SAVING UTILITIES
# ---------------------------------------------------------
def save_results_file(data_dict, save_path, filename):
    """Save numerical results to compressed numpy format."""
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, filename + ".npz"), **data_dict)

# ---------------------------------------------------------
# MAIN PINN POST-PROCESSOR CLASS  
# ---------------------------------------------------------
class PINNPostProcessor:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config
        self.model = None

        # Setup results directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_name = os.path.join(base_dir, "results")
        os.makedirs(self.path_name, exist_ok=True)
    
    def train(self):
        """Train the PINN model."""
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
    
    def plot_results(self, true_params):
        """Generate all report-ready plots."""
        # Training diagnostics
        plot_parameters(self.model.parameters, true_params, self.path_name)
        plot_losses(self.model.loss_u_array, self.model.loss_pde_1_array, 
                   self.model.loss_pde_2_array, self.path_name)
        
        # Get model prediction
        X, Y = np.meshgrid(self.config['xrange'], self.config['yrange'])
        XY = tf.Variable(np.column_stack([X.flatten(), Y.flatten()]), dtype=tf.float32)
        
        model_output = self.model.model_up(XY, training=False)
        u_pred_flat, _ = tf.split(model_output, 2, axis=1)
        u_pred = np.reshape(u_pred_flat.numpy(), (self.config['n'], self.config['n']))
        
        # Ground truth
        u_true = self.config['u_n']
        
        # Main comparison plot (for report body)
        plot_pattern_comparison(u_true, u_pred, self.config['Lx'], 
                              self.config['Ly'], self.path_name)
        
        # Individual plots (for supplementary material)
        plot_field(u_pred, self.config['Lx'], self.config['Ly'], 
                  "PINN Prediction", self.config.get('cmap', 'viridis'), 
                  self.path_name, 'u_prediction')
        
        # Error analysis
        plot_error_analysis(u_pred, u_true, self.config['Lx'], 
                          self.config['Ly'], self.path_name)

    def simulate_pattern(self, step_forward, modelfun=True):
        """Simulate pattern using learned parameters."""
        n = self.config['n']
        u0 = 0.1 * np.ones(n**2)
        perturb = np.random.normal(0, 0.01, n**2)
        y0 = u0 + perturb
        t = np.linspace(0, 10000, 1000)
        
        # Get learned parameters
        c_new = self.model.final_parameters
        epsilon, delta, gamma = c_new[0], c_new[1], c_new[2]
        
        # Solve ODE
        sol = odeint(step_forward, y0, t, args=(epsilon, delta, gamma, self.config['dx'], modelfun))
        u_final = np.reshape(sol[-1], (n, n))
        
        # Plot result
        plot_field(u_final, self.config['Lx'], self.config['Ly'], 
                  "Simulated Pattern", self.config.get('cmap', 'viridis'),
                  self.path_name, 'u_simulated')
        
        return u_final
    
    def save_results(self, u_final, u_true, true_params):
        """Save quantitative results."""
        # Calculate metrics
        mse = np.mean((u_final - u_true)**2)
        mae = np.mean(np.abs(u_final - u_true))
        
        pred_params = np.array(self.model.final_parameters[:3])
        true_params_array = np.array(true_params)

        # Parameter errors
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.where(true_params_array != 0,
                               (pred_params - true_params_array) / true_params_array,
                               np.inf)

        # Comprehensive results dictionary
        data = {
            "final_parameters": np.array(self.model.final_parameters, dtype=float),
            "true_parameters": true_params_array,
            "mse": mse,
            "mae": mae,
            "relative_parameter_error": rel_error,
            "u_final": np.array(u_final, dtype=float),
            "u_true": np.array(u_true, dtype=float)
        }

        save_results_file(data, self.path_name, "results")
        
        # Print summary for log
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"MSE: {mse:.2e}")
        print(f"MAE: {mae:.2e}")
        for i, name in enumerate(['ε', 'δ', 'γ']):
            print(f"Parameter {name}: {pred_params[i]:.4f} (true: {true_params_array[i]:.4f})")