import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import tensorflow as tf
from scipy.integrate import odeint


#Plotting each of the parameters that are being trained
def plot_parameters(parameters, true_values, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    param_names = ["epsilon", "delta", "gamma"]
    
    for i, val in enumerate(true_values):
        plt.figure()
        
        param_data = parameters[i]
        true_val = true_values[i]
        
        plt.plot(param_data, label=f'Predicted {param_names[i]}')
        plt.axhline(true_val, linestyle=":", color='r', label=f'True Value ({true_val})')
        
        # --- * FIXED: Epsilon always 0-1, others auto-scale * ---
        if param_names[i] == "epsilon":
            # Force epsilon to always show 0 to 1 range
            plt.ylim(0.1, 0.9)
        else:
            # For delta and gamma, keep auto-scaling with padding
            data_min = np.min(param_data)
            data_max = np.max(param_data)
            
            if data_min == data_max:
                # It's a constant plot (delta or gamma)
                padding = 0.1 * abs(true_val) # 10% padding
                if padding == 0: padding = 0.1 # Handle case where val is 0
                plt.ylim(true_val - padding, true_val + padding)
            else:
                # It's a converging plot
                padding = 0.1 * (data_max - data_min) # 10% padding
                if padding == 0: padding = 0.1
                plt.ylim(data_min - padding, data_max + padding)
        # --- * END FIX * ---
            
        plt.xlabel("Iterations (x50)") # Since we log every 50 steps
        plt.ylabel(f"Parameter Value")
        plt.title(f"Convergence of {param_names[i]}") # Added Title
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add tight_layout to prevent cropping
        plt.tight_layout() 
        plt.savefig(f"{save_path}/{param_names[i]}.png")
        plt.close()

#plotting the losses of u and losses of pde in a plot
def plot_losses(loss_u, loss_pde1, loss_pde2, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Log scale
    plt.figure()
    plt.plot(loss_u, label="L_Data (loss_u)")
    plt.plot(loss_pde1, label="Auxillary Equation")
    plt.plot(loss_pde2, label="Simplified Swift-Hohenberg Equation")
    plt.yscale("log")
    plt.xlabel("Iterations (x50)")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Loss Convergence (Log Scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_log.png")
    plt.close()

    # Linear scale
    plt.figure()
    plt.plot(loss_u, label="L_Data (loss_u)")
    plt.plot(loss_pde1, label="Auxillary Equation")
    plt.plot(loss_pde2, label="Simplified Swift-Hohenberg Equation")
    plt.xlabel("Iterations (x50)")
    plt.ylabel("Loss (Linear Scale)")
    plt.title("Loss Convergence (Linear Scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_linear.png")
    plt.close()


#Plots the whole PINN output
def plot_field(u_field, Lx, Ly, cmap='RdBu', save_path=None, filename=None):
    extent = [0, Lx, 0, Ly]
    plt.figure()
    plt.imshow(u_field, extent=extent, origin='lower', cmap=cmap, 
               interpolation='bilinear')  # 0 to 1 range
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Generated Pattern: {filename}") # Added Title
    plt.colorbar()
    
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        # Add tight_layout to prevent cropping
        plt.tight_layout()
        plt.savefig(f"{save_path}/{filename}.png")
        plt.savefig(f"{save_path}/{filename}.pdf")
    plt.close()

# --- * NEW FUNCTION 1: ABSOLUTE ERROR * ---
def plot_error_heatmap(u_pred, u_true, Lx, Ly, cmap='RdBu', save_path=None, filename=None):
    """Plots the absolute error (u_pred - u_true) as a heatmap."""
    error_field = u_pred - u_true
    extent = [0, Lx, 0, Ly]
    plt.figure()
    
    # Center the colormap at 0
    vmax = np.max(np.abs(error_field))
    if vmax == 0: vmax = 1e-6 # Avoid vmin=vmax=0
    vmin = -vmax
    
    plt.imshow(error_field, extent=extent, origin='lower', cmap=cmap, 
                 interpolation='bilinear', vmin=vmin, vmax=vmax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Absolute Error: {filename}")
    plt.colorbar(label="Pred - True")
    
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{filename}.png")
    plt.close()

# --- * NEW FUNCTION 2: RELATIVE ERROR (As requested) * ---
def plot_relative_error_heatmap(u_pred, u_true, Lx, Ly, cmap='RdBu', save_path=None, filename=None):
    """Plots the relative error ((u_pred - u_true) / u_true) as a heatmap."""
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = (u_pred - u_true) / u_true
    
    # Cap at 99th percentile for visualization, ignoring NaNs/Infs
    vmax = np.nanpercentile(np.abs(relative_error[np.isfinite(relative_error)]), 99)
    if vmax == 0 or not np.isfinite(vmax): vmax = 1.0 # Sensible default
    vmin = -vmax
    
    relative_error[~np.isfinite(relative_error)] = 0.0 # Set NaNs/Infs to 0 for plotting
    relative_error = np.clip(relative_error, vmin, vmax) # Clip to the percentile
    
    extent = [0, Lx, 0, Ly]
    plt.figure()
    plt.imshow(relative_error, extent=extent, origin='lower', cmap=cmap, 
                 interpolation='bilinear', vmin=vmin, vmax=vmax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Relative Error: {filename}\n(Viz capped at {vmax*100:.1f}%)")
    plt.colorbar(label="(Pred - True) / True")
    
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{filename}.png")
    plt.close()

# *** FIXED: Was plotting weights instead of gradnorms ***
def plot_gradnorms(gradnorm_u, gradnorm_pde1, gradnorm_pde2, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure()
    plt.plot(gradnorm_u, label="GradNorm Data Loss")
    plt.plot(gradnorm_pde1, label="GradNorm PDE1")
    plt.plot(gradnorm_pde2, label="GradNorm PDE2")
    plt.xlabel("Iterations (x50)")
    plt.ylabel("Gradient Norm (L2)")
    plt.title("GradNorm Evolution")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_path}/gradnorms.png")
    plt.close()

# *** NEW: Plot loss weights evolution ***
def plot_loss_weights(weight_data, weight_pde1, weight_pde2, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure()
    plt.plot(weight_data, label="Weight Data Loss")
    plt.plot(weight_pde1, label="Weight PDE1")
    plt.plot(weight_pde2, label="Weight PDE2")
    plt.xlabel("Iterations (x50)")
    plt.ylabel("Loss Weight")
    plt.title("Loss Weight Evolution (GradNorm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_weights.png")
    plt.close()

# Saving the files
def save_results_file(data_dict, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    np.savez(os.path.join(save_path, filename + ".npz"), **data_dict)

#Full training class
class PINNPostProcessor:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config
        self.model = None

        # Absolute path to the folder where main.py lives
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Create /results in that same folder
        self.path_name = os.path.join(base_dir, "results")
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
    
    #Literally just plotting all the results we got
    def plot_results(self, true_params):
        plot_parameters(self.model.parameters, true_params, self.path_name)
        plot_losses(self.model.loss_u_array, self.model.loss_pde_1_array, self.model.loss_pde_2_array, self.path_name)
        
        # Reconstructed field
        X, Y = np.meshgrid(self.config['xrange'], self.config['yrange'])
        XY = tf.Variable(np.column_stack([X.flatten(), Y.flatten()]), dtype=tf.float32)

        # 1. Call the correct model, 'model_up'
        model_output = self.model.model_up(XY, training=False)
        
        # 2. Split the output to get only the first column (u_pred)
        u_pred_flat, _ = tf.split(model_output, 2, axis=1)
        
        # 3. Reshape the u_pred tensor for plotting
        u_pred = np.reshape(u_pred_flat.numpy(), (self.config['n'], self.config['n']))

        plot_field(u_pred, self.config['Lx'], self.config['Ly'], cmap=self.config['cmap'], save_path=self.path_name, filename='u_approx_pinn')
        
        # Get the ground truth data
        u_true = self.config['u_n']
        
        # 1. Plot Absolute Error
        plot_error_heatmap(
            u_pred, u_true, 
            self.config['Lx'], self.config['Ly'], 
            cmap='RdBu',
            save_path=self.path_name, 
            filename='u_absolute_error'
        )
        
        # 2. Plot Relative Error
        plot_relative_error_heatmap(
            u_pred, u_true, 
            self.config['Lx'], self.config['Ly'], 
            cmap='RdBu',
            save_path=self.path_name, 
            filename='u_relative_error'
        )

        # *** FIXED: Pass correct gradnorm arrays ***
        plot_gradnorms(
            self.model.gradnorm_u_array,      # Was: self.model.weight_data
            self.model.gradnorm_pde1_array,
            self.model.gradnorm_pde2_array,
            self.path_name
        )
        
        # *** NEW: Also plot the loss weights separately ***
        plot_loss_weights(
            self.model.weight_data_array,
            self.model.weight_pde1_array,
            self.model.weight_pde2_array,
            self.path_name
        )
    
    # *** FIXED: simulate_pattern now properly passes c_new ***
    def simulate_pattern(self, step_forward, modelfun=True):
        n = self.config['n']
        u0 = 0.1 * np.ones(n**2)
        perturb = np.random.normal(0, 0.01, n**2)
        y0 = u0 + perturb
        t = np.linspace(0, 10000, 1000)  # Added number of time points for stability
        c_new = self.model.final_parameters
        
        # *** FIXED: Properly unpack parameters for odeint ***
        # The step_forward function expects: (y, t, epsilon, delta, gamma, dx, modelfun)
        epsilon, delta, gamma = c_new[0], c_new[1], c_new[2]
        
        # Pass parameters correctly to odeint
        sol = odeint(step_forward, y0, t, args=(epsilon, delta, gamma, self.config['dx'], modelfun))
        
        u_final = np.reshape(sol[-1], (n, n))
        plot_field(u_final, self.config['Lx'], self.config['Ly'], cmap=self.config['cmap'], save_path=self.path_name, filename='u_simulated')
        return u_final
    
    def save_results(self, u_final, u_true, true_params):
        mse = np.mean((u_final - u_true)**2)

        pred_params = np.array(self.model.final_parameters[:3])
        true_params_array = np.array(true_params)

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.where(true_params_array != 0,
                                 (pred_params - true_params_array) / true_params_array,
                                 np.inf)

        data = {
            "final_parameters": np.array(self.model.final_parameters, dtype=float),
            "mse": mse,
            "relative_error": rel_error,
            "u_final": np.array(u_final, dtype=float)
        }

        save_results_file(data, self.path_name, "results")