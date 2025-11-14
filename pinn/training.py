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
        
        # --- *** NEW AXIS FIX *** ---
        data_min = np.min(param_data)
        data_max = np.max(param_data)
        
        if data_min == data_max:
            # It's a constant plot (delta or gamma)
            padding = 0.1 * abs(true_val) # 10% padding
            if padding == 0: padding = 0.1 # Handle case where val is 0
            plt.ylim(true_val - padding, true_val + padding)
        else:
            # It's a converging plot (epsilon)
            padding = 0.1 * (data_max - data_min) # 10% padding
            if padding == 0: padding = 0.1
            plt.ylim(data_min - padding, data_max + padding)
        # --- *** END FIX *** ---
            
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

        # *** --- START OF FIX (from v1.1) --- ***
        # 1. Call the correct model, 'model_up'
        model_output = self.model.model_up(XY, training=False)
        
        # 2. Split the output to get only the first column (u_pred)
        u_pred_flat, _ = tf.split(model_output, 2, axis=1)
        
        # 3. Reshape the u_pred tensor for plotting
        u_pred = np.reshape(u_pred_flat.numpy(), (self.config['n'], self.config['n']))
        # *** --- END OF FIX --- ***

        plot_field(u_pred, self.config['Lx'], self.config['Ly'], cmap=self.config['cmap'], save_path=self.path_name, filename='u_approx_pinn')
    
    def simulate_pattern(self, step_forward, modelfun=True):
        n = self.config['n']
        u0 = 0.1 * np.ones(n**2)
        perturb = np.random.normal(0, 0.01, n**2)
        y0 = u0 + perturb
        t = np.linspace(0, 10000)
        c_new = self.model.final_parameters
        
        # *** --- START OF FIX (from v1.1) --- ***
        # Removed the faulty if/else block.
        sol = odeint(step_forward, y0, t, args=(c_new, self.config['dx'], modelfun))
        # *** --- END OF FIX --- ***
        
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
