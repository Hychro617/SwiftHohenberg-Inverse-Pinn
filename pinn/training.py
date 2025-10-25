import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import tensorflow as tf

class TrainingManager:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        try:
            self.path_name = f"paper_images/{self.config['model_name']}_no_br/noise_level_{self.config['noise']}"
            os.makedirs(self.path_name)
        except FileExistsError:
            print("Directory already exists, possible rewriting")
            pass
    
    def train_model(self):
        self.PINN_model = self.model_class(
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
            self.config['sigma2'],
        )
        self.PINN_model.train(self.config['length_total'], self.config['step_size'])
        return self.PINN_model
    
    def plot_results(self, c_original):
        os.makedirs(self.path_name, exist_ok=True)
        
        # Plot parameters
        for i, par in enumerate(c_original):
            self._plot_parameter(i, par)
        
        # Plot losses
        self._plot_losses()
        
        # Plot reconstructed pattern
        self._plot_reconstructed_pattern()
        
        # Save arrays
        self._save_data_arrays()
    
    def _plot_parameter(self, idx, true_value):
        plt.figure()
        plt.plot(self.PINN_model.parameters[idx])
        plt.yscale("log")
        plt.axhline(y=true_value, linestyle=":")
        plt.xlabel("Iterations")
        plt.ylabel(f"D_{idx}")
        plt.savefig(f"{self.path_name}/d{idx}.png")
        plt.savefig(f"{self.path_name}/d{idx}.pdf")
        plt.close()
    
    def _plot_losses(self):
        # Log scale losses
        plt.figure()
        plt.plot(self.PINN_model.loss_u_array, label="loss_u")
        plt.plot(self.PINN_model.loss_pde_u_array, label="loss_pde_u")
        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(f"{self.path_name}/loss_plot_log.png")
        plt.savefig(f"{self.path_name}/loss_plot_log.pdf")
        plt.close()
        
        # Linear scale losses
        plt.figure()
        plt.plot(self.PINN_model.loss_u_array, label="loss_u")
        plt.plot(self.PINN_model.loss_pde_u_array, label="loss_pde_u")
        plt.xlabel("Iterations")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(f"{self.path_name}/loss_plot.png")
        plt.savefig(f"{self.path_name}/loss_plot.pdf")
        plt.close()
    
    def _plot_reconstructed_pattern(self):
        X, Y = np.meshgrid(self.config['xrange'], self.config['yrange'])
        X, Y = tf.Variable(X.flatten()[:, None], dtype=tf.float32), tf.Variable(
            Y.flatten()[:, None], dtype=tf.float32
        )
        plt.imshow(
            np.reshape(self.PINN_model.model_u(tf.concat([X, Y], 1)), (self.config['n'], self.config['n'])),
            cmap=self.config.get('cmap', cm.Spectral)
        )
        plt.savefig(f"{self.path_name}/u_approx_pinn.png")
        plt.savefig(f"{self.path_name}/u_approx_pinn.pdf")
        plt.close()
    
    def _save_data_arrays(self):
        filename = f"{self.path_name}/saved_param_arrays"
        with open(filename, "wb") as outfile:
            pickle.dump(self.PINN_model.parameters, outfile)
        
        filename = f"{self.path_name}/saved_losses_arrays"
        with open(filename, "wb") as outfile:
            pickle.dump([
                self.PINN_model.loss_u_array,
                self.PINN_model.loss_pde_u_array,
            ], outfile)
    
    def simulate_new_pattern(self, step_forward, modelfun=True):
        c_new = self.PINN_model.final_parameters
        n = self.config['n']
        
        u0 = 0.1 * np.ones(n**2)
        perturbation1 = np.random.normal(0, 0.01, (n**2))
        y0 = u0 + perturbation1
        tlen = 10000
        t = np.linspace(0, tlen)
        
        from scipy.integrate import odeint
        if modelfun:
            solb = odeint(step_forward, y0, t, args=(c_new, self.config['dx']))
        else:
            solb = odeint(step_forward, y0, t, args=(c_new, self.config['dx'], modelfun))
        
        u_tp_new = np.reshape(solb[-1], (n, n))
        plt.imshow(u_tp_new, cmap=self.config.get('cmap', cm.Spectral))
        plt.savefig(f"{self.path_name}/u_approx.png")
        plt.savefig(f"{self.path_name}/u_approx.pdf")
        plt.close()
        
        return u_tp_new
    
    def save_results(self, u_tp_new, u_original, c_original):
        MSE_u = np.mean((u_tp_new - u_original) ** 2)
        relative_error = (self.PINN_model.final_parameters[:3] - c_original) / c_original
        
        objects_to_save = [
            self.PINN_model.final_parameters,
            MSE_u,
            relative_error,
            u_tp_new,
        ]
        
        filename = f"{self.path_name}/saved_matrices"
        with open(filename, "wb") as outfile:
            pickle.dump(objects_to_save, outfile)