import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import tensorflow as tf
from scipy.integrate import odeint


#Plotting each of the parameters that are being trained, here it is only epsilon
def plot_parameters(parameters, true_values, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, val in enumerate(true_values):
        plt.figure()
        plt.plot(parameters[i])
        plt.axhline(val, linestyle=":")
        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel(f"D_{i}")
        plt.savefig(f"{save_path}/D_{i}.png")
        plt.close()

#plotting the losses of u and losses of pde in a plot

def plot_losses(loss_u, loss_pde_u, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    # Log scale
    plt.figure()
    plt.plot(loss_u, label="loss_u")
    plt.plot(loss_pde_u, label="loss_pde_u")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig(f"{save_path}/loss_log.png")
    plt.close()
    
    # Linear scale
    plt.figure()
    plt.plot(loss_u, label="loss_u")
    plt.plot(loss_pde_u, label="loss_pde_u")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig(f"{save_path}/loss_linear.png")
    plt.close()

#Plots the whole PINN output
def plot_field(u_field, Lx, Ly, cmap='RdBu', save_path=None, filename=None):
    extent = [0, Lx, 0, Ly]
    plt.figure()
    
    # Use 0-1 normalization (like your good plot)
    u_normalized = (u_field - u_field.min()) / (u_field.max() - u_field.min())
    
    plt.imshow(u_normalized, extent=extent, origin='lower', cmap='RdBu', 
               interpolation='bilinear')  # 0 to 1 range
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(ticks=[0, 0.25, 0.5, 0.75, 1])
    
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}.png")
        plt.savefig(f"{save_path}/{filename}.pdf")
    plt.close()

# Saving the files
def save_pickle(data, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/{filename}.pkl", "wb") as f:
        pickle.dump(data, f)

#Full training class
class PINNPostProcessor:
    def __init__(self, model_class, config):
        self.model_class = model_class
        self.config = config
        self.model = None
        self.path_name = f"paper_images/{self.config['model_name']}_no_br/noise_level_{self.config['noise']}"
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
        plot_losses(self.model.loss_u_array, self.model.loss_pde_u_array, self.path_name)
        
        # Reconstructed field
        X, Y = np.meshgrid(self.config['xrange'], self.config['yrange'])
        XY = tf.Variable(np.column_stack([X.flatten(), Y.flatten()]), dtype=tf.float32)
        u_pred = np.reshape(self.model.model_u(XY), (self.config['n'], self.config['n']))
        plot_field(u_pred, self.config['Lx'], self.config['Ly'], save_path=self.path_name, filename='u_approx_pinn')
    
    def simulate_pattern(self, step_forward, modelfun=True):
        n = self.config['n']
        u0 = 0.1 * np.ones(n**2)
        perturb = np.random.normal(0, 0.01, n**2)
        y0 = u0 + perturb
        t = np.linspace(0, 10000)
        c_new = self.model.final_parameters
        
        if modelfun:
            sol = odeint(step_forward, y0, t, args=(c_new, self.config['dx']))
        else:
            sol = odeint(step_forward, y0, t, args=(c_new, self.config['dx'], modelfun))
        
        u_final = np.reshape(sol[-1], (n, n))
        plot_field(u_final, self.config['Lx'], self.config['Ly'], save_path=self.path_name, filename='u_simulated')
        return u_final
    
    def save_results(self, u_final, u_true, true_params):
        mse = np.mean((u_final - u_true)**2)
        rel_error = (self.model.final_parameters[:3] - true_params)/true_params
        save_pickle([self.model.final_parameters, mse, rel_error, u_final], self.path_name, "results")