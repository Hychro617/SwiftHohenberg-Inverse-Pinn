import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2  
import time
from models import RBF_PINNs 
from training import PINNPostProcessor
import tensorflow as tf
import os
import random
from typing import Union

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

L_DOMAIN = 16 * np.pi          # Physical domain size
n = 64                         # Grid resolution (64x64)
dx = L_DOMAIN / (n - 1)        # Grid spacing

MODEL_CONFIG = {
    'nodes': 300,
    'n': n,
    'noise': 0,
    'learning_rates': [1e-5, 2e-3],
    'length_app': 20000,
    'length_total': 120000,
    'batchsize': 256,
    'step_size': 2000,
    'model_name': "image_based_experiment_n64_visualized",
    'sigma2': 2,
    'tol': 1e-7,
}
#These are only needed for plots if you know the true values of the system
C_ORIGINAL = [0.05, 0.406, 0.196]

def load_pattern_array(path: Union[str, Path], n: int = None):
    """Load and preprocess pattern array from .npy or image file, with resizing.
    The data is normalized to the range [-1, 1].
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".npy":
        u_tp = np.load(str(path)).astype(float)

    elif ext in [".png", ".jpg", ".jpeg", ".jpf", ".jfif", ".bmp", ".tiff"]:
        #Use cv2.IMREAD_GRAYSCALE to get a single channel
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        u_tp = img.astype(float)

        #Normalise to [-1, 1]
        u_min = u_tp.min()
        u_max = u_tp.max()
        if u_max != u_min:
            u_tp = 2 * ((u_tp - u_min) / (u_max - u_min)) - 1
        else:
        #Handle case where image is uniform (all pixels the same)
            u_tp = np.zeros_like(u_tp)
            logger.warning("Image is uniform; normalized to 0.0.")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Resize if needed
    if n is not None and u_tp.shape[0] != n:
        original_shape = u_tp.shape
        u_tp = cv2.resize(u_tp, (n, n), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Resized input from {original_shape} to {u_tp.shape}")
        
    logger.info(f"Data range after normalization: min={u_tp.min():.4f}, max={u_tp.max():.4f}")
    return u_tp


def plot_pattern(u_tp: np.ndarray, title: str = "Loaded and Normalized Pattern"):
    """Displays the loaded and resized pattern array for verification."""
    plt.figure(figsize=(6, 6))
    im = plt.imshow(u_tp, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    
    # Add colorbar for clarity
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Amplitude u(x, y)')

    plt.title(title)
    plt.xlabel(f"X-coordinate (0 to {u_tp.shape[1]-1})")
    plt.ylabel(f"Y-coordinate (0 to {u_tp.shape[0]-1})")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show() # Display the figure

def get_physical_grid(L_DOMAIN: float, n: int):
    """Generate physical coordinates for PINN training"""
    x = np.linspace(0, L_DOMAIN, n)
    y = np.linspace(0, L_DOMAIN, n)
    dx = x[1] - x[0]
    return x, y, dx

def main():
    start = time.time()
    logger.info("Starting Swift-Hohenberg PINN data preparation")

    x_phys, y_phys, dx = get_physical_grid(L_DOMAIN, n)

    #Update this path to point to your actual data file.
    data_path = Path("C:/Users/Zach Mollatt/Documents/Git/SwiftHohenberg-Inverse-Pinn/data/stripes.npy")
    
    try:
        u_tp = load_pattern_array(data_path, n=n)
        logger.info(f"Pattern array loaded (normalized) with final shape: {u_tp.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required data file not found: {data_path}\n"
            "Please fix the path or provide the file."
        )

    #Visualise the loaded
    plot_pattern(u_tp, title=f"Input Pattern u({n}x{n}) Normalized to [-1, 1]")


    config = MODEL_CONFIG.copy()
    config.update({
        'dx': dx,
        'xrange': x_phys,
        'yrange': y_phys, 
        'u_n': u_tp, # Pass the normalized data
        'Lx': L_DOMAIN,
        'Ly': L_DOMAIN,
        'cmap': 'RdBu_r',
        'run_path': 'Results'

    })

    logger.info("Initializing PINN trainer (using dummy class for demonstration)")
    trainer = PINNPostProcessor(RBF_PINNs, config)

    logger.info("Starting model training")
    trainer.train() 

    end = time.time()
    elapsed = end - start
    logger.info(f"Total execution time: {elapsed:.2f} seconds")
    
    logger.info("Generating plots and saving results") 
    trainer.plot_results(C_ORIGINAL)
    trainer.save_results(u_tp, C_ORIGINAL)
    logger.info("Data preparation and visualization complete")

if __name__ == "__main__":
    main()