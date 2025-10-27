#!/usr/bin/env python3
"""
Swift-Hohenberg PINN parameter estimation using image data
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models import RBF_PINNs
from training import PINNPostProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

L_DOMAIN = 50           # Physical domain size
n = 512                 # Grid resolution (512x512)
dx = L_DOMAIN / (n - 1) # Grid spacing

MODEL_CONFIG = {
    'nodes': 1000,
    'n': n,
    'noise': 0,
    'learning_rates': [1e-5, 2e-3],
    'length_app': 40000,
    'length_total': 600000,
    'batchsize': 1024,
    'step_size': 2000,
    'model_name': "image_based_experiment",
    'sigma2': 10 / 10,
    'tol': 1e-7,
}

C_ORIGINAL = [0.5, 0.406, 0.196]  # Reference parameters for comparison

def load_pattern_image(image_path: str, n: int = None):
    """Load and preprocess pattern image to 0-1 and resize to n x n if specified"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u_tp = (gray - gray.min()) / (gray.max() - gray.min())
    
    if n is not None and gray.shape[0] != n:
        u_tp = cv2.resize(u_tp, (n, n), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Resized image to {n}x{n}")
    return u_tp

def get_physical_grid(L_DOMAIN: float, n: int):
    """Generate physical coordinates for PINN training"""
    x = np.linspace(0, L_DOMAIN, n)
    y = np.linspace(0, L_DOMAIN, n)
    dx = x[1] - x[0]
    return x, y, dx

def main():
    logger.info("Starting Swift-Hohenberg PINN training")

    x_phys, y_phys, dx = get_physical_grid(L_DOMAIN, n)

    image_path = Path("C:/Users/Zach Mollatt/Documents/Git/SwiftHohenberg-Inverse-Pinn/data/pattern_eps0.600_delta0.406_gamma0.196_PINN.png")

    try:
        u_tp = load_pattern_image(image_path, n=n)
        logger.info(f"Pattern image loaded with shape: {u_tp.shape}")
    except FileNotFoundError:
        logger.warning("Image not found, generating synthetic pattern instead")
        from pde_utils import generate_synthetic_pattern
        u_tp, _ = generate_synthetic_pattern(n, C_ORIGINAL, dx)

    config = MODEL_CONFIG.copy()
    config.update({
        'dx': dx,
        'xrange': x_phys,
        'yrange': y_phys,
        'u_n': u_tp,
        'Lx': L_DOMAIN,
        'Ly': L_DOMAIN,
        'cmap': 'RdBu',
    })
    
    logger.info("Initializing PINN trainer")
    trainer = PINNPostProcessor(RBF_PINNs, config)
    
    logger.info("Starting model training")
    trainer.train()
    
    logger.info("Generating plots and saving results")
    trainer.plot_results(C_ORIGINAL)

    from pde_utils import step_forward2
    u_tp_new = trainer.simulate_pattern(step_forward2)
    trainer.save_results(u_tp_new, u_tp, C_ORIGINAL)

    logger.info("Training and analysis complete")

if __name__ == "__main__":
    main()
