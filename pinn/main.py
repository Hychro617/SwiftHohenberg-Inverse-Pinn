#!/usr/bin/env python3
"""
Swift-Hohenberg PINN parameter estimation using array data
(Version with n=100 and NO data normalization)
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Still needed for resizing

from models import RBF_PINNs # Assumes you are using the corresponding models.py
from training import PINNPostProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

L_DOMAIN = 50        # Physical domain size
n = 100              # Grid resolution (100x100)
dx = L_DOMAIN / (n - 1) # Grid spacing

MODEL_CONFIG = {
    'nodes': 265,
    'n': n,
    'noise': 0,
    'learning_rates': [1e-5, 2e-3],
    'length_app': 1000, # Initial data-fitting phase length
    'length_total': 600000,
    'batchsize': 1024,
    'step_size': 2000,
    'model_name': "image_based_experiment_n100_raw", # Updated model name
    'sigma2': 10 / 10,
    'tol': 1e-7,
}

# Reference parameters for comparison (matches the data filename's epsilon)
C_ORIGINAL = [0.6, 0.406, 0.196]

def load_pattern_array(array_path: str, n: int = None):
    """Load and preprocess pattern array from .npy file and resize"""
    try:
        u_data = np.load(str(array_path))
    except FileNotFoundError:
        logger.error(f"Could not load array: {array_path}")
        raise

    u_tp = u_data.astype(float)

    if n is not None and u_tp.shape[0] != n:
        original_shape = u_tp.shape
        # Use cv2.resize for high-quality interpolation
        u_tp = cv2.resize(u_tp, (n, n), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Resized array from {original_shape} to ({n}, {n})")
    return u_tp

def get_physical_grid(L_DOMAIN: float, n: int):
    """Generate physical coordinates for PINN training"""
    x = np.linspace(0, L_DOMAIN, n)
    y = np.linspace(0, L_DOMAIN, n)
    dx = x[1] - x[0]
    return x, y, dx

def main():
    logger.info("Starting Swift-Hohenberg PINN training (n=100, raw data)")

    x_phys, y_phys, dx = get_physical_grid(L_DOMAIN, n)

    data_path = Path("C:/Users/Zach Mollatt/Documents/Git/SwiftHohenberg-Inverse-Pinn/data/pattern_eps0.600_delta0.406_gamma0.196_PINN.npy")

    try:
        u_tp = load_pattern_array(data_path, n=n)
        logger.info(f"Pattern array loaded (raw) with final shape: {u_tp.shape}")
    except FileNotFoundError:
        logger.warning(f"Array file not found at {data_path}, generating synthetic pattern instead")
        # Ensure pde_utils is available if this block is needed
        try:
            from pde_utils import generate_synthetic_pattern
            u_tp_syn, _ = generate_synthetic_pattern(n, C_ORIGINAL, dx)

            # --- NO NORMALIZATION FOR SYNTHETIC DATA ---
            u_tp = u_tp_syn.astype(float)
            # --- END NO SYNTHETIC NORMALIZATION ---

            logger.info(f"Generated raw synthetic pattern with shape: {u_tp.shape}")
        except ImportError:
            logger.error("pde_utils not found. Cannot generate synthetic data.")
            return # Exit if synthetic data generation fails


    # --- Check data range (will show raw range) ---
    logger.info(f"Data range after loading (raw): min={u_tp.min()}, max={u_tp.max()}")

    config = MODEL_CONFIG.copy()
    config.update({
        'dx': dx,
        'xrange': x_phys,
        'yrange': y_phys,
        'u_n': u_tp, # Pass the raw data
        'Lx': L_DOMAIN,
        'Ly': L_DOMAIN,
        'cmap': 'RdBu',
    })

    logger.info("Initializing PINN trainer")
    trainer = PINNPostProcessor(RBF_PINNs, config)

    logger.info("Starting model training")
    trainer.train() # This will run the training loop in models.py

    logger.info("Generating plots and saving results")
    trainer.plot_results(C_ORIGINAL)

    # Ensure pde_utils is available if this block is needed
    try:
        from pde_utils import step_forward2
        u_tp_new = trainer.simulate_pattern(step_forward2)
        trainer.save_results(u_tp_new, u_tp, C_ORIGINAL)
    except ImportError:
        logger.warning("pde_utils not found. Skipping simulation and saving steps.")


    logger.info("Training and analysis complete")

if __name__ == "__main__":
    main()