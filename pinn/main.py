#!/usr/bin/env python3
"""
Main script for Swift-Hohenberg PINN parameter estimation using image data
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import cv2
from models import RBF_PINNs
from training import TrainingManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pattern_image(image_path, n=None):
    """Load and preprocess pattern image"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u_tp = (gray - gray.min()) / (gray.max() - gray.min())
    
    # Resize if n is specified and different from current size
    if n is not None and gray.shape[0] != n:
        u_tp = cv2.resize(u_tp, (n, n))
        logger.info(f"Resized image from {gray.shape} to ({n}, {n})")
    
    return u_tp

# Configuration
MODEL_CONFIG = {
    'nodes': 300,
    'n': 512,  # This should match your resolution of image
    'noise': 0,
    'learning_rates': [0.00001, 0.002],
    'length_app': 40000,
    'length_total': 600000,
    'batchsize': 128,
    'step_size': 2000,
    'model_name': "image_based_experiment",
    'sigma2': 10 / 50,
    'tol': 1e-7,
}

# PDE parameters (these are what we're trying to estimate)
C_ORIGINAL = [0.5, 0.406, 0.196]  # Reference values for comparison
L_DOMAIN = 50 # this should match the length of your picture

def get_domain_config(n):
    """Generate domain configuration for given n"""
    dx = L_DOMAIN / n
    xrange = yrange = np.linspace(0, L_DOMAIN, n, endpoint=True)
    return dx, xrange, yrange

def main():
    """Main execution function"""
    logger.info("Starting Swift-Hohenberg PINN training with image data")
    
    # Configuration
    n = MODEL_CONFIG['n']
    dx, xrange, yrange = get_domain_config(n)
    
    # Load pattern image
    image_path = "C:/Users/Zach Mollatt/Documents/Git/SwiftHohenberg-Inverse-Pinn/data/pattern_eps0.799_delta0.406_gamma0.196_PINN.png"  # Update this path
    logger.info(f"Loading pattern image from {image_path}")
    
    try:
        u_tp_toy = load_pattern_image(image_path, n=n)
        logger.info(f"Loaded image with shape: {u_tp_toy.shape}")
    except FileNotFoundError as e:
        logger.error(f"Image loading failed: {e}")
        # Generate a synthetic pattern as fallback
        logger.info("Generating synthetic pattern as fallback")
        from pde_utils import generate_synthetic_pattern
        u_tp_toy, _ = generate_synthetic_pattern(n, C_ORIGINAL, dx)
    
    # Prepare configuration
    config = MODEL_CONFIG.copy()
    config.update({
        'dx': dx,
        'xrange': xrange,
        'yrange': yrange,
        'u_n': u_tp_toy,
        'cmap': 'viridis'
    })
    
    # Train model
    logger.info("Initializing training manager")
    trainer = TrainingManager(RBF_PINNs, config)
    
    logger.info("Starting model training")
    trainer.train_model()
    
    # Generate plots and results
    logger.info("Generating plots and results")
    trainer.plot_results(C_ORIGINAL)
    
    # Simulate new pattern with estimated parameters
    logger.info("Simulating new pattern with estimated parameters")
    from pde_utils import step_forward2
    u_tp_new = trainer.simulate_new_pattern(step_forward2)
    
    # Save results
    logger.info("Saving results")
    trainer.save_results(u_tp_new, u_tp_toy, C_ORIGINAL)
    
    logger.info("Training and analysis complete!")

if __name__ == "__main__":
    main()