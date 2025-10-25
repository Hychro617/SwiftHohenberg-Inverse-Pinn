# Inverse Problems in Turing Patterns (IPTP)

This repository contains the code for the paper:  
> *“Unraveling biochemical spatial patterns: machine learning approaches to the inverse problem of stationary Turing patterns.”*

The framework provides machine learning–based methods for solving inverse problems associated with stationary Turing patterns, focusing on identifying system parameters that give rise to observed spatial structures.

---

## Overview

The paper explores the use of **Physics-Informed Neural Networks (PINNs)** and **Radial Basis Function (RBF) networks** to infer parameters of reaction–diffusion systems that produce characteristic Turing-type spatial patterns.  

This adaptation extends the original formulation to study the **Swift–Hohenberg equation**, a canonical model for spatial pattern formation and bifurcation analysis.

### Swift–Hohenberg Equation

$$
\frac{\partial u}{\partial t} = \varepsilon u - (1 + \nabla^2)^2 u - u^3
$$

where:

- `u(x, t)`: order parameter (scalar field)  
- `ε`: bifurcation or control parameter (learned by the network)  
- `∇²`: Laplacian operator  
- The nonlinear term `-u³` stabilises the emergent pattern amplitudes  

In this adaptation:
- **δ** and **γ** are fixed constants.  
- **ε (epsilon)** is the sole trainable parameter optimised to reproduce the observed steady-state patterns.

---

## Datasets

All datasets are generated **within the code** using forward simulations of the Swift–Hohenberg equation.  

- Initial conditions are generated as small perturbations around a homogeneous state.  
- Steady-state patterns are obtained by numerically integrating the PDE over time.  
- These generated patterns are then used to train the PINN to infer ε.

This approach allows full control over the data and ensures reproducibility of both numerical and synthetic experiments.

---

## Requirements

| Library | Version |
|---------|---------|
| Python | 3.10.11 |
| TensorFlow | 2.13.0 |
| NumPy | 1.24.3 |
| Matplotlib | 3.7.2 |
| SciPy | 1.11.1 |
| OpenCV | 4.8.1.78 |
| Shapely | 2.0.3 |

Install dependencies via:

```bash
pip install -r requirements.txt
