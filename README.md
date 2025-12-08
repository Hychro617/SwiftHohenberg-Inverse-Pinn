# Inverse Swift–Hohenberg PINN (IPTP)

This repository contains a **Physics-Informed Neural Network (PINN)** implementation to solve the **inverse problem** for the **Swift–Hohenberg (SH) equation**, specifically calculating the control parameter $\varepsilon$ from digitally generated spatial patterns.

The methodology is detailed in the paper:
“Unraveling biochemical spatial patterns: machine learning approaches to the inverse problem of stationary Turing patterns.”

---

## Technical Overview and Innovation

The framework's core strength lies in its specialised architecture and **robust 3-Phase Training Strategy** coupled with **GradNorm dynamic loss weighting** to stabilise the inverse parameter solve.

### 1. RBF-Based PINN Architecture

This implementation utilises a **Radial Basis Function (RBF) Neural Network** to approximate the pattern field $u(x, y)$, rather than a traditional Multi-Layer Perceptron (MLP). The RBF architecture is chosen for its ability to produce **smoother spatial derivatives**, which is beneficial when enforcing the physics constraints of a Partial Differential Equation (PDE).

### 2. The 3-Phase Training Strategy

The training is strictly partitioned to sequentially introduce complexity, ensuring accurate function approximation before attempting parameter inversion.

* **Phase 1: Pure Data Fit** (Iterations $0$ to $20,000$).
    * **Goal:** The network is trained solely on the input pattern data, $u(x, y)$, to accurately approximate the observed steady-state pattern.
    * **Loss:** $\lambda_{\text{PDE}} = 0$ (Physics constraints are ignored).
* **Phase 2: Physics-Constrained Fit** (Iterations $20,000$ to $50,000$).
    * **Goal:** Introduce the PDE constraints to make the network's internal representation physically consistent.
    * **Parameter $\varepsilon$:** $\varepsilon$ remains fixed. Only network weights are updated.
* **Phase 3: Inverse Parameter Solve** (Iterations $50,000$ to $120,000$).
    * **Goal:** Simultaneously solve the parameter inverse problem.
    * **Action:** The control parameter $\varepsilon$ is **unfrozen and updated** alongside the network weights, minimising both data and full PDE residuals.

### 3. GradNorm Loss Weighting

To dynamically balance the influence of the multiple loss components (data loss $L_u$, and two PDE residuals $L_{\text{PDE1}}$, $L_{\text{PDE2}}$), the network implements the **GradNorm algorithm**.

This technique automatically adjusts the loss weights ($\lambda_i$) based on the magnitude of the gradients with respect to the network parameters, preventing any single loss term from dominating or stalling training.

---

## Swift–Hohenberg Equation (SH)

The target system is the canonical $4^{th}$ order SH equation, simplified to its **steady-state approximation**:

$$\frac{\partial u}{\partial t} = \varepsilon u - (1 + \nabla^2)^2 u - u^3 \approx 0$$

To calculate the PDE residual, the $4^{th}$ order equation is rewritten as two coupled $2^{nd}$ order equations, requiring the PINN to predict two fields: $u$ and an auxiliary variable $p$:

1.  $$p = \nabla^2 u$$ (Physics Constraint 1: Defining the auxiliary variable $p$)
2.  $$0 = \varepsilon u - p - \nabla^2 p - u^3$$ (Physics Constraint 2: Simplified steady-state SH residual)

| Parameter | Role in PINN | Value (Training) |
| :--- | :--- | :--- |
| $\varepsilon$ (Epsilon) | Trainable (Phase 3 only) | Starts at $\approx 0.5$ |
| $\delta$ (Delta) | Fixed Constant | $0.406$ |
| $\gamma$ (Gamma) | Fixed Constant | $0.196$ |

---

## Datasets and Data Handling

Pattern inputs are loaded from disk (`.npy` or image files), normalised, and then used to train the PINN.

* **Pattern Input:** A $64 \times 64$ grid of the pattern amplitude $u(x, y)$.
* **Data Preprocessing:** The `load_pattern_array` utility automatically loads, resizes, and normalises the input pattern to the standard range $[-1, 1]$.

---

## Requirements

The analysis and training code requires the following Python environment and library versions. **Note that these specific versions are critical for TensorFlow stability and reproducibility.**

| Library | Version |
| :--- | :--- |
| Python | $3.9.16$ |
| TensorFlow | $2.11.0$ |
| NumPy | $1.22.1$ |
| Matplotlib | $3.8.4$ |
| SciPy | $1.9.3$ |
| opencv-python | $4.9.0.80$ |
| Shapely | $2.0.3$ |

---
## Installation

Install all dependencies via a `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
