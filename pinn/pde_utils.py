import numpy as np
from scipy.ndimage import laplace
from scipy.integrate import odeint

def step_forward1(y, t, c, dx):
    n = int(np.sqrt(len(y)))
    u = np.reshape(y, (n, n))
    epsilon, delta, gamma = c[0], c[1], c[2]

    laplacianu = laplace(u, mode="wrap") / dx**2
    biharmonic = laplace(laplacianu, mode="wrap") / dx**2
    dudt = epsilon*u - delta*u**2 - gamma*u**3 - (u + 2*laplacianu + biharmonic)

    return dudt.flatten()

def step_forward2(y, t, c, dx, reaction):
    n = int(np.sqrt(len(y)))
    u = np.reshape(y, (n, n))
    epsilon, delta, gamma = c[0], c[1], c[2]

    laplacianu = laplace(u, mode="wrap") / dx**2
    biharmonic = laplace(laplacianu, mode="wrap") / dx**2
    dudt = epsilon*u - delta*u**2 - gamma*u**3 - (u + 2*laplacianu + biharmonic)

    return dudt.flatten()

def sh_pat(u, mod_pars, dx):
    epsilon, delta, gamma = mod_pars[0], mod_pars[1], mod_pars[2]
    
    laplacian_u = laplace(u, mode="wrap") / dx**2
    biharmonic_u = laplace(laplacian_u, mode="wrap") / dx**2
    sh_operator = u + 2 * laplacian_u + biharmonic_u
    non_linear = epsilon * u - delta * u**2 - gamma * u**3
    f = non_linear - sh_operator
    
    return f

def generate_synthetic_pattern(n, c_original, dx):
    """Generate synthetic pattern as fallback if image loading fails"""
    L_domain = 10 * np.pi
    dx = L_domain / n
    
    u0 = 0.01 * np.ones(n**2)
    perturbation1 = np.random.normal(0, 0.01, (n**2))
    y0 = u0 + perturbation1
    
    tlen = 1000
    t = np.linspace(0, tlen, tlen)
    
    solb = odeint(step_forward2, y0, t, args=(c_original, dx, sh_pat))
    u_tp = np.reshape(solb[-1], (n, n))
    u_tp_toy = (u_tp - u_tp.min()) / (u_tp.max() - u_tp.min())
    
    return u_tp_toy, dx