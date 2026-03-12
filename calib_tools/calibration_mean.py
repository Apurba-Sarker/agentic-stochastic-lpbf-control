"""
Deterministic mean-fit: Powell optimizer matching mean exp W and D.
Provides starting point for MCMC.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def deterministic_mean_fit(target_w_m, target_d_m, sigma_m, melt_dims_fn,
                            maxiter=250, verbose=True):
    """
    Fit (eta, alpha_mult, z_scale) to match mean experimental W and D.

    Returns
    -------
    best    : dict  {eta, alpha_mult, z_scale}
    history : pd.DataFrame  optimizer trace
    """
    history = []

    def obj(x):
        eta, alpha_mult, z_scale = x
        w_um, d_um = melt_dims_fn(np.array([[eta, alpha_mult, z_scale]], dtype=float), float(sigma_m))
        w = float(np.asarray(w_um)[0]) * 1e-6
        d = float(np.asarray(d_um)[0]) * 1e-6
        ew = (w - target_w_m) / (target_w_m + 1e-12)
        ed = (d - target_d_m) / (target_d_m + 1e-12)
        J  = float(ew**2 + ed**2)
        history.append({"eta": eta, "alpha_mult": alpha_mult, "z_scale": z_scale,
                         "width_um": w*1e6, "depth_um": d*1e6, "J": J})
        return J

    x0     = np.array([0.45, 1.0, 0.6], dtype=float)
    bounds = [(0.15, 1.2), (0.6, 1.6), (0.15, 1.0)]
    res    = minimize(obj, x0, method="Powell", bounds=bounds,
                      options={"maxiter": int(maxiter), "xtol": 1e-4, "ftol": 1e-6})

    if verbose:
        print(f"[mean-fit] J={res.fun:.3e}  "
              f"eta={res.x[0]:.4f}  alpha={res.x[1]:.4f}  z_scale={res.x[2]:.4f}")

    best = {"eta": float(res.x[0]), "alpha_mult": float(res.x[1]), "z_scale": float(res.x[2])}
    return best, pd.DataFrame(history)
