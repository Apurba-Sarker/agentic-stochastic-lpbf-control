"""
control_tools/calibration.py
=============================
Everything needed to go from calibration_for_printing.json → PathThermalInputs.

Public API
----------
load_calib(path)                      → dict
sample_theta(calib, seed)             → (eta, alpha_mult, z_scale)
make_inputs(calib, cfg, eta, alpha_mult, z_scale) → ptf.PathThermalInputs
make_inputs_from_calib(calib, cfg)    → (inputs, mode_str, eta, alpha_mult, z_scale)
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import path_temperature_field as ptf

try:
    from tools.config import MaterialProps
except Exception:
    MaterialProps = None

from .config import TrackConfig


def load_calib(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def sample_theta(calib: dict, seed: int) -> tuple[float, float, float]:
    """
    Draw one posterior sample (eta, alpha_mult, z_scale) from the calibration.
    Uses the Cholesky factor L if present and valid, else recomputes from Sigma.
    """
    mu = np.asarray(calib["stochastic_params"]["mu"], dtype=float)
    L  = np.asarray(calib["stochastic_params"].get("L"), dtype=float)

    if L.shape != (3, 3):
        Sigma = np.asarray(calib["stochastic_params"]["Sigma"], dtype=float)
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(3))

    rng = np.random.default_rng(seed)
    th  = mu + L @ rng.standard_normal(3)
    th[0] = np.clip(th[0], 0.15, 1.20)   # eta
    th[1] = np.clip(th[1], 0.60, 1.60)   # alpha_mult
    th[2] = np.clip(th[2], 0.15, 1.00)   # z_scale
    return float(th[0]), float(th[1]), float(th[2])


def make_inputs(
    calib: dict,
    cfg: TrackConfig,
    eta: float,
    alpha_mult: float,
    z_scale: float,
) -> ptf.PathThermalInputs:
    """Build a PathThermalInputs from calibration JSON + TrackConfig + sampled theta."""
    P_W     = float(calib["process_params"]["P_W"])
    V_mmps  = float(calib["process_params"]["V_mmps"])
    sigma_m = float(calib["process_params"]["sigma_m"])
    material = calib.get("material", "IN718")

    if MaterialProps is not None:
        mp = MaterialProps(name=str(material))
        RHO, CP, T0, T_MELT = float(mp.RHO), float(mp.CP), float(mp.T0), float(mp.T_MELT)
    else:
        # Hard-coded IN718-like fallback (matches your original scripts)
        RHO, CP, T0, T_MELT = 8190.0, 505.0, 300.0, 1600.0

    return ptf.PathThermalInputs(
        P_W=P_W, V_mmps=V_mmps, sigma_m=sigma_m,
        RHO=RHO, CP=CP, T0=T0, T_MELT=T_MELT,
        eta=eta, alpha_mult=alpha_mult, z_scale=z_scale,
    )


def make_inputs_from_calib(
    calib: dict,
    cfg: TrackConfig,
) -> tuple[ptf.PathThermalInputs, str, float, float, float]:
    """
    Convenience wrapper: samples (or uses mean) theta and returns
    (inputs, mode_str, eta, alpha_mult, z_scale).
    """
    mu = np.asarray(calib["stochastic_params"]["mu"], dtype=float)

    if cfg.stochastic_run:
        eta, alpha_mult, z_scale = sample_theta(calib, cfg.seed)
        mode = "posterior_draw"
    else:
        eta, alpha_mult, z_scale = float(mu[0]), float(mu[1]), float(mu[2])
        mode = "posterior_mean"

    inputs = make_inputs(calib, cfg, eta, alpha_mult, z_scale)
    return inputs, mode, eta, alpha_mult, z_scale
