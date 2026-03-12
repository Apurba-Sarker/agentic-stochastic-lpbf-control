#!/usr/bin/env python3
"""
lshape_4layer_multilayer_depth.py

4-layer 3D L-shape LPBF simulation with optional closed-loop power control.

Features:
- 4-layer horizontal-hatch L-shape trajectory
- Multilayer 3D Goldak kernel (lower layers heat upper layers)
- Uncontrolled layer-by-layer depth evolution
- Analytical-inverse power controller with iterative spike correction
- 4x4 time-lapse temperature distributions
- All original depth/power plots plus before/after control comparison

Assumes:
- path_temperature_field.py is importable (same folder)
- calibration_for_printing.json exists (or edit CALIB_JSON)

Outputs (uncontrolled):
- lshape_xy_path_layers.pdf
- depth_vs_pathfraction_4layers.pdf
- depth_delta_vs_layer1.pdf
- depth_stats_by_layer.pdf
- depth_summary_by_layer.csv
- timelapse_temperature_4x4_layers.pdf

Outputs (control, only when RUN_CONTROL=True):
- ctrl_uncontrolled_vs_controlled_depth_by_layer.pdf
- ctrl_depth_stats_before_after.pdf
- ctrl_power_stats_by_layer.pdf
- ctrl_power_schedule_by_layer.pdf
- ctrl_summary_by_layer.csv
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import path_temperature_field as ptf


# ============================================================
# User-tunable settings
# ============================================================
N_LAYERS = 4

# Geometry (L-shape = [0,L]x[0,W] minus [notch_x,L]x[notch_y,W])
L_MM = 3.0
W_MM = 3.0
NOTCH_X_MM = 1.5
NOTCH_Y_MM = 1.5

# Path discretization
HATCH_SPACING_UM = 110.0
DX_UM = 20.0

# Layer stack
LAYER_THICKNESS_UM = 60.0

# Depth evaluation
BEHIND_UM = 40.0
EVAL_STRIDE = 10
# Z_MAX_DEPTH_UM: how far below the current layer surface to probe for the
# melt-pool boundary.  Must be larger than the expected max melt depth
# (~120-135 um) but small enough that the temperature field has dropped
# below T_MELT before hitting the bottom of the probe window.
# Setting this to 520 um (= full build height) was the root cause of
# artificially deep melt-pool readings: accumulated inter-layer heat kept
# T > T_MELT all the way to the substrate.  150 um (~2.5x layer thickness)
# is the correct physical window.
Z_MAX_UM = 150.0      # was 520.0 — FIXED: probe only ~2.5x layer thickness
Z_SAMPLES = 100       # was 320  — fewer samples needed over smaller window

# Thermal history filters
TAU_HIST_MS = None       # None = full inter-layer carryover; try 10-20 to limit
HISTORY_RADIUS_UM = 500.0

# Time-axis construction
TURN_SLOWDOWN = 1.05
TURN_ANGLE_DEG = 70.0
JUMP_THRESHOLD_UM = 80.0
LASER_OFF_FOR_JUMPS = True
INTERLAYER_DWELL_MS = 0.0

# Files
CALIB_JSON = "calibration_for_printing.json"
OUTDIR = Path("outputs_lshape_4layer")

# Time-lapse settings
TIMELAPSE_NX = 90
TIMELAPSE_NY = 90
TIMELAPSE_HISTORY_RADIUS_UM = 700.0
TIMELAPSE_TAU_HIST_MS = 20.0
TIMELAPSE_RASTER_MARGIN_MM = 0.05
TIMELAPSE_PROGRESS_DILATE_UM = 0.6 * HATCH_SPACING_UM
TIMELAPSE_VMAX_K = 2200.0

# -- Control -------------------------------------------------------
RUN_CONTROL = True   # set False to skip the controller entirely


@dataclass
class CtrlCfg:
    # depth grid — must match the corrected Z_MAX_UM / Z_SAMPLES above
    z_max_um: float = Z_MAX_UM   # 150 um: physically correct probe window
    z_samples: int = Z_SAMPLES   # 100 samples over 150 um = 1.5 um/step

    # thermal history for control evaluations
    history_radius_um: float = 400.0
    tau_hist_ms: float | None = 10.0
    behind_um: float = BEHIND_UM

    # time-axis (mirrors top-level settings)
    turn_slowdown: float = TURN_SLOWDOWN
    turn_angle_deg: float = TURN_ANGLE_DEG
    jump_threshold_um: float = JUMP_THRESHOLD_UM
    laser_off_for_jumps: bool = LASER_OFF_FOR_JUMPS
    interlayer_dwell_ms: float = INTERLAYER_DWELL_MS

    # controller knobs
    eval_stride: int = EVAL_STRIDE
    pmin_w: float = 80.0
    pmax_w: float = 400.0
    dp_max_w: float = 60.0
    spike_threshold_um: float = 1.5
    spike_fix_window: int = 12
    n_fix_passes: int = 5
    target_quantile: float = 0.55

    # iterative outer loop
    max_outer_passes: int = 8
    flat_tol_um: float = 1.5
    improve_tol_um: float = 0.15

    # jump-safe eval filtering
    skip_jump_neighbors: bool = True
    jump_neighbor_radius: int = 2
    skip_first_eval_point: bool = True


CTRL = CtrlCfg()


# ============================================================
# Plot style
# ============================================================
def setup_latex_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.dpi": 120,
                "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 12,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })


# ============================================================
# Inputs
# ============================================================
def load_inputs_from_calibration(calib_json_path: str):
    with open(calib_json_path, "r") as f:
        cal = json.load(f)
    mu = np.array(cal["stochastic_params"]["mu"], dtype=float)
    inputs = ptf.PathThermalInputs(
        P_W=float(cal["process_params"]["P_W"]),
        V_mmps=float(cal["process_params"]["V_mmps"]),
        sigma_m=float(cal["process_params"]["sigma_m"]),
        RHO=8190.0,
        CP=505.0,
        T0=300.0,
        T_MELT=1600.0,
        eta=float(mu[0]),
        alpha_mult=float(mu[1]),
        z_scale=float(mu[2]),
    )
    return inputs, cal


# ============================================================
# Trajectory
# ============================================================
def generate_horizontal_L_trajectory(
    length_mm=3.0,
    width_mm=3.0,
    notch_x_mm=1.5,
    notch_y_mm=1.5,
    hatch_spacing_um=110.0,
    dx_um=20.0,
):
    """
    Horizontal serpentine hatch for L-shaped footprint.
    Returns (N, 2) trajectory in meters.

    Each new hatch line is added as a single first point (no interpolated connector).
    This keeps the inter-line gap as one 110 um step, which ptf.build_time_axis_from_trajectory
    detects as a laser-off jump (jump_threshold_um=80 um). Interpolating the connector
    into 20 um micro-steps would fool the jump detector and heat the repositioning path.
    """
    Lm = length_mm * 1e-3
    Wm = width_mm * 1e-3
    notch_x = notch_x_mm * 1e-3
    notch_y = notch_y_mm * 1e-3
    hatch = hatch_spacing_um * 1e-6
    dx = dx_um * 1e-6

    ys = np.arange(0.0, Wm + 1e-15, hatch)
    traj = []

    for i, y in enumerate(ys):
        intervals = [(0.0, Lm)] if y <= notch_y + 1e-12 else [(0.0, notch_x)]
        left_to_right = (i % 2 == 0)

        for (xa, xb) in intervals:
            xs = (
                np.arange(xa, xb + 1e-15, dx)
                if left_to_right
                else np.arange(xb, xa - 1e-15, -dx)
            )
            # Single jump point: dist > jump_threshold -> laser off during reposition
            traj.append([float(xs[0]), float(y)])
            for x in xs[1:]:
                traj.append([float(x), float(y)])

    return np.asarray(traj, dtype=float)


# ============================================================
# Multilayer source construction
# ============================================================
def build_multilayer_sources(
    base_traj_xy,
    n_layers,
    layer_thickness_um,
    inputs,
    *,
    turn_slowdown=1.05,
    turn_angle_deg=70.0,
    jump_threshold_um=80.0,
    laser_off_for_jumps=True,
    interlayer_dwell_ms=0.0,
):
    """
    Stack the same XY trajectory at successive z levels.
    Returns (xyz_all, t_all, dt_all, layer_start_idxs, N, z_step_m, t_layer_s,
             t_local, dt_local).
    """
    base_traj_xy = np.asarray(base_traj_xy, dtype=float)

    t_local, dt_local = ptf.build_time_axis_from_trajectory(
        base_traj_xy,
        inputs.V_mmps,
        turn_slowdown=turn_slowdown,
        turn_angle_deg=turn_angle_deg,
        jump_threshold_um=jump_threshold_um,
        laser_off_for_jumps=laser_off_for_jumps,
    )

    N = base_traj_xy.shape[0]
    z_step = layer_thickness_um * 1e-6

    xyz_blocks, t_blocks, dt_blocks = [], [], []
    layer_start_idxs = []
    t_offset = 0.0

    for ell in range(n_layers):
        z = ell * z_step
        xyz = np.column_stack([base_traj_xy, np.full(N, z, dtype=float)])
        xyz_blocks.append(xyz)
        t_blocks.append(t_local + t_offset)
        dt_blocks.append(dt_local.copy())
        layer_start_idxs.append(ell * N)
        t_offset = float(t_offset + t_local[-1] + interlayer_dwell_ms * 1e-3)

    return (
        np.vstack(xyz_blocks),
        np.concatenate(t_blocks),
        np.concatenate(dt_blocks),
        np.asarray(layer_start_idxs, dtype=int),
        N,
        z_step,
        float(t_local[-1]),
        np.asarray(t_local),
        np.asarray(dt_local),
    )


# ============================================================
# Core depth kernel (3D multilayer)
# ============================================================
def depth_profile_multilayer_at_point(
    xyz_all,
    t_all,
    dt_all,
    eval_global_idx,
    inputs,
    z_probe_um,
    *,
    layer_local_traj,
    layer_start_idx,
    history_radius_um=300.0,
    tau_hist_ms=2.0,
    behind_um=40.0,
    P_src_W=None,
):
    """
    Evaluate melt-pool depth at one point using the full multilayer 3D source history.
    Depth is positive downward from the current layer surface.
    Returns (Tz, depth_um).
    """
    local_idx = int(eval_global_idx - layer_start_idx)
    p_xy = ptf._point_for_depth_eval(layer_local_traj, local_idx, behind_um * 1e-6)
    z_surface = float(xyz_all[eval_global_idx, 2])
    p_eval = np.array([float(p_xy[0]), float(p_xy[1]), z_surface], dtype=float)

    src_xyz = xyz_all[: eval_global_idx + 1]
    t_s = t_all[: eval_global_idx + 1]
    w_s = dt_all[: eval_global_idx + 1].copy()
    t_eval = float(t_all[eval_global_idx])

    r_hist = float(history_radius_um) * 1e-6
    keep = np.sum((src_xyz - p_eval[None, :]) ** 2, axis=1) <= (r_hist * r_hist)
    if tau_hist_ms is not None:
        keep &= (t_eval - t_s) <= (float(tau_hist_ms) * 1e-3)
    if not np.any(keep):
        keep[-1] = True

    src_xyz = src_xyz[keep]
    t_s = t_s[keep]
    w_s = w_s[keep]

    if P_src_W is not None:
        P0 = max(float(inputs.P_W), 1e-12)
        P_s = np.asarray(P_src_W[: eval_global_idx + 1], dtype=float)[keep]
        w_s = w_s * (P_s / P0)

    k_val = ptf.get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = ptf._prefactor(inputs.P_W, inputs.RHO, inputs.CP) * inputs.eta

    dt_arr = np.maximum(t_eval - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m ** 2
    lam_pow = np.power(lam, 1.5)

    z_probe_um = np.asarray(z_probe_um, dtype=float)
    z_abs = z_surface + z_probe_um * 1e-6

    dx2 = (p_eval[0] - src_xyz[:, 0]) ** 2
    dy2 = (p_eval[1] - src_xyz[:, 1]) ** 2
    dz2 = ((z_abs[:, None] - src_xyz[None, :, 2]) * inputs.z_scale) ** 2

    r2 = dx2[None, :] + dy2[None, :] + dz2
    term = np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :]
    Tz = inputs.T0 + pref * np.sum(term * w_s[None, :], axis=1)

    above = Tz >= inputs.T_MELT
    if not np.any(above):
        return Tz, 0.0

    idx_last = int(np.max(np.where(above)[0]))
    if idx_last >= len(z_probe_um) - 1:
        return Tz, float(z_probe_um[-1])

    T1, T2 = float(Tz[idx_last]), float(Tz[idx_last + 1])
    z1, z2 = float(z_probe_um[idx_last]), float(z_probe_um[idx_last + 1])
    frac = (inputs.T_MELT - T1) / (T2 - T1) if abs(T2 - T1) > 1e-12 else 0.0
    return Tz, float(z1 + frac * (z2 - z1))


# ============================================================
# Depth evolution (uncontrolled)
# ============================================================
def depth_evolution_for_layer(
    base_traj_xy,
    xyz_all,
    t_all,
    dt_all,
    inputs,
    layer_idx,
    n_local,
    z_probe_um,
    dt_local,
    *,
    eval_stride=10,
    history_radius_um=300.0,
    tau_hist_ms=2.0,
    behind_um=40.0,
):
    """
    Compute depth vs path-fraction for one layer.
    Skips eval points where behind_um would land outside the trajectory
    (first point and first-point-of-each-line after a jump) and forward-fills.
    """
    start = layer_idx * n_local
    idxs_local = np.arange(0, n_local, int(eval_stride), dtype=int)

    # Mark eval points that are too close to a line start to give a valid depth.
    # The first point of each hatch line is a laser-off jump point (behind_um would
    # land outside the domain).  But even a few points in, the melt pool hasn't
    # re-established: an eval only 2-3 points into a line reads a spuriously low
    # depth because the thermal history is too shallow.  Skip the first eval_stride
    # points after every jump so the first evaluated point always has at least
    # eval_stride source points of history on the new line.
    seg_dists = np.linalg.norm(np.diff(base_traj_xy, axis=0), axis=1)
    skip_mask = np.zeros(n_local, dtype=bool)
    skip_mask[0] = True
    jump_locs = np.where(np.concatenate([[False], seg_dists > (JUMP_THRESHOLD_UM * 1e-6)]))[0]
    for j in jump_locs:
        skip_mask[j: min(n_local, j + eval_stride + 1)] = True

    depths = np.full(len(idxs_local), np.nan, dtype=float)
    peak_temps = np.zeros(len(idxs_local), dtype=float)

    for k, li in enumerate(idxs_local):
        if skip_mask[li]:
            continue   # too close to line start — forward-fill below
        Tz, d = depth_profile_multilayer_at_point(
            xyz_all=xyz_all,
            t_all=t_all,
            dt_all=dt_all,
            eval_global_idx=int(start + li),
            inputs=inputs,
            z_probe_um=z_probe_um,
            layer_local_traj=base_traj_xy,
            layer_start_idx=start,
            history_radius_um=history_radius_um,
            tau_hist_ms=tau_hist_ms,
            behind_um=behind_um,
        )
        depths[k] = d
        peak_temps[k] = float(np.max(Tz))

    # Forward-fill NaN startup values from depth restarts
    fv = next((depths[k] for k in range(len(depths)) if not np.isnan(depths[k])), 0.0)
    for k in range(len(depths)):
        if np.isnan(depths[k]):
            depths[k] = fv
        else:
            fv = depths[k]

    frac = idxs_local / max(n_local - 1, 1)
    return idxs_local, frac, depths, peak_temps


# ============================================================
# Time-lapse helpers
# ============================================================
def _temperature_field_xy_on_plane_at_time(
    xyz_all, t_all, dt_all, inputs, t_eval, z_plane_m, X_m, Y_m,
    *, history_radius_um=1200.0, tau_hist_ms=None, P_src_W=None, chunk_size=2000,
):
    src_mask = t_all <= float(t_eval)
    if not np.any(src_mask):
        return np.full_like(X_m, inputs.T0, dtype=float)

    src_xyz = xyz_all[src_mask]
    t_s = t_all[src_mask]
    w_s = dt_all[src_mask].copy()

    if P_src_W is not None:
        P0 = max(float(inputs.P_W), 1e-12)
        w_s = w_s * (np.asarray(P_src_W, dtype=float)[src_mask] / P0)

    if tau_hist_ms is not None:
        keep_t = (float(t_eval) - t_s) <= (float(tau_hist_ms) * 1e-3)
        if np.any(keep_t):
            src_xyz, t_s, w_s = src_xyz[keep_t], t_s[keep_t], w_s[keep_t]

    r_hist = float(history_radius_um) * 1e-6
    xmin, xmax = float(np.min(X_m)), float(np.max(X_m))
    ymin, ymax = float(np.min(Y_m)), float(np.max(Y_m))
    z0 = float(z_plane_m)

    dx_box = np.maximum.reduce([xmin - src_xyz[:, 0], np.zeros(len(src_xyz)), src_xyz[:, 0] - xmax])
    dy_box = np.maximum.reduce([ymin - src_xyz[:, 1], np.zeros(len(src_xyz)), src_xyz[:, 1] - ymax])
    dz_box = np.abs(z0 - src_xyz[:, 2]) * float(inputs.z_scale)
    keep_s = (dx_box**2 + dy_box**2 + dz_box**2) <= r_hist**2
    if np.any(keep_s):
        src_xyz, t_s, w_s = src_xyz[keep_s], t_s[keep_s], w_s[keep_s]
    else:
        src_xyz, t_s, w_s = src_xyz[-1:], t_s[-1:], w_s[-1:]

    k_val = ptf.get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = ptf._prefactor(inputs.P_W, inputs.RHO, inputs.CP) * inputs.eta
    dt_arr = np.maximum(float(t_eval) - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m ** 2
    lam_pow = lam ** 1.5

    sx, sy, sz = src_xyz[:, 0], src_xyz[:, 1], src_xyz[:, 2]
    z_scale = float(inputs.z_scale)
    Xf, Yf = X_m.ravel(), Y_m.ravel()
    Tf = np.empty(Xf.size, dtype=float)

    for i0 in range(0, Xf.size, chunk_size):
        i1 = min(i0 + chunk_size, Xf.size)
        Xc = Xf[i0:i1]
        Yc = Yf[i0:i1]
        Zc = np.full(i1 - i0, z0)
        r2 = ((Xc[:, None] - sx)**2 + (Yc[:, None] - sy)**2
              + ((Zc[:, None] - sz) * z_scale)**2)
        Tf[i0:i1] = inputs.T0 + pref * np.sum(
            np.exp(-3.0 * r2 / lam) / lam_pow * w_s, axis=1)

    return Tf.reshape(X_m.shape)


def _visited_mask_from_path_points(X_m, Y_m, path_xy_m, radius_um):
    if path_xy_m is None or len(path_xy_m) == 0:
        return np.zeros_like(X_m, dtype=bool)
    pts = np.asarray(path_xy_m, dtype=float)
    r2 = (float(radius_um) * 1e-6) ** 2
    Xf, Yf = X_m.ravel(), Y_m.ravel()
    mask = np.zeros(Xf.shape[0], dtype=bool)
    for i in range(0, len(pts), 600):
        P = pts[i:i + 600]
        mask |= np.any(
            (Xf[:, None] - P[None, :, 0])**2 + (Yf[:, None] - P[None, :, 1])**2 <= r2,
            axis=1,
        )
        if np.all(mask):
            break
    return mask.reshape(X_m.shape)


def _lshape_footprint_mask(X_m, Y_m, L_mm, W_mm, notch_x_mm, notch_y_mm):
    x_mm, y_mm = X_m * 1e3, Y_m * 1e3
    in_box = (x_mm >= 0) & (x_mm <= L_mm) & (y_mm >= 0) & (y_mm <= W_mm)
    in_notch = (x_mm >= notch_x_mm) & (x_mm <= L_mm) & (y_mm >= notch_y_mm) & (y_mm <= W_mm)
    return in_box & (~in_notch)


def _snapshot_time_and_progress_index(t_all, n_local, layer_idx):
    s, e = layer_idx * n_local, (layer_idx + 1) * n_local
    t_mid = 0.5 * (float(t_all[s]) + float(t_all[e - 1]))
    g_idx = int(np.searchsorted(t_all, t_mid, side="right") - 1)
    return t_mid, max(0, min(g_idx, len(t_all) - 1))


def make_timelapse_4x4_temperature_plot(xyz_all, t_all, dt_all, base_traj, n_local, inputs, outpath):
    setup_latex_style()

    margin = TIMELAPSE_RASTER_MARGIN_MM
    x_mm = np.linspace(-margin, L_MM + margin, TIMELAPSE_NX)
    y_mm = np.linspace(-margin, W_MM + margin, TIMELAPSE_NY)
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
    X_m, Y_m = X_mm * 1e-3, Y_mm * 1e-3

    lshape_mask = _lshape_footprint_mask(X_m, Y_m, L_MM, W_MM, NOTCH_X_MM, NOTCH_Y_MM)

    fig, axes = plt.subplots(4, 4, figsize=(13.6, 12.2), constrained_layout=True)
    last_im = None

    # colormap with white for masked / nonexistent region
    cmap = plt.cm.get_cmap("hot").copy()
    cmap.set_bad(color="white")

    for r in range(4):
        t_snap, g_idx = _snapshot_time_and_progress_index(t_all, n_local, r)

        for c in range(4):
            ax = axes[r, c]
            z_plane_m = c * (LAYER_THICKNESS_UM * 1e-6)

            T = _temperature_field_xy_on_plane_at_time(
                xyz_all,
                t_all,
                dt_all,
                inputs,
                t_snap,
                z_plane_m,
                X_m,
                Y_m,
                history_radius_um=TIMELAPSE_HISTORY_RADIUS_UM,
                tau_hist_ms=TIMELAPSE_TAU_HIST_MS,
                chunk_size=1500,
            )

            # which part of this plane physically exists / has been reached
            if c > r:
                exists_mask = np.zeros_like(lshape_mask, dtype=bool)
            elif c < r:
                exists_mask = lshape_mask.copy()
            else:
                s, e = c * n_local, (c + 1) * n_local
                if g_idx >= s:
                    local_cut = min(g_idx, e - 1) - s
                    pts = base_traj[: local_cut + 1]
                else:
                    pts = np.empty((0, 2))

                exists_mask = (
                    _visited_mask_from_path_points(
                        X_m, Y_m, pts, TIMELAPSE_PROGRESS_DILATE_UM
                    ) & lshape_mask
                )

            # mask nonexistent region so it becomes white instead of black
            T_plot = np.ma.masked_where(~exists_mask, T)

            im = ax.imshow(
                T_plot,
                extent=[x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()],
                origin="lower",
                vmin=0.0,
                vmax=TIMELAPSE_VMAX_K,
                cmap=cmap,
                interpolation="nearest",
                aspect="equal",
            )
            last_im = im

            # L-shape outline
            ax.plot(
                [0, L_MM, L_MM, NOTCH_X_MM, NOTCH_X_MM, 0, 0],
                [0, 0, NOTCH_Y_MM, NOTCH_Y_MM, W_MM, W_MM, 0],
                color="black",
                linewidth=1.0,
            )

            # minimal titles only on top row
            if r == 0:
                ax.set_title(rf"Layer {c+1}", fontsize=20)

            # minimal row label only on first column
            if c == 0:
                ax.set_ylabel(rf"$t={1e3*t_snap:.0f}\,\mathrm{{ms}}$", fontsize=20)

            ax.set_xlim(x_mm.min(), x_mm.max())
            ax.set_ylim(y_mm.min(), y_mm.max())

            # bottom row x-labels only
            if r < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$x$ (mm)", fontsize=20)

            # only first column keeps y ticks
            if c > 0:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=20)

    # one global y label instead of repeating inside every first-column axis
    fig.text(0.035, 0.5, r"", rotation=90, va="center", ha="center", fontsize=20)

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.015)
    cbar.set_label(r"Temperature (K)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Controller helpers
# ============================================================
def _build_valid_eval_indices(n_local, dt_local, cfg: CtrlCfg):
    """Return eval indices that are jump-safe (skip first point and jump neighbours)."""
    idxs = np.arange(0, n_local, int(cfg.eval_stride), dtype=int)
    if not cfg.skip_jump_neighbors:
        return idxs
    invalid = np.zeros(n_local, dtype=bool)
    for j in np.where(np.asarray(dt_local) <= 0.0)[0]:
        # Before jump: guard jump_neighbor_radius points (end-of-line spike region).
        # After jump: guard eval_stride + jump_neighbor_radius points so the first
        # eval on the new line has at least eval_stride source points of thermal history.
        # Without this the first eval lands only 2-3 pts in and reads a spuriously
        # low depth (cold-start artifact visible as the unrealistic dip in the plots).
        invalid[max(0, int(j) - cfg.jump_neighbor_radius):
                min(n_local, int(j) + cfg.eval_stride + cfg.jump_neighbor_radius + 1)] = True
    if cfg.skip_first_eval_point:
        invalid[0] = True
    kept = [i for i in idxs if not invalid[i]]
    return np.asarray(kept if kept else [min(cfg.eval_stride, n_local - 1)], dtype=int)


def _clip_power(P, P_ref, cfg: CtrlCfg):
    P = float(np.clip(P, cfg.pmin_w, cfg.pmax_w))
    P = float(np.clip(P, P_ref - cfg.dp_max_w, P_ref + cfg.dp_max_w))
    return float(np.clip(P, cfg.pmin_w, cfg.pmax_w))


def _depth_at(xyz_all, t_all, dt_all, inputs, z_um, layer_idx, n_local,
              local_idx, P_all, cfg: CtrlCfg, base_traj):
    gi = layer_idx * n_local + int(local_idx)
    return depth_profile_multilayer_at_point(
        xyz_all, t_all, dt_all, gi, inputs, z_um,
        layer_local_traj=base_traj,
        layer_start_idx=layer_idx * n_local,
        history_radius_um=cfg.history_radius_um,
        tau_hist_ms=cfg.tau_hist_ms,
        behind_um=cfg.behind_um,
        P_src_W=P_all,
    )


def _analytical_inverse(xyz_all, t_all, dt_all, inputs, z_um, iz_target,
                        layer_idx, n_local, predict_local_idx,
                        ctrl_start, ctrl_end, P_all, P_prev, cfg: CtrlCfg, base_traj):
    """One analytical-inverse power update for one control window."""
    predict_local_idx = min(int(predict_local_idx), n_local - 1)

    Tz_full, _ = _depth_at(xyz_all, t_all, dt_all, inputs, z_um,
                            layer_idx, n_local, predict_local_idx, P_all, cfg, base_traj)
    T_full = float(Tz_full[iz_target])

    P_zero = P_all.copy()
    gs = layer_idx * n_local + ctrl_start
    ge = layer_idx * n_local + ctrl_end
    P_zero[gs:ge] = 0.0

    Tz_fix, _ = _depth_at(xyz_all, t_all, dt_all, inputs, z_um,
                           layer_idx, n_local, predict_local_idx, P_zero, cfg, base_traj)
    T_fix = float(Tz_fix[iz_target])

    T_ctrl = T_full - T_fix
    if abs(T_ctrl) < 0.5:
        return float(P_all[gs])

    P_cur = float(P_all[gs]) if float(P_all[gs]) >= 1.0 else float(inputs.P_W)
    return _clip_power(P_cur * (inputs.T_MELT - T_fix) / T_ctrl, P_prev, cfg)


def _compute_depths(xyz_all, t_all, dt_all, inputs, z_um, layer_idx, n_local,
                    idxs_local, P_all, cfg: CtrlCfg, base_traj):
    depths = np.zeros(len(idxs_local), dtype=float)
    for k, li in enumerate(idxs_local):
        _, d = _depth_at(xyz_all, t_all, dt_all, inputs, z_um,
                         layer_idx, n_local, int(li), P_all, cfg, base_traj)
        depths[k] = d
    return depths


def run_controller_one_layer(xyz_all, t_all, dt_all, inputs, layer_idx, n_local,
                              z_um, cfg: CtrlCfg, P_all, dt_local, base_traj):
    """
    Two-pass analytical-inverse controller for one layer, with iterative outer loop.

    Pass 1  stride-by-stride inverse update.
    Pass 2  symmetric spike correction (fixes both dips and peaks).
    Outer   repeat until max|d - target| < flat_tol_um or improvement stalls.

    Key fix: _safe_predict_idx() prevents predict_idx from crossing a hatch-line
    boundary.  Without this guard, predict_idx lands on the cold start of the next
    line, making the inverse see artificially low T and over-boost power, which
    creates the persistent depth dip observed in earlier runs.
    """
    P0 = float(inputs.P_W)
    layer_slice = slice(layer_idx * n_local, (layer_idx + 1) * n_local)

    idxs_eval = _build_valid_eval_indices(n_local, dt_local, cfg)

    # Precompute segment distances for line-boundary clamping
    _seg_dists = np.linalg.norm(np.diff(base_traj, axis=0), axis=1)
    _jump_m = cfg.jump_threshold_um * 1e-6

    def _safe_predict_idx(ctrl_start, ctrl_end):
        """Last index in [ctrl_start, ctrl_end) that stays on the same hatch line.
        Stops before any segment that is a jump (dist > jump_threshold), ensuring
        predict_idx never crosses into the next (cold) line."""
        best = int(ctrl_start)
        for k in range(int(ctrl_start), min(int(ctrl_end), n_local)):
            if k < len(_seg_dists) and _seg_dists[k] > _jump_m:
                break
            best = k
        return best

    # Uncontrolled baseline
    depths_before = _compute_depths(xyz_all, t_all, dt_all, inputs, z_um,
                                    layer_idx, n_local, idxs_eval, P_all, cfg, base_traj)

    # Target: 0.55-quantile (skip first 5 evals for startup transient)
    arr = depths_before[5:] if len(depths_before) > 5 else depths_before
    target = float(np.quantile(arr, cfg.target_quantile))
    iz_target = int(np.argmin(np.abs(z_um - target)))

    prev_max_err = None
    for outer in range(cfg.max_outer_passes):

        # -- Pass 1: stride-by-stride inverse ---------------------------------
        P_prev = float(np.mean(P_all[layer_slice])) if np.any(P_all[layer_slice] > 0) else P0
        for idx in idxs_eval:
            idx = int(idx)
            ctrl_start = idx
            ctrl_end = min(n_local, idx + cfg.eval_stride)
            predict_idx = _safe_predict_idx(ctrl_start, ctrl_end)
            P_new = _analytical_inverse(
                xyz_all, t_all, dt_all, inputs, z_um, iz_target,
                layer_idx, n_local, predict_idx,
                ctrl_start, ctrl_end, P_all, P_prev, cfg, base_traj,
            )
            gs = layer_idx * n_local + ctrl_start
            ge = layer_idx * n_local + ctrl_end
            P_all[gs:ge] = P_new
            P_prev = P_new

        # -- Pass 2: symmetric spike correction -------------------------------
        # Precompute line_start[i] = index of the most recent jump point at or
        # before i.  Used to prevent the spike-fix ctrl window from reaching back
        # across a line boundary into the previous hatch line, which would corrupt
        # already-correct power values and create the residual dip artifact.
        _line_start = np.zeros(n_local, dtype=int)
        _cur_ls = 0
        for _i in range(n_local):
            if dt_local[_i] <= 0:
                _cur_ls = _i
            _line_start[_i] = _cur_ls

        for _ in range(cfg.n_fix_passes):
            depths_now = _compute_depths(xyz_all, t_all, dt_all, inputs, z_um,
                                         layer_idx, n_local, idxs_eval, P_all, cfg, base_traj)
            spikes = idxs_eval[
                (depths_now < target - cfg.spike_threshold_um) |
                (depths_now > target + cfg.spike_threshold_um)
            ]
            if len(spikes) == 0:
                break
            for sp in spikes:
                sp = int(sp)
                offset = cfg.spike_fix_window // 3
                cs = max(0, sp - cfg.spike_fix_window + offset)
                # Clamp cs so it never reaches back past the start of the current
                # hatch line.  Without this, the window spans the inter-line jump,
                # zeroing heat on the previous line and corrupting its power schedule.
                cs = max(cs, int(_line_start[sp]))
                ce = min(n_local, sp + offset)
                if ce <= cs:
                    continue
                Tz_f, _ = _depth_at(xyz_all, t_all, dt_all, inputs, z_um,
                                     layer_idx, n_local, sp, P_all, cfg, base_traj)
                T_f = float(Tz_f[iz_target])
                P_zero = P_all.copy()
                gs = layer_idx * n_local + cs
                ge = layer_idx * n_local + ce
                P_zero[gs:ge] = 0.0
                Tz_fix, _ = _depth_at(xyz_all, t_all, dt_all, inputs, z_um,
                                       layer_idx, n_local, sp, P_zero, cfg, base_traj)
                T_fix = float(Tz_fix[iz_target])
                T_ctrl = T_f - T_fix
                if abs(T_ctrl) < 0.5:
                    continue
                P_cur = float(np.mean(P_all[gs:ge])) if gs < ge else P0
                P_all[gs:ge] = _clip_power(P_cur * (inputs.T_MELT - T_fix) / T_ctrl, P_cur, cfg)

        # -- Convergence check ------------------------------------------------
        depths_iter = _compute_depths(xyz_all, t_all, dt_all, inputs, z_um,
                                      layer_idx, n_local, idxs_eval, P_all, cfg, base_traj)
        max_err = float(np.max(np.abs(depths_iter - target)))
        rms_err = float(np.sqrt(np.mean((depths_iter - target) ** 2)))
        print(f"  outer {outer+1:02d}: target={target:.2f} um | "
              f"max|d-target|={max_err:.3f} um | rms={rms_err:.3f} um")
        if max_err <= cfg.flat_tol_um:
            break
        if prev_max_err is not None and (prev_max_err - max_err) < cfg.improve_tol_um:
            break
        prev_max_err = max_err

    depths_after = _compute_depths(xyz_all, t_all, dt_all, inputs, z_um,
                                   layer_idx, n_local, idxs_eval, P_all, cfg, base_traj)
    return P_all, idxs_eval, depths_before, depths_after, target


def run_multilayer_controller(base_traj, xyz_all, t_all, dt_all, inputs,
                               n_local, dt_local, cfg: CtrlCfg):
    """Control all layers sequentially; later layers see the corrected power history."""
    z_um = np.linspace(0.0, cfg.z_max_um, cfg.z_samples)
    P_all = np.full_like(t_all, float(inputs.P_W), dtype=float)
    results = []

    for ell in range(N_LAYERS):
        print(f"\n=== Layer {ell+1}/{N_LAYERS} ===")
        P_all, idxs_u, d_before, d_after, target = run_controller_one_layer(
            xyz_all, t_all, dt_all, inputs, ell, n_local,
            z_um, cfg, P_all, dt_local, base_traj,
        )
        frac = idxs_u / max(n_local - 1, 1)
        P_layer = P_all[ell * n_local:(ell + 1) * n_local].copy()
        print(f"  Target={target:.2f} um | "
              f"Before {d_before.min():.1f}..{d_before.max():.1f} um | "
              f"After  {d_after.min():.1f}..{d_after.max():.1f} um")
        results.append({
            "layer": ell + 1,
            "idxs_eval": idxs_u,
            "frac_eval": frac,
            "depth_before_um": d_before,
            "depth_after_um": d_after,
            "target_um": target,
            "P_layer_W": P_layer,
            "P_mean_W": float(np.mean(P_layer)),
            "P_min_W": float(np.min(P_layer)),
            "P_max_W": float(np.max(P_layer)),
        })

    return P_all, results


# ============================================================
# Plotting
# ============================================================
def save_uncontrolled_plots(results_unc, xyz_all, t_all, dt_all,
                            base_traj, n_local, inputs, outdir):
    setup_latex_style()
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV summary
    rows = [{
        "Layer": r["layer"],
        "Mean depth (um)": float(np.mean(r["depth_um"])),
        "Min depth (um)":  float(np.min(r["depth_um"])),
        "Max depth (um)":  float(np.max(r["depth_um"])),
        "Depth std (um)":  float(np.std(r["depth_um"])),
    } for r in results_unc]
    pd.DataFrame(rows).to_csv(outdir / "depth_summary_by_layer.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # Plot 1: XY path
    plt.figure(figsize=(7.2, 6.2))
    for ell in range(N_LAYERS):
        s = ell * n_local
        e = (ell + 1) * n_local
        xy = xyz_all[s:e, :2] * 1e3  # mm
        plt.plot(xy[:, 0], xy[:, 1], color="black", linewidth=0.8)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"$x$ (mm)")
    plt.ylabel(r"$y$ (mm)")
    #plt.legend(); 
    plt.tight_layout()
    plt.savefig(outdir / "lshape_xy_path_layers.pdf", bbox_inches="tight")
    plt.savefig(outdir / "lshape_xy_path_layers.png", dpi=220, bbox_inches="tight"); plt.close()

    # Plot 2: Depth vs path fraction
    plt.figure(figsize=(8.2, 4.8))
    for r in results_unc:
        plt.plot(r["frac"], r["depth_um"], linewidth=1.5, label=rf"Layer {r['layer']}")
    plt.xlabel(r"Path fraction"); plt.ylabel(r"Melt-pool depth ($\mu$m)")
    plt.title(r"Predicted melt-pool depth vs path fraction (4 layers, same power)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "depth_vs_pathfraction_4layers.pdf", bbox_inches="tight")
    plt.savefig(outdir / "depth_vs_pathfraction_4layers.png", dpi=220, bbox_inches="tight"); plt.close()

    # Plot 3: Depth delta vs layer 1
    base_depth, base_frac = results_unc[0]["depth_um"], results_unc[0]["frac"]
    plt.figure(figsize=(8.2, 4.8))
    for r in results_unc[1:]:
        plt.plot(base_frac, r["depth_um"] - base_depth, linewidth=1.4,
                 label=rf"Layer {r['layer']} $-$ Layer 1")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(r"Path fraction"); plt.ylabel(r"Depth change ($\mu$m)")
    # plt.title(r"Layer-to-layer depth increase relative to Layer 1")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "depth_delta_vs_layer1.pdf", bbox_inches="tight")
    plt.savefig(outdir / "depth_delta_vs_layer1.png", dpi=220, bbox_inches="tight"); plt.close()

    # Plot 4: Stats by layer
    layers = np.arange(1, N_LAYERS + 1)
    plt.figure(figsize=(6.8, 4.8))
    plt.plot(layers, [np.mean(r["depth_um"]) for r in results_unc],
             marker="o", linewidth=1.8, label="Mean")
    plt.plot(layers, [np.max(r["depth_um"]) for r in results_unc],
             marker="o", linewidth=1.4, label="Max")
    plt.plot(layers, [np.min(r["depth_um"]) for r in results_unc],
             marker="o", linewidth=1.4, label="Min")
    plt.xticks(layers); plt.xlabel(r"Layer"); plt.ylabel(r"Depth ($\mu$m)")
    plt.title(r"Depth statistics by layer"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "depth_stats_by_layer.pdf", bbox_inches="tight")
    plt.savefig(outdir / "depth_stats_by_layer.png", dpi=220, bbox_inches="tight"); plt.close()

    # Plot 5: Timelapse
    make_timelapse_4x4_temperature_plot(
        xyz_all, t_all, dt_all, base_traj, n_local, inputs,
        outpath=outdir / "timelapse_temperature_4x4_layers.pdf",
    )

def compute_case_stats(depth_um, startup_skip=5, spike_threshold_um=1.5):
    """
    Compute publication-style depth statistics from one depth trace.

    Returns:
      baseline_um
      std_um
      mean_spike_peak_um
      mean_spike_overshoot_um
      spike_count
    """
    d = np.asarray(depth_um, dtype=float)

    if len(d) <= startup_skip:
        arr = d.copy()
    else:
        arr = d[startup_skip:].copy()

    baseline = float(np.mean(arr))
    std_val = float(np.std(arr))

    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            if arr[i] > baseline + spike_threshold_um:
                peaks.append(arr[i])

    if len(peaks) == 0:
        mean_peak = baseline
        mean_overshoot = 0.0
        spike_count = 0
    else:
        peaks = np.asarray(peaks, dtype=float)
        mean_peak = float(np.mean(peaks))
        mean_overshoot = float(np.mean(peaks - baseline))
        spike_count = int(len(peaks))

    return {
        "baseline_um": baseline,
        "std_um": std_val,
        "mean_spike_peak_um": mean_peak,
        "mean_spike_overshoot_um": mean_overshoot,
        "spike_count": spike_count,
    }

def save_control_plots(ctrl_results, outdir):
    setup_latex_style()
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # CSV summary (existing style)
    # ------------------------------------------------------------
    rows = [{
        "Layer":             r["layer"],
        "Target depth (um)": r["target_um"],
        "Before mean (um)":  float(np.mean(r["depth_before_um"])),
        "Before min (um)":   float(np.min(r["depth_before_um"])),
        "Before max (um)":   float(np.max(r["depth_before_um"])),
        "Before std (um)":   float(np.std(r["depth_before_um"])),
        "After mean (um)":   float(np.mean(r["depth_after_um"])),
        "After min (um)":    float(np.min(r["depth_after_um"])),
        "After max (um)":    float(np.max(r["depth_after_um"])),
        "After std (um)":    float(np.std(r["depth_after_um"])),
        "Mean power (W)":    r["P_mean_W"],
        "Min power (W)":     r["P_min_W"],
        "Max power (W)":     r["P_max_W"],
    } for r in ctrl_results]

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "ctrl_summary_by_layer.csv", index=False)

    print("\nLayer summary:")
    print(df.to_string(index=False))

    # ------------------------------------------------------------
    # Publication-style stats for paper table
    # ------------------------------------------------------------
    pub_rows = []
    for r in ctrl_results:
        before_stats = compute_case_stats(r["depth_before_um"])
        after_stats  = compute_case_stats(r["depth_after_um"])

        std_before = before_stats["std_um"]
        std_after  = after_stats["std_um"]
        ov_before  = before_stats["mean_spike_overshoot_um"]
        ov_after   = after_stats["mean_spike_overshoot_um"]

        std_red = 100.0 * (std_before - std_after) / max(std_before, 1e-12)
        if ov_before > 1e-12:
            ov_red = 100.0 * (ov_before - ov_after) / ov_before
        else:
            ov_red = 0.0

        pub_rows.append({
            "Layer": r["layer"],

            "Baseline before (um)": before_stats["baseline_um"],
            "Baseline after (um)": after_stats["baseline_um"],

            "Std before (um)": std_before,
            "Std after (um)": std_after,

            "Mean spike peak before (um)": before_stats["mean_spike_peak_um"],
            "Mean spike peak after (um)": after_stats["mean_spike_peak_um"],

            "Mean spike overshoot before (um)": ov_before,
            "Mean spike overshoot after (um)": ov_after,

            "Spike count before": before_stats["spike_count"],
            "Spike count after": after_stats["spike_count"],

            "Std reduction (%)": std_red,
            "Overshoot reduction (%)": ov_red,
        })

    df_pub = pd.DataFrame(pub_rows)
    df_pub.to_csv(outdir / "ctrl_publication_stats_by_layer.csv", index=False)

    print("\nPublication-style statistics:")
    print(df_pub.to_string(index=False))

    # ------------------------------------------------------------
    # Plot A: Uncontrolled vs controlled per layer
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, r in zip(axes.ravel(), ctrl_results):
        ax.plot(
            r["frac_eval"], r["depth_before_um"],
            linewidth=1.3, linestyle="-", color="#000000", label="Uncontrolled"
        )
        ax.plot(
            r["frac_eval"], r["depth_after_um"],
            linewidth=1.6, linestyle="-", color="#00B8D4", label="Controlled"
        )

        ymin = min(np.min(r["depth_before_um"]), np.min(r["depth_after_um"]))
        ymax = max(np.max(r["depth_before_um"]), np.max(r["depth_after_um"]))
        #ax.set_ylim(ymin - 1.0, ymax + 6.0)
        ax.set_ylim(ymin - 1.0, ymax + 6.0)

        ax.set_title(rf"Layer {r['layer']}")
        ax.set_xlabel("Path fraction")
        ax.set_ylabel(r"Depth ($\mu$m)")
        ax.grid(False)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.00, 1.02),
            fontsize=20,
            frameon=True
        )

    fig.tight_layout()
    fig.savefig(
        outdir / "ctrl_uncontrolled_vs_controlled_depth_by_layer.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        outdir / "ctrl_uncontrolled_vs_controlled_depth_by_layer.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Plot B: Power schedule by layer
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, r in zip(axes.ravel(), ctrl_results):
        frac = np.linspace(0.0, 1.0, len(r["P_layer_W"]))
        ax.plot(frac, r["P_layer_W"], linewidth=1.2)
        # ax.axhline(
        #     # r["P_mean_W"], linestyle="--", linewidth=1.0,
        #     # label=f"Mean {r['P_mean_W']:.0f} W"
        # )
        ax.set_title(rf"Layer {r['layer']} power")
        ax.set_xlabel("Path fraction")
        ax.set_ylabel("Power (W)")
        ax.grid(False)
        # ax.legend(fontsize=20)

    fig.suptitle("", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "ctrl_power_schedule_by_layer.pdf", bbox_inches="tight")
    fig.savefig(outdir / "ctrl_power_schedule_by_layer.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Plot C: Mean depth before / after with std bars
    # ------------------------------------------------------------
    layers = np.array([r["layer"] for r in ctrl_results])
    bmean = np.array([np.mean(r["depth_before_um"]) for r in ctrl_results])
    amean = np.array([np.mean(r["depth_after_um"])  for r in ctrl_results])
    bstd  = np.array([np.std(r["depth_before_um"])  for r in ctrl_results])
    astd  = np.array([np.std(r["depth_after_um"])   for r in ctrl_results])

    x = np.arange(len(layers))
    w = 0.36

    plt.figure(figsize=(8.6, 5.0))
    plt.bar(x - w/2, bmean, w, label="Before")
    plt.bar(x + w/2, amean, w, label="After")
    plt.errorbar(x - w/2, bmean, yerr=bstd, fmt="none", capsize=3)
    plt.errorbar(x + w/2, amean, yerr=astd, fmt="none", capsize=3)
    plt.axhline(
        float(np.mean([r["target_um"] for r in ctrl_results])),
        linestyle="--", linewidth=1.0, label="Avg target"
    )
    plt.xticks(x, [f"Layer {k}" for k in layers])
    plt.ylabel(r"Depth ($\mu$m)")
    plt.title("Layer-wise depth statistics before/after control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ctrl_depth_stats_before_after.pdf", bbox_inches="tight")
    plt.savefig(outdir / "ctrl_depth_stats_before_after.png",
                dpi=220, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot D: Power statistics by layer
    # ------------------------------------------------------------
    plt.figure(figsize=(7.8, 4.8))
    plt.plot(
        layers, [r["P_mean_W"] for r in ctrl_results],
        marker="o", linewidth=1.8, label="Mean power"
    )
    plt.plot(
        layers, [r["P_max_W"] for r in ctrl_results],
        marker="o", linewidth=1.3, label="Max power"
    )
    plt.plot(
        layers, [r["P_min_W"] for r in ctrl_results],
        marker="o", linewidth=1.3, label="Min power"
    )
    plt.axhline(CTRL.pmin_w, linestyle=":", linewidth=1.0, label=r"$P_{\min}$")
    plt.axhline(CTRL.pmax_w, linestyle=":", linewidth=1.0, label=r"$P_{\max}$")
    plt.xticks(layers)
    plt.xlabel("Layer")
    plt.ylabel("Power (W)")
    plt.title("Controlled power statistics by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ctrl_power_stats_by_layer.pdf", bbox_inches="tight")
    plt.savefig(outdir / "ctrl_power_stats_by_layer.png",
                dpi=220, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot E: Publication-style std and overshoot reduction
    # ------------------------------------------------------------
    std_red = df_pub["Std reduction (%)"].to_numpy(dtype=float)
    ov_red  = df_pub["Overshoot reduction (%)"].to_numpy(dtype=float)

    x = np.arange(len(layers))
    w = 0.36

    plt.figure(figsize=(8.4, 4.8))
    plt.bar(x - w/2, std_red, w, label="Std reduction")
    plt.bar(x + w/2, ov_red,  w, label="Overshoot reduction")
    plt.xticks(x, [f"Layer {k}" for k in layers])
    plt.ylabel("Reduction (%)")
    plt.title("Layer-wise control improvement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ctrl_improvement_by_layer.pdf", bbox_inches="tight")
    plt.savefig(outdir / "ctrl_improvement_by_layer.png",
                dpi=220, bbox_inches="tight")
    plt.close()
# ============================================================
# Main
# ============================================================
def main():
    setup_latex_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    inputs, _ = load_inputs_from_calibration(CALIB_JSON)

    base_traj = generate_horizontal_L_trajectory(
        length_mm=L_MM, width_mm=W_MM,
        notch_x_mm=NOTCH_X_MM, notch_y_mm=NOTCH_Y_MM,
        hatch_spacing_um=HATCH_SPACING_UM, dx_um=DX_UM,
    )

    (xyz_all, t_all, dt_all, layer_starts,
     n_local, z_step_m, t_layer_s,
     t_local, dt_local) = build_multilayer_sources(
        base_traj_xy=base_traj,
        n_layers=N_LAYERS,
        layer_thickness_um=LAYER_THICKNESS_UM,
        inputs=inputs,
        turn_slowdown=TURN_SLOWDOWN,
        turn_angle_deg=TURN_ANGLE_DEG,
        jump_threshold_um=JUMP_THRESHOLD_UM,
        laser_off_for_jumps=LASER_OFF_FOR_JUMPS,
        interlayer_dwell_ms=INTERLAYER_DWELL_MS,
    )

    print(f"Base power:           {inputs.P_W} W")
    print(f"Points per layer:     {n_local}")
    print(f"Layer scan time:      {t_layer_s:.4f} s")
    print(f"Total simulated time: {t_all[-1]:.4f} s")
    print(f"Jump points/layer:    {int(np.sum(dt_local <= 0))}")

    # -- Uncontrolled depth simulation --------------------------------------
    z_probe_um = np.linspace(0.0, Z_MAX_UM, Z_SAMPLES)
    results_unc = []
    for ell in range(N_LAYERS):
        idxs, frac, depths, peak_temps = depth_evolution_for_layer(
            base_traj_xy=base_traj,
            xyz_all=xyz_all, t_all=t_all, dt_all=dt_all,
            inputs=inputs,
            layer_idx=ell, n_local=n_local,
            z_probe_um=z_probe_um, dt_local=dt_local,
            eval_stride=EVAL_STRIDE,
            history_radius_um=HISTORY_RADIUS_UM,
            tau_hist_ms=TAU_HIST_MS,
            behind_um=BEHIND_UM,
        )
        results_unc.append({
            "layer": ell + 1, "idxs": idxs,
            "frac": frac, "depth_um": depths, "peakT_K": peak_temps,
        })

    save_uncontrolled_plots(
        results_unc, xyz_all, t_all, dt_all, base_traj, n_local, inputs, OUTDIR)

    # -- Controller ---------------------------------------------------------
    if RUN_CONTROL:
        print(f"\nRunning controller  (target_quantile={CTRL.target_quantile}, "
              f"max_outer_passes={CTRL.max_outer_passes})")
        _, ctrl_results = run_multilayer_controller(
            base_traj, xyz_all, t_all, dt_all,
            inputs, n_local, dt_local, CTRL,
        )
        save_control_plots(ctrl_results, OUTDIR)

    print(f"\nAll outputs saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()