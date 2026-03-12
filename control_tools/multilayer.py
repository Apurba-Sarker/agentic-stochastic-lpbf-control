"""
control_tools/multilayer.py
============================
3D multilayer LPBF simulation and control — lifted verbatim from the
standalone ``lshape_4layer_multilayer_depth.py`` to guarantee identical
numerical results when run through the pipeline.

Public API
----------
generate_lshape_trajectory(cfg)              → (N, 2) metres
build_multilayer_sources(base_traj, cfg, inputs) → MultilayerSources
depth_profile_multilayer_at_point(...)       → (Tz, depth_um)
depth_evolution_for_layer(...)               → (idxs, frac, depths, peakT)
run_controller_one_layer(...)                → (P_all, idxs_eval, d_before, d_after, target)
run_multilayer_controller(...)               → (P_all, results_list)
run_multilayer_uncontrolled(...)             → results_list
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import path_temperature_field as ptf

from .config import LShapeConfig


# ============================================================
# Dataclass to carry multilayer build results
# ============================================================
@dataclass
class MultilayerSources:
    xyz_all: np.ndarray          # (N_total, 3)
    t_all: np.ndarray            # (N_total,)
    dt_all: np.ndarray           # (N_total,)
    layer_start_idxs: np.ndarray # (n_layers,)
    n_local: int
    z_step_m: float
    t_layer_s: float
    t_local: np.ndarray          # (n_local,)  single-layer time
    dt_local: np.ndarray         # (n_local,)  single-layer dt_heat


# ============================================================
# Trajectory (L-shape horizontal serpentine)
# ============================================================
def generate_lshape_trajectory(cfg: LShapeConfig) -> np.ndarray:
    """
    Horizontal serpentine hatch for L-shaped footprint.
    Returns (N, 2) trajectory in metres.

    Each new hatch line is added as a single first point (no interpolated
    connector). This keeps the inter-line gap as one ~110 µm step, which
    ptf.build_time_axis_from_trajectory detects as a laser-off jump
    (jump_threshold_um=80 µm).
    """
    Lm = cfg.length_mm * 1e-3
    Wm = cfg.width_mm * 1e-3
    notch_x = cfg.notch_x_mm * 1e-3
    notch_y = cfg.notch_y_mm * 1e-3
    hatch = cfg.hatch_spacing_um * 1e-6
    dx = cfg.dx_um * 1e-6

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
            traj.append([float(xs[0]), float(y)])
            for x in xs[1:]:
                traj.append([float(x), float(y)])

    return np.asarray(traj, dtype=float)


# ============================================================
# Multilayer source construction
# ============================================================
def build_multilayer_sources(
    base_traj_xy: np.ndarray,
    cfg: LShapeConfig,
    inputs: ptf.PathThermalInputs,
) -> MultilayerSources:
    """
    Stack the same XY trajectory at successive z levels.
    Identical to the standalone ``build_multilayer_sources()`` function.
    """
    base_traj_xy = np.asarray(base_traj_xy, dtype=float)

    t_local, dt_local = ptf.build_time_axis_from_trajectory(
        base_traj_xy,
        inputs.V_mmps,
        turn_slowdown=cfg.turn_slowdown,
        turn_angle_deg=cfg.turn_angle_deg,
        jump_threshold_um=cfg.jump_threshold_um,
        laser_off_for_jumps=cfg.laser_off_for_jumps,
    )

    N = base_traj_xy.shape[0]
    z_step = cfg.layer_thickness_um * 1e-6

    xyz_blocks, t_blocks, dt_blocks = [], [], []
    layer_start_idxs = []
    t_offset = 0.0

    for ell in range(cfg.n_layers):
        z = ell * z_step
        xyz = np.column_stack([base_traj_xy, np.full(N, z, dtype=float)])
        xyz_blocks.append(xyz)
        t_blocks.append(t_local + t_offset)
        dt_blocks.append(dt_local.copy())
        layer_start_idxs.append(ell * N)
        t_offset = float(t_offset + t_local[-1] + cfg.interlayer_dwell_ms * 1e-3)

    return MultilayerSources(
        xyz_all=np.vstack(xyz_blocks),
        t_all=np.concatenate(t_blocks),
        dt_all=np.concatenate(dt_blocks),
        layer_start_idxs=np.asarray(layer_start_idxs, dtype=int),
        n_local=N,
        z_step_m=z_step,
        t_layer_s=float(t_local[-1]),
        t_local=np.asarray(t_local),
        dt_local=np.asarray(dt_local),
    )


# ============================================================
# Core depth kernel (3D multilayer) — VERBATIM from standalone
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
    Evaluate melt-pool depth at one point using the full multilayer 3D source
    history. Depth is positive downward from the current layer surface.
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
# Depth evolution (uncontrolled) — VERBATIM
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
    cfg: LShapeConfig,
):
    """
    Compute depth vs path-fraction for one layer (uncontrolled).
    Skips eval points near jump boundaries and forward-fills.
    """
    eval_stride = cfg.eval_stride
    history_radius_um = cfg.history_radius_um
    tau_hist_ms = cfg.tau_hist_ms
    behind_um = cfg.behind_um

    start = layer_idx * n_local
    idxs_local = np.arange(0, n_local, int(eval_stride), dtype=int)

    seg_dists = np.linalg.norm(np.diff(base_traj_xy, axis=0), axis=1)
    skip_mask = np.zeros(n_local, dtype=bool)
    skip_mask[0] = True
    jump_locs = np.where(np.concatenate([[False], seg_dists > (cfg.jump_threshold_um * 1e-6)]))[0]
    for j in jump_locs:
        skip_mask[j: min(n_local, j + eval_stride + 1)] = True

    depths = np.full(len(idxs_local), np.nan, dtype=float)
    peak_temps = np.zeros(len(idxs_local), dtype=float)

    for k, li in enumerate(idxs_local):
        if skip_mask[li]:
            continue
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

    # Forward-fill NaN startup values
    fv = next((depths[k] for k in range(len(depths)) if not np.isnan(depths[k])), 0.0)
    for k in range(len(depths)):
        if np.isnan(depths[k]):
            depths[k] = fv
        else:
            fv = depths[k]

    frac = idxs_local / max(n_local - 1, 1)
    return idxs_local, frac, depths, peak_temps


# ============================================================
# Uncontrolled wrapper (all layers)
# ============================================================
def run_multilayer_uncontrolled(
    base_traj: np.ndarray,
    src: MultilayerSources,
    inputs: ptf.PathThermalInputs,
    cfg: LShapeConfig,
) -> list[dict]:
    """Run uncontrolled depth evaluation for all layers."""
    z_probe_um = np.linspace(0.0, cfg.z_max_um, cfg.z_samples)
    results = []

    for ell in range(cfg.n_layers):
        idxs, frac, depths, peak_temps = depth_evolution_for_layer(
            base_traj_xy=base_traj,
            xyz_all=src.xyz_all,
            t_all=src.t_all,
            dt_all=src.dt_all,
            inputs=inputs,
            layer_idx=ell,
            n_local=src.n_local,
            z_probe_um=z_probe_um,
            dt_local=src.dt_local,
            cfg=cfg,
        )
        results.append({
            "layer": ell + 1,
            "idxs": idxs,
            "frac": frac,
            "depth_um": depths,
            "peakT_K": peak_temps,
        })

    return results


# ============================================================
# Controller helpers — VERBATIM from standalone
# ============================================================
def _build_valid_eval_indices(n_local, dt_local, cfg: LShapeConfig):
    """Return eval indices that are jump-safe."""
    idxs = np.arange(0, n_local, int(cfg.eval_stride), dtype=int)
    if not cfg.ctrl_skip_jump_neighbors:
        return idxs
    invalid = np.zeros(n_local, dtype=bool)
    for j in np.where(np.asarray(dt_local) <= 0.0)[0]:
        invalid[max(0, int(j) - cfg.ctrl_jump_neighbor_radius):
                min(n_local, int(j) + cfg.eval_stride + cfg.ctrl_jump_neighbor_radius + 1)] = True
    if cfg.ctrl_skip_first_eval_point:
        invalid[0] = True
    kept = [i for i in idxs if not invalid[i]]
    return np.asarray(kept if kept else [min(cfg.eval_stride, n_local - 1)], dtype=int)


def _clip_power(P, P_ref, cfg: LShapeConfig):
    P = float(np.clip(P, cfg.pmin_w, cfg.pmax_w))
    P = float(np.clip(P, P_ref - cfg.dp_max_w, P_ref + cfg.dp_max_w))
    return float(np.clip(P, cfg.pmin_w, cfg.pmax_w))


def _depth_at(xyz_all, t_all, dt_all, inputs, z_um, layer_idx, n_local,
              local_idx, P_all, cfg: LShapeConfig, base_traj):
    gi = layer_idx * n_local + int(local_idx)
    return depth_profile_multilayer_at_point(
        xyz_all, t_all, dt_all, gi, inputs, z_um,
        layer_local_traj=base_traj,
        layer_start_idx=layer_idx * n_local,
        history_radius_um=cfg.ctrl_history_radius_um,
        tau_hist_ms=cfg.ctrl_tau_hist_ms,
        behind_um=cfg.behind_um,
        P_src_W=P_all,
    )


def _analytical_inverse(xyz_all, t_all, dt_all, inputs, z_um, iz_target,
                        layer_idx, n_local, predict_local_idx,
                        ctrl_start, ctrl_end, P_all, P_prev, cfg: LShapeConfig, base_traj):
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
                    idxs_local, P_all, cfg: LShapeConfig, base_traj):
    depths = np.zeros(len(idxs_local), dtype=float)
    for k, li in enumerate(idxs_local):
        _, d = _depth_at(xyz_all, t_all, dt_all, inputs, z_um,
                         layer_idx, n_local, int(li), P_all, cfg, base_traj)
        depths[k] = d
    return depths


def run_controller_one_layer(xyz_all, t_all, dt_all, inputs, layer_idx, n_local,
                              z_um, cfg: LShapeConfig, P_all, dt_local, base_traj):
    """
    Two-pass analytical-inverse controller for one layer, with iterative outer loop.
    VERBATIM from standalone to guarantee identical results.
    """
    P0 = float(inputs.P_W)
    layer_slice = slice(layer_idx * n_local, (layer_idx + 1) * n_local)

    idxs_eval = _build_valid_eval_indices(n_local, dt_local, cfg)

    # Precompute segment distances for line-boundary clamping
    _seg_dists = np.linalg.norm(np.diff(base_traj, axis=0), axis=1)
    _jump_m = cfg.jump_threshold_um * 1e-6

    def _safe_predict_idx(ctrl_start, ctrl_end):
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
    for outer in range(cfg.ctrl_max_outer_passes):

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
        if max_err <= cfg.ctrl_flat_tol_um:
            break
        if prev_max_err is not None and (prev_max_err - max_err) < cfg.ctrl_improve_tol_um:
            break
        prev_max_err = max_err

    depths_after = _compute_depths(xyz_all, t_all, dt_all, inputs, z_um,
                                   layer_idx, n_local, idxs_eval, P_all, cfg, base_traj)
    return P_all, idxs_eval, depths_before, depths_after, target


def run_multilayer_controller(
    base_traj: np.ndarray,
    src: MultilayerSources,
    inputs: ptf.PathThermalInputs,
    cfg: LShapeConfig,
) -> tuple[np.ndarray, list[dict]]:
    """Control all layers sequentially; later layers see corrected power history."""
    z_um = np.linspace(0.0, cfg.z_max_um, cfg.z_samples)
    P_all = np.full_like(src.t_all, float(inputs.P_W), dtype=float)
    results = []

    for ell in range(cfg.n_layers):
        print(f"\n=== Layer {ell+1}/{cfg.n_layers} ===")
        P_all, idxs_u, d_before, d_after, target = run_controller_one_layer(
            src.xyz_all, src.t_all, src.dt_all, inputs, ell, src.n_local,
            z_um, cfg, P_all, src.dt_local, base_traj,
        )
        frac = idxs_u / max(src.n_local - 1, 1)
        P_layer = P_all[ell * src.n_local:(ell + 1) * src.n_local].copy()
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
