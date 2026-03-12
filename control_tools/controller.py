"""
control_tools/controller.py
============================
Two-pass analytical inverse controller for melt-pool depth uniformity.

Pass 1  — stride-by-stride forward inverse: for each evaluation window
           solve analytically for the power that would bring the predicted
           temperature at the target depth to T_melt.

Pass 2  — targeted spike fix: iterate up to N_FIX_PASSES times, each time
           finding points where depth < (target − SPIKE_THRESHOLD) and
           re-solving the power for those windows.

If cfg.power_levels_path is set the controller snaps every power decision to
the nearest allowed value from that file (quantised-power mode).

Public API
----------
run_controller(traj, inputs, cfg)
    → (P_ctrl, depths_u, depths_ctrl, idxs_u, fr, target)
pick_abc_points(idxs_u, depths_u, depths_ctrl, P_un, P_ctrl, cfg)
    → list[int]   (indices into idxs_u)
"""

from __future__ import annotations
import numpy as np
import path_temperature_field as ptf
from .config import TrackConfig


# ── Quantised-power helper ────────────────────────────────────────────────
def quantize_power(P: float, levels: np.ndarray | None) -> float:
    """
    Snap P to the nearest value in *levels*.
    If levels is None (continuous mode) returns P unchanged.
    """
    if levels is None or len(levels) == 0:
        return P
    idx = int(np.argmin(np.abs(levels - P)))
    return float(levels[idx])


# ── Low-level depth helper ────────────────────────────────────────────────
def depth_at(
    traj: np.ndarray,
    inputs,
    z_um: np.ndarray,
    idx: int,
    P_src_W: np.ndarray,
    cfg: TrackConfig,
) -> tuple[np.ndarray, float]:
    Tz, d = ptf.compute_depth_profile_at_point(
        traj, inputs, int(idx), z_um,
        P_src_W=P_src_W,
        **cfg.ptf_kw(),
    )
    return np.asarray(Tz, float), float(d)


# ── Analytical single-window inverse step ────────────────────────────────
def analytical_inverse_step(
    traj: np.ndarray,
    inputs,
    z_um: np.ndarray,
    iz_target: int,
    predict_idx: int,
    ctrl_start: int,
    ctrl_end: int,
    P_schedule: np.ndarray,
    P_prev: float,
    cfg: TrackConfig,
    power_levels: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Solve analytically for the power in [ctrl_start, ctrl_end) that drives
    T[iz_target] → T_melt at predict_idx.

    Returns (P_new, depth_at_predict_idx).
    """
    N = len(traj)
    predict_idx = min(int(predict_idx), N - 1)

    Tz_full, d_full = depth_at(traj, inputs, z_um, predict_idx, P_schedule, cfg)
    T_full = float(Tz_full[iz_target])

    # Contribution with control window zeroed
    P_zero = P_schedule.copy()
    P_zero[ctrl_start:ctrl_end] = 0.0
    Tz_fixed, _ = depth_at(traj, inputs, z_um, predict_idx, P_zero, cfg)
    T_fixed = float(Tz_fixed[iz_target])

    T_ctrl = T_full - T_fixed
    if abs(T_ctrl) < 0.5:
        return quantize_power(float(P_schedule[ctrl_start]), power_levels), d_full

    P_cur = float(P_schedule[ctrl_start])
    if P_cur < 1.0:
        P_cur = float(inputs.P_W)

    P_new = P_cur * (inputs.T_MELT - T_fixed) / T_ctrl
    P_new = np.clip(P_new, cfg.pmin_w, cfg.pmax_w)
    P_new = np.clip(P_new, P_prev - cfg.dp_max_w, P_prev + cfg.dp_max_w)
    P_new = float(np.clip(P_new, cfg.pmin_w, cfg.pmax_w))

    # ── snap to allowed levels if quantised mode ──────────────────────────
    P_new = quantize_power(P_new, power_levels)

    return P_new, d_full


# ── ABC point selection ───────────────────────────────────────────────────
def pick_abc_points(
    idxs_u: np.ndarray,
    depths_u: np.ndarray,
    depths_ctrl: np.ndarray,
    P_un: np.ndarray,
    P_ctrl: np.ndarray,
    cfg: TrackConfig,
) -> list[int]:
    """
    Select cfg.abc_k points that show the largest depth difference AND
    power change, with minimum spatial separation.

    Returns a list of indices into idxs_u (not raw trajectory indices).
    """
    idxs_u    = np.asarray(idxs_u, int)
    du        = np.asarray(depths_u, float)
    dc        = np.asarray(depths_ctrl, float)
    P_un_arr  = np.asarray(P_un, float)
    P_ctrl_arr= np.asarray(P_ctrl, float)

    diff = np.abs(du - dc)
    dp   = np.abs(P_ctrl_arr[idxs_u] - P_un_arr[idxs_u])

    cand = np.where((diff >= cfg.abc_diff_min_um) & (dp >= cfg.abc_dp_min_w))[0]
    if cand.size < cfg.abc_k:
        cand = np.where(diff >= cfg.abc_diff_min_um)[0]
    if cand.size < cfg.abc_k:
        cand = np.arange(len(du), dtype=int)

    order = cand[np.argsort(-diff[cand])]

    Ntot    = int(idxs_u[-1]) + 1
    min_sep = max(10, int(cfg.abc_min_sep_frac * Ntot))

    chosen = []
    for j in order:
        if len(chosen) >= cfg.abc_k:
            break
        if not chosen or all(
            abs(idxs_u[j] - idxs_u[c]) >= min_sep for c in chosen
        ):
            chosen.append(int(j))

    # Fallback: relax spacing
    for j in order:
        if len(chosen) >= cfg.abc_k:
            break
        if int(j) not in chosen:
            chosen.append(int(j))

    return chosen[: cfg.abc_k]


# ── Main controller entry-point ───────────────────────────────────────────
def run_controller(
    traj: np.ndarray,
    inputs,
    cfg: TrackConfig,
    power_levels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Two-pass analytical inverse controller.

    Parameters
    ----------
    traj         : (N, 2) trajectory in metres
    inputs       : ptf.PathThermalInputs
    cfg          : TrackConfig
    power_levels : optional sorted 1-D array of allowed power values (W).
                   When provided every power decision is snapped to the
                   nearest value in this set (quantised-power mode).
                   Pass None for continuous (default) control.

    Returns
    -------
    P_ctrl      : (N,) controlled power schedule
    depths_u    : uncontrolled depths at eval points
    depths_ctrl : controlled depths at eval points
    idxs_u      : trajectory indices of eval points
    fr          : path fractions at eval points
    target      : internal depth target (µm)
    """
    N   = len(traj)
    P0  = float(inputs.P_W)
    z_um = np.linspace(0.0, -cfg.z_max_um, cfg.z_samples)

    # Quantisation mode summary
    if power_levels is not None and len(power_levels) > 0:
        lvl_str = ", ".join(f"{v:.0f}" for v in power_levels[:8])
        if len(power_levels) > 8:
            lvl_str += f" … ({len(power_levels)} levels)"
        print(f"  [Quantised power] allowed levels (W): {lvl_str}")
    else:
        power_levels = None   # normalise: treat empty array same as None
        print("  [Continuous power] no quantisation applied")

    # ── Uncontrolled baseline ─────────────────────────────────────────────
    print("[1/4] Uncontrolled baseline...")
    P_uncontrolled = np.full(N, P0)
    idxs_u, depths_u = ptf.compute_depth_evolution_over_path(
        traj, inputs, z_um,
        eval_stride=cfg.eval_stride,
        P_src_W=P_uncontrolled,
        **cfg.ptf_kw(),
    )
    idxs_u  = idxs_u.astype(int)
    depths_u = depths_u.astype(float)

    target    = float(np.quantile(depths_u[min(5, len(depths_u)):], cfg.target_quantile))
    iz_target = int(np.argmin(np.abs(z_um - target)))
    fr        = idxs_u / max(1, N - 1)

    print(f"  Depth range: {depths_u.min():.2f}..{depths_u.max():.2f} µm  "
          f"| target: {target:.2f} µm")

    # ── Pass 1: stride-by-stride inverse ─────────────────────────────────
    print("[2/4] Pass 1: stride-by-stride inverse...")
    P_ctrl = np.full(N, P0)
    P_prev = P0

    for k, idx in enumerate(idxs_u):
        idx = int(idx)
        if k % 100 == 0:
            print(f"  step {k}/{len(idxs_u)}  idx={idx}  P={P_prev:.1f} W")

        ctrl_start  = idx
        ctrl_end    = min(N, idx + cfg.eval_stride)
        predict_idx = min(N - 1, idx + cfg.eval_stride)

        P_new, _ = analytical_inverse_step(
            traj, inputs, z_um, iz_target,
            predict_idx, ctrl_start, ctrl_end,
            P_ctrl, P_prev, cfg,
            power_levels=power_levels,
        )
        P_ctrl[ctrl_start:ctrl_end] = P_new
        P_prev = P_new

    # ── Pass 2: spike fix ─────────────────────────────────────────────────
    print(f"[3/4] Pass 2: spike fix ({cfg.n_fix_passes} iterations)...")
    for fix_pass in range(cfg.n_fix_passes):
        _, depths_current = ptf.compute_depth_evolution_over_path(
            traj, inputs, z_um,
            eval_stride=cfg.eval_stride,
            P_src_W=P_ctrl,
            **cfg.ptf_kw(),
        )
        depths_current = depths_current.astype(float)

        spike_mask      = depths_current < (target - cfg.spike_threshold_um)
        spike_positions = idxs_u[spike_mask]
        spike_depths    = depths_current[spike_mask]

        if len(spike_positions) == 0:
            print(f"  Pass {fix_pass + 1}: no spikes remaining")
            break

        print(f"  Pass {fix_pass + 1}: {len(spike_positions)} spikes, "
              f"worst={spike_depths.min():.2f} µm")

        for sp_idx in spike_positions:
            sp_idx     = int(sp_idx)
            offset     = cfg.spike_fix_window // 3
            ctrl_start = max(0, sp_idx - cfg.spike_fix_window + offset)
            ctrl_end   = min(N, sp_idx + offset)

            Tz_full, _ = depth_at(traj, inputs, z_um, sp_idx, P_ctrl, cfg)
            T_full = float(Tz_full[iz_target])

            P_zero = P_ctrl.copy()
            P_zero[ctrl_start:ctrl_end] = 0.0
            Tz_fixed, _ = depth_at(traj, inputs, z_um, sp_idx, P_zero, cfg)
            T_fixed = float(Tz_fixed[iz_target])

            T_contrib = T_full - T_fixed
            if abs(T_contrib) < 0.5:
                continue

            P_cur    = float(np.mean(P_ctrl[ctrl_start:ctrl_end]))
            P_needed = P_cur * (inputs.T_MELT - T_fixed) / T_contrib
            P_needed = float(np.clip(P_needed, cfg.pmin_w, cfg.pmax_w))

            # ── snap to allowed levels if quantised mode ──────────────
            P_needed = quantize_power(P_needed, power_levels)

            P_ctrl[ctrl_start:ctrl_end] = P_needed

    # ── Final controlled depths ───────────────────────────────────────────
    print("[4/4] Computing final controlled depth evolution...")
    _, depths_ctrl = ptf.compute_depth_evolution_over_path(
        traj, inputs, z_um,
        eval_stride=cfg.eval_stride,
        P_src_W=P_ctrl,
        **cfg.ptf_kw(),
    )
    depths_ctrl = depths_ctrl.astype(float)

    return P_ctrl, depths_u, depths_ctrl, idxs_u, fr, target


# ── Stochastic ensemble controller ──────────────────────────────────────
def run_stochastic_ensemble(
    traj: np.ndarray,
    calib: dict,
    cfg: TrackConfig,
    n_realizations: int = 10,
    power_levels: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full controller N times, each with a DIFFERENT posterior draw
    of (eta, alpha_mult, z_scale).  Each realization produces its own
    power schedule and controlled depth profile.

    This makes the control genuinely stochastic: the power schedule changes
    across realizations because the model parameters change.

    Parameters
    ----------
    traj            : (N, 2) trajectory in metres
    calib           : calibration dict (from calibration_for_printing.json)
    cfg             : TrackConfig
    n_realizations  : number of posterior draws (default 10)
    power_levels    : optional quantised power levels
    verbose         : print progress

    Returns
    -------
    dict with keys:
        n_realizations : int
        seeds          : list[int]
        thetas         : list of (eta, alpha_mult, z_scale) per realization
        P_schedules    : (n_realizations, N) array of power schedules
        depths_ctrl_all: (n_realizations, n_eval) controlled depths
        depths_u_all   : (n_realizations, n_eval) uncontrolled depths
        targets        : (n_realizations,) depth targets per realization
        idxs_u         : (n_eval,) evaluation indices (shared)
        fr             : (n_eval,) path fractions (shared)
        # ── Ensemble statistics ──
        P_consensus    : (N,) median power schedule across realizations
        depth_ctrl_mean: (n_eval,) mean controlled depth
        depth_ctrl_std : (n_eval,) std of controlled depth
        depth_u_mean   : (n_eval,) mean uncontrolled depth
        depth_u_std    : (n_eval,) std of uncontrolled depth
        std_reduction_per_real: (n_realizations,) % std reduction per realization
        std_reduction_mean: float
        std_reduction_std : float
    """
    from .calibration import sample_theta, make_inputs

    N = len(traj)
    base_seed = cfg.seed

    P_all     = []
    d_ctrl_all = []
    d_u_all    = []
    targets_all = []
    thetas     = []
    seeds      = []
    idxs_shared = None
    fr_shared   = None

    for i in range(n_realizations):
        seed_i = base_seed + i
        seeds.append(seed_i)

        # ── Draw from posterior ───────────────────────────────────────────
        eta_i, alpha_i, z_i = sample_theta(calib, seed_i)
        thetas.append((eta_i, alpha_i, z_i))

        # ── Build inputs for this draw ────────────────────────────────────
        inputs_i = make_inputs(calib, cfg, eta_i, alpha_i, z_i)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Realization {i+1}/{n_realizations}  "
                  f"(seed={seed_i}, η={eta_i:.4f}, α={alpha_i:.4f}, z={z_i:.4f})")
            print(f"{'='*60}")

        # ── Run full controller with this parameter draw ──────────────────
        P_ctrl_i, depths_u_i, depths_ctrl_i, idxs_u_i, fr_i, target_i = (
            run_controller(traj, inputs_i, cfg, power_levels=power_levels)
        )

        P_all.append(P_ctrl_i)
        d_ctrl_all.append(depths_ctrl_i)
        d_u_all.append(depths_u_i)
        targets_all.append(target_i)

        if idxs_shared is None:
            idxs_shared = idxs_u_i
            fr_shared   = fr_i

    # ── Stack into arrays ─────────────────────────────────────────────────
    P_all      = np.array(P_all)           # (n_real, N)
    d_ctrl_all = np.array(d_ctrl_all)      # (n_real, n_eval)
    d_u_all    = np.array(d_u_all)         # (n_real, n_eval)
    targets_all = np.array(targets_all)    # (n_real,)

    # ── Consensus power schedule (pointwise median) ───────────────────────
    P_consensus = np.median(P_all, axis=0)

    # ── Ensemble statistics ───────────────────────────────────────────────
    d_ctrl_mean = d_ctrl_all.mean(axis=0)
    d_ctrl_std  = d_ctrl_all.std(axis=0)
    d_u_mean    = d_u_all.mean(axis=0)
    d_u_std     = d_u_all.std(axis=0)

    # Per-realization std reduction
    std_reductions = []
    for i in range(n_realizations):
        s_u = float(np.std(d_u_all[i, 5:]))
        s_c = float(np.std(d_ctrl_all[i, 5:]))
        red = 100.0 * (1.0 - s_c / s_u) if s_u > 0 else 0.0
        std_reductions.append(red)
    std_reductions = np.array(std_reductions)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stochastic Ensemble Summary ({n_realizations} realizations)")
        print(f"{'='*60}")
        print(f"  Std reduction: {std_reductions.mean():.1f}% ± {std_reductions.std():.1f}%")
        print(f"  Range: [{std_reductions.min():.1f}%, {std_reductions.max():.1f}%]")
        print(f"  Controlled depth std (mean across path): {d_ctrl_std.mean():.3f} µm")
        print(f"  Power schedule std (mean across path): {P_all.std(axis=0).mean():.1f} W")

    return {
        "n_realizations":       n_realizations,
        "seeds":                seeds,
        "thetas":               thetas,
        "P_schedules":          P_all,
        "depths_ctrl_all":      d_ctrl_all,
        "depths_u_all":         d_u_all,
        "targets":              targets_all,
        "idxs_u":               idxs_shared,
        "fr":                   fr_shared,
        "P_consensus":          P_consensus,
        "depth_ctrl_mean":      d_ctrl_mean,
        "depth_ctrl_std":       d_ctrl_std,
        "depth_u_mean":         d_u_mean,
        "depth_u_std":          d_u_std,
        "std_reduction_per_real": std_reductions,
        "std_reduction_mean":   float(std_reductions.mean()),
        "std_reduction_std":    float(std_reductions.std()),
    }