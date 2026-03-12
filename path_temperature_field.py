"""
Path-based temperature field + melt depth evolution using a Green's-function summation
generalized to an arbitrary 2D scan path.

Model form (surface field at z=0):
  T(X,Y,t) = T0 + C * eta * Σ_i [ dt_heat_i * exp( -3 * r_i^2 / lam_i ) / lam_i^(3/2) ]
  lam_i = 12 * alpha * (t - t_i) + sigma^2
  r_i^2 = (X-x_i)^2 + (Y-y_i)^2 + (z*z_scale)^2

Key practical fixes implemented here:
  1) "Laser-off" travel moves:
     If the polyline contains long jump segments (e.g., hatch end -> next hatch start),
     we still advance time (cooling), but we set dt_heat=0 so no heat is deposited.
  2) Time-based history cutoff (tau_hist_ms):
     Prevents unphysical runaway accumulation in spiral/looping paths when revisiting regions.
  3) Depth evaluation behind-the-beam:
     Depth is evaluated at a point behind the current index to avoid artificial resets.

Compatibility:
  - Provides compute_surface_temperature_field(...) expected by older plotting code
  - Provides pick_turning_snapshot_by_heading(...) expected by your plot script
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PathThermalInputs:
    # Process
    P_W: float
    V_mmps: float
    sigma_m: float  # meters

    # Material
    RHO: float
    CP: float
    T0: float
    T_MELT: float

    # Calibrated parameters (constant within a run)
    eta: float
    alpha_mult: float
    z_scale: float


def get_k_in718_like(T: float) -> float:
    """k(T) ≈ 0.016*T + 6.39  [W/m/K] (simple IN718-like fit)."""
    return 0.016 * T + 6.39


def _prefactor(P_W: float, RHO: float, CP: float) -> float:
    """Matches your analytical prefactor."""
    return (6.0 * np.sqrt(3.0) * P_W) / (RHO * CP * np.pi * np.sqrt(np.pi))


def densify_trajectory(traj_xy_m: np.ndarray, step_um: float) -> np.ndarray:
    """
    Densify a polyline so that consecutive points are separated by <= step_um.
    """
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    if traj_xy_m.shape[0] < 2:
        return traj_xy_m.copy()

    step_m = float(step_um) * 1e-6
    out = [traj_xy_m[0].copy()]
    for i in range(1, traj_xy_m.shape[0]):
        p0 = traj_xy_m[i - 1]
        p1 = traj_xy_m[i]
        d = float(np.linalg.norm(p1 - p0))
        if d <= 1e-15:
            continue
        n = max(1, int(np.ceil(d / step_m)))
        for k in range(1, n + 1):
            out.append(p0 + (p1 - p0) * (k / n))
    return np.asarray(out, dtype=float)


def build_time_axis_from_trajectory(
    traj_xy_m: np.ndarray,
    V_mmps: float,
    *,
    turn_slowdown: float = 1.0,
    turn_angle_deg: float = 70.0,
    jump_threshold_um: float = 250.0,
    laser_off_for_jumps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build cumulative time t_i and per-source dt_heat_i from a polyline trajectory.

    Returns:
      t_src  : (N,) cumulative time at each source point [s]
      dt_heat: (N,) effective heat-weight [s] (0 for travel/jumps if laser_off_for_jumps=True)
    """
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    assert traj_xy_m.ndim == 2 and traj_xy_m.shape[1] == 2

    N = traj_xy_m.shape[0]
    if N == 1:
        return np.zeros(1), np.zeros(1)

    seg = traj_xy_m[1:] - traj_xy_m[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    V_mps = float(V_mmps) * 1e-3
    V_mps = max(V_mps, 1e-12)

    dt_seg = seg_len / V_mps  # always advances time

    # Corner slowdown (optional)
    if float(turn_slowdown) > 1.0 and N >= 3:
        v0 = seg[:-1]
        v1 = seg[1:]
        n0 = np.linalg.norm(v0, axis=1)
        n1 = np.linalg.norm(v1, axis=1)
        mask = (n0 > 1e-15) & (n1 > 1e-15)
        cosang = np.zeros_like(n0)
        cosang[mask] = np.sum(v0[mask] * v1[mask], axis=1) / (n0[mask] * n1[mask])
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        sharp = ang >= float(turn_angle_deg)
        for i, is_sharp in enumerate(sharp, start=1):
            if is_sharp:
                dt_seg[i - 1] *= float(turn_slowdown)
                dt_seg[i] *= float(turn_slowdown)

    # Map segment times -> per-point heat weights
    dt_heat = np.empty(N, dtype=float)
    dt_heat[0] = dt_seg[0]
    dt_heat[1:] = dt_seg

    # Laser-off for long jumps (heat=0, time still advances)
    if laser_off_for_jumps and jump_threshold_um is not None:
        jump_m = float(jump_threshold_um) * 1e-6
        long_seg = seg_len > jump_m
        if np.any(long_seg):
            dt_heat[1:][long_seg] = 0.0
            if long_seg[0]:
                dt_heat[0] = 0.0

    t_src = np.zeros(N, dtype=float)
    t_src[1:] = np.cumsum(dt_seg)
    return t_src, dt_heat


def _point_for_depth_eval(traj_xy_m: np.ndarray, idx: int, behind_m: float) -> np.ndarray:
    """Point located behind idx along the path by behind_m."""
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    N = traj_xy_m.shape[0]
    idx = int(np.clip(idx, 0, N - 1))
    behind_m = float(max(behind_m, 0.0))
    if behind_m <= 0.0 or idx == 0:
        return traj_xy_m[idx].copy()

    remaining = behind_m
    p = traj_xy_m[idx].copy()
    j = idx
    while remaining > 0 and j > 0:
        p0 = traj_xy_m[j - 1]
        p1 = traj_xy_m[j]
        v = p1 - p0
        d = float(np.linalg.norm(v))
        if d <= 1e-15:
            j -= 1
            continue
        if remaining >= d:
            remaining -= d
            p = p0.copy()
            j -= 1
        else:
            u = v / d
            p = p1 - u * remaining
            remaining = 0.0
    return p


def temperature_field_at_time(
    X_um: np.ndarray,
    Y_um: np.ndarray,
    traj_xy_m: np.ndarray,
    inputs: PathThermalInputs,
    eval_idx: int,
    *,
    history_radius_um: float = 450.0,
    tau_hist_ms: Optional[float] = 6.0,
    turn_slowdown: float = 1.0,
    turn_angle_deg: float = 70.0,
    jump_threshold_um: float = 250.0,
    laser_off_for_jumps: bool = True,
    P_src_W: Optional[np.ndarray] = None,
) -> np.ndarray:
    """2D surface temperature field at z=0 (for plotting)."""
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    N = traj_xy_m.shape[0]
    eval_idx = int(np.clip(eval_idx, 0, N - 1))

    t_src, dt_heat = build_time_axis_from_trajectory(
        traj_xy_m, inputs.V_mmps,
        turn_slowdown=turn_slowdown, turn_angle_deg=turn_angle_deg,
        jump_threshold_um=jump_threshold_um, laser_off_for_jumps=laser_off_for_jumps,
    )
    t_eval = float(t_src[eval_idx])

    x_src = traj_xy_m[: eval_idx + 1, 0]
    y_src = traj_xy_m[: eval_idx + 1, 1]
    t_s = t_src[: eval_idx + 1]
    w_s = dt_heat[: eval_idx + 1]

    # spatial + time history cutoff
    r_hist = float(history_radius_um) * 1e-6
    x0 = float(traj_xy_m[eval_idx, 0])
    y0 = float(traj_xy_m[eval_idx, 1])
    keep = ((x_src - x0) ** 2 + (y_src - y0) ** 2) <= (r_hist * r_hist)
    if tau_hist_ms is not None:
        tau = float(tau_hist_ms) * 1e-3
        keep &= (t_eval - t_s) <= tau
    if not np.any(keep):
        keep[-1] = True

    x_src = x_src[keep]
    y_src = y_src[keep]
    t_s = t_s[keep]
    w_s = w_s[keep]

    # variable power schedule: linear scaling
    if P_src_W is not None:
        P_src_W = np.asarray(P_src_W, dtype=float)
        P_s = P_src_W[: eval_idx + 1][keep]
        P0 = max(float(inputs.P_W), 1e-12)
        w_s = w_s * (P_s / P0)

    k_val = get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = _prefactor(inputs.P_W, inputs.RHO, inputs.CP) * inputs.eta

    dt_arr = np.maximum(t_eval - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m**2
    lam_pow = np.power(lam, 1.5)

    x_um = np.asarray(X_um, dtype=float)
    y_um = np.asarray(Y_um, dtype=float)
    X, Y = np.meshgrid(x_um, y_um, indexing="xy")
    X = X.reshape(-1) * 1e-6
    Y = Y.reshape(-1) * 1e-6

    # chunked sum to reduce memory
    chunk_pts = 4096
    out = np.empty(X.shape[0], dtype=float)
    for s in range(0, X.shape[0], chunk_pts):
        e = min(s + chunk_pts, X.shape[0])
        xp = X[s:e]
        yp = Y[s:e]
        dx2 = (xp[:, None] - x_src[None, :]) ** 2
        dy2 = (yp[:, None] - y_src[None, :]) ** 2
        r2 = dx2 + dy2
        term = np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :]
        out[s:e] = inputs.T0 + pref * np.sum(term * w_s[None, :], axis=1)

    return out.reshape(len(y_um), len(x_um))


# --- Backward-compatible alias expected by your plot script ---
def compute_surface_temperature_field(
    traj_xy_m: np.ndarray,
    inputs: PathThermalInputs,
    x_um: np.ndarray,
    y_um: np.ndarray,
    t_eval_s: float,
    src_stop_idx: int,
    chunk_pts: int = 2048,
    use_jax_if_available: bool = True,
    turn_slowdown: float = 1.0,
    turn_angle_deg: float = 70.0,
    history_radius_um: float = 450.0,
    tau_hist_ms: Optional[float] = 6.0,
    jump_threshold_um: float = 250.0,
    laser_off_for_jumps: bool = True,
    P_src_W: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compatibility wrapper: your plot code calls this old API.
    We ignore t_eval_s (because we compute time internally from eval_idx),
    and we evaluate the field at eval_idx=src_stop_idx.
    """
    _ = t_eval_s
    return temperature_field_at_time(
        X_um=x_um,
        Y_um=y_um,
        traj_xy_m=traj_xy_m,
        inputs=inputs,
        eval_idx=int(src_stop_idx),
        history_radius_um=history_radius_um,
        tau_hist_ms=tau_hist_ms,
        turn_slowdown=turn_slowdown,
        turn_angle_deg=turn_angle_deg,
        jump_threshold_um=jump_threshold_um,
        laser_off_for_jumps=laser_off_for_jumps,
        P_src_W=P_src_W,
    )


def compute_depth_profile_at_point(
    traj_xy_m: np.ndarray,
    inputs: PathThermalInputs,
    eval_idx: int,
    z_um: np.ndarray,
    history_radius_um: float = 450.0,
    behind_um: float = 40.0,
    use_jax_if_available: bool = True,
    turn_slowdown: float = 1.0,
    turn_angle_deg: float = 70.0,
    jump_threshold_um: float = 250.0,
    laser_off_for_jumps: bool = True,
    tau_hist_ms: Optional[float] = 6.0,
    P_src_W: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """T(z) and melt depth at a point behind eval_idx along the path."""
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    N = traj_xy_m.shape[0]
    eval_idx = int(np.clip(eval_idx, 0, N - 1))

    t_src, dt_heat = build_time_axis_from_trajectory(
        traj_xy_m, inputs.V_mmps,
        turn_slowdown=turn_slowdown, turn_angle_deg=turn_angle_deg,
        jump_threshold_um=jump_threshold_um, laser_off_for_jumps=laser_off_for_jumps,
    )
    t_eval = float(t_src[eval_idx])

    p_eval = _point_for_depth_eval(traj_xy_m, eval_idx, float(behind_um) * 1e-6)
    x_eval, y_eval = float(p_eval[0]), float(p_eval[1])

    x_s = traj_xy_m[: eval_idx + 1, 0]
    y_s = traj_xy_m[: eval_idx + 1, 1]
    t_s = t_src[: eval_idx + 1]
    w_s = dt_heat[: eval_idx + 1]

    # spatial + time history cutoff
    r_hist = float(history_radius_um) * 1e-6
    dx = x_eval - x_s
    dy = y_eval - y_s
    keep = (dx * dx + dy * dy) <= (r_hist * r_hist)
    if tau_hist_ms is not None:
        tau = float(tau_hist_ms) * 1e-3
        keep &= (t_eval - t_s) <= tau
    if not np.any(keep):
        keep[-1] = True

    x_s = x_s[keep]
    y_s = y_s[keep]
    t_s = t_s[keep]
    w_s = w_s[keep]

    # variable power schedule: linear scaling
    if P_src_W is not None:
        P_src_W = np.asarray(P_src_W, dtype=float)
        P_s = P_src_W[: eval_idx + 1][keep]
        P0 = max(float(inputs.P_W), 1e-12)
        w_s = w_s * (P_s / P0)

    dxy2 = (x_eval - x_s) ** 2 + (y_eval - y_s) ** 2

    k_val = get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = _prefactor(inputs.P_W, inputs.RHO, inputs.CP) * inputs.eta

    dt_arr = np.maximum(t_eval - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m**2
    lam_pow = np.power(lam, 1.5)

    z_m = (np.asarray(z_um, dtype=float) * 1e-6)
    z2 = (z_m * inputs.z_scale) ** 2

    # Compute T(z)
    if use_jax_if_available:
        try:
            import jax
            import jax.numpy as jnp

            dxy2_j = jnp.asarray(dxy2)
            lam_j = jnp.asarray(lam)
            lam_pow_j = jnp.asarray(lam_pow)
            w_j = jnp.asarray(w_s)
            z2_j = jnp.asarray(z2)
            pref_j = jnp.asarray(pref)
            T0_j = jnp.asarray(inputs.T0)

            @jax.jit
            def _eval_Tz():
                r2 = z2_j[:, None] + dxy2_j[None, :]
                term = jnp.exp(-3.0 * r2 / lam_j[None, :]) / lam_pow_j[None, :]
                s = jnp.sum(term * w_j[None, :], axis=1)
                return T0_j + pref_j * s

            Tz = np.array(_eval_Tz())
        except Exception:
            r2 = z2[:, None] + dxy2[None, :]
            term = np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :]
            Tz = inputs.T0 + pref * np.sum(term * w_s[None, :], axis=1)
    else:
        r2 = z2[:, None] + dxy2[None, :]
        term = np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :]
        Tz = inputs.T0 + pref * np.sum(term * w_s[None, :], axis=1)

    # Melt depth via threshold crossing
    above = Tz >= inputs.T_MELT
    if not np.any(above):
        return Tz, 0.0

    idx_last = int(np.max(np.where(above)[0]))
    if idx_last >= len(z_um) - 1:
        return Tz, float(z_um[-1])

    # linear interpolation between idx_last (above) and idx_last+1 (below)
    T1 = float(Tz[idx_last])
    T2 = float(Tz[idx_last + 1])
    z1 = float(z_um[idx_last])
    z2u = float(z_um[idx_last + 1])
    if abs(T2 - T1) < 1e-12:
        depth = z1
    else:
        frac = (inputs.T_MELT - T1) / (T2 - T1)
        depth = z1 + frac * (z2u - z1)

    return Tz, float(depth)


def compute_depth_evolution_over_path(
    traj_xy_m: np.ndarray,
    inputs: PathThermalInputs,
    z_um: np.ndarray,
    *,
    eval_stride: int = 1,
    history_radius_um: float = 450.0,
    behind_um: float = 40.0,
    tau_hist_ms: Optional[float] = 6.0,
    turn_slowdown: float = 1.0,
    turn_angle_deg: float = 70.0,
    jump_threshold_um: float = 250.0,
    laser_off_for_jumps: bool = True,
    P_src_W: Optional[np.ndarray] = None,
    use_jax_if_available: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Depth vs index along path."""
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    N = traj_xy_m.shape[0]
    eval_stride = max(int(eval_stride), 1)

    idxs = np.arange(0, N, eval_stride, dtype=int)
    depths = np.zeros_like(idxs, dtype=float)

    for k, idx in enumerate(idxs):
        _, d = compute_depth_profile_at_point(
            traj_xy_m, inputs, int(idx), z_um,
            history_radius_um=history_radius_um,
            behind_um=behind_um,
            use_jax_if_available=use_jax_if_available,
            turn_slowdown=turn_slowdown,
            turn_angle_deg=turn_angle_deg,
            jump_threshold_um=jump_threshold_um,
            laser_off_for_jumps=laser_off_for_jumps,
            tau_hist_ms=tau_hist_ms,
            P_src_W=P_src_W,
        )
        depths[k] = d

    return idxs, depths


# ---------- utilities expected by plot script ----------

def pick_turning_snapshot_by_heading(traj_xy_m: np.ndarray) -> int:
    """
    Return an index near the sharpest turn (largest heading change).
    Used only for picking a "nice" snapshot time for plotting.
    """
    traj_xy_m = np.asarray(traj_xy_m, dtype=float)
    if traj_xy_m.shape[0] < 5:
        return max(0, traj_xy_m.shape[0] // 2)

    v = traj_xy_m[1:] - traj_xy_m[:-1]
    n = np.linalg.norm(v, axis=1)
    good = n > 1e-15
    v[good] = v[good] / n[good][:, None]

    # heading change between consecutive segments
    dp = np.sum(v[1:] * v[:-1], axis=1)
    dp = np.clip(dp, -1.0, 1.0)
    ang = np.arccos(dp)  # radians
    k = int(np.argmax(ang)) + 1  # +1 to map to point index
    return int(np.clip(k, 0, traj_xy_m.shape[0] - 1))


def pick_turning_snapshot_horizontal(traj_xy_m: np.ndarray) -> int:
    """Legacy alias. Just use heading-based pick."""
    return pick_turning_snapshot_by_heading(traj_xy_m)
