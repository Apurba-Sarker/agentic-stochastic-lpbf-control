"""
control_tools/cross_sections.py
================================
XZ (across-track) and SZ (along-track) cross-section computation and saving.

Key guarantees:
  - NO dt=0 singular spikes:
      t_eval = t_src[i] + eps
      sources: [:i] (strictly before i)
  - Contours always computed on TRUE temperature field (unclipped)
  - Consistent color coding across ALL plots:
      Uncontrolled = black
      Controlled   = cyan (#00B8D4)
  - Adds ensemble “spread” plotters:
      save_field_with_contour_spread(...)
      save_overlay_spread(...)
    which overlay mean + quantile contours (qlo/qhi) with stronger visibility.

Public API
----------
xz_cross_section(traj, inputs, *, eval_idx, cfg, P_src_W)
sz_cross_section(traj, inputs, *, eval_idx, cfg, P_src_W)

save_scalebar_only(...)
save_temperature_colorbar_only(...)
save_field_with_contour(...)
save_overlay(...)
save_field_with_contour_spread(...)
save_overlay_spread(...)
save_all_xsections(...)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import path_temperature_field as ptf
from .config import TrackConfig


# ──────────────────────────────────────────────────────────────────────────
# Global style: consistent colors everywhere
# ──────────────────────────────────────────────────────────────────────────
COLOR_UNCONTROLLED = "#000000"   # black
COLOR_CONTROLLED   = "#00B8D4"   # cyan


# ── Internal: build history sources for a given eval point ───────────────
def _history_sources(traj_xy_m, inputs, i, x0, y0, P_src_W, cfg: TrackConfig):
    """
    Build history sources (x_s, y_s, t_s, w_s) to evaluate temperature at (x0,y0).

    CRITICAL: Avoid dt=0 singular spikes by summing sources strictly BEFORE i.
      - t_eval = t_src[i] + eps
      - sources: [:i] (NOT [:i+1])

    Filters:
      - within history_radius_um
      - within tau_hist_ms (optional)
    """
    t_src, dt_heat = ptf.build_time_axis_from_trajectory(
        traj_xy_m, inputs.V_mmps,
        turn_slowdown=cfg.turn_slowdown,
        turn_angle_deg=cfg.turn_angle_deg,
        jump_threshold_um=cfg.jump_threshold_um,
        laser_off_for_jumps=cfg.laser_off_for_jumps,
    )

    eps = 1e-9
    t_eval = float(t_src[i]) + eps

    i0 = int(i)
    x_s = traj_xy_m[:i0, 0]
    y_s = traj_xy_m[:i0, 1]
    t_s = t_src[:i0]
    w_s = dt_heat[:i0]

    # Safety: if i0==0 (caller should clamp i>=1)
    if x_s.size == 0:
        x_s = traj_xy_m[:1, 0]
        y_s = traj_xy_m[:1, 1]
        t_s = t_src[:1]
        w_s = dt_heat[:1]

    # spatial window
    r_hist = cfg.history_radius_um * 1e-6
    dxs = x0 - x_s
    dys = y0 - y_s
    keep = (dxs ** 2 + dys ** 2) <= r_hist ** 2

    # time window (optional)
    if cfg.tau_hist_ms is not None:
        tau = cfg.tau_hist_ms * 1e-3
        keep &= (t_eval - t_s) <= tau

    # never allow empty: keep nearest-in-space among the pre-i sources
    if not np.any(keep):
        j = int(np.argmin(dxs * dxs + dys * dys))
        keep[j] = True

    x_s = x_s[keep]
    y_s = y_s[keep]
    t_s = t_s[keep]
    w_s = w_s[keep]

    # If a power schedule exists, scale weights by P_s / P0, matching [:i]
    if P_src_W is not None:
        P_arr = np.asarray(P_src_W, dtype=float)
        P_s_all = P_arr[:i0] if P_arr.size >= i0 else P_arr
        if P_s_all.size == 0:
            P_s_all = P_arr[:1]
        P_s = P_s_all[keep]
        P0 = max(float(inputs.P_W), 1e-12)
        w_s = w_s * (P_s / P0)

    return x_s, y_s, t_s, w_s, t_eval


# ── Standalone temperature colorbar (paper-ready) ─────────────────────────
def save_temperature_colorbar_only(
    outpath: str,
    vmin: float,
    vmax: float,
    cmap: str = "hot",
    label: str = "Temperature (K)",
    orientation: str = "vertical",
    figsize=(1.2, 4.0),
    labelsize: int = 16,
    ticksize: int = 14,
    dpi: int = 300,
    show_endpoints: bool = False,
    endpoint_mode: str = "text",   # "text" or "ticks"
    force_ticks: bool = True,
):
    import matplotlib as mpl

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.35, 0.05, 0.30, 0.90])

    norm = mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax))
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=mpl.cm.get_cmap(cmap), norm=norm, orientation=orientation
    )
    cb.set_label(label, fontsize=labelsize)
    cb.ax.tick_params(labelsize=ticksize)

    if force_ticks:
        ticks = np.array([300, 400, 600, 800, 1000, 1200, 1400, 1600], dtype=float)
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.0f}" for t in ticks])

    if show_endpoints:
        if endpoint_mode == "ticks":
            ticks = cb.get_ticks()
            ticks = np.unique(np.r_[float(vmin), ticks, float(vmax)])
            cb.set_ticks(ticks)
            labs = []
            for t in ticks:
                if abs(t - float(vmin)) < 1e-9:
                    labs.append(f"{t:.0f} (min)")
                elif abs(t - float(vmax)) < 1e-9:
                    labs.append(f"{t:.0f} (max)")
                else:
                    labs.append(f"{t:.0f}")
            cb.set_ticklabels(labs)
        else:
            cb.ax.text(0.5, -0.06, f"min: {float(vmin):.0f} K",
                       transform=cb.ax.transAxes, ha="center", va="top",
                       fontsize=ticksize)
            cb.ax.text(0.5,  1.06, f"max: {float(vmax):.0f} K",
                       transform=cb.ax.transAxes, ha="center", va="bottom",
                       fontsize=ticksize)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── XZ cross-section ──────────────────────────────────────────────────────
def xz_cross_section(
    traj_xy_m: np.ndarray,
    inputs,
    *,
    eval_idx: int,
    cfg: TrackConfig,
    P_src_W: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Across-track (X) × depth (Z) temperature field.
    Returns (x_um, z_um, T) where T.shape == (nz, nx).
    """
    traj_xy_m = np.asarray(traj_xy_m, float)
    N = traj_xy_m.shape[0]
    i = int(np.clip(eval_idx, 1, N - 1))

    # local tangent & normal
    t = traj_xy_m[i] - traj_xy_m[i - 1]
    tn = float(np.linalg.norm(t))
    t = np.array([1.0, 0.0], float) if tn < 1e-15 else t / tn
    nv = np.array([-t[1], t[0]], dtype=float)

    p_eval = ptf._point_for_depth_eval(traj_xy_m, i, cfg.behind_um * 1e-6)
    x0, y0 = float(p_eval[0]), float(p_eval[1])

    x_s, y_s, t_s, w_s, t_eval = _history_sources(
        traj_xy_m, inputs, i, x0, y0, P_src_W, cfg
    )

    k_val = ptf.get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = (
        (6.0 * np.sqrt(3.0) * inputs.P_W)
        / (inputs.RHO * inputs.CP * np.pi * np.sqrt(np.pi))
        * inputs.eta
    )

    dt_arr = np.maximum(t_eval - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m ** 2
    lam_pow = np.power(lam, 1.5)

    x_um = np.linspace(-cfg.xz_xspan_um, cfg.xz_xspan_um, cfg.xz_nx)
    z_um = np.linspace(-cfg.xz_zmax_um, 0.0, cfg.xz_nz)

    pxy = np.stack(
        [x0 + (x_um * 1e-6) * nv[0], y0 + (x_um * 1e-6) * nv[1]],
        axis=1,
    )
    dxy2 = (pxy[:, 0:1] - x_s[None, :]) ** 2 + (pxy[:, 1:2] - y_s[None, :]) ** 2
    z_m = z_um * 1e-6
    z2 = (z_m * inputs.z_scale) ** 2

    T = np.empty((len(z_um), len(x_um)), float)
    for iz, zz2 in enumerate(z2):
        r2 = zz2 + dxy2
        term = np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :]
        s = np.sum(term * w_s[None, :], axis=1)
        T[iz, :] = inputs.T0 + pref * s

    return x_um, z_um, T


# ── SZ cross-section ──────────────────────────────────────────────────────
def sz_cross_section(
    traj_xy_m: np.ndarray,
    inputs,
    *,
    eval_idx: int,
    cfg: TrackConfig,
    P_src_W: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Along-track (S) × depth (Z) temperature field.
    Returns (s_um, z_um, T) where T.shape == (nz, ns).
    """
    traj_xy_m = np.asarray(traj_xy_m, float)
    N = traj_xy_m.shape[0]
    i = int(np.clip(eval_idx, 1, N - 1))

    t = traj_xy_m[i] - traj_xy_m[i - 1]
    tn = float(np.linalg.norm(t))
    t = np.array([1.0, 0.0], float) if tn < 1e-15 else t / tn

    p_eval = ptf._point_for_depth_eval(traj_xy_m, i, cfg.behind_um * 1e-6)
    x0, y0 = float(p_eval[0]), float(p_eval[1])

    x_s, y_s, t_s, w_s, t_eval = _history_sources(
        traj_xy_m, inputs, i, x0, y0, P_src_W, cfg
    )

    k_val = ptf.get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = (
        (6.0 * np.sqrt(3.0) * inputs.P_W)
        / (inputs.RHO * inputs.CP * np.pi * np.sqrt(np.pi))
        * inputs.eta
    )

    dt_arr = np.maximum(t_eval - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m ** 2
    lam_pow = np.power(lam, 1.5)

    s_um = np.linspace(-cfg.sz_sspan_um, cfg.sz_sspan_um, cfg.sz_ns)
    z_um = np.linspace(-cfg.sz_zmax_um, 0.0, cfg.sz_nz)

    pxy = np.stack(
        [x0 + (s_um * 1e-6) * t[0], y0 + (s_um * 1e-6) * t[1]],
        axis=1,
    )
    dxy2 = (pxy[:, 0:1] - x_s[None, :]) ** 2 + (pxy[:, 1:2] - y_s[None, :]) ** 2
    z_m = z_um * 1e-6
    z2 = (z_m * inputs.z_scale) ** 2

    T = np.empty((len(z_um), len(s_um)), float)
    for iz, zz2 in enumerate(z2):
        r2 = zz2 + dxy2
        term = np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :]
        s = np.sum(term * w_s[None, :], axis=1)
        T[iz, :] = inputs.T0 + pref * s

    return s_um, z_um, T


# ── Save helpers ──────────────────────────────────────────────────────────
def save_scalebar_only(out_png: Path, scalebar_um: float = 100.0) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.6, 0.8), dpi=320)
    ax.axis("off")
    ax.plot([0.1, 0.9], [0.45, 0.45], lw=5.0, solid_capstyle="butt", color="black")
    ax.text(0.5, 0.65, rf"{scalebar_um:.0f} $\mu$m", ha="center", va="bottom", fontsize=18)
    fig.savefig(out_png, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_field_base(ax, x_um, z_um, T, T_melt, cfg: TrackConfig):
    """
    Background image may be clipped for visualization; contours must use TRUE T.
    Return imshow handle.
    """
    if cfg.use_farimani_clip:
        Tplot = np.minimum(T, T_melt)
        vmax = float(T_melt)
    else:
        Tplot = T
        vmax = float(max(T_melt + 400.0, np.nanpercentile(T, 99.5)))

    im = ax.imshow(
        Tplot,
        origin="lower",
        extent=[x_um.min(), x_um.max(), z_um.min(), z_um.max()],
        aspect="auto",
        cmap="hot",
        vmin=float(cfg.xsec_vmin_k),
        vmax=float(vmax),
        interpolation="bilinear",
        zorder=1,
    )
    return im


def _add_scalebar(ax, x_um, z_um, cfg: TrackConfig, scalebar_um: float, color="white"):
    xr = x_um.max() - x_um.min()
    zr = z_um.max() - z_um.min()
    x0 = x_um.min() + cfg.scalebar_pad * xr
    y0 = z_um.min() + cfg.scalebar_pad * zr

    # White bar + black outline improves visibility on hot colormap
    ax.plot([x0, x0 + scalebar_um], [y0, y0], lw=5.0, solid_capstyle="butt", color="white", zorder=5)
    ax.plot([x0, x0 + scalebar_um], [y0, y0], lw=2.0, solid_capstyle="butt", color="black", alpha=0.35, zorder=4)

    ax.text(
        x0 + 0.5 * scalebar_um,
        y0 + 0.05 * zr,
        rf"{scalebar_um:.0f} $\mu$m",
        ha="center",
        va="bottom",
        color=color,
        fontsize=18,
        fontweight="bold",
        zorder=6,
    )


def _contour_with_stroke(ax, x_um, z_um, T, level, color, lw, alpha=1.0, stroke=True, zorder=10):
    """
    Draw contour with optional white under-stroke for visibility.
    """
    try:
        if stroke:
            ax.contour(x_um, z_um, T, levels=[level], colors=["white"], linewidths=lw + 1.6, alpha=0.40, zorder=zorder)
        ax.contour(x_um, z_um, T, levels=[level], colors=[color], linewidths=lw, alpha=alpha, zorder=zorder + 1)
    except Exception:
        pass


def save_field_with_contour(
    out_png: Path,
    x_um: np.ndarray,
    z_um: np.ndarray,
    T: np.ndarray,
    T_melt: float,
    *,
    title: str,
    cfg: TrackConfig,
    scalebar_um: float | None = None,
    contour_color: str = COLOR_CONTROLLED,
) -> None:
    """
    Single-run field + melt contour (no spread).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sb_um = scalebar_um if scalebar_um is not None else cfg.scalebar_um
    fig, ax = plt.subplots(figsize=(4.3, 3.1), dpi=320)

    _plot_field_base(ax, x_um, z_um, T, T_melt, cfg)

    # contour on TRUE T (unclipped) with stroke
    _contour_with_stroke(ax, x_um, z_um, T, T_melt, contour_color, lw=3.0, alpha=0.98, stroke=True)

    _add_scalebar(ax, x_um, z_um, cfg, sb_um)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    out_png: Path,
    x_um: np.ndarray,
    z_um: np.ndarray,
    T_u: np.ndarray,
    T_c: np.ndarray,
    T_melt: float,
    *,
    title: str,
    cfg: TrackConfig,
    scalebar_um: float | None = None,
) -> None:
    """
    Single-run overlay:
      background = controlled field (or clipped-controlled)
      contours = uncontrolled (black) and controlled (cyan)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sb_um = scalebar_um if scalebar_um is not None else cfg.scalebar_um
    fig, ax = plt.subplots(figsize=(4.3, 3.1), dpi=320)

    _plot_field_base(ax, x_um, z_um, T_c, T_melt, cfg)

    # Controlled (cyan) then Uncontrolled (black)
    _contour_with_stroke(ax, x_um, z_um, T_c, T_melt, COLOR_CONTROLLED, lw=3.8, alpha=0.98, stroke=True)
    _contour_with_stroke(ax, x_um, z_um, T_u, T_melt, COLOR_UNCONTROLLED, lw=3.4, alpha=0.90, stroke=True)

    _add_scalebar(ax, x_um, z_um, cfg, sb_um)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Ensemble / distribution-spread versions
# ──────────────────────────────────────────────────────────────────────────
def save_field_with_contour_spread(
    out_png: Path,
    x_um: np.ndarray,
    z_um: np.ndarray,
    T_runs: np.ndarray,   # (n_runs, nz, nx)
    T_melt: float,
    *,
    title: str,
    cfg: TrackConfig,
    scalebar_um: float | None = None,
    contour_color: str = COLOR_CONTROLLED,
    qlo: float = 0.10,
    qhi: float = 0.90,
    shadow_alpha: float = 0.45,   # darker cloud
) -> None:
    """
    Ensemble field:
      - background: mean temperature field
      - mean melt contour (strong)
      - qlo/qhi melt contours (shadow, darker so visible)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sb_um = scalebar_um if scalebar_um is not None else cfg.scalebar_um

    T_runs = np.asarray(T_runs, float)
    T_mean = T_runs.mean(axis=0)
    T_lo = np.quantile(T_runs, qlo, axis=0)
    T_hi = np.quantile(T_runs, qhi, axis=0)

    fig, ax = plt.subplots(figsize=(4.3, 3.1), dpi=320)

    _plot_field_base(ax, x_um, z_um, T_mean, T_melt, cfg)

    # shadow contours (qlo/qhi)
    _contour_with_stroke(ax, x_um, z_um, T_lo, T_melt, contour_color, lw=3.0, alpha=shadow_alpha, stroke=False, zorder=8)
    _contour_with_stroke(ax, x_um, z_um, T_hi, T_melt, contour_color, lw=3.0, alpha=shadow_alpha, stroke=False, zorder=8)

    # mean contour on top (stroke on)
    _contour_with_stroke(ax, x_um, z_um, T_mean, T_melt, contour_color, lw=3.8, alpha=0.98, stroke=True, zorder=12)

    _add_scalebar(ax, x_um, z_um, cfg, sb_um)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_overlay_spread(
    out_png: Path,
    x_um: np.ndarray,
    z_um: np.ndarray,
    Tu_runs: np.ndarray,  # (n_runs, nz, nx)
    Tc_runs: np.ndarray,  # (n_runs, nz, nx)
    T_melt: float,
    *,
    title: str,
    cfg: TrackConfig,
    scalebar_um: float | None = None,
    qlo: float = 0.10,
    qhi: float = 0.90,
    shadow_alpha: float = 0.45,   # darker cloud
) -> None:
    """
    Ensemble overlay:
      - background: mean controlled temperature field
      - controlled mean contour (cyan, strong)
      - controlled qlo/qhi contour “shadow” (cyan, visible)
      - uncontrolled mean contour (black)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sb_um = scalebar_um if scalebar_um is not None else cfg.scalebar_um

    Tu_runs = np.asarray(Tu_runs, float)
    Tc_runs = np.asarray(Tc_runs, float)

    Tu_mean = Tu_runs.mean(axis=0)
    Tc_mean = Tc_runs.mean(axis=0)
    Tc_lo = np.quantile(Tc_runs, qlo, axis=0)
    Tc_hi = np.quantile(Tc_runs, qhi, axis=0)

    fig, ax = plt.subplots(figsize=(4.3, 3.1), dpi=320)

    _plot_field_base(ax, x_um, z_um, Tc_mean, T_melt, cfg)

    # controlled shadow
    _contour_with_stroke(ax, x_um, z_um, Tc_lo, T_melt, COLOR_CONTROLLED, lw=3.0, alpha=shadow_alpha, stroke=False, zorder=9)
    _contour_with_stroke(ax, x_um, z_um, Tc_hi, T_melt, COLOR_CONTROLLED, lw=3.0, alpha=shadow_alpha, stroke=False, zorder=9)

    # controlled mean + uncontrolled mean (both with stroke)
    _contour_with_stroke(ax, x_um, z_um, Tc_mean, T_melt, COLOR_CONTROLLED, lw=3.9, alpha=0.98, stroke=True, zorder=12)
    _contour_with_stroke(ax, x_um, z_um, Tu_mean, T_melt, COLOR_UNCONTROLLED, lw=3.5, alpha=0.92, stroke=True, zorder=12)

    _add_scalebar(ax, x_um, z_um, cfg, sb_um)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ── Colorbar range helper ─────────────────────────────────────────────────
def _pick_colorbar_range(
    cfg: TrackConfig,
    inputs,
    T_fields: list[np.ndarray],
) -> tuple[float, float]:
    vmin = float(cfg.xsec_vmin_k)

    if cfg.use_farimani_clip:
        vmax = float(inputs.T_MELT)
        return vmin, vmax

    p995 = [float(np.nanpercentile(T, 99.5)) for T in T_fields if T.size > 0]
    vmax = max(p995) if len(p995) else float(inputs.T_MELT + 400.0)
    vmax = float(max(vmax, float(inputs.T_MELT) + 50.0))
    return vmin, vmax


# ── Existing single-run driver (kept compatible) ──────────────────────────
def save_all_xsections(
    traj: np.ndarray,
    inputs,
    cfg: TrackConfig,
    labels: list[str],
    abc_ks: list[int],
    idxs_u: np.ndarray,
    fr: np.ndarray,
    P_uncontrolled: np.ndarray,
    P_ctrl: np.ndarray,
    xsec_dir: Path,
) -> None:
    xsec_dir.mkdir(parents=True, exist_ok=True)

    save_scalebar_only(xsec_dir / f"scalebar_{cfg.scalebar_um:.0f}um.png", cfg.scalebar_um)

    # representative fields for colorbar scaling
    idx_rep = int(idxs_u[abc_ks[0]])
    _, _, Tuxz_rep = xz_cross_section(traj, inputs, eval_idx=idx_rep, cfg=cfg, P_src_W=P_uncontrolled)
    _, _, Tcxz_rep = xz_cross_section(traj, inputs, eval_idx=idx_rep, cfg=cfg, P_src_W=P_ctrl)

    vmin, vmax = _pick_colorbar_range(cfg, inputs, [Tuxz_rep, Tcxz_rep])

    save_temperature_colorbar_only(
        str(xsec_dir / "xsec_temperature_colorbar.png"),
        vmin=300.0,
        vmax=1600.0,
        cmap="hot",
        label="Temperature (K)",
        figsize=(1.2, 4.0),
        labelsize=30,
        ticksize=30,
        dpi=660,
        show_endpoints=True,
        endpoint_mode="text",
    )

    for lab, ksel in zip(labels, abc_ks):
        idx0 = int(idxs_u[ksel])
        f0 = float(fr[ksel])
        frac_tag = f"{f0:.3f}".replace(".", "p")

        # XZ
        x_u, z_u, Tuxz = xz_cross_section(traj, inputs, eval_idx=idx0, cfg=cfg, P_src_W=P_uncontrolled)
        x_c, z_c, Tcxz = xz_cross_section(traj, inputs, eval_idx=idx0, cfg=cfg, P_src_W=P_ctrl)

        save_field_with_contour(
            xsec_dir / f"A{lab}_uncontrolled_XZ_frac_{frac_tag}.png",
            x_u, z_u, Tuxz, inputs.T_MELT,
            title=f"Uncontrolled X–Z ({lab})", cfg=cfg,
            contour_color=COLOR_UNCONTROLLED,
        )
        save_field_with_contour(
            xsec_dir / f"A{lab}_controlled_XZ_frac_{frac_tag}.png",
            x_c, z_c, Tcxz, inputs.T_MELT,
            title=f"Controlled X–Z ({lab})", cfg=cfg,
            contour_color=COLOR_CONTROLLED,
        )
        save_overlay(
            xsec_dir / f"A{lab}_overlay_XZ_frac_{frac_tag}.png",
            x_u, z_u, Tuxz, Tcxz, inputs.T_MELT,
            title=f"Overlay X–Z ({lab})", cfg=cfg,
        )

        # SZ
        s_u, z_u2, Tusz = sz_cross_section(traj, inputs, eval_idx=idx0, cfg=cfg, P_src_W=P_uncontrolled)
        s_c, z_c2, Tcsz = sz_cross_section(traj, inputs, eval_idx=idx0, cfg=cfg, P_src_W=P_ctrl)

        save_field_with_contour(
            xsec_dir / f"A{lab}_uncontrolled_SZ_frac_{frac_tag}.png",
            s_u, z_u2, Tusz, inputs.T_MELT,
            title=f"Uncontrolled S–Z ({lab})", cfg=cfg, scalebar_um=200.0,
            contour_color=COLOR_UNCONTROLLED,
        )
        save_field_with_contour(
            xsec_dir / f"A{lab}_controlled_SZ_frac_{frac_tag}.png",
            s_c, z_c2, Tcsz, inputs.T_MELT,
            title=f"Controlled S–Z ({lab})", cfg=cfg, scalebar_um=200.0,
            contour_color=COLOR_CONTROLLED,
        )
        save_overlay(
            xsec_dir / f"A{lab}_overlay_SZ_frac_{frac_tag}.png",
            s_u, z_u2, Tusz, Tcsz, inputs.T_MELT,
            title=f"Overlay S–Z ({lab})", cfg=cfg, scalebar_um=200.0,
        )

    print(f"  Cross-sections saved → {xsec_dir.resolve()}")