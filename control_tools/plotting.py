"""
control_tools/plotting.py
==========================
All plotting routines shared across track types.

Public API
----------
setup_rcparams()
plot_depth_uncontrolled(fr, depths_u, outdir)
plot_depth_controlled(fr, depths_ctrl, outdir)
plot_compare_depth(fr, depths_u, depths_ctrl, labels, abc_ks, outdir)
plot_power_schedule(fr, idxs_u, P_ctrl, outdir)
plot_temp_field_and_path(traj, T, x_um, y_um, T_melt, snap_xy_m, cfg, save_path)
plot_scan_path(traj, save_path)
plot_depth_evolution(fracs, depths_um, title, save_path)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .config import TrackConfig


def setup_rcparams() -> None:
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })


# ── Depth plots ────────────────────────────────────────────────────────────
def plot_depth_uncontrolled(
    fr: np.ndarray,
    depths_u: np.ndarray,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.plot(fr, depths_u, linewidth=3.0, label="Uncontrolled", color="tab:blue")
    ax.set_xlabel("Path fraction")
    ax.set_ylabel(r"Melt depth ($\mu$m)")
    ax.set_title("Uncontrolled depth")
    ax.grid(False)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "depth_uncontrolled.png", dpi=340)
    plt.close(fig)


def plot_depth_controlled(
    fr: np.ndarray,
    depths_ctrl: np.ndarray,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.plot(fr, depths_ctrl, linewidth=3.0, label="Controlled", color="tab:orange")
    ax.set_xlabel("Path fraction")
    ax.set_ylabel(r"Melt depth ($\mu$m)")
    ax.set_title("Controlled depth (analytical inverse, two-pass)")
    ax.grid(False)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "depth_controlled.png", dpi=340)
    plt.close(fig)


def plot_compare_depth_ensemble(
    fr: np.ndarray,
    depths_u_runs: np.ndarray,      # (n_runs, n_eval)
    depths_c_runs: np.ndarray,      # (n_runs, n_eval)
    labels: list[str],
    abc_ks: list[int],
    outdir: Path,
    qlo: float = 0.10,
    qhi: float = 0.90,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Colors (consistent across ALL figures)
    C_U = "#000000"   # uncontrolled = black
    C_C = "#00B8D4"   # controlled = cyan

    du_mean = depths_u_runs.mean(axis=0)
    dc_mean = depths_c_runs.mean(axis=0)

    du_lo = np.quantile(depths_u_runs, qlo, axis=0)
    du_hi = np.quantile(depths_u_runs, qhi, axis=0)
    dc_lo = np.quantile(depths_c_runs, qlo, axis=0)
    dc_hi = np.quantile(depths_c_runs, qhi, axis=0)

    fig, ax = plt.subplots(figsize=(14.6, 6.0))

    # --- darker bands, pushed BEHIND lines ---
    ax.fill_between(fr, du_lo, du_hi, color=C_U, alpha=0.20, linewidth=0.0, zorder=1)
    ax.fill_between(fr, dc_lo, dc_hi, color=C_C, alpha=0.22, linewidth=0.0, zorder=1)

    # OPTIONAL but very effective: show a few individual runs faintly to reveal “spikiness”
    # (keeps paper-clean; just 3 traces)
    if depths_u_runs.shape[0] >= 3:
        for k in [0, depths_u_runs.shape[0]//2, depths_u_runs.shape[0]-1]:
            ax.plot(fr, depths_u_runs[k], color=C_U, alpha=0.12, linewidth=1.0, zorder=2)
            ax.plot(fr, depths_c_runs[k], color=C_C, alpha=0.12, linewidth=1.0, zorder=2)

    # --- mean curves on TOP, thicker ---
    ax.plot(fr, du_mean, linewidth=3.6, color=C_U, label="Uncontrolled", zorder=5)
    ax.plot(fr, dc_mean, linewidth=4.4, color=C_C, label="Controlled", zorder=6)

    ax.set_xlabel("Path fraction")
    ax.set_ylabel(r"Melt depth ($\mu$m)")

    # If you want positive tick labels even though depth is plotted downward:
    ax.yaxis.set_major_formatter(lambda y, pos: f"{abs(y):.0f}")

    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)

    # A/B/C markers (place on controlled mean)
    for lab, ksel in zip(labels, abc_ks):
        f0 = float(fr[ksel])
        y0 = float(dc_mean[ksel])
        ax.plot([f0], [y0], marker="o", markersize=8, color="black", zorder=10)
        ax.text(f0 + 0.01, y0 + 2, lab, fontsize=22, fontweight="bold", zorder=10)

    fig.tight_layout()
    fig.savefig(outdir / "compare_depth.png", dpi=340)
    plt.close(fig)


def plot_power_schedule_ensemble(
    fr: np.ndarray,
    idxs_u: np.ndarray,
    P_ctrl_runs: np.ndarray,
    outdir: Path,
    qlo: float = 0.10,
    qhi: float = 0.90,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    P_samp_runs = []
    for r in range(P_ctrl_runs.shape[0]):
        P_ctrl = P_ctrl_runs[r]
        P_samp = np.array([P_ctrl[int(i)] for i in idxs_u], dtype=float)
        P_samp_runs.append(P_samp)
    P_samp_runs = np.asarray(P_samp_runs, float)

    P_mean = P_samp_runs.mean(axis=0)
    P_lo = np.quantile(P_samp_runs, qlo, axis=0)
    P_hi = np.quantile(P_samp_runs, qhi, axis=0)

    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    ax.fill_between(fr, P_lo, P_hi, alpha=0.28, linewidth=0.0, zorder=1)  # darker cloud
    ax.plot(fr, P_mean, linewidth=3.6, zorder=5)                          # thicker mean

    ax.set_xlabel("Path fraction")
    ax.set_ylabel("Power (W)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir / "power_schedule.png", dpi=340)
    plt.close(fig)


def plot_compare_depth(
    fr: np.ndarray,
    depths_u: np.ndarray,
    depths_ctrl: np.ndarray,
    labels: list[str],
    abc_ks: list[int],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.6, 6.0))
    #ax.plot(fr, depths_u,    linewidth=2.6, linestyle="-", color="tab:blue",   label="Uncontrolled")
    #ax.plot(fr, depths_ctrl, linewidth=3.0, label="Controlled", color="tab:orange")
    # ax.plot(fr, depths_ctrl, linewidth=3.2, linestyle="-",  color="tab:blue", label="Controlled")
    # ax.plot(fr, depths_u,    linewidth=2.5, linestyle=":", color="#000000", label="Uncontrolled")
    ax.plot(fr, depths_u,    linewidth=3.2, linestyle="-", color="#000000", label="Uncontrolled")
    ax.plot(fr, depths_ctrl, linewidth=3.8, linestyle="-",  color="#00B8D4", label="Controlled")
    ax.set_xlabel("Path fraction")
    ax.set_ylabel(r"Melt depth ($\mu$m)")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{abs(y):.0f}")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)

    for lab, ksel in zip(labels, abc_ks):
        f0 = float(fr[ksel])
        y0 = float(depths_ctrl[ksel])
        ax.plot([f0], [y0], marker="o", markersize=8, color="black")
        ax.text(f0 + 0.01, y0 + 2, lab, fontsize=22, fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "compare_depth.png", dpi=340)
    plt.close(fig)


def plot_power_schedule(
    fr: np.ndarray,
    idxs_u: np.ndarray,
    P_ctrl: np.ndarray,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    P_samp = np.array([P_ctrl[int(i)] for i in idxs_u])
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    ax.plot(fr, P_samp, linewidth=3.0)
    ax.set_xlabel("Path fraction")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power schedule (two-pass inverse)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir / "power_schedule.png", dpi=340)
    plt.close(fig)


# ── Temperature field snapshot ────────────────────────────────────────────
def plot_temp_field_and_path(
    traj: np.ndarray,
    T: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    T_melt: float,
    snap_xy_m: np.ndarray,
    cfg: TrackConfig,
    save_path: Path,
    title: str = "Surface temperature",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(
        T, origin="lower",
        extent=[x_um.min(), x_um.max(), y_um.min(), y_um.max()],
        aspect="equal",
    )
    fig.colorbar(im, ax=ax).set_label("Temperature (K)")
    try:
        ax.contour(x_um, y_um, T, levels=[T_melt], linewidths=1.5)
    except Exception:
        pass

    pts_um = traj * 1e6
    p0 = pts_um[:-1];  p1 = pts_um[1:]
    d  = np.linalg.norm(p1 - p0, axis=1)
    segs = np.stack([p0, p1], axis=1)[d <= 1.5 * cfg.dx_um]
    ax.add_collection(LineCollection(segs, linewidths=0.8, color="white", alpha=0.5))
    ax.plot(snap_xy_m[0] * 1e6, snap_xy_m[1] * 1e6, "o",
            markersize=7, color="cyan", label="snapshot")
    ax.set_xlabel(r"x ($\mu$m)")
    ax.set_ylabel(r"y ($\mu$m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Scan path coloured by position ───────────────────────────────────────
def plot_scan_path(traj: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    traj_um = traj * 1e6
    N = len(traj_um)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = plt.cm.jet(np.linspace(0, 1, N))
    p0 = traj_um[:-1];  p1 = traj_um[1:]
    segs = np.stack([p0, p1], axis=1)
    lc   = LineCollection(segs, colors=colors[:-1], linewidths=1.8, alpha=0.9)
    ax.add_collection(lc)
    ax.plot(traj_um[0,  0], traj_um[0,  1], "o",  ms=9, color="black",
            markerfacecolor="white", markeredgewidth=2, zorder=10, label="Start")
    ax.plot(traj_um[-1, 0], traj_um[-1, 1], "^",  ms=10, color="red",
            zorder=10, label="End")
    ax.set_xlim(traj_um[:, 0].min() - 150, traj_um[:, 0].max() + 150)
    ax.set_ylim(traj_um[:, 1].min() - 150, traj_um[:, 1].max() + 150)
    ax.set_aspect("equal")
    ax.set_xlabel(r"x [$\mu$m]", fontsize=13)
    ax.set_ylabel(r"y [$\mu$m]", fontsize=13)
    ax.set_title("Scan path", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ── Uncontrolled depth-evolution (for farimani-style plots) ───────────────
def plot_depth_evolution(
    fracs: np.ndarray,
    depths_um: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(fracs, depths_um, linewidth=2.0, color="steelblue")
    ax.set_xlabel("Path fraction")
    ax.set_ylabel(r"Melt depth ($\mu$m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
