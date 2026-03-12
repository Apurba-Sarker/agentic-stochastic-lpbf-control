# ============================================================
# Publication-quality KDE + 2D scatter distribution plots
# Fixes:
#  1) Legends never overlap (placed above axes + reserved margin)
#  2) Smoother KDE (use Scott bw by default + optional SavGol smoothing)
#  3) Full boxed axes (all 4 spines visible + top/right ticks)
# ============================================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from matplotlib.patches import Ellipse


# ---------------- Professional publication settings ----------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 20,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.3,
})


def _boxed_axes(ax):
    """Show a full box around plot + top/right ticks."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.tick_params(top=False, right=False, direction="out")


def _odd_leq(n: int) -> int:
    """Largest odd integer <= n."""
    return n if (n % 2 == 1) else n - 1


def _maybe_smooth(y, smooth=True, window=51, poly=3):
    """
    Optional Savitzky–Golay smoothing to remove KDE wiggles
    without changing trends too much.
    """
    y = np.asarray(y, float)
    if (not smooth) or (len(y) < 7):
        return y

    window = min(window, _odd_leq(len(y)))
    if window < 7:
        return y

    ys = savgol_filter(y, window_length=window, polyorder=min(poly, window - 1), mode="interp")
    return np.clip(ys, 0.0, None)


def _kde_curve(data, grid, bw_mult=1.0):
    """
    KDE evaluated on 'grid'.

    bw_mult is a multiplier on Scott's factor:
      - bw_mult = 1.0  -> Scott (smooth)
      - bw_mult > 1.0  -> smoother
      - bw_mult < 1.0  -> sharper (can get wiggly)
    """
    data = np.asarray(data, float)
    if len(data) < 3:
        return np.zeros_like(grid)

    kde_scott = gaussian_kde(data, bw_method="scott")
    scott_factor = kde_scott.factor  # scalar
    kde = gaussian_kde(data, bw_method=scott_factor * float(bw_mult))
    return kde(grid)


def _confidence_ellipse(
    x, y, ax,
    n_std=2.0,
    facecolor="orange",
    alpha=0.15,
    edgecolor="darkorange",
    linewidth=2.5,
):
    """Add an n-std confidence ellipse based on covariance."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or len(y) < 3:
        return

    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(ellipse)


def _save_png_and_pdf(fig, save_path: str, dpi_png: int = 300):
    """Save both PNG (high-res) and PDF (vector)."""
    root, ext = os.path.splitext(save_path)
    base = root if ext.lower() in {".png", ".pdf"} else save_path

    out_dir = os.path.dirname(base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(base + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")


def plot_distribution_kde(
    exp_w, exp_d,
    sim_w, sim_d,
    raw_w, raw_d,     # kept for API compatibility (unused)
    cal_w, cal_d,     # kept for API compatibility (unused)
    save_path: str,
    *,
    sim_label: str = "Distribution-fit",
    info_text: str | None = None,
    bw_mult: float = 1.0,
    smooth_kde: bool = True,
    smooth_window: int = 61,
    smooth_poly: int = 3,
    grid_points: int = 1200,
):
    """
    Two-panel KDE plot for Width and Depth.

    Fixes:
      - Legend placed ABOVE the plots (no overlap)
      - Reserve top margin explicitly (subplots_adjust(top=...))
      - Smoother KDE via bw_mult + optional SavGol smoothing
      - Boxed axes (all 4 spines visible)
    """
    exp_w = np.asarray(exp_w, float)
    exp_d = np.asarray(exp_d, float)
    sim_w = np.asarray(sim_w, float)
    sim_d = np.asarray(sim_d, float)

    w_all = np.concatenate([exp_w, sim_w])
    d_all = np.concatenate([exp_d, sim_d])

    w_pad = (w_all.max() - w_all.min()) * 0.12
    d_pad = (d_all.max() - d_all.min()) * 0.12

    gw = np.linspace(w_all.min() - w_pad, w_all.max() + w_pad, grid_points)
    gd = np.linspace(d_all.min() - d_pad, d_all.max() + d_pad, grid_points)

    ew = _maybe_smooth(_kde_curve(exp_w, gw, bw_mult=bw_mult),
                       smooth=smooth_kde, window=smooth_window, poly=smooth_poly)
    sw = _maybe_smooth(_kde_curve(sim_w, gw, bw_mult=bw_mult),
                       smooth=smooth_kde, window=smooth_window, poly=smooth_poly)
    ed = _maybe_smooth(_kde_curve(exp_d, gd, bw_mult=bw_mult),
                       smooth=smooth_kde, window=smooth_window, poly=smooth_poly)
    sd = _maybe_smooth(_kde_curve(sim_d, gd, bw_mult=bw_mult),
                       smooth=smooth_kde, window=smooth_window, poly=smooth_poly)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Width
    ax = axes[0]
    ax.plot(gw, ew, linewidth=3.5, label="Experimental (KDE)",
            color="#1f77b4", zorder=3, solid_capstyle="round")
    ax.plot(gw, sw, linewidth=3.5, label=sim_label,
            color="#ff7f0e", zorder=3, solid_capstyle="round")
    ax.set_xlabel(r"Width ($\mu$m)")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.8)
    _boxed_axes(ax)

    # Depth
    ax = axes[1]
    ax.plot(gd, ed, linewidth=3.5, label="Experimental (KDE)",
            color="#1f77b4", zorder=3, solid_capstyle="round")
    ax.plot(gd, sd, linewidth=3.5, label=sim_label,
            color="#ff7f0e", zorder=3, solid_capstyle="round")
    ax.set_xlabel(r"Depth ($\mu$m)")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.8)
    _boxed_axes(ax)

    # Shared legend ABOVE both plots (no overlap)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.98,
        fontsize=20,
    )

    if info_text:
        fig.text(
            0.98, 0.02, info_text,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.95),
            fontsize=14,
        )

    # Reserve top space for legend explicitly
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.78, wspace=0.30)

    _save_png_and_pdf(fig, save_path, dpi_png=300)
    plt.close(fig)


def plot_scatter_wd(
    exp_w, exp_d,
    sim_w, sim_d,
    raw_w, raw_d,    # kept for API compatibility (unused)
    cal_w, cal_d,    # kept for API compatibility (unused)
    save_path: str,
    *,
    sim_label: str = "Predicted Distribution",
    info_text: str | None = None,
):
    """
    2D scatter plot with confidence ellipses.

    Fixes:
      - Legend moved ABOVE the plot (no overlap)
      - Boxed axes
      - True square aspect (equal scaling)
    """
    exp_w = np.asarray(exp_w, float)
    exp_d = np.asarray(exp_d, float)
    sim_w = np.asarray(sim_w, float)
    sim_d = np.asarray(sim_d, float)

    w_all = np.concatenate([exp_w, sim_w])
    d_all = np.concatenate([exp_d, sim_d])

    w_min, w_max = w_all.min(), w_all.max()
    d_min, d_max = d_all.min(), d_all.max()
    w_range = w_max - w_min
    d_range = d_max - d_min

    max_range = max(w_range, d_range)
    w_center = (w_min + w_max) / 2
    d_center = (d_min + d_max) / 2

    padding = 0.15
    w_lim = [w_center - max_range * (0.5 + padding),
             w_center + max_range * (0.5 + padding)]
    d_lim = [d_center - max_range * (0.5 + padding),
             d_center + max_range * (0.5 + padding)]

    # fig, ax = plt.subplots(1, 1, figsize=(14, 6.5))
    fig, ax = plt.subplots(figsize=(6.8, 6.5))

    # Thin sim cloud to ≤400 points for visual clarity (ellipses use full arrays)
    rng_thin = np.random.default_rng(42)
    max_sim_pts = 400
    if len(sim_w) > max_sim_pts:
        idx = rng_thin.choice(len(sim_w), size=max_sim_pts, replace=False)
        sim_w_plot = sim_w[idx]
        sim_d_plot = sim_d[idx]
    else:
        sim_w_plot = sim_w
        sim_d_plot = sim_d

    # Ellipses — drawn on FULL arrays for correct covariance, sim behind exp
    _confidence_ellipse(sim_w, sim_d, ax, n_std=2.0,
                        facecolor="orange", alpha=0.18,
                        edgecolor="darkorange", linewidth=2.5)
    _confidence_ellipse(exp_w, exp_d, ax, n_std=2.0,
                        facecolor="blue", alpha=0.20,
                        edgecolor="darkblue", linewidth=2.5)

    # Points — sim thinned for clarity, exp full
    ax.scatter(sim_w_plot, sim_d_plot, s=12, alpha=0.35,
               label=sim_label,
               color="#ff7f0e", edgecolors="none",
               zorder=3)
    ax.scatter(exp_w, exp_d, s=150, alpha=0.90,
               label="Experimental Distribution",
               color="#1f77b4", edgecolors="darkblue",
               linewidths=1.5, zorder=5)

    ax.set_xlabel(r"Width ($\mu$m)")
    ax.set_ylabel(r"Depth ($\mu$m)")

    ax.set_xlim(w_lim)
    ax.set_ylim(d_lim)
    #ax.set_aspect("equal", adjustable="box")

    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    _boxed_axes(ax)

    if info_text:
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.95),
            fontsize=14,
        )

    # Legend ABOVE (no overlap with points)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        fontsize=18,
    )

    # Reserve top space for legend explicitly
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.80)

    _save_png_and_pdf(fig, save_path, dpi_png=300)
    plt.close(fig)


# ---------------- Example usage (optional) ----------------
if __name__ == "__main__":
    # Replace these with your arrays
    # exp_w, exp_d = ...
    # sim_w, sim_d = ...

    # Dummy example:
    rng = np.random.default_rng(0)
    exp_w = 141 + 1.5 * rng.standard_normal(16)
    exp_d = 122 + 2.0 * rng.standard_normal(16)
    sim_w = 141.5 + 2.8 * rng.standard_normal(900)
    sim_d = 122.3 + 2.5 * rng.standard_normal(900)

    raw_w = raw_d = cal_w = cal_d = np.array([])

    # KDE plot: try bw_mult=1.0 (Scott), or 1.15 for extra smooth
    plot_distribution_kde(
        exp_w, exp_d, sim_w, sim_d, raw_w, raw_d, cal_w, cal_d,
        save_path="dist_fixed",
        bw_mult=1.10,            # smoother than your old 0.6
        smooth_kde=True,
        smooth_window=61,
        smooth_poly=3,
        sim_label="Distribution-fit",
    )

    # Scatter plot
    plot_scatter_wd(
        exp_w, exp_d, sim_w, sim_d, raw_w, raw_d, cal_w, cal_d,
        save_path="scatter_fixed",
        sim_label="Predicted Distribution",
    )