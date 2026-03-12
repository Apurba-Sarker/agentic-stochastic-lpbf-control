"""
control_tools/plotting_multilayer.py
=====================================
All plotting routines for the multilayer L-shape case.
Lifted from lshape_4layer_multilayer_depth.py to guarantee identical figures.

Public API
----------
save_uncontrolled_plots(results_unc, src, base_traj, inputs, cfg, outdir)
save_control_plots(ctrl_results, cfg, outdir)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import LShapeConfig
from .plotting import setup_rcparams


def _setup_latex_style():
    """Match the standalone's style."""
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


def compute_case_stats(depth_um, startup_skip=5, spike_threshold_um=1.5):
    """Publication-style depth statistics from one depth trace."""
    d = np.asarray(depth_um, dtype=float)
    arr = d[startup_skip:].copy() if len(d) > startup_skip else d.copy()

    baseline = float(np.mean(arr))
    std_val = float(np.std(arr))

    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            if arr[i] > baseline + spike_threshold_um:
                peaks.append(arr[i])

    if len(peaks) == 0:
        return {"baseline_um": baseline, "std_um": std_val,
                "mean_spike_peak_um": baseline, "mean_spike_overshoot_um": 0.0,
                "spike_count": 0}

    peaks = np.asarray(peaks, dtype=float)
    return {
        "baseline_um": baseline, "std_um": std_val,
        "mean_spike_peak_um": float(np.mean(peaks)),
        "mean_spike_overshoot_um": float(np.mean(peaks - baseline)),
        "spike_count": int(len(peaks)),
    }


def save_uncontrolled_plots(
    results_unc: list[dict],
    src,  # MultilayerSources
    base_traj: np.ndarray,
    inputs,
    cfg: LShapeConfig,
    outdir: Path,
) -> list[str]:
    """Generate all uncontrolled multilayer plots. Returns list of saved paths."""
    _setup_latex_style()
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    try:
        import pandas as pd
        rows = [{
            "Layer": r["layer"],
            "Mean depth (um)": float(np.mean(r["depth_um"])),
            "Min depth (um)":  float(np.min(r["depth_um"])),
            "Max depth (um)":  float(np.max(r["depth_um"])),
            "Depth std (um)":  float(np.std(r["depth_um"])),
        } for r in results_unc]
        df = pd.DataFrame(rows)
        csv_path = outdir / "depth_summary_by_layer.csv"
        df.to_csv(csv_path, index=False)
        saved.append(str(csv_path))
        print(df.to_string(index=False))
    except ImportError:
        pass

    # Plot 1: XY path
    fig = plt.figure(figsize=(7.2, 6.2))
    for ell in range(cfg.n_layers):
        s = ell * src.n_local
        e = (ell + 1) * src.n_local
        xy = src.xyz_all[s:e, :2] * 1e3
        plt.plot(xy[:, 0], xy[:, 1], color="black", linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"$x$ (mm)")
    plt.ylabel(r"$y$ (mm)")
    plt.tight_layout()
    p = outdir / "lshape_xy_path_layers.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # Plot 2: Depth vs path fraction
    fig = plt.figure(figsize=(8.2, 4.8))
    for r in results_unc:
        plt.plot(r["frac"], r["depth_um"], linewidth=1.5, label=rf"Layer {r['layer']}")
    plt.xlabel(r"Path fraction")
    plt.ylabel(r"Melt-pool depth ($\mu$m)")
    plt.title(r"Predicted melt-pool depth vs path fraction")
    plt.legend()
    plt.tight_layout()
    p = outdir / "depth_vs_pathfraction_layers.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # Plot 3: Depth delta vs layer 1
    if len(results_unc) > 1:
        base_depth, base_frac = results_unc[0]["depth_um"], results_unc[0]["frac"]
        fig = plt.figure(figsize=(8.2, 4.8))
        for r in results_unc[1:]:
            plt.plot(base_frac, r["depth_um"] - base_depth, linewidth=1.4,
                     label=rf"Layer {r['layer']} $-$ Layer 1")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel(r"Path fraction")
        plt.ylabel(r"Depth change ($\mu$m)")
        plt.legend()
        plt.tight_layout()
        p = outdir / "depth_delta_vs_layer1.png"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # Plot 4: Stats by layer
    layers = np.arange(1, cfg.n_layers + 1)
    fig = plt.figure(figsize=(6.8, 4.8))
    plt.plot(layers, [np.mean(r["depth_um"]) for r in results_unc],
             marker="o", linewidth=1.8, label="Mean")
    plt.plot(layers, [np.max(r["depth_um"]) for r in results_unc],
             marker="o", linewidth=1.4, label="Max")
    plt.plot(layers, [np.min(r["depth_um"]) for r in results_unc],
             marker="o", linewidth=1.4, label="Min")
    plt.xticks(layers)
    plt.xlabel(r"Layer")
    plt.ylabel(r"Depth ($\mu$m)")
    plt.title(r"Depth statistics by layer")
    plt.legend()
    plt.tight_layout()
    p = outdir / "depth_stats_by_layer.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    return saved


def save_control_plots(
    ctrl_results: list[dict],
    cfg: LShapeConfig,
    outdir: Path,
) -> list[str]:
    """Generate all controlled multilayer plots. Returns list of saved paths."""
    _setup_latex_style()
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    try:
        import pandas as pd
        rows = [{
            "Layer":             r["layer"],
            "Target depth (um)": r["target_um"],
            "Before mean (um)":  float(np.mean(r["depth_before_um"])),
            "Before std (um)":   float(np.std(r["depth_before_um"])),
            "After mean (um)":   float(np.mean(r["depth_after_um"])),
            "After std (um)":    float(np.std(r["depth_after_um"])),
            "Mean power (W)":    r["P_mean_W"],
        } for r in ctrl_results]
        df = pd.DataFrame(rows)
        csv_path = outdir / "ctrl_summary_by_layer.csv"
        df.to_csv(csv_path, index=False)
        saved.append(str(csv_path))
        print("\nLayer summary:")
        print(df.to_string(index=False))

        # Publication stats
        pub_rows = []
        for r in ctrl_results:
            bs = compute_case_stats(r["depth_before_um"])
            as_ = compute_case_stats(r["depth_after_um"])
            std_red = 100.0 * (bs["std_um"] - as_["std_um"]) / max(bs["std_um"], 1e-12)
            ov_red = (100.0 * (bs["mean_spike_overshoot_um"] - as_["mean_spike_overshoot_um"])
                      / max(bs["mean_spike_overshoot_um"], 1e-12)) if bs["mean_spike_overshoot_um"] > 1e-12 else 0.0
            pub_rows.append({
                "Layer": r["layer"],
                "Std before (um)": bs["std_um"],
                "Std after (um)": as_["std_um"],
                "Std reduction (%)": std_red,
                "Overshoot reduction (%)": ov_red,
            })
        df_pub = pd.DataFrame(pub_rows)
        csv_path = outdir / "ctrl_publication_stats_by_layer.csv"
        df_pub.to_csv(csv_path, index=False)
        saved.append(str(csv_path))
    except ImportError:
        pass

    n_layers = len(ctrl_results)

    # Plot A: Uncontrolled vs controlled per layer
    ncols = min(n_layers, 2)
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True, sharey=True)
    if n_layers == 1:
        axes = np.array([axes])
    axes_flat = axes.ravel()
    for idx, r in enumerate(ctrl_results):
        ax = axes_flat[idx]
        ax.plot(r["frac_eval"], r["depth_before_um"],
                linewidth=1.3, linestyle="-", color="#000000", label="Uncontrolled")
        ax.plot(r["frac_eval"], r["depth_after_um"],
                linewidth=1.6, linestyle="-", color="#00B8D4", label="Controlled")
        ymin = min(np.min(r["depth_before_um"]), np.min(r["depth_after_um"]))
        ymax = max(np.max(r["depth_before_um"]), np.max(r["depth_after_um"]))
        ax.set_ylim(ymin - 1.0, ymax + 6.0)
        ax.set_title(rf"Layer {r['layer']}")
        ax.set_xlabel("Path fraction")
        ax.set_ylabel(r"Depth ($\mu$m)")
        ax.grid(False)
        ax.legend(loc="upper left", fontsize=12, frameon=True)
    for idx in range(len(ctrl_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.tight_layout()
    p = outdir / "ctrl_uncontrolled_vs_controlled_depth_by_layer.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # Plot B: Power schedule by layer
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharey=True)
    if n_layers == 1:
        axes = np.array([axes])
    axes_flat = axes.ravel()
    for idx, r in enumerate(ctrl_results):
        ax = axes_flat[idx]
        frac = np.linspace(0.0, 1.0, len(r["P_layer_W"]))
        ax.plot(frac, r["P_layer_W"], linewidth=1.2)
        ax.set_title(rf"Layer {r['layer']} power")
        ax.set_xlabel("Path fraction")
        ax.set_ylabel("Power (W)")
        ax.grid(False)
    for idx in range(len(ctrl_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.tight_layout()
    p = outdir / "ctrl_power_schedule_by_layer.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # Plot C: Depth stats bar chart
    layers = np.array([r["layer"] for r in ctrl_results])
    bmean = np.array([np.mean(r["depth_before_um"]) for r in ctrl_results])
    amean = np.array([np.mean(r["depth_after_um"])  for r in ctrl_results])
    bstd  = np.array([np.std(r["depth_before_um"])  for r in ctrl_results])
    astd  = np.array([np.std(r["depth_after_um"])   for r in ctrl_results])
    x = np.arange(len(layers))
    w = 0.36
    fig = plt.figure(figsize=(8.6, 5.0))
    plt.bar(x - w/2, bmean, w, label="Before")
    plt.bar(x + w/2, amean, w, label="After")
    plt.errorbar(x - w/2, bmean, yerr=bstd, fmt="none", capsize=3)
    plt.errorbar(x + w/2, amean, yerr=astd, fmt="none", capsize=3)
    plt.axhline(float(np.mean([r["target_um"] for r in ctrl_results])),
                linestyle="--", linewidth=1.0, label="Avg target")
    plt.xticks(x, [f"Layer {k}" for k in layers])
    plt.ylabel(r"Depth ($\mu$m)")
    plt.title("Layer-wise depth statistics before/after control")
    plt.legend()
    plt.tight_layout()
    p = outdir / "ctrl_depth_stats_before_after.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # Plot D: Power statistics by layer
    fig = plt.figure(figsize=(7.8, 4.8))
    plt.plot(layers, [r["P_mean_W"] for r in ctrl_results],
             marker="o", linewidth=1.8, label="Mean power")
    plt.plot(layers, [r["P_max_W"] for r in ctrl_results],
             marker="o", linewidth=1.3, label="Max power")
    plt.plot(layers, [r["P_min_W"] for r in ctrl_results],
             marker="o", linewidth=1.3, label="Min power")
    plt.axhline(cfg.pmin_w, linestyle=":", linewidth=1.0, label=r"$P_{\min}$")
    plt.axhline(cfg.pmax_w, linestyle=":", linewidth=1.0, label=r"$P_{\max}$")
    plt.xticks(layers)
    plt.xlabel("Layer")
    plt.ylabel("Power (W)")
    plt.title("Controlled power statistics by layer")
    plt.legend()
    plt.tight_layout()
    p = outdir / "ctrl_power_stats_by_layer.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    return saved
