from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrackConfig
from .calibration import load_calib, make_inputs_from_calib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_base_module():
    """
    Import the existing standalone L-shape multilayer script as a module.
    """
    _ensure_repo_root_on_path()
    return importlib.import_module("control_tools.lshape_4layer_multilayer_depth")


def _patch_base_globals(base, cfg: TrackConfig) -> None:
    """
    Push TrackConfig values into the existing standalone script so its internal
    functions run with the same settings the agent selected from the track JSON.
    """
    # Geometry
    base.N_LAYERS = int(getattr(cfg, "n_layers", 4))
    base.L_MM = float(getattr(cfg, "length_mm", 3.0))
    base.W_MM = float(getattr(cfg, "width_mm", 3.0))
    base.NOTCH_X_MM = float(getattr(cfg, "notch_x_mm", 1.5))
    base.NOTCH_Y_MM = float(getattr(cfg, "notch_y_mm", 1.5))

    # Trajectory
    base.HATCH_SPACING_UM = float(cfg.hatch_spacing_um)
    base.DX_UM = float(cfg.dx_um)

    # Layering
    base.LAYER_THICKNESS_UM = float(getattr(cfg, "layer_thickness_um", 60.0))

    # Depth evaluation
    base.BEHIND_UM = float(cfg.behind_um)
    base.EVAL_STRIDE = int(cfg.eval_stride)
    base.Z_MAX_UM = float(cfg.z_max_um)
    base.Z_SAMPLES = int(cfg.z_samples)

    # Thermal-history filters
    base.TAU_HIST_MS = None if cfg.tau_hist_ms is None else float(cfg.tau_hist_ms)
    base.HISTORY_RADIUS_UM = float(cfg.history_radius_um)

    # Time-axis construction
    base.TURN_SLOWDOWN = float(cfg.turn_slowdown)
    base.TURN_ANGLE_DEG = float(cfg.turn_angle_deg)
    base.JUMP_THRESHOLD_UM = float(cfg.jump_threshold_um)
    base.LASER_OFF_FOR_JUMPS = bool(cfg.laser_off_for_jumps)

    # Files / outputs
    base.CALIB_JSON = str(cfg.calib_path)
    base.OUTDIR = Path(cfg.outdir)

    # Keep the standalone script's control switch on
    base.RUN_CONTROL = True

    # Patch the dataclass instance CTRL used internally by the standalone script
    ctrl = base.CTRL
    ctrl.z_max_um = float(cfg.z_max_um)
    ctrl.z_samples = int(cfg.z_samples)
    ctrl.history_radius_um = float(cfg.history_radius_um)
    ctrl.tau_hist_ms = None if cfg.tau_hist_ms is None else float(cfg.tau_hist_ms)
    ctrl.behind_um = float(cfg.behind_um)

    ctrl.turn_slowdown = float(cfg.turn_slowdown)
    ctrl.turn_angle_deg = float(cfg.turn_angle_deg)
    ctrl.jump_threshold_um = float(cfg.jump_threshold_um)
    ctrl.laser_off_for_jumps = bool(cfg.laser_off_for_jumps)

    ctrl.eval_stride = int(cfg.eval_stride)
    ctrl.pmin_w = float(cfg.pmin_w)
    ctrl.pmax_w = float(cfg.pmax_w)
    ctrl.dp_max_w = float(cfg.dp_max_w)
    ctrl.spike_threshold_um = float(cfg.spike_threshold_um)
    ctrl.spike_fix_window = int(cfg.spike_fix_window)
    ctrl.n_fix_passes = int(cfg.n_fix_passes)
    ctrl.target_quantile = float(cfg.target_quantile)


def _summarize_layer_stats(ctrl_results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Build simple aggregate statistics for the agent.
    """
    before_std = np.array([np.std(r["depth_before_um"]) for r in ctrl_results], dtype=float)
    after_std = np.array([np.std(r["depth_after_um"]) for r in ctrl_results], dtype=float)

    before_mean = float(np.mean(before_std))
    after_mean = float(np.mean(after_std))

    improvement = 100.0 * (before_mean - after_mean) / max(before_mean, 1e-12)

    return {
        "uncontrolled_std_um": before_mean,
        "controlled_std_um": after_mean,
        "improvement_pct": improvement,
    }


def _quality_from_improvement(improvement_pct: float) -> tuple[str, str]:
    """
    Simple agent-facing quality label.
    """
    if improvement_pct >= 95.0:
        return "EXCELLENT", "accept"
    if improvement_pct >= 75.0:
        return "GOOD", "accept"
    return "NEEDS_IMPROVEMENT", "adjust_and_rerun"


def run_lshape_control(
    cfg: TrackConfig,
    *,
    include_extra_3d: bool | None = None,
) -> dict[str, Any]:
    """
    Run the 4-layer L-shape multilayer control backend using the existing
    standalone L-shape script, but package the result in the same style as the
    ControlAgent's 2D backend.

    Returns a dict with agent-friendly summary fields.
    """
    base = _import_base_module()
    _patch_base_globals(base, cfg)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load calibration and build inputs using the same calibration helpers
    # calib = load_calib(cfg.calib_path)
    # inputs, mode_str, eta, alpha_mult, z_scale = make_inputs_from_calib(calib, cfg)
    calib = load_calib(cfg.calib_path)

    cfg_lshape = TrackConfig(**vars(cfg))
    cfg_lshape.stochastic_run = False

    inputs, mode_str, eta, alpha_mult, z_scale = make_inputs_from_calib(calib, cfg_lshape)


    # Build 1-layer base XY trajectory
    base_traj = base.generate_horizontal_L_trajectory(
        length_mm=base.L_MM,
        width_mm=base.W_MM,
        notch_x_mm=base.NOTCH_X_MM,
        notch_y_mm=base.NOTCH_Y_MM,
        hatch_spacing_um=base.HATCH_SPACING_UM,
        dx_um=base.DX_UM,
    )

    # Stack layers in 3D
    (
        xyz_all,
        t_all,
        dt_all,
        layer_start_idxs,
        n_local,
        z_step_m,
        t_layer_s,
        t_local,
        dt_local,
    ) = base.build_multilayer_sources(
        base_traj_xy=base_traj,
        n_layers=base.N_LAYERS,
        layer_thickness_um=base.LAYER_THICKNESS_UM,
        inputs=inputs,
        turn_slowdown=base.TURN_SLOWDOWN,
        turn_angle_deg=base.TURN_ANGLE_DEG,
        jump_threshold_um=base.JUMP_THRESHOLD_UM,
        laser_off_for_jumps=base.LASER_OFF_FOR_JUMPS,
        interlayer_dwell_ms=getattr(base, "INTERLAYER_DWELL_MS", 0.0),
    )

    # Uncontrolled multilayer depth evolution
    z_probe_um = np.linspace(0.0, cfg.z_max_um, int(cfg.z_samples))
    unc_results: list[dict[str, Any]] = []

    for ell in range(base.N_LAYERS):
        idxs_local, frac, depths, peak_temps = base.depth_evolution_for_layer(
            base_traj_xy=base_traj,
            xyz_all=xyz_all,
            t_all=t_all,
            dt_all=dt_all,
            inputs=inputs,
            layer_idx=ell,
            n_local=n_local,
            z_probe_um=z_probe_um,
            dt_local=dt_local,
            eval_stride=cfg.eval_stride,
            history_radius_um=cfg.history_radius_um,
            tau_hist_ms=cfg.tau_hist_ms,
            behind_um=cfg.behind_um,
        )
        unc_results.append(
            {
                "layer": ell + 1,
                "idxs_local": idxs_local,
                "frac": frac,
                "depth_um": depths,
                "peak_temps_k": peak_temps,
            }
        )

    # Save the standard uncontrolled figures from your original script
    base.save_uncontrolled_plots(
        unc_results,
        xyz_all,
        t_all,
        dt_all,
        base_traj,
        n_local,
        inputs,
        outdir,
    )

    # Controlled multilayer backend
    P_all, ctrl_results = base.run_multilayer_controller(
        base_traj=base_traj,
        xyz_all=xyz_all,
        t_all=t_all,
        dt_all=dt_all,
        inputs=inputs,
        n_local=n_local,
        dt_local=dt_local,
        cfg=base.CTRL,
    )

    # Save the standard control figures from your original script
    base.save_control_plots(ctrl_results, outdir)

    # Optional extras
    if include_extra_3d is None:
        include_extra_3d = bool(getattr(cfg, "save_multilayer_3d", False)) or bool(
            getattr(cfg, "save_fraction60_diag", False)
        )

    if include_extra_3d:
        from .lshape_figures import generate_extra_lshape_figures

        generate_extra_lshape_figures(cfg)

    # Aggregate stats for the agent
    stats = _summarize_layer_stats(ctrl_results)
    quality, action = _quality_from_improvement(stats["improvement_pct"])

    saved_files = [
        "lshape_xy_path_layers.pdf",
        "depth_vs_pathfraction_4layers.pdf",
        "depth_delta_vs_layer1.pdf",
        "depth_stats_by_layer.pdf",
        "depth_summary_by_layer.csv",
        "ctrl_uncontrolled_vs_controlled_depth_by_layer.pdf",
        "ctrl_power_schedule_by_layer.pdf",
        "ctrl_depth_stats_before_after.pdf",
        "ctrl_power_stats_by_layer.pdf",
        "ctrl_summary_by_layer.csv",
    ]

    if bool(getattr(cfg, "save_fraction60_diag", False)):
        saved_files.append("timelapse_3d_lshape_fraction60_tworow_updated.pdf")
    if bool(getattr(cfg, "save_multilayer_3d", False)):
        saved_files.append("timelapse_temperature_4x4_layers.pdf")

    return {
        "track_type": "lshape_4layer",
        "outdir": str(outdir),
        "mode": mode_str,
        "eta": float(eta),
        "alpha_mult": float(alpha_mult),
        "z_scale": float(z_scale),
        "n_layers": int(base.N_LAYERS),
        "n_local": int(n_local),
        "layer_results": ctrl_results,
        "uncontrolled_results": unc_results,
        "uncontrolled_std_um": stats["uncontrolled_std_um"],
        "controlled_std_um": stats["controlled_std_um"],
        "improvement_pct": stats["improvement_pct"],
        "quality": quality,
        "action": action,
        "saved_files": saved_files,
    }
