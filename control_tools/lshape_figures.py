from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .config import TrackConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_base():
    _ensure_repo_root_on_path()
    return importlib.import_module("control_tools.lshape_4layer_multilayer_depth")


def _import_diag():
    _ensure_repo_root_on_path()
    return importlib.import_module("control_tools.lshape_diagonal_element")


def _patch_base_globals(base, cfg: TrackConfig) -> None:
    """
    Mirror the same config-to-script patching used in lshape_backend.py.
    """
    base.N_LAYERS = int(getattr(cfg, "n_layers", 4))
    base.L_MM = float(getattr(cfg, "length_mm", 3.0))
    base.W_MM = float(getattr(cfg, "width_mm", 3.0))
    base.NOTCH_X_MM = float(getattr(cfg, "notch_x_mm", 1.5))
    base.NOTCH_Y_MM = float(getattr(cfg, "notch_y_mm", 1.5))

    base.HATCH_SPACING_UM = float(cfg.hatch_spacing_um)
    base.DX_UM = float(cfg.dx_um)

    base.LAYER_THICKNESS_UM = float(getattr(cfg, "layer_thickness_um", 60.0))

    base.BEHIND_UM = float(cfg.behind_um)
    base.EVAL_STRIDE = int(cfg.eval_stride)
    base.Z_MAX_UM = float(cfg.z_max_um)
    base.Z_SAMPLES = int(cfg.z_samples)

    base.TAU_HIST_MS = None if cfg.tau_hist_ms is None else float(cfg.tau_hist_ms)
    base.HISTORY_RADIUS_UM = float(cfg.history_radius_um)

    base.TURN_SLOWDOWN = float(cfg.turn_slowdown)
    base.TURN_ANGLE_DEG = float(cfg.turn_angle_deg)
    base.JUMP_THRESHOLD_UM = float(cfg.jump_threshold_um)
    base.LASER_OFF_FOR_JUMPS = bool(cfg.laser_off_for_jumps)

    base.CALIB_JSON = str(cfg.calib_path)
    base.OUTDIR = Path(cfg.outdir)


def generate_multilayer_timelapse(cfg: TrackConfig) -> None:
    """
    Generate the 4x4 multilayer time-lapse figure using the original script helper.
    """
    base = _import_base()
    _patch_base_globals(base, cfg)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs, _ = base.load_inputs_from_calibration(str(_repo_root() / base.CALIB_JSON))

    base_traj = base.generate_horizontal_L_trajectory(
        length_mm=base.L_MM,
        width_mm=base.W_MM,
        notch_x_mm=base.NOTCH_X_MM,
        notch_y_mm=base.NOTCH_Y_MM,
        hatch_spacing_um=base.HATCH_SPACING_UM,
        dx_um=base.DX_UM,
    )

    (
        xyz_all,
        t_all,
        dt_all,
        _layer_starts,
        n_local,
        _z_step_m,
        _t_layer_s,
        _t_local,
        _dt_local,
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

    base.make_timelapse_4x4_temperature_plot(
        xyz_all=xyz_all,
        t_all=t_all,
        dt_all=dt_all,
        base_traj=base_traj,
        n_local=n_local,
        inputs=inputs,
        outpath=outdir / "timelapse_temperature_4x4_layers.pdf",
    )


def generate_fraction60_diagonal(cfg: TrackConfig) -> None:
    """
    Generate the publication-style fraction=0.6 diagonal figure using the
    existing diagonal-element script.
    """
    base = _import_base()
    diag = _import_diag()

    _patch_base_globals(base, cfg)

    # The diagonal script imports the base module internally, so we patch both
    # the already-imported base module and the one visible inside diag.
    if hasattr(diag, "base"):
        _patch_base_globals(diag.base, cfg)

    # Run its existing main() driver
    diag.main()


def generate_extra_lshape_figures(cfg: TrackConfig) -> None:
    """
    Convenience wrapper used by the backend.
    """
    if bool(getattr(cfg, "save_multilayer_3d", False)):
        generate_multilayer_timelapse(cfg)

    if bool(getattr(cfg, "save_fraction60_diag", False)):
        generate_fraction60_diagonal(cfg)
