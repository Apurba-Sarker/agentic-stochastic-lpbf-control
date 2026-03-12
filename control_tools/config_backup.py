"""
control_tools/config.py
=======================
TrackConfig dataclass — holds every tunable parameter for one run.
TRACK_REGISTRY — default configs keyed by track_type string.
load_track_config(path) — merges a user JSON with registry defaults.
"""

from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Literal

TrackType = Literal["horizontal", "45deg", "triangle"]


@dataclasses.dataclass
class TrackConfig:
    # ── identity ──────────────────────────────────────────────────────────
    track_type: TrackType = "horizontal"
    outdir: str = "out_control"
    calib_path: str = "calibration_for_printing.json"

    # ── geometry ──────────────────────────────────────────────────────────
    track_mm: float = 6.0          # horizontal only: track length
    width_mm: float = 3.0          # horizontal only: total width
    domain_mm: float = 3.0         # 45deg: square side
    side_mm: float = 3.0           # triangle: outer side
    hatch_spacing_um: float = 110.0
    dx_um: float = 20.0

    # ── depth grid ────────────────────────────────────────────────────────
    z_max_um: float = 520.0
    z_samples: int = 420

    # ── thermal history window ────────────────────────────────────────────
    history_radius_um: float = 300.0
    tau_hist_ms: float = 10.0
    behind_um: float = 0.0

    # ── speed model ───────────────────────────────────────────────────────
    turn_slowdown: float = 1.05
    turn_angle_deg: float = 70.0

    # ── jump / laser-off ──────────────────────────────────────────────────
    jump_threshold_um: float = 80.0
    laser_off_for_jumps: bool = True

    # ── controller ────────────────────────────────────────────────────────
    eval_stride: int = 10
    pmin_w: float = 80.0
    pmax_w: float = 400.0
    dp_max_w: float = 60.0
    spike_threshold_um: float = 1.5
    spike_fix_window: int = 12
    n_fix_passes: int = 5
    target_quantile: float = 0.55

    # ── quantised power levels ────────────────────────────────────────────
    # Path to a file listing allowed power values (W), one per line or JSON array.
    # Empty string = no quantisation (continuous control).
    # Resolved relative to the project root if not absolute.
    power_levels_path: str = ""

    # ── stochastic ────────────────────────────────────────────────────────
    stochastic_run: bool = True
    seed: int = 42

    # ── cross-sections ────────────────────────────────────────────────────
    save_xsections: bool = True
    xz_xspan_um: float = 240.0
    xz_zmax_um: float = 200.0
    xz_nx: int = 220
    xz_nz: int = 220
    sz_sspan_um: float = 800.0
    sz_zmax_um: float = 220.0
    sz_ns: int = 320
    sz_nz: int = 240
    scalebar_um: float = 200.0
    scalebar_pad: float = 0.08
    xsec_vmin_k: float = 300.0
    use_farimani_clip: bool = False

    # ── ABC selection ─────────────────────────────────────────────────────
    abc_k: int = 3
    abc_diff_min_um: float = 6.0
    abc_dp_min_w: float = 10.0
    abc_min_sep_frac: float = 0.06

    # ── uncontrolled surface-temp snapshot ────────────────────────────────
    plot_domain_mm: float = 4.0
    nx_field: int = 260
    ny_field: int = 260
    n_depth_eval: int = 320

    # ── misc ──────────────────────────────────────────────────────────────
    use_jax_if_available: bool = False

    def ptf_kw(self) -> dict:
        """Shared kwargs forwarded to most ptf.* calls."""
        return dict(
            history_radius_um=self.history_radius_um,
            behind_um=self.behind_um,
            use_jax_if_available=self.use_jax_if_available,
            turn_slowdown=self.turn_slowdown,
            turn_angle_deg=self.turn_angle_deg,
            jump_threshold_um=self.jump_threshold_um,
            laser_off_for_jumps=self.laser_off_for_jumps,
            tau_hist_ms=self.tau_hist_ms,
        )


# ── Registry of per-pattern defaults ──────────────────────────────────────
TRACK_REGISTRY: dict[str, TrackConfig] = {
    "horizontal": TrackConfig(
        track_type="horizontal",
        outdir="out_control_horizontal",
        track_mm=6.0,
        width_mm=3.0,
        hatch_spacing_um=110.0,
        dx_um=20.0,
        history_radius_um=300.0,
        turn_angle_deg=70.0,
        jump_threshold_um=80.0,
    ),
    "45deg": TrackConfig(
        track_type="45deg",
        outdir="out_control_45deg",
        domain_mm=3.0,
        hatch_spacing_um=110.0,
        dx_um=20.0,
        history_radius_um=300.0,
        turn_angle_deg=70.0,
        jump_threshold_um=80.0,
    ),
    "triangle": TrackConfig(
        track_type="triangle",
        outdir="out_control_triangle",
        side_mm=3.0,
        hatch_spacing_um=86.6,
        dx_um=20.0,
        history_radius_um=450.0,
        turn_angle_deg=60.0,
        jump_threshold_um=250.0,
    ),
}


def load_track_config(track_json_path: str | Path) -> TrackConfig:
    """
    Load a track JSON file and return a fully-populated TrackConfig.

    Required key in JSON:  ``"track_type"``  ("horizontal" | "45deg" | "triangle").
    All other keys are optional and override the registry defaults.
    """
    data: dict = json.loads(Path(track_json_path).read_text())
    track_type: str = data.pop("track_type", "horizontal")

    if track_type not in TRACK_REGISTRY:
        raise ValueError(
            f"Unknown track_type={track_type!r}. "
            f"Must be one of {list(TRACK_REGISTRY)}"
        )

    base = dataclasses.asdict(TRACK_REGISTRY[track_type])
    valid_keys = {f.name for f in dataclasses.fields(TrackConfig)}

    for k, v in data.items():
        if k in valid_keys:
            base[k] = v

    base["track_type"] = track_type
    return TrackConfig(**base)