"""
control_agent/tools.py
=======================
Concrete implementations of every tool the Ollama agent can call.

Each function is registered in TOOL_REGISTRY and follows the signature:
    fn(args: dict, state: AgentState) -> str   (human-readable result)

AgentState holds runtime mutable state (loaded config, computed results).
"""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Add project root to path so control_tools and path_temperature_field are importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import path_temperature_field as ptf
from control_tools.config import TrackConfig, load_track_config
from control_tools.trajectory import generate_trajectory
from control_tools.calibration import load_calib, make_inputs_from_calib
from control_tools import controller as ctrl_mod
from control_tools import cross_sections as xsec_mod
from control_tools import plotting as plot_mod

TRACKS_DIR  = _ROOT / "control_tools" / "tracks"
POWER_DIR   = _ROOT / "control_tools" / "power_levels"


# ── Agent state ───────────────────────────────────────────────────────────
@dataclass
class AgentState:
    cfg: TrackConfig | None = None
    calib: dict | None = None
    inputs: object = None          # ptf.PathThermalInputs
    mode: str = ""
    eta: float = 0.0
    alpha_mult: float = 0.0
    z_scale: float = 0.0
    traj: np.ndarray | None = None

    # ── quantised power levels (None = continuous) ────────────────────────
    power_levels: np.ndarray | None = None
    power_levels_source: str = ""   # human-readable label shown in summaries

    # controller outputs
    P_ctrl: np.ndarray | None = None
    P_uncontrolled: np.ndarray | None = None
    depths_u: np.ndarray | None = None
    depths_ctrl: np.ndarray | None = None
    idxs_u: np.ndarray | None = None
    fr: np.ndarray | None = None
    target: float = 0.0

    # abc selection
    labels: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    abc_ks: list[int] = field(default_factory=list)

    # stochastic ensemble results
    ensemble_result: dict | None = None


# ── Helpers ───────────────────────────────────────────────────────────────
def _resolve_track_file(name_or_path: str) -> Path:
    """
    Accept "horizontal", "45deg", "triangle", or a full/relative path.
    Returns an absolute Path to the JSON file.
    """
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p

    # bare name → look in tracks dir
    candidate = TRACKS_DIR / f"{name_or_path}.json"
    if candidate.exists():
        return candidate

    # path relative to cwd
    if p.exists():
        return p.resolve()

    # path relative to project root
    rp = _ROOT / p
    if rp.exists():
        return rp

    raise FileNotFoundError(
        f"Track file not found for {name_or_path!r}. "
        f"Available: {[f.stem for f in TRACKS_DIR.glob('*.json')]}"
    )


def _load_levels_from_path(path: Path) -> np.ndarray:
    """
    Load power levels from a file.

    Supported formats
    -----------------
    • .json  — a JSON array of numbers, e.g.  [100, 150, 200, 250, 300]
    • .txt / .csv / anything else — one number per line (blank lines ignored)
    """
    text = path.read_text().strip()
    if path.suffix.lower() == ".json":
        import json
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON power-levels file must contain a list, got {type(data)}")
        levels = np.array([float(v) for v in data], dtype=float)
    else:
        lines  = [ln.strip() for ln in text.splitlines() if ln.strip()]
        levels = np.array([float(ln.split(",")[0]) for ln in lines], dtype=float)

    levels = np.unique(levels)          # sort + deduplicate
    if len(levels) == 0:
        raise ValueError(f"No valid power levels found in {path}")
    return levels


# ── Tool: list_tracks ─────────────────────────────────────────────────────
def tool_list_tracks(args: dict, state: AgentState) -> str:
    files = sorted(TRACKS_DIR.glob("*.json"))
    if not files:
        return "No track files found in control_tools/tracks/."
    lines = ["Available track files:"]
    for f in files:
        lines.append(f"  • {f.stem}  ({f})")
    return "\n".join(lines)


# ── Tool: load_power_levels ───────────────────────────────────────────────
def tool_load_power_levels(args: dict, state: AgentState) -> str:
    """
    Load a discrete set of allowed power values for the quantised-power
    controller mode.

    args
    ----
    file : str
        Path or bare filename (without extension) to load.
        Resolution order:
          1. Absolute path if it exists.
          2. control_tools/power_levels/<file>          (exact name)
          3. control_tools/power_levels/<file>.json
          4. control_tools/power_levels/<file>.txt
          5. Path relative to project root.
        Pass "none" or "" to *clear* quantisation (return to continuous mode).
    """
    file_arg = str(args.get("file", "")).strip()

    # ── clear quantisation ────────────────────────────────────────────────
    if file_arg.lower() in ("none", "", "clear", "off", "continuous"):
        state.power_levels        = None
        state.power_levels_source = ""
        return "Quantised-power mode cleared. Controller will use continuous power."

    # ── resolve path ──────────────────────────────────────────────────────
    POWER_DIR.mkdir(parents=True, exist_ok=True)   # ensure folder exists

    candidates = [
        Path(file_arg),                               # absolute / as-is
        POWER_DIR / file_arg,                         # exact name in power_levels/
        POWER_DIR / f"{file_arg}.json",               # .json extension
        POWER_DIR / f"{file_arg}.txt",                # .txt  extension
        _ROOT / file_arg,                             # relative to project root
    ]
    resolved = None
    for c in candidates:
        if c.exists():
            resolved = c.resolve()
            break

    if resolved is None:
        # List what IS available to help the user
        available = sorted(POWER_DIR.glob("*")) if POWER_DIR.exists() else []
        av_str = (
            "\n".join(f"  • {f.name}" for f in available)
            if available
            else "  (folder is empty)"
        )
        return (
            f"ERROR: Could not find power-levels file for {file_arg!r}.\n"
            f"Looked in: {POWER_DIR}\n"
            f"Available files:\n{av_str}\n\n"
            f"Tip: create control_tools/power_levels/ and place a .json or .txt file there."
        )

    # ── load & validate ───────────────────────────────────────────────────
    try:
        levels = _load_levels_from_path(resolved)
    except Exception as exc:
        return f"ERROR loading {resolved}: {exc}"

    state.power_levels        = levels
    state.power_levels_source = str(resolved.name)

    level_str = ", ".join(f"{v:.1f}" for v in levels)
    return (
        f"Power levels loaded from: {resolved}\n"
        f"  {len(levels)} discrete levels (W): {level_str}\n"
        f"  Range: {levels.min():.1f} – {levels.max():.1f} W\n"
        f"Controller will now snap every power decision to the nearest allowed value."
    )


# ── Tool: list_power_levels ───────────────────────────────────────────────
def tool_list_power_levels(args: dict, state: AgentState) -> str:
    """List available power-level files in control_tools/power_levels/."""
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(POWER_DIR.glob("*"))
    lines = [f"Power-level files in {POWER_DIR}:"]
    if not files:
        lines.append("  (none found — create .json or .txt files here)")
    else:
        for f in files:
            lines.append(f"  • {f.name}")
    if state.power_levels is not None:
        lines.append(
            f"\nCurrently active: {state.power_levels_source}  "
            f"({len(state.power_levels)} levels)"
        )
    else:
        lines.append("\nCurrently active: none (continuous power)")
    return "\n".join(lines)


# ── Tool: load_track ──────────────────────────────────────────────────────
def tool_load_track(args: dict, state: AgentState) -> str:
    track_file = args.get("track_file", "horizontal")
    path       = _resolve_track_file(str(track_file))
    state.cfg  = load_track_config(path)

    calib_path = Path(state.cfg.calib_path)
    if not calib_path.is_absolute():
        calib_path = _ROOT / calib_path
    state.calib = load_calib(calib_path)

    state.inputs, state.mode, state.eta, state.alpha_mult, state.z_scale = (
        make_inputs_from_calib(state.calib, state.cfg)
    )

    # Build + densify trajectory
    raw_traj   = generate_trajectory(state.cfg)
    state.traj = ptf.densify_trajectory(raw_traj, step_um=state.cfg.dx_um)

    cfg = state.cfg
    summary = (
        f"Loaded track: {path.stem}\n"
        f"  type={cfg.track_type}, outdir={cfg.outdir}\n"
        f"  mode={state.mode}, seed={cfg.seed}\n"
        f"  eta={state.eta:.4f}, alpha_mult={state.alpha_mult:.4f}, "
        f"z_scale={state.z_scale:.4f}\n"
        f"  P={state.inputs.P_W:.1f} W, V={state.inputs.V_mmps:.1f} mm/s\n"
        f"  Trajectory: {len(state.traj)} points\n"
    )

    if cfg.track_type == "horizontal":
        summary += (
            f"  Geometry: track={cfg.track_mm} mm, width={cfg.width_mm} mm, "
            f"hatch={cfg.hatch_spacing_um} µm\n"
        )
    elif cfg.track_type == "45deg":
        summary += (
            f"  Geometry: domain={cfg.domain_mm} mm, hatch={cfg.hatch_spacing_um} µm\n"
        )
    elif cfg.track_type == "triangle":
        summary += (
            f"  Geometry: side={cfg.side_mm} mm, hatch={cfg.hatch_spacing_um} µm\n"
        )

    # Auto-load power levels from config if specified
    if cfg.power_levels_path:
        pl_args = {"file": cfg.power_levels_path}
        pl_result = tool_load_power_levels(pl_args, state)
        summary += f"\n{pl_result}"
    else:
        if state.power_levels is not None:
            summary += (
                f"\n  [Quantised power] keeping previously loaded levels: "
                f"{state.power_levels_source} ({len(state.power_levels)} levels)"
            )

    return summary


# ── Tool: run_uncontrolled ────────────────────────────────────────────────
def tool_run_uncontrolled(args: dict, state: AgentState) -> str:
    if state.cfg is None or state.traj is None:
        return "ERROR: No track loaded. Call load_track first."

    cfg    = state.cfg
    inputs = state.inputs
    traj   = state.traj
    N      = len(traj)
    z_um   = np.linspace(0.0, -cfg.z_max_um, cfg.z_samples)
    P0     = float(inputs.P_W)

    state.P_uncontrolled = np.full(N, P0)

    t0 = time.time()
    idxs_u, depths_u = ptf.compute_depth_evolution_over_path(
        traj, inputs, z_um,
        eval_stride=cfg.eval_stride,
        P_src_W=state.P_uncontrolled,
        **cfg.ptf_kw(),
    )
    elapsed = time.time() - t0

    state.idxs_u  = idxs_u.astype(int)
    state.depths_u = depths_u.astype(float)
    state.fr       = state.idxs_u / max(1, N - 1)

    mn, mx = state.depths_u.min(), state.depths_u.max()
    std    = np.std(state.depths_u[5:])

    return (
        f"Uncontrolled baseline complete ({elapsed:.1f}s).\n"
        f"  Depth range: {mn:.2f} .. {mx:.2f} µm\n"
        f"  Std (excluding first 5 pts): {std:.3f} µm\n"
        f"  Eval points: {len(state.idxs_u)}"
    )


# ── Tool: run_controller ──────────────────────────────────────────────────
def tool_run_controller(args: dict, state: AgentState) -> str:
    if state.cfg is None or state.traj is None:
        return "ERROR: No track loaded. Call load_track first."

    plot_mod.setup_rcparams()
    t0 = time.time()

    (
        state.P_ctrl,
        state.depths_u,
        state.depths_ctrl,
        state.idxs_u,
        state.fr,
        state.target,
    ) = ctrl_mod.run_controller(
        state.traj,
        state.inputs,
        state.cfg,
        power_levels=state.power_levels,   # ← pass quantisation levels
    )

    # Uncontrolled power array (needed for cross-sections)
    N = len(state.traj)
    state.P_uncontrolled = np.full(N, float(state.inputs.P_W))

    # ABC selection
    state.abc_ks = ctrl_mod.pick_abc_points(
        state.idxs_u,
        state.depths_u,
        state.depths_ctrl,
        state.P_uncontrolled,
        state.P_ctrl,
        state.cfg,
    )

    elapsed  = time.time() - t0
    std_u    = float(np.std(state.depths_u[5:]))
    std_c    = float(np.std(state.depths_ctrl[5:]))
    improve  = 100.0 * (1.0 - std_c / std_u) if std_u > 0 else 0.0
    range_u  = state.depths_u.max()  - state.depths_u.min()
    range_c  = state.depths_ctrl.max() - state.depths_ctrl.min()

    lines = [
        f"Controller complete ({elapsed:.1f}s).",
        f"  Target depth (internal): {state.target:.2f} µm",
        f"  Uncontrolled — range: {range_u:.2f} µm, std: {std_u:.3f} µm",
        f"  Controlled   — range: {range_c:.2f} µm, std: {std_c:.3f} µm",
        f"  Std improvement: {improve:.1f}%",
        f"  ABC points (frac): " + ", ".join(
            f"{state.labels[i]}={state.fr[k]:.3f}"
            for i, k in enumerate(state.abc_ks)
        ),
    ]

    # Tell the user which power mode was active
    if state.power_levels is not None:
        lines.append(
            f"  [Quantised power] {len(state.power_levels)} levels from "
            f"'{state.power_levels_source}'"
        )
        unique_used = np.unique(state.P_ctrl)
        lines.append(
            f"  Unique power values used: "
            + ", ".join(f"{v:.0f}" for v in unique_used)
        )
    else:
        lines.append("  [Continuous power] no quantisation applied")

    return "\n".join(lines)


# ── Tool: generate_plots ──────────────────────────────────────────────────
def tool_generate_plots(args: dict, state: AgentState) -> str:
    if state.cfg is None:
        return "ERROR: No track loaded."
    if state.depths_ctrl is None and state.depths_u is None:
        return "ERROR: No results to plot. Run run_controller (or run_uncontrolled) first."

    include_xsections = bool(args.get("include_xsections", True))
    include_scan_path = bool(args.get("include_scan_path", True))

    cfg    = state.cfg
    outdir = _ROOT / cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    saved  = []
    plot_mod.setup_rcparams()

    # ── Scan path ────────────────────────────────────────────────────────
    if include_scan_path and state.traj is not None:
        sp = outdir / "scan_path.png"
        plot_mod.plot_scan_path(state.traj, sp)
        saved.append(str(sp))

    # ── Depth plots (controlled run available) ────────────────────────────
    if state.depths_ctrl is not None and state.fr is not None:
        plot_mod.plot_depth_uncontrolled(state.fr, state.depths_u, outdir)
        saved.append(str(outdir / "depth_uncontrolled.png"))

        plot_mod.plot_depth_controlled(state.fr, state.depths_ctrl, outdir)
        saved.append(str(outdir / "depth_controlled.png"))

        plot_mod.plot_compare_depth(
            state.fr, state.depths_u, state.depths_ctrl,
            state.labels, state.abc_ks, outdir,
        )
        saved.append(str(outdir / "compare_depth.png"))

        plot_mod.plot_power_schedule(
            state.fr, state.idxs_u, state.P_ctrl, outdir
        )
        saved.append(str(outdir / "power_schedule.png"))

    elif state.depths_u is not None and state.fr is not None:
        # Uncontrolled only
        plot_mod.plot_depth_evolution(
            state.fr, state.depths_u,
            f"{cfg.track_type} — uncontrolled melt depth",
            outdir / "depth_uncontrolled.png",
        )
        saved.append(str(outdir / "depth_uncontrolled.png"))

    # ── Cross-sections ───────────────────────────────────────────────────
    if (
        include_xsections
        and cfg.save_xsections
        and state.depths_ctrl is not None
        and state.abc_ks
    ):
        xsec_dir = outdir / "cross_sections"
        xsec_mod.save_all_xsections(
            state.traj,
            state.inputs,
            cfg,
            state.labels,
            state.abc_ks,
            state.idxs_u,
            state.fr,
            state.P_uncontrolled,
            state.P_ctrl,
            xsec_dir,
        )
        saved.append(str(xsec_dir))

    return (
        f"Plots saved to {outdir.resolve()}:\n"
        + "\n".join(f"  {s}" for s in saved)
    )


# ── Tool: run_full_pipeline ───────────────────────────────────────────────
def tool_run_full_pipeline(args: dict, state: AgentState) -> str:
    results = []

    # 0. (Optional) load power levels before anything else if specified
    power_file = str(args.get("power_levels_file", "")).strip()
    if power_file:
        results.append(tool_load_power_levels({"file": power_file}, state))

    # 1. Load
    load_args = {"track_file": args.get("track_file", "horizontal")}
    results.append(tool_load_track(load_args, state))

    # 2. Controller (includes uncontrolled baseline internally)
    results.append(tool_run_controller({}, state))

    # 3. Plots
    plot_args = {
        "include_xsections": args.get("include_xsections", True),
        "include_scan_path": True,
    }
    results.append(tool_generate_plots(plot_args, state))

    return "\n\n".join(results)


# ── Tool: run_stochastic_ensemble ─────────────────────────────────────────
def tool_run_stochastic_ensemble(args: dict, state: AgentState) -> str:
    """
    Run the full controller N times, each with a DIFFERENT posterior draw
    of (eta, alpha_mult, z_scale).  Produces N power schedules and a
    consensus schedule.  This makes the control genuinely stochastic.

    args
    ----
    n_realizations : int (default 10, min 3, max 30)
    """
    if state.cfg is None or state.traj is None:
        return "ERROR: No track loaded. Call load_track first."
    if state.calib is None:
        return "ERROR: No calibration loaded."

    n_real = int(args.get("n_realizations", 10))
    n_real = int(np.clip(n_real, 3, 30))

    plot_mod.setup_rcparams()
    t0 = time.time()

    ens = ctrl_mod.run_stochastic_ensemble(
        state.traj,
        state.calib,
        state.cfg,
        n_realizations=n_real,
        power_levels=state.power_levels,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Store ensemble result
    state.ensemble_result = ens

    # Also update state with consensus (median) schedule for plotting
    state.P_ctrl     = ens["P_consensus"]
    state.depths_u   = ens["depth_u_mean"]
    state.depths_ctrl = ens["depth_ctrl_mean"]
    state.idxs_u     = ens["idxs_u"]
    state.fr         = ens["fr"]
    state.target     = float(np.median(ens["targets"]))

    N = len(state.traj)
    state.P_uncontrolled = np.full(N, float(state.inputs.P_W))

    # ABC selection on consensus results
    state.abc_ks = ctrl_mod.pick_abc_points(
        state.idxs_u, state.depths_u, state.depths_ctrl,
        state.P_uncontrolled, state.P_ctrl, state.cfg,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    sr_mean = ens["std_reduction_mean"]
    sr_std  = ens["std_reduction_std"]
    sr_min  = float(ens["std_reduction_per_real"].min())
    sr_max  = float(ens["std_reduction_per_real"].max())
    d_std_mean = float(ens["depth_ctrl_std"].mean())
    p_std_mean = float(ens["P_schedules"].std(axis=0).mean())

    lines = [
        f"Stochastic ensemble complete ({elapsed:.1f}s, {n_real} realizations).",
        f"  Std reduction: {sr_mean:.1f}% ± {sr_std:.1f}%  "
        f"[{sr_min:.1f}%, {sr_max:.1f}%]",
        f"  Controlled depth uncertainty (mean σ along path): {d_std_mean:.3f} µm",
        f"  Power schedule uncertainty (mean σ along path): {p_std_mean:.1f} W",
        f"  Consensus target depth: {state.target:.2f} µm",
        "",
        "  Per-realization breakdown:",
    ]
    for i, (theta, sr) in enumerate(
        zip(ens["thetas"], ens["std_reduction_per_real"])
    ):
        lines.append(
            f"    #{i+1}: η={theta[0]:.4f}, α={theta[1]:.4f}, "
            f"z={theta[2]:.4f} → std_red={sr:.1f}%"
        )

    if state.power_levels is not None:
        lines.append(
            f"\n  [Quantised power] {len(state.power_levels)} levels from "
            f"'{state.power_levels_source}'"
        )

    return "\n".join(lines)


# ── Tool: evaluate_control ────────────────────────────────────────────────
def tool_evaluate_control(args: dict, state: AgentState) -> str:
    """
    Evaluate control quality.  When NEEDS_IMPROVEMENT, returns concrete
    suggested_adjustments for eval_stride / n_fix_passes / spike_fix_window
    that the agent should apply via adjust_controller_settings before rerunning.
    """
    import json as _json

    if state.depths_ctrl is None or state.depths_u is None:
        return "ERROR: No control results. Run run_controller first."

    std_u   = float(np.std(state.depths_u[5:]))
    std_c   = float(np.std(state.depths_ctrl[5:]))
    improve = 100.0 * (1.0 - std_c / std_u) if std_u > 0 else 0.0

    # Spike analysis
    spike_thr   = 5.0
    spike_mask  = state.depths_ctrl < (state.target - spike_thr)
    n_spikes    = int(np.sum(spike_mask))
    worst_spike = float(state.depths_ctrl[spike_mask].min()) if n_spikes > 0 else state.target
    spike_over  = float(state.target - worst_spike) if n_spikes > 0 else 0.0

    # Power schedule
    P_min  = float(state.P_ctrl.min())
    P_max  = float(state.P_ctrl.max())
    P_eval = state.P_ctrl[state.idxs_u]
    max_dp = float(np.max(np.abs(np.diff(P_eval)))) if len(P_eval) > 1 else 0.0

    # Current settings
    cur_stride    = state.cfg.eval_stride
    cur_passes    = state.cfg.n_fix_passes
    cur_spike_win = state.cfg.spike_fix_window

    issues = []
    suggested = {}

    # Quality tier
    if improve > 95:
        quality = "EXCELLENT"
    elif improve > 80:
        quality = "GOOD"
    else:
        quality = "NEEDS_IMPROVEMENT"
        issues.append(f"Std reduction only {improve:.1f}%.")
        if cur_stride > 5:
            suggested["eval_stride"] = max(3, cur_stride // 2)
            issues.append(
                f"Suggest eval_stride {cur_stride} → {suggested['eval_stride']}."
            )

    if n_spikes > 5:
        issues.append(f"{n_spikes} spikes, worst overshoot {spike_over:.1f} µm.")
        if cur_passes < 8:
            suggested["n_fix_passes"] = min(10, cur_passes + 3)
            issues.append(
                f"Suggest n_fix_passes {cur_passes} → {suggested['n_fix_passes']}."
            )
        if spike_over > 15 and cur_spike_win < 20:
            suggested["spike_fix_window"] = min(24, cur_spike_win + 6)
            issues.append(
                f"Suggest spike_fix_window {cur_spike_win} → "
                f"{suggested['spike_fix_window']}."
            )
    elif n_spikes > 0:
        issues.append(f"{n_spikes} minor spike(s), overshoot {spike_over:.1f} µm.")

    if P_min < 50:
        issues.append(f"Power drops to {P_min:.0f} W — below practical minimum.")
    if max_dp > 100:
        issues.append(f"Max power jump {max_dp:.0f} W between eval points.")

    if state.power_levels is not None:
        n_used = len(np.unique(state.P_ctrl))
        issues.append(f"Quantised: {n_used}/{len(state.power_levels)} levels used.")
        if improve < 80:
            issues.append("Quantised power may limit control. Try finer levels.")

    action = "accept"
    if quality == "NEEDS_IMPROVEMENT" and suggested:
        action = "adjust_and_rerun"

    return _json.dumps({
        "quality": quality, "action": action,
        "std_reduction_pct":    round(improve, 2),
        "std_uncontrolled":     round(std_u, 3),
        "std_controlled":       round(std_c, 3),
        "target_depth_um":      round(state.target, 2),
        "residual_spikes":      n_spikes,
        "worst_spike_um":       round(worst_spike, 2) if n_spikes > 0 else None,
        "spike_overshoot_um":   round(spike_over, 2),
        "power_range_W":        f"{P_min:.0f}–{P_max:.0f}",
        "max_power_jump_W":     round(max_dp, 1),
        "current_settings": {
            "eval_stride": cur_stride, "n_fix_passes": cur_passes,
            "spike_fix_window": cur_spike_win,
        },
        "suggested_adjustments": suggested,
        "issues": issues,
    }, indent=2)


# ── Tool: adjust_controller_settings ──────────────────────────────────────
def tool_adjust_controller_settings(args: dict, state: AgentState) -> str:
    """
    Modify controller hyperparameters in the loaded TrackConfig BEFORE
    rerunning run_controller.  Adjustable:

      eval_stride        (int, 3–30)   — control window size
      n_fix_passes       (int, 1–12)   — spike correction iterations
      spike_fix_window   (int, 6–30)   — spike correction spatial window
      spike_threshold_um (float, 0.5–10) — spike detection threshold
      target_quantile    (float, 0.3–0.8) — depth target percentile
    """
    if state.cfg is None:
        return "ERROR: No track loaded. Call load_track first."

    changed = []

    if "eval_stride" in args:
        old = state.cfg.eval_stride
        new = int(np.clip(int(args["eval_stride"]), 3, 30))
        state.cfg.eval_stride = new
        changed.append(f"eval_stride: {old} → {new}")

    if "n_fix_passes" in args:
        old = state.cfg.n_fix_passes
        new = int(np.clip(int(args["n_fix_passes"]), 1, 12))
        state.cfg.n_fix_passes = new
        changed.append(f"n_fix_passes: {old} → {new}")

    if "spike_fix_window" in args:
        old = state.cfg.spike_fix_window
        new = int(np.clip(int(args["spike_fix_window"]), 6, 30))
        state.cfg.spike_fix_window = new
        changed.append(f"spike_fix_window: {old} → {new}")

    if "spike_threshold_um" in args:
        old = state.cfg.spike_threshold_um
        new = float(np.clip(float(args["spike_threshold_um"]), 0.5, 10.0))
        state.cfg.spike_threshold_um = new
        changed.append(f"spike_threshold_um: {old} → {new}")

    if "target_quantile" in args:
        old = state.cfg.target_quantile
        new = float(np.clip(float(args["target_quantile"]), 0.3, 0.8))
        state.cfg.target_quantile = new
        changed.append(f"target_quantile: {old} → {new}")

    if not changed:
        return "No parameters changed. Pass eval_stride, n_fix_passes, etc."

    return (
        "Controller settings adjusted:\n"
        + "\n".join(f"  {c}" for c in changed)
        + "\n\nCall run_controller to rerun with new settings."
    )


# ── Tool: validate_calibration ────────────────────────────────────────────
def tool_validate_calibration(args: dict, state: AgentState) -> str:
    """Load and validate a calibration JSON before running control."""
    import json as _json

    path_arg = str(args.get("path", "calibration_for_printing.json")).strip()
    path = Path(path_arg)
    if not path.is_absolute():
        path = _ROOT / path_arg
    if not path.exists():
        return f"ERROR: Calibration file not found: {path}"

    try:
        with open(path) as f:
            calib = _json.load(f)
    except Exception as exc:
        return f"ERROR reading {path}: {exc}"

    issues = []
    ar = calib.get("mcmc_diagnostics", {}).get("accept_rate", None)
    if ar is not None:
        if ar < 0.20:
            issues.append(f"WARNING: Accept rate {ar:.4f} low — may not have converged.")
        elif ar > 0.80:
            issues.append(f"WARNING: Accept rate {ar:.4f} high — under-explored.")

    stds = calib.get("stochastic_params", {}).get("stds", [])
    zero_stds = [i for i, s in enumerate(stds) if s == 0.0]
    if zero_stds:
        names = ["eta", "alpha_mult", "z_scale"]
        issues.append(f"NOTE: {[names[i] for i in zero_stds if i < 3]} have zero std.")

    pp = calib.get("process_params", {})
    mu = calib.get("stochastic_params", {}).get("mu", [])

    lines = [
        f"Calibration loaded: {path}",
        f"  Material: {calib.get('material', '?')}",
        f"  P={pp.get('P_W','?')} W, V={pp.get('V_mmps','?')} mm/s",
    ]
    if ar is not None:
        lines.append(f"  Accept rate: {ar:.4f}")
    if mu:
        lines.append(f"  Posterior: η={mu[0]:.4f}, α={mu[1]:.4f}, z={mu[2]:.4f}")
    if issues:
        lines += [f"  {i}" for i in issues]
    else:
        lines.append("  Quality: OK")

    return "\n".join(lines)


# ── Tool registry ─────────────────────────────────────────────────────────
TOOL_REGISTRY: dict[str, Callable[[dict, AgentState], str]] = {
    "list_tracks":                tool_list_tracks,
    "load_track":                 tool_load_track,
    "run_uncontrolled":           tool_run_uncontrolled,
    "run_controller":             tool_run_controller,
    "generate_plots":             tool_generate_plots,
    "run_full_pipeline":          tool_run_full_pipeline,
    "run_stochastic_ensemble":    tool_run_stochastic_ensemble,
    "load_power_levels":          tool_load_power_levels,
    "list_power_levels":          tool_list_power_levels,
    "evaluate_control":           tool_evaluate_control,
    "adjust_controller_settings": tool_adjust_controller_settings,
    "validate_calibration":       tool_validate_calibration,
}