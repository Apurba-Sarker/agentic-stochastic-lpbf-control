"""
control_tools/trajectory.py
============================
Pure-numpy trajectory generators for all three scan patterns.

All functions return float64 ndarray of shape (N, 2) in **metres**.

Public API
----------
generate_trajectory(cfg)                       → dispatches by cfg.track_type
generate_horizontal_trajectory(...)
generate_45deg_trajectory(...)
generate_concentric_triangle_trajectory(...)
"""

from __future__ import annotations
import numpy as np
from .config import TrackConfig


# ── Horizontal (serpentine) hatch ─────────────────────────────────────────
def generate_horizontal_trajectory(
    length_mm: float,
    width_mm: float,
    hatch_spacing_um: float,
    dx_um: float,
) -> np.ndarray:
    """Boustrophedon horizontal hatch. Returns (N, 2) metres."""
    length_m = length_mm * 1e-3
    width_m  = width_mm  * 1e-3
    hatch_m  = hatch_spacing_um * 1e-6
    dx_m     = dx_um * 1e-6

    ys   = np.arange(0.0, width_m + 1e-15, hatch_m)
    traj = []
    for i, y in enumerate(ys):
        xs = (
            np.arange(0.0, length_m + 1e-15, dx_m)
            if i % 2 == 0
            else np.arange(length_m, -1e-15, -dx_m)
        )
        for x in xs:
            traj.append([x, y])
    return np.asarray(traj, dtype=float)


# ── 45-degree boustrophedon hatch ─────────────────────────────────────────
def generate_45deg_trajectory(
    domain_mm: float,
    hatch_spacing_um: float,
    dx_um: float,
) -> np.ndarray:
    """
    Full 45-degree boustrophedon hatch covering the [0, domain_mm]^2 square.

    Tracks run at 45° and are stitched together by boundary-edge connector
    segments → zero jumps. Returns (N, 2) metres.
    """
    domain_m = domain_mm        * 1e-3
    hatch_m  = hatch_spacing_um * 1e-6
    dx_m     = dx_um            * 1e-6

    angle_rad = np.radians(45.0)
    d_vec = np.array([ np.cos(angle_rad),  np.sin(angle_rad)])
    p_vec = np.array([-np.sin(angle_rad),  np.cos(angle_rad)])

    centre    = np.array([domain_m / 2.0, domain_m / 2.0])
    half_span = domain_m * np.sqrt(2.0) / 2.0

    n_tracks  = int(np.ceil(2.0 * half_span / hatch_m)) + 1
    p_offsets = np.linspace(-half_span, half_span, n_tracks)

    def clip_to_square(origin, direction, L):
        t_lo, t_hi = -np.inf, np.inf
        for dim in range(2):
            if abs(direction[dim]) < 1e-15:
                if origin[dim] < -1e-9 or origin[dim] > L + 1e-9:
                    return None
            else:
                t1 = (0.0 - origin[dim]) / direction[dim]
                t2 = (L   - origin[dim]) / direction[dim]
                t_lo = max(t_lo, min(t1, t2))
                t_hi = min(t_hi, max(t1, t2))
        if t_hi <= t_lo + 1e-9:
            return None
        return t_lo, t_hi

    def connector(p_from, p_to, step):
        dist = np.linalg.norm(p_to - p_from)
        if dist < 1e-9:
            return np.empty((0, 2), float)
        n  = max(2, int(np.ceil(dist / step)) + 1)
        ts = np.linspace(0.0, 1.0, n)[1:]   # skip t=0 (= end of previous track)
        return p_from[None, :] + ts[:, None] * (p_to - p_from)[None, :]

    raw_tracks  = []
    track_count = 0
    for po in p_offsets:
        origin = centre + po * p_vec
        result = clip_to_square(origin, d_vec, domain_m)
        if result is None:
            continue
        t_lo, t_hi = result
        n_pts = max(2, int(np.ceil((t_hi - t_lo) / dx_m)) + 1)
        ts  = np.linspace(t_lo, t_hi, n_pts)
        pts = origin[None, :] + ts[:, None] * d_vec[None, :]
        if track_count % 2 == 1:
            pts = pts[::-1]
        raw_tracks.append(pts)
        track_count += 1

    if not raw_tracks:
        raise RuntimeError("No 45-degree tracks generated — check domain/hatch parameters.")

    traj_all = [raw_tracks[0]]
    for i in range(len(raw_tracks) - 1):
        conn = connector(raw_tracks[i][-1], raw_tracks[i + 1][0], dx_m)
        if len(conn):
            traj_all.append(conn)
        traj_all.append(raw_tracks[i + 1])

    return np.vstack(traj_all).astype(float)


# ── Concentric equilateral triangle hatch ─────────────────────────────────
def generate_concentric_triangle_trajectory(
    side_mm: float,
    hatch_spacing_um: float,
    dx_um: float,
) -> np.ndarray:
    """
    Concentric equilateral triangles spiraling inward, fully connected.

    - Outermost triangle: side=side_mm, equilateral, pointing UP
    - Each inner triangle inset by hatch_spacing_um perpendicular to each side
    - Connector segments stitch end→start with zero jumps
    - Scan direction: CCW (bottom-left → bottom-right → apex → …)

    Returns (N, 2) metres.
    """
    s0      = float(side_mm)
    hatch_m = hatch_spacing_um * 1e-6
    dx_m    = dx_um * 1e-6

    # inradius of equilateral triangle r = s / (2√3)
    r0 = s0 * 1e-3 * np.sqrt(3) / 6.0
    n_tri = int(r0 / hatch_m)

    cx = s0 * 1e-3 / 2.0
    cy = r0   # centroid y (inradius above base)

    def triangle_vertices(side_m, cx, cy):
        r_in  = side_m / (2.0 * np.sqrt(3))
        r_out = 2.0 * r_in
        return np.array([
            [cx - side_m / 2.0, cy - r_in],   # bottom-left
            [cx + side_m / 2.0, cy - r_in],   # bottom-right
            [cx,                cy + r_out],   # apex
        ])

    def sample_edge(p0, p1, step):
        dist = np.linalg.norm(p1 - p0)
        n    = max(2, int(np.ceil(dist / step)) + 1)
        ts   = np.linspace(0.0, 1.0, n)[:-1]   # exclude endpoint
        return p0[None, :] + ts[:, None] * (p1 - p0)[None, :]

    def triangle_path(side_m, cx, cy, step):
        v = triangle_vertices(side_m, cx, cy)
        return np.vstack([sample_edge(v[i], v[(i + 1) % 3], step) for i in range(3)])

    def connector(p_from, p_to, step):
        dist = np.linalg.norm(p_to - p_from)
        if dist < 1e-9:
            return np.empty((0, 2), float)
        n  = max(2, int(np.ceil(dist / step)) + 1)
        ts = np.linspace(0.0, 1.0, n)[1:]
        return p_from[None, :] + ts[:, None] * (p_to - p_from)[None, :]

    traj_all = []
    side_k   = s0 * 1e-3

    for k in range(n_tri + 1):
        if side_k < 1e-4:
            break
        pts = triangle_path(side_k, cx, cy, dx_m)
        if k == 0:
            traj_all.append(pts)
        else:
            conn = connector(traj_all[-1][-1], pts[0], dx_m)
            if len(conn):
                traj_all.append(conn)
            traj_all.append(pts)
        side_k -= 2.0 * np.sqrt(3) * hatch_m

    if not traj_all:
        raise RuntimeError("No triangle loops generated — check side/hatch parameters.")

    return np.vstack(traj_all).astype(float)


# ── Dispatcher ────────────────────────────────────────────────────────────
def generate_trajectory(cfg: TrackConfig) -> np.ndarray:
    """Generate the raw trajectory for any supported track_type."""
    if cfg.track_type == "horizontal":
        return generate_horizontal_trajectory(
            cfg.track_mm, cfg.width_mm, cfg.hatch_spacing_um, cfg.dx_um
        )
    elif cfg.track_type == "45deg":
        return generate_45deg_trajectory(
            cfg.domain_mm, cfg.hatch_spacing_um, cfg.dx_um
        )
    elif cfg.track_type == "triangle":
        return generate_concentric_triangle_trajectory(
            cfg.side_mm, cfg.hatch_spacing_um, cfg.dx_um
        )
    else:
        raise ValueError(f"Unknown track_type={cfg.track_type!r}")
