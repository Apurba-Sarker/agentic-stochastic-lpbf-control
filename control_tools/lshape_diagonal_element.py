#!/usr/bin/env python3
"""
lshape_diagonal_element_fraction60_tworow_updated.py

Two-row publication figure for the L-shape multilayer case:
- (a) top-left: auxiliary 3D CAD-style L-shape geometry
- (b) top-right: 2D track diagram with arrows and the f=0.6 point
- (c) bottom row: 4 diagonal temperature panels (Layers 1..4), each taken at
      path fraction = 0.6 of that layer
- right: shared temperature colorbar only for bottom row
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import path_temperature_field as ptf
import lshape_4layer_multilayer_depth as base


# ---------------------------------------------------------------------
# User knobs
# ---------------------------------------------------------------------
OUTNAME = "timelapse_3d_lshape_fraction60_tworow_updated"

Z_VISUAL_SCALE = 15.0

TOP_NX = 70
TOP_NY = 70
SIDE_NZ = 26
CUT_NX = 90
CUT_NZ = 70

HISTORY_RADIUS_UM = 1000.0
TAU_HIST_MS = 20.0

CUT_TAU_HIST_MS = 4.0
CUT_T_MIN_K = 1300.0

TMIN_K = 0.0
TMAX_K = 2200.0

VIEW_ELEV = 24
VIEW_AZIM = -58

WIRE_COLOR = "0.20"
WIRE_LW = 0.55

SHOW_CUT_PLANE = True
CUT_PLANE_ALPHA = 0.95

EPS_MM = 1e-4

SNAPSHOT_FRACTION = 0.60

# font scaling
FONT_SCALE = 1.20

# CAD-style colors
CAD_FACE = "#9a9a9a"
CAD_EDGE = "#4d4d4d"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
def apply_pub_style():
    base.setup_latex_style()

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 18 * FONT_SCALE,
        "axes.labelsize": 18 * FONT_SCALE,
        "xtick.labelsize": 16 * FONT_SCALE,
        "ytick.labelsize": 16 * FONT_SCALE,
        "legend.fontsize": 14 * FONT_SCALE,
        "axes.titlesize": 18 * FONT_SCALE,
    })


# ---------------------------------------------------------------------
# Generic 3D temperature evaluator
# ---------------------------------------------------------------------
def temperature_field_on_surface_at_time(
    xyz_all,
    t_all,
    dt_all,
    inputs,
    t_eval,
    X_m,
    Y_m,
    Z_m,
    *,
    history_radius_um=1200.0,
    tau_hist_ms=None,
    P_src_W=None,
    chunk_size=2500,
):
    src_mask = np.asarray(t_all) <= float(t_eval)
    if not np.any(src_mask):
        return np.full_like(X_m, float(inputs.T0), dtype=float)

    src_xyz = np.asarray(xyz_all, dtype=float)[src_mask]
    t_s = np.asarray(t_all, dtype=float)[src_mask]
    w_s = np.asarray(dt_all, dtype=float)[src_mask].copy()

    if P_src_W is not None:
        P0 = max(float(inputs.P_W), 1e-12)
        w_s *= np.asarray(P_src_W, dtype=float)[src_mask] / P0

    if tau_hist_ms is not None:
        keep_t = (float(t_eval) - t_s) <= (float(tau_hist_ms) * 1e-3)
        if np.any(keep_t):
            src_xyz = src_xyz[keep_t]
            t_s = t_s[keep_t]
            w_s = w_s[keep_t]

    r_hist = float(history_radius_um) * 1e-6
    xmin, xmax = float(np.min(X_m)), float(np.max(X_m))
    ymin, ymax = float(np.min(Y_m)), float(np.max(Y_m))
    zmin, zmax = float(np.min(Z_m)), float(np.max(Z_m))

    dx_box = np.maximum.reduce([
        xmin - src_xyz[:, 0],
        np.zeros(len(src_xyz)),
        src_xyz[:, 0] - xmax
    ])
    dy_box = np.maximum.reduce([
        ymin - src_xyz[:, 1],
        np.zeros(len(src_xyz)),
        src_xyz[:, 1] - ymax
    ])
    dz_box = np.maximum.reduce([
        zmin - src_xyz[:, 2],
        np.zeros(len(src_xyz)),
        src_xyz[:, 2] - zmax
    ])
    dz_box *= float(inputs.z_scale)
    keep_s = (dx_box**2 + dy_box**2 + dz_box**2) <= r_hist**2

    if np.any(keep_s):
        src_xyz = src_xyz[keep_s]
        t_s = t_s[keep_s]
        w_s = w_s[keep_s]
    else:
        src_xyz = src_xyz[-1:]
        t_s = t_s[-1:]
        w_s = w_s[-1:]

    k_val = ptf.get_k_in718_like(inputs.T_MELT)
    alpha = (k_val / (inputs.RHO * inputs.CP)) * inputs.alpha_mult
    pref = ptf._prefactor(inputs.P_W, inputs.RHO, inputs.CP) * inputs.eta

    dt_arr = np.maximum(float(t_eval) - t_s, 1e-12)
    lam = 12.0 * alpha * dt_arr + inputs.sigma_m**2
    lam_pow = lam ** 1.5

    sx, sy, sz = src_xyz[:, 0], src_xyz[:, 1], src_xyz[:, 2]
    z_scale = float(inputs.z_scale)

    Xf = np.asarray(X_m, dtype=float).ravel()
    Yf = np.asarray(Y_m, dtype=float).ravel()
    Zf = np.asarray(Z_m, dtype=float).ravel()
    Tf = np.empty_like(Xf, dtype=float)

    for i0 in range(0, Xf.size, int(chunk_size)):
        i1 = min(i0 + int(chunk_size), Xf.size)
        Xc = Xf[i0:i1]
        Yc = Yf[i0:i1]
        Zc = Zf[i0:i1]

        r2 = (
            (Xc[:, None] - sx[None, :]) ** 2
            + (Yc[:, None] - sy[None, :]) ** 2
            + ((Zc[:, None] - sz[None, :]) * z_scale) ** 2
        )

        Tf[i0:i1] = float(inputs.T0) + pref * np.sum(
            np.exp(-3.0 * r2 / lam[None, :]) / lam_pow[None, :] * w_s[None, :],
            axis=1,
        )

    return Tf.reshape(np.shape(X_m))


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def lshape_polygon_xy_mm():
    return np.array([
        [0.0, 0.0],
        [base.L_MM, 0.0],
        [base.L_MM, base.NOTCH_Y_MM],
        [base.NOTCH_X_MM, base.NOTCH_Y_MM],
        [base.NOTCH_X_MM, base.W_MM],
        [0.0, base.W_MM],
        [0.0, 0.0],
    ], dtype=float)


def lshape_mask_xy(X_m, Y_m):
    return base._lshape_footprint_mask(
        X_m, Y_m,
        base.L_MM, base.W_MM,
        base.NOTCH_X_MM, base.NOTCH_Y_MM
    )


def valid_x_limit_for_y_mm(y_mm: float) -> float:
    if y_mm <= base.NOTCH_Y_MM + 1e-12:
        return base.L_MM
    return base.NOTCH_X_MM


def make_clipped_hot_cmap():
    return plt.get_cmap("hot").copy()


def masked_surface(ax, X_mm, Y_mm, Z_mm, T, valid_mask, cmap, norm, alpha_scale=1.0):
    Tm = np.ma.masked_where(~valid_mask, T)
    facecolors = cmap(norm(Tm))
    facecolors[..., -1] = np.where(np.ma.getmaskarray(Tm), 0.0, alpha_scale)
    ax.plot_surface(
        X_mm, Y_mm, Z_mm,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )


def solid_surface(ax, X_mm, Y_mm, Z_mm, color=CAD_FACE, edgecolor=CAD_EDGE, alpha=1.0):
    ax.plot_surface(
        X_mm, Y_mm, Z_mm,
        color=color,
        edgecolor=edgecolor,
        linewidth=0.25,
        antialiased=True,
        shade=True,
        alpha=alpha,
    )


def plot_wire_outline(ax, z_bot_mm, z_top_mm, color=WIRE_COLOR, alpha=1.0):
    poly = lshape_polygon_xy_mm()
    ax.plot(poly[:, 0], poly[:, 1], z_bot_mm, color=color, lw=WIRE_LW, alpha=alpha)
    ax.plot(poly[:, 0], poly[:, 1], z_top_mm, color=color, lw=WIRE_LW, alpha=alpha)
    for x, y in poly[:-1]:
        ax.plot([x, x], [y, y], [z_bot_mm, z_top_mm], color=color, lw=WIRE_LW, alpha=alpha)


# ---------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------
def snapshot_time_at_layer_fraction(t_all, n_local, layer_idx, frac=0.60):
    frac = float(np.clip(frac, 0.0, 1.0))
    i0 = int(layer_idx * n_local)
    i1 = int(min((layer_idx + 1) * n_local - 1, len(t_all) - 1))
    idx = i0 + int(round(frac * (i1 - i0)))
    idx = max(i0, min(idx, i1))
    return float(t_all[idx]), idx


def end_of_layer_time(t_all, n_local, layer_idx):
    end_idx = min((layer_idx + 1) * n_local - 1, len(t_all) - 1)
    return float(t_all[end_idx])


def active_layer_and_last_source_idx(t_all, n_local, t_snap):
    idx = int(np.searchsorted(np.asarray(t_all), t_snap, side="right") - 1)
    idx = max(0, min(idx, len(t_all) - 1))
    layer_idx = min(int(idx // n_local), base.N_LAYERS - 1)
    return layer_idx, idx


# ---------------------------------------------------------------------
# Draw one physical layer
# ---------------------------------------------------------------------
def draw_top_surface(ax, xyz_all, t_all, dt_all, inputs, t_snap, layer_idx, cmap, norm, alpha_scale=1.0):
    h_mm = base.LAYER_THICKNESS_UM * 1e-3
    z_top_mm = (layer_idx + 1) * h_mm

    x_mm = np.linspace(0.0, base.L_MM, TOP_NX)
    y_mm = np.linspace(0.0, base.W_MM, TOP_NY)
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
    X_m = X_mm * 1e-3
    Y_m = Y_mm * 1e-3
    Z_m = np.full_like(X_m, z_top_mm * 1e-3)

    T = temperature_field_on_surface_at_time(
        xyz_all, t_all, dt_all, inputs, t_snap,
        X_m, Y_m, Z_m,
        history_radius_um=HISTORY_RADIUS_UM,
        tau_hist_ms=TAU_HIST_MS,
        chunk_size=1500,
    )

    mask = lshape_mask_xy(X_m, Y_m)
    masked_surface(
        ax,
        X_mm, Y_mm, np.full_like(X_mm, z_top_mm * Z_VISUAL_SCALE),
        T, mask, cmap, norm, alpha_scale=alpha_scale
    )


def draw_front_face(ax, xyz_all, t_all, dt_all, inputs, t_snap, layer_idx, cmap, norm, alpha_scale=1.0):
    h_mm = base.LAYER_THICKNESS_UM * 1e-3
    z_bot_mm = layer_idx * h_mm
    z_top_mm = (layer_idx + 1) * h_mm

    x_mm = np.linspace(0.0, base.L_MM, TOP_NX)
    z_mm = np.linspace(z_bot_mm, z_top_mm, SIDE_NZ)
    X_mm, Z_mm = np.meshgrid(x_mm, z_mm)
    Y_mm = np.zeros_like(X_mm)

    T = temperature_field_on_surface_at_time(
        xyz_all, t_all, dt_all, inputs, t_snap,
        X_mm * 1e-3, Y_mm * 1e-3, Z_mm * 1e-3,
        history_radius_um=HISTORY_RADIUS_UM,
        tau_hist_ms=TAU_HIST_MS,
        chunk_size=1200,
    )

    masked_surface(
        ax,
        X_mm, Y_mm + EPS_MM, Z_mm * Z_VISUAL_SCALE,
        T, np.ones_like(T, dtype=bool), cmap, norm,
        alpha_scale=alpha_scale
    )


def draw_left_face(ax, xyz_all, t_all, dt_all, inputs, t_snap, layer_idx, cmap, norm, alpha_scale=1.0):
    h_mm = base.LAYER_THICKNESS_UM * 1e-3
    z_bot_mm = layer_idx * h_mm
    z_top_mm = (layer_idx + 1) * h_mm

    y_mm = np.linspace(0.0, base.W_MM, TOP_NY)
    z_mm = np.linspace(z_bot_mm, z_top_mm, SIDE_NZ)
    Y_mm, Z_mm = np.meshgrid(y_mm, z_mm)
    X_mm = np.zeros_like(Y_mm)

    T = temperature_field_on_surface_at_time(
        xyz_all, t_all, dt_all, inputs, t_snap,
        X_mm * 1e-3, Y_mm * 1e-3, Z_mm * 1e-3,
        history_radius_um=HISTORY_RADIUS_UM,
        tau_hist_ms=TAU_HIST_MS,
        chunk_size=1200,
    )

    masked_surface(
        ax,
        X_mm + EPS_MM, Y_mm, Z_mm * Z_VISUAL_SCALE,
        T, np.ones_like(T, dtype=bool), cmap, norm,
        alpha_scale=alpha_scale
    )


def draw_notch_faces(ax, xyz_all, t_all, dt_all, inputs, t_snap, layer_idx, cmap, norm, alpha_scale=1.0):
    h_mm = base.LAYER_THICKNESS_UM * 1e-3
    z_bot_mm = layer_idx * h_mm
    z_top_mm = (layer_idx + 1) * h_mm

    x_mm = np.linspace(base.NOTCH_X_MM, base.L_MM, max(10, TOP_NX // 2))
    z_mm = np.linspace(z_bot_mm, z_top_mm, SIDE_NZ)
    X_mm, Z_mm = np.meshgrid(x_mm, z_mm)
    Y_mm = np.full_like(X_mm, base.NOTCH_Y_MM)

    T1 = temperature_field_on_surface_at_time(
        xyz_all, t_all, dt_all, inputs, t_snap,
        X_mm * 1e-3, Y_mm * 1e-3, Z_mm * 1e-3,
        history_radius_um=HISTORY_RADIUS_UM,
        tau_hist_ms=TAU_HIST_MS,
        chunk_size=1200,
    )

    masked_surface(
        ax,
        X_mm, Y_mm - EPS_MM, Z_mm * Z_VISUAL_SCALE,
        T1, np.ones_like(T1, dtype=bool), cmap, norm,
        alpha_scale=alpha_scale
    )

    y_mm = np.linspace(base.NOTCH_Y_MM, base.W_MM, max(10, TOP_NY // 2))
    z_mm = np.linspace(z_bot_mm, z_top_mm, SIDE_NZ)
    Y_mm, Z_mm = np.meshgrid(y_mm, z_mm)
    X_mm = np.full_like(Y_mm, base.NOTCH_X_MM)

    T2 = temperature_field_on_surface_at_time(
        xyz_all, t_all, dt_all, inputs, t_snap,
        X_mm * 1e-3, Y_mm * 1e-3, Z_mm * 1e-3,
        history_radius_um=HISTORY_RADIUS_UM,
        tau_hist_ms=TAU_HIST_MS,
        chunk_size=1200,
    )

    masked_surface(
        ax,
        X_mm - EPS_MM, Y_mm, Z_mm * Z_VISUAL_SCALE,
        T2, np.ones_like(T2, dtype=bool), cmap, norm,
        alpha_scale=alpha_scale
    )


def draw_cut_plane(ax, xyz_all, t_all, dt_all, inputs, t_snap, active_layer_idx, last_src_idx, max_layer_idx, cmap, norm):
    if not SHOW_CUT_PLANE:
        return
    if max_layer_idx != active_layer_idx:
        return

    h_mm = base.LAYER_THICKNESS_UM * 1e-3
    z_top_mm = (active_layer_idx + 1) * h_mm
    z_bot_mm = max(0.0, z_top_mm - h_mm)

    y_cut_m = float(xyz_all[last_src_idx, 1])
    y_cut_mm = 1e3 * y_cut_m

    x_max_mm = valid_x_limit_for_y_mm(y_cut_mm)
    if x_max_mm <= 0.0:
        return

    x_mm = np.linspace(0.0, x_max_mm, CUT_NX)
    z_mm = np.linspace(z_bot_mm, z_top_mm, CUT_NZ)
    X_mm, Z_mm = np.meshgrid(x_mm, z_mm)
    Y_mm = np.full_like(X_mm, y_cut_mm)

    T = temperature_field_on_surface_at_time(
        xyz_all, t_all, dt_all, inputs, t_snap,
        X_mm * 1e-3, Y_mm * 1e-3, Z_mm * 1e-3,
        history_radius_um=HISTORY_RADIUS_UM,
        tau_hist_ms=CUT_TAU_HIST_MS,
        chunk_size=1800,
    )

    valid = T >= CUT_T_MIN_K

    masked_surface(
        ax,
        X_mm, Y_mm, Z_mm * Z_VISUAL_SCALE,
        T, valid, cmap, norm,
        alpha_scale=CUT_PLANE_ALPHA
    )

    ax.plot(
        [0.0, x_max_mm],
        [y_cut_mm, y_cut_mm],
        [z_top_mm * Z_VISUAL_SCALE, z_top_mm * Z_VISUAL_SCALE],
        color="white", lw=0.9, alpha=0.9
    )


# ---------------------------------------------------------------------
# Column drawing
# ---------------------------------------------------------------------
def draw_stacked_column(
    ax,
    xyz_all, t_all, dt_all, inputs,
    t_snap,
    column_max_layer_idx,
    active_layer_idx,
    last_src_idx,
    cmap, norm,
    n_local=None,
):
    if column_max_layer_idx > active_layer_idx:
        return

    for ell in range(column_max_layer_idx + 1):
        if ell < active_layer_idx and n_local is not None:
            t_draw = end_of_layer_time(t_all, n_local, ell)
        else:
            t_draw = t_snap

        draw_top_surface(ax, xyz_all, t_all, dt_all, inputs, t_draw, ell, cmap, norm)
        draw_front_face(ax, xyz_all, t_all, dt_all, inputs, t_draw, ell, cmap, norm)
        draw_left_face(ax, xyz_all, t_all, dt_all, inputs, t_draw, ell, cmap, norm)
        draw_notch_faces(ax, xyz_all, t_all, dt_all, inputs, t_draw, ell, cmap, norm)

        h_mm = base.LAYER_THICKNESS_UM * 1e-3
        z_bot_mm = ell * h_mm
        z_top_mm = (ell + 1) * h_mm
        plot_wire_outline(ax, z_bot_mm * Z_VISUAL_SCALE, z_top_mm * Z_VISUAL_SCALE, alpha=0.95)

    if column_max_layer_idx == active_layer_idx:
        draw_cut_plane(
            ax, xyz_all, t_all, dt_all, inputs,
            t_snap, active_layer_idx, last_src_idx,
            column_max_layer_idx, cmap, norm
        )


# ---------------------------------------------------------------------
# New top-row panels
# ---------------------------------------------------------------------
def draw_auxiliary_lshape_geometry(ax):
    """
    CAD-style grey auxiliary geometry.
    All labels are outside the box.
    """
    h_mm = base.LAYER_THICKNESS_UM * 1e-3
    total_h_mm = base.N_LAYERS * h_mm
    z_total_mm = total_h_mm * Z_VISUAL_SCALE

    x_mm = np.linspace(0.0, base.L_MM, 50)
    y_mm = np.linspace(0.0, base.W_MM, 50)

    for ell in range(base.N_LAYERS):
        z_bot = ell * h_mm
        z_top = (ell + 1) * h_mm

        # top face
        X, Y = np.meshgrid(x_mm, y_mm)
        Z = np.full_like(X, z_top * Z_VISUAL_SCALE)
        mask = lshape_mask_xy(X * 1e-3, Y * 1e-3)
        Zm = np.ma.masked_where(~mask, Z)
        ax.plot_surface(
            X, Y, Zm,
            color=CAD_FACE, edgecolor=CAD_EDGE, linewidth=0.20,
            shade=True, alpha=1.0
        )

        # front face y=0
        Xf, Zf = np.meshgrid(np.linspace(0.0, base.L_MM, 40), np.linspace(z_bot, z_top, 10))
        Yf = np.zeros_like(Xf)
        ax.plot_surface(
            Xf, Yf, Zf * Z_VISUAL_SCALE,
            color=CAD_FACE, edgecolor=CAD_EDGE, linewidth=0.20,
            shade=True, alpha=1.0
        )

        # left face x=0
        Yl, Zl = np.meshgrid(np.linspace(0.0, base.W_MM, 40), np.linspace(z_bot, z_top, 10))
        Xl = np.zeros_like(Yl)
        ax.plot_surface(
            Xl, Yl, Zl * Z_VISUAL_SCALE,
            color=CAD_FACE, edgecolor=CAD_EDGE, linewidth=0.20,
            shade=True, alpha=1.0
        )

        # notch faces
        Xn, Zn = np.meshgrid(np.linspace(base.NOTCH_X_MM, base.L_MM, 25), np.linspace(z_bot, z_top, 10))
        Yn = np.full_like(Xn, base.NOTCH_Y_MM)
        ax.plot_surface(
            Xn, Yn, Zn * Z_VISUAL_SCALE,
            color=CAD_FACE, edgecolor=CAD_EDGE, linewidth=0.20,
            shade=True, alpha=1.0
        )

        Yn2, Zn2 = np.meshgrid(np.linspace(base.NOTCH_Y_MM, base.W_MM, 25), np.linspace(z_bot, z_top, 10))
        Xn2 = np.full_like(Yn2, base.NOTCH_X_MM)
        ax.plot_surface(
            Xn2, Yn2, Zn2 * Z_VISUAL_SCALE,
            color=CAD_FACE, edgecolor=CAD_EDGE, linewidth=0.20,
            shade=True, alpha=1.0
        )

        plot_wire_outline(ax, z_bot * Z_VISUAL_SCALE, z_top * Z_VISUAL_SCALE, color="0.18", alpha=1.0)

    # layer labels outside on left
    for ell in range(base.N_LAYERS):
        z_mid = (ell + 0.5) * h_mm * Z_VISUAL_SCALE
        ax.text(
            -0.38, 0.20, z_mid,
            f"Layer {ell+1}",
            fontsize=15 * FONT_SCALE,
            va="center",
            ha="left"
        )

    # height guide outside on right
    x_anno = base.L_MM + 0.25
    y_anno = 0.00
    ax.plot([x_anno, x_anno], [y_anno, y_anno], [0.0, z_total_mm], color="k", lw=1.2)
    ax.plot([x_anno - 0.03, x_anno + 0.03], [y_anno, y_anno], [0.0, 0.0], color="k", lw=1.0)
    ax.plot([x_anno - 0.03, x_anno + 0.03], [y_anno, y_anno], [z_total_mm, z_total_mm], color="k", lw=1.0)

    ax.text(
        base.L_MM + 0.34, 0.02, 0.62 * z_total_mm,
        rf"$N={base.N_LAYERS}$",
        fontsize=15 * FONT_SCALE,
        va="center", ha="left"
    )
    ax.text(
        base.L_MM + 0.34, 0.02, 0.42 * z_total_mm,
        rf"$h={base.LAYER_THICKNESS_UM:.0f}\,\mu$m",
        fontsize=15 * FONT_SCALE,
        va="center", ha="left"
    )

    ax.view_init(elev=24, azim=-58)
    ax.set_xlim(-0.45, base.L_MM + 0.65)
    ax.set_ylim(-0.05, base.W_MM + 0.03)
    ax.set_zlim(0.0, z_total_mm + 0.05)
    ax.set_box_aspect((base.L_MM + 1.10, base.W_MM + 0.08, max(z_total_mm, 1e-12)))
    ax.set_proj_type("persp")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        except Exception:
            pass
        try:
            axis.line.set_color((1, 1, 1, 0))
        except Exception:
            pass

    # panel label only
    ax.text2D(0.02, 0.98, r"(a)", transform=ax.transAxes,
              fontsize=18 * FONT_SCALE, va="top", ha="left")


def draw_track_diagram(ax, base_traj_xy, frac=0.60):
    """
    Top-right track layout with 5 arrows showing scan direction.
    """
    xy_mm = np.asarray(base_traj_xy, dtype=float) * 1e3
    x_mm = xy_mm[:, 0]
    y_mm = xy_mm[:, 1]

    idx = int(round(frac * (len(x_mm) - 1)))
    idx = max(0, min(idx, len(x_mm) - 1))

    ax.plot(x_mm, y_mm, lw=1.0, color="k")

    poly = lshape_polygon_xy_mm()
    ax.plot(poly[:, 0], poly[:, 1], lw=1.0, color="0.45", linestyle="--")

    ax.scatter([x_mm[0]], [y_mm[0]], s=36, marker="o", zorder=5)
    ax.scatter([x_mm[-1]], [y_mm[-1]], s=36, marker="s", zorder=5)
    ax.scatter([x_mm[idx]], [y_mm[idx]], s=80, marker="*", color="crimson", zorder=6)

    # 5 arrows in different places
    arrow_ids = np.linspace(50, len(x_mm) - 60, 5, dtype=int)
    for k in arrow_ids:
        dx = x_mm[k + 6] - x_mm[k]
        dy = y_mm[k + 6] - y_mm[k]
        if dx * dx + dy * dy < 1e-12:
            continue
        ax.annotate(
            "",
            xy=(x_mm[k + 6], y_mm[k + 6]),
            xytext=(x_mm[k], y_mm[k]),
            arrowprops=dict(
                arrowstyle="->",
                color="tab:blue",
                lw=1.3,
                mutation_scale=12,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4
        )

    # emphasize near selected fraction point
    k0 = max(0, min(idx, len(x_mm) - 8))
    ax.annotate(
        "",
        xy=(x_mm[k0 + 7], y_mm[k0 + 7]),
        xytext=(x_mm[k0], y_mm[k0]),
        arrowprops=dict(
            arrowstyle="->",
            color="crimson",
            lw=1.7,
            mutation_scale=14,
        ),
        zorder=6
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, base.L_MM + 0.05)
    ax.set_ylim(-0.05, base.W_MM + 0.05)
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$y$ (mm)")
    ax.grid(False)

    # panel label only
    ax.text(0.02, 0.98, r"(b)", transform=ax.transAxes,
            fontsize=18 * FONT_SCALE, va="top", ha="left")


# ---------------------------------------------------------------------
# New two-row figure
# ---------------------------------------------------------------------
def make_two_row_fraction60_figure(xyz_all, t_all, dt_all, n_local, inputs, base_traj, outbase: Path):
    apply_pub_style()

    fig = plt.figure(figsize=(20, 10), constrained_layout=False)
    gs = GridSpec(
        2, 5,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.07],
        height_ratios=[0.92, 1.12],
        wspace=0.03,
        hspace=0.08,
        left=0.03, right=0.94, top=0.96, bottom=0.06,
    )

    cmap = make_clipped_hot_cmap()
    norm = colors.Normalize(vmin=TMIN_K, vmax=TMAX_K, clip=False)

    # ----------------------------
    # Top-left: auxiliary 3D geometry
    # ----------------------------
    ax_aux = fig.add_subplot(gs[0, 0:2], projection="3d")
    draw_auxiliary_lshape_geometry(ax_aux)

    # ----------------------------
    # Top-right: track diagram
    # ----------------------------
    ax_track = fig.add_subplot(gs[0, 2:4])
    draw_track_diagram(ax_track, base_traj, frac=SNAPSHOT_FRACTION)

    # ----------------------------
    # Bottom row: diagonal snapshots
    # ----------------------------
    bottom_axes = []
    for c in range(4):
        t_snap, last_src_idx = snapshot_time_at_layer_fraction(
            t_all, n_local, c, frac=SNAPSHOT_FRACTION
        )
        active_layer_idx, _ = active_layer_and_last_source_idx(t_all, n_local, t_snap)

        ax = fig.add_subplot(gs[1, c], projection="3d")
        bottom_axes.append(ax)

        draw_stacked_column(
            ax,
            xyz_all, t_all, dt_all, inputs,
            t_snap=t_snap,
            column_max_layer_idx=c,
            active_layer_idx=active_layer_idx,
            last_src_idx=last_src_idx,
            cmap=cmap,
            norm=norm,
            n_local=n_local,
        )

        z_total_mm = base.N_LAYERS * base.LAYER_THICKNESS_UM * 1e-3 * Z_VISUAL_SCALE

        ax.view_init(elev=VIEW_ELEV, azim=-45)
        ax.set_xlim(-0.02, base.L_MM)
        ax.set_ylim(-0.02, base.W_MM)
        ax.set_zlim(0.0, z_total_mm + 0.01)
        ax.set_box_aspect((base.L_MM, base.W_MM, max(z_total_mm, 1e-12)))
        ax.set_proj_type("persp")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.fill = False
                axis.pane.set_edgecolor((1, 1, 1, 0))
            except Exception:
                pass
            try:
                axis.line.set_color((1, 1, 1, 0))
            except Exception:
                pass

        # bottom text only
        ax.text2D(
            0.50, -0.10,
            rf"Layer {c+1} at $f={SNAPSHOT_FRACTION:.1f}$",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=16 * FONT_SCALE
        )

    # panel label for bottom row
    bottom_axes[0].text2D(
        0.02, 0.98, r"(c)",
        transform=bottom_axes[0].transAxes,
        fontsize=18 * FONT_SCALE,
        va="top", ha="left"
    )

    # ----------------------------
    # Colorbar only for bottom row
    # ----------------------------
    cax = fig.add_subplot(gs[1, 4])
    norm_cbar = colors.Normalize(vmin=TMIN_K, vmax=TMAX_K)
    sm = cm.ScalarMappable(norm=norm_cbar, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"Temperature (K)", fontsize=18 * FONT_SCALE)
    cbar.set_ticks(np.arange(TMIN_K, TMAX_K + 1, 500))
    cbar.ax.tick_params(labelsize=14 * FONT_SCALE)

    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main():
    outdir = base.OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    inputs, _ = base.load_inputs_from_calibration(str(THIS_DIR / base.CALIB_JSON))

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
        interlayer_dwell_ms=base.INTERLAYER_DWELL_MS,
    )

    outbase = outdir / OUTNAME
    make_two_row_fraction60_figure(
        xyz_all, t_all, dt_all, n_local, inputs, base_traj, outbase
    )

    print(f"Saved: {outbase.with_suffix('.pdf')}")
    print(f"Saved: {outbase.with_suffix('.png')}")


if __name__ == "__main__":
    main()