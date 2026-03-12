# \"\"\"
# Top-level calibration pipeline.
# Called by both run_calib.py (CLI) and calib_agent (LLM).

# z_scale is ALWAYS taken from the Powell mean-fit result for the specific
# (material, P, V, spot) case.  It is never hardcoded.
# \"\"\"
import os, json
import numpy as np
import pandas as pd

from .config import get_material
from .dataset import filter_data, get_exp_arrays, spot_d4sigma_to_sigma_m
from .analytical_model_jax import make_melt_dims_jax
from .calibration_mean import deterministic_mean_fit
from .calibration_mcmc import StochasticCalibrationMCMC
from .plotting import plot_distribution_kde, plot_scatter_wd


def run_calibration(
    df,
    material: str,
    P: float,
    V: float,
    spot_um: float,
    out_dir: str = "calib_outputs",
    n_steps: int = 7500,
    burn_in: int = 4000,
    n_ensemble: int = 30,
    n_predict: int = 2000,
    verbose: bool = True,
    seed: int = 123,
) -> dict:
    # \"\"\"
    # Full MCMC calibration pipeline for one (material, P, V, spot) case.

    # Steps
    # -----
    # 1. Filter dataset to requested (material, P, V, spot)
    # 2. Compute sigma_m from spot_um  (sigma_m = spot_um / 4, in meters)
    # 3. Build JAX thermal model (JIT-compiled once per call)
    # 4. Powell mean-fit  ->  eta, alpha_mult, z_scale   (all float from optimizer)
    # 5. MCMC  ->  mu_eta, mu_alpha, std_eta, std_alpha   (z_scale fixed from step 4)
    # 6. Generate prediction ensemble
    # 7. Save: KDE plot, scatter plot, Excel, JSON

    # Parameters
    # ----------
    # df         : pd.DataFrame from calib_tools.load_dataset()
    # material   : e.g. "IN718", "IN625"
    # P          : laser power [W]
    # V          : scan speed [mm/s]
    # spot_um    : D4sigma beam diameter [µm]  -- drives sigma_m and data filtering
    # out_dir    : output directory (created if absent)
    # n_steps    : MCMC chain length      (default 7500; use 800 for fast testing)
    # burn_in    : burn-in steps          (default 4000; use 200 for fast testing)
    # n_ensemble : forward samples/step   (default 30;   use 15 for fast testing)
    # n_predict  : samples for plots      (default 2000)
    # verbose    : print progress
    # seed       : RNG seed

    # Returns
    # -------
    # dict with keys:
    #     case_id, material, P, V, spot_um, sigma_m, n_points,
    #     meanfit  -- {eta, alpha_mult, z_scale}  from Powell optimizer
    #     mcmc     -- {mu_eta, mu_alpha, std_eta, std_alpha, z_scale,
    #                  accept_rate, std_bounds_used}
    #     paths    -- {dist_png, scatter_png, excel, json}
    #     sim_summary -- {W_mean, W_std, D_mean, D_std}
    # \"\"\"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Filter data
    # ------------------------------------------------------------------
    df_case = filter_data(df, material, P, V, spot_um)
    exp_w, exp_d = get_exp_arrays(df_case)
    n_pts = len(exp_w)

    if n_pts < 4:
        raise RuntimeError(
            f"Only {n_pts} data points for {material} P={P}W V={V}mm/s "
            f"spot={spot_um}um -- need at least 4 for MCMC calibration."
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Calibrating: {material}  P={P} W  V={V} mm/s  spot={spot_um} um")
        print(f"  Data points : {n_pts}")
        print(f"  Exp W : {np.mean(exp_w):.2f} +/- {np.std(exp_w, ddof=1):.2f} um")
        print(f"  Exp D : {np.mean(exp_d):.2f} +/- {np.std(exp_d, ddof=1):.2f} um")
        print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 2. sigma_m from spot diameter
    # ------------------------------------------------------------------
    sigma_m = spot_d4sigma_to_sigma_m(spot_um)
    if verbose:
        print(f"  sigma_m = {sigma_m*1e6:.2f} um  (= spot_um / 4 = {spot_um}/4)")

    # ------------------------------------------------------------------
    # 3. JAX thermal model
    # ------------------------------------------------------------------
    mat = get_material(material)
    melt_dims_fn = make_melt_dims_jax(P_W=P, V_mmps=V, material_props=mat)
    _ = melt_dims_fn(np.ones((1,  3), dtype=float), sigma_m)   # warm-up JIT
    _ = melt_dims_fn(np.ones((64, 3), dtype=float), sigma_m)

    # ------------------------------------------------------------------
    # 4. Deterministic mean-fit
    # ------------------------------------------------------------------
    best, hist_df = deterministic_mean_fit(
        target_w_m=float(np.mean(exp_w)) * 1e-6,
        target_d_m=float(np.mean(exp_d)) * 1e-6,
        sigma_m=sigma_m,
        melt_dims_fn=melt_dims_fn,
        verbose=verbose,
    )
    # best = {eta, alpha_mult, z_scale}  -- all from optimizer, none hardcoded

    # ------------------------------------------------------------------
    # 5. MCMC calibration
    #    z_scale is passed from best["z_scale"] -- the mean-fit result
    #    for THIS specific (material, P, V, spot) case
    # ------------------------------------------------------------------
    cal = StochasticCalibrationMCMC(
        exp_widths_um=exp_w,
        exp_depths_um=exp_d,
        z_scale=best["z_scale"],   # from mean-fit, NOT a hardcoded constant
    )
    mcmc_res = cal.calibrate(
        melt_dims_fn=melt_dims_fn,
        sigma_m=sigma_m,
        init_from_meanfit=best,
        n_steps=n_steps,
        burn_in=burn_in,
        n_samples=n_ensemble,    # ← correct name
        verbose=verbose,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 6. Prediction ensemble
    # ------------------------------------------------------------------
    sim_w, sim_d, sim_params = cal.generate_predictions(
        melt_dims_fn=melt_dims_fn,
        sigma_m=sigma_m,
        n_samples=n_predict,
        seed=seed + 1000,
    )

    if verbose:
        print(f"\n  Predicted W : {np.mean(sim_w):.2f} +/- {np.std(sim_w, ddof=1):.2f} um")
        print(f"  Predicted D : {np.mean(sim_d):.2f} +/- {np.std(sim_d, ddof=1):.2f} um")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    spot_tag = int(round(spot_um))
    case_id  = f"{material}_P{int(round(P))}_V{int(round(V))}_spot{spot_tag}"

    dist_png    = os.path.join(out_dir, f"dist_{case_id}.png")
    scatter_png = os.path.join(out_dir, f"scatter_{case_id}.png")
    excel_path  = os.path.join(out_dir, f"results_{case_id}.xlsx")
    json_path   = os.path.join(out_dir, f"calibration_{case_id}.json")

    title = f"{material}  P={int(P)}W  V={int(V)}mm/s  spot={spot_um:.1f}um"
    plot_distribution_kde(exp_w, exp_d, sim_w, sim_d, dist_png,    title=title)
    plot_scatter_wd      (exp_w, exp_d, sim_w, sim_d, scatter_png, title=title)

    # Excel
    summary = {
        "material": material, "P_W": P, "V_mmps": V, "spot_um": spot_um,
        "sigma_m":  sigma_m,  "n_points": n_pts,
        "exp_W_mean": float(np.mean(exp_w)), "exp_W_std": float(np.std(exp_w, ddof=1)),
        "exp_D_mean": float(np.mean(exp_d)), "exp_D_std": float(np.std(exp_d, ddof=1)),
        # Mean-fit results (all from optimizer)
        "meanfit_eta":     best["eta"],
        "meanfit_alpha":   best["alpha_mult"],
        "meanfit_z_scale": best["z_scale"],   # <-- NOT hardcoded, from optimizer
        # MCMC results
        "mcmc_mu_eta":         float(cal.mu_star[0]),
        "mcmc_mu_alpha":       float(cal.mu_star[1]),
        "mcmc_std_eta":        float(cal.sigma_star[0]),
        "mcmc_std_alpha":      float(cal.sigma_star[1]),
        "mcmc_z_scale_fixed":  float(cal.z_scale),   # same as meanfit_z_scale
        "mcmc_accept_rate":    float(mcmc_res["accept_rate"]),
        "mcmc_std_bound_eta_lo":    mcmc_res["std_bounds_used"]["std_eta"][0],
        "mcmc_std_bound_eta_hi":    mcmc_res["std_bounds_used"]["std_eta"][1],
        "mcmc_std_bound_alpha_lo":  mcmc_res["std_bounds_used"]["std_alpha"][0],
        "mcmc_std_bound_alpha_hi":  mcmc_res["std_bounds_used"]["std_alpha"][1],
        "sim_W_mean": float(np.mean(sim_w)), "sim_W_std": float(np.std(sim_w, ddof=1)),
        "sim_D_mean": float(np.mean(sim_d)), "sim_D_std": float(np.std(sim_d, ddof=1)),
    }
    chain_df = pd.DataFrame(
        mcmc_res["chain"], columns=["mu_eta", "mu_alpha", "std_eta", "std_alpha"]
    )
    chain_df["logp"]     = mcmc_res["logp"]
    chain_df["accepted"] = mcmc_res["accepted"]
    pred_df = pd.DataFrame({
        "Width_um": sim_w, "Depth_um": sim_d,
        "eta": sim_params[:, 0], "alpha_mult": sim_params[:, 1],
        "z_scale": sim_params[:, 2],
    })
    exp_df = pd.DataFrame({"Width_um": exp_w, "Depth_um": exp_d})

    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        pd.DataFrame([summary]).to_excel(w, sheet_name="Summary",          index=False)
        pred_df.to_excel  (w, sheet_name="Predictions",       index=False)
        exp_df.to_excel   (w, sheet_name="Experimental",      index=False)
        chain_df.to_excel (w, sheet_name="MCMC_Chain",        index=True)
        hist_df.to_excel  (w, sheet_name="MeanFit_History",   index=False)

    # JSON (for downstream print simulation use)
    calib_json = {
        "case_id":  case_id,
        "material": material,
        "process_params": {"P_W": P, "V_mmps": V, "sigma_m": sigma_m},
        "meanfit": {
            "eta":       best["eta"],
            "alpha_mult":best["alpha_mult"],
            "z_scale":   best["z_scale"],   # from optimizer, not hardcoded
        },
        "stochastic_params": {
            "mu_eta":        float(cal.mu_star[0]),
            "mu_alpha":      float(cal.mu_star[1]),
            "std_eta":       float(cal.sigma_star[0]),
            "std_alpha":     float(cal.sigma_star[1]),
            "z_scale_fixed": float(cal.z_scale),   # = meanfit z_scale
        },
        "mcmc_diagnostics": {
            "accept_rate": float(mcmc_res["accept_rate"]),
            "n_effective": int(mcmc_res["n_effective"]),
            "n_steps": n_steps, "burn_in": burn_in,
            "std_bounds_used": mcmc_res["std_bounds_used"],
        },
    }
    with open(json_path, "w") as f:
        json.dump(calib_json, f, indent=2)

    if verbose:
        print(f"\n  dist_png    : {dist_png}")
        print(f"  scatter_png : {scatter_png}")
        print(f"  excel       : {excel_path}")
        print(f"  json        : {json_path}")

    return {
        "case_id":  case_id,
        "material": material,
        "P": P, "V": V, "spot_um": spot_um, "sigma_m": sigma_m, "n_points": n_pts,
        "meanfit": best,
        "mcmc": {
            "mu_eta":       float(cal.mu_star[0]),
            "mu_alpha":     float(cal.mu_star[1]),
            "std_eta":      float(cal.sigma_star[0]),
            "std_alpha":    float(cal.sigma_star[1]),
            "z_scale":      float(cal.z_scale),
            "accept_rate":  float(mcmc_res["accept_rate"]),
            "std_bounds_used": mcmc_res["std_bounds_used"],
        },
        "paths": {
            "dist_png":    dist_png,
            "scatter_png": scatter_png,
            "excel":       excel_path,
            "json":        json_path,
        },
        "sim_summary": {
            "W_mean": float(np.mean(sim_w)), "W_std": float(np.std(sim_w, ddof=1)),
            "D_mean": float(np.mean(sim_d)), "D_std": float(np.std(sim_d, ddof=1)),
        },
    }
