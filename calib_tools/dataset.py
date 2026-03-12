# \"\"\"
# Load and filter Master_TrackList_Measurements.xlsx.
# Supports full dataset (IN718, IN625, all P/V/spot combos).
# \"\"\"
import os
import numpy as np
import pandas as pd

MATERIAL_COL = "Material"
POWER_COL    = "Laser Power (W)"
SPEED_COL    = "Laser Scan Speed (mm/s)"
SPOT_COL     = "Est. Spot Diameter (D4\u03c3 = Dg)"
WIDTH_COL    = "Width (\u00b5m)"
DEPTH_COL    = "Depth (\u00b5m)"


def load_dataset(xlsx_path: str) -> pd.DataFrame:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Dataset not found: {xlsx_path}")
    xls = pd.ExcelFile(xlsx_path)
    sheet = "Data" if "Data" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    required = [MATERIAL_COL, POWER_COL, SPEED_COL, WIDTH_COL, DEPTH_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")
    return df


def list_cases(df: pd.DataFrame, material=None) -> pd.DataFrame:
    d = df.copy()
    if material is not None:
        d = d[d[MATERIAL_COL] == material]
    d = d.dropna(subset=[POWER_COL, SPEED_COL, WIDTH_COL, DEPTH_COL])
    if SPOT_COL in d.columns:
        grp = d.groupby([MATERIAL_COL, POWER_COL, SPEED_COL, SPOT_COL]).size().reset_index(name="n_measurements")
        grp = grp.sort_values([MATERIAL_COL, POWER_COL, SPEED_COL, SPOT_COL])
    else:
        grp = d.groupby([MATERIAL_COL, POWER_COL, SPEED_COL]).size().reset_index(name="n_measurements")
        grp = grp.sort_values([MATERIAL_COL, POWER_COL, SPEED_COL])
    return grp.reset_index(drop=True)


def filter_data(df, material, P, V, spot_um=None, spot_atol_um=2.0):
    # \"\"\"
    # Filter to (material, P, V, [spot]) case.
    # Raises RuntimeError with available options if no data found.
    # \"\"\"
    d = df.copy()
    d = d[d[MATERIAL_COL] == material].copy()
    d = d[np.isclose(d[POWER_COL].astype(float), float(P), atol=0.1)].copy()
    d = d[np.isclose(d[SPEED_COL].astype(float), float(V), atol=0.5)].copy()

    if len(d) == 0:
        avail = list_cases(df, material)
        raise RuntimeError(
            f"No data for {material} P={P}W V={V}mm/s.\n"
            f"Available:\n{avail.to_string(index=False)}"
        )

    if spot_um is not None and SPOT_COL in d.columns:
        d_spot = d[np.isclose(d[SPOT_COL].astype(float), float(spot_um), atol=float(spot_atol_um))].copy()
        if len(d_spot) == 0:
            avail_spots = sorted(d[SPOT_COL].dropna().unique())
            raise RuntimeError(
                f"No data for {material} P={P}W V={V}mm/s spot={spot_um}um (+/-{spot_atol_um}um).\n"
                f"Available spots: {[round(s,2) for s in avail_spots]}"
            )
        d = d_spot

    d = d.dropna(subset=[WIDTH_COL, DEPTH_COL]).copy()
    if len(d) == 0:
        raise RuntimeError(f"Data matched but all rows missing Width/Depth for {material} P={P}W V={V}mm/s.")
    return d.reset_index(drop=True)


def get_exp_arrays(df):
    return df[WIDTH_COL].astype(float).values, df[DEPTH_COL].astype(float).values


def spot_d4sigma_to_sigma_m(spot_um: float) -> float:
    # \"\"\"D4sigma (um) -> sigma_m (m). Convention: sigma_m = D4sigma/4.\"\"\"
    return (float(spot_um) / 4.0) * 1e-6
