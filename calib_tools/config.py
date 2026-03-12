"""
calib_tools/config.py
=====================
Material thermal properties loader.

Properties are stored as JSON files in calib_tools/properties/.
Each file is named <MATERIAL>.json (e.g. IN718.json, IN625.json).

To add a new material:
    1. Copy calib_tools/properties/TEMPLATE.json
    2. Rename it to YOUR_MATERIAL.json
    3. Fill in RHO, CP, T_MELT, T0  (all in SI units)
    4. Use material="YOUR_MATERIAL" in run_calibration()

No code changes needed.
"""
import os
import json
from dataclasses import dataclass

# Folder where material JSON files live (same directory as this file)
_PROPS_DIR = os.path.join(os.path.dirname(__file__), "properties")


@dataclass(frozen=True)
class MaterialProps:
    """
    Thermal properties for the Goldak analytical model.

    Parameters (all SI)
    -------------------
    name   : material identifier string
    RHO    : density [kg/m^3]
    CP     : specific heat capacity [J/(kg·K)]
    T_MELT : melting / liquidus temperature [K]
    T0     : initial / preheat temperature [K]
    """
    name:   str   = "IN718"
    RHO:    float = 8190.0
    CP:     float = 435.0
    T_MELT: float = 1563.15   # 1290 + 273.15 K
    T0:     float = 423.15    # 150  + 273.15 K


def get_material(name: str) -> MaterialProps:
    """
    Load MaterialProps for *name* from calib_tools/properties/<name>.json.

    Search order
    ------------
    1. calib_tools/properties/<name>.json   (primary)
    2. calib_tools/properties/<NAME>.json   (uppercase fallback)
    3. Hard-coded IN718 defaults            (with a warning)

    Parameters
    ----------
    name : str
        Material name, e.g. "IN718", "IN625", "316L".
        Must match the filename in the properties/ folder (case-sensitive on Linux).

    Returns
    -------
    MaterialProps
    """
    # Try exact name, then uppercase
    candidates = [
        os.path.join(_PROPS_DIR, f"{name}.json"),
        os.path.join(_PROPS_DIR, f"{name.upper()}.json"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Strip comment keys (start with "_")
            props = {k: v for k, v in data.items() if not k.startswith("_")}

            required = {"name", "RHO", "CP", "T_MELT", "T0"}
            missing  = required - props.keys()
            if missing:
                raise KeyError(
                    f"Material file '{path}' is missing required keys: {missing}\n"
                    f"See calib_tools/properties/TEMPLATE.json for the expected format."
                )

            return MaterialProps(
                name   = str(props["name"]),
                RHO    = float(props["RHO"]),
                CP     = float(props["CP"]),
                T_MELT = float(props["T_MELT"]),
                T0     = float(props["T0"]),
            )

    # Not found -- warn and return IN718 defaults
    print(
        f"[WARNING] No properties file found for '{name}' in {_PROPS_DIR}/\n"
        f"          Expected: {os.path.join(_PROPS_DIR, name + '.json')}\n"
        f"          Falling back to IN718 defaults.  "
        f"Copy TEMPLATE.json to add your material."
    )
    return MaterialProps(name=name)


def list_materials() -> list[str]:
    """
    Return names of all materials with a properties file in properties/.
    Useful for checking what's available before running calibration.
    """
    if not os.path.isdir(_PROPS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_PROPS_DIR)
        if f.endswith(".json") and not f.startswith("TEMPLATE") and not f.startswith("_")
    )
