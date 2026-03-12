"""
calib_agent/agent.py

Agent decisions:
  1. Calls get_case_info first -> reads n_points -> selects MCMC hyperparameters
  2. Evaluates convergence diagnostics after calibration completes
  3. Autonomously reruns with stronger settings if convergence is poor
"""
import json
import os
import numpy as np
import time
from pathlib import Path
from typing import Callable

import ollama
from .prompts import SYSTEM_PROMPT

_ROOT = Path(__file__).resolve().parent.parent

# ── Module-level state so evaluate_calibration can read the last result ───
_last_calibration_result: dict | None = None


# ── Tool implementations ──────────────────────────────────────────────────

def _find_dataset():
    env = os.environ.get("CALIB_DATASET")
    if env and os.path.isfile(env):
        return env
    for p in [
        _ROOT / "Master_TrackList_Measurements.xlsx",
        _ROOT / "data" / "Master_TrackList_Measurements.xlsx",
        _ROOT / "NIST.xlsx",
    ]:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Dataset not found. Place Master_TrackList_Measurements.xlsx in the "
        "project root or set CALIB_DATASET environment variable."
    )


def tool_list_cases(args: dict) -> str:
    import calib_tools as ct
    material = args.get("material", None)
    df = ct.load_dataset(_find_dataset())
    cases = ct.list_cases(df, material=material or None)
    return cases.to_string(index=False)


def tool_get_case_info(args: dict) -> str:
    """
    Inspect a specific case BEFORE calibrating.  Returns number of data
    points, experimental statistics, and recommended MCMC hyperparameters.
    """
    import numpy as np
    import calib_tools as ct

    material = str(args.get("material", "IN718"))
    P        = float(args.get("P_W", 285.0))
    V        = float(args.get("V_mmps", 960.0))
    spot_um  = float(args.get("spot_um", 80.0))

    try:
        df      = ct.load_dataset(_find_dataset())
        df_case = ct.filter_data(df, material, P, V, spot_um)
        exp_w, exp_d = ct.get_exp_arrays(df_case)
        n_pts = len(exp_w)
    except Exception as exc:
        return json.dumps({
            "error": f"No data for {material} P={P} V={V} spot={spot_um}. {exc}",
            "n_points": 0,
        })

    if n_pts == 0:
        return json.dumps({
            "error": f"No data for {material} P={P} V={V} spot={spot_um}",
            "n_points": 0,
        })

    if n_pts >= 20:
        rec = dict(n_steps=5000,  burn_in=2500, n_ensemble=25,
                   note="Large dataset -- moderate chain sufficient.")
    elif n_pts >= 10:
        rec = dict(n_steps=7500,  burn_in=4000, n_ensemble=30,
                   note="Standard dataset -- default settings.")
    elif n_pts >= 6:
        rec = dict(n_steps=10000, burn_in=5000, n_ensemble=40,
                   note="Small dataset -- extended chain for stability.")
    else:
        rec = dict(n_steps=12000, burn_in=6000, n_ensemble=50,
                   note="Very few points -- long chain, wide posterior.")

    return json.dumps({
        "material": material, "P_W": P, "V_mmps": V, "spot_um": spot_um,
        "n_points": n_pts,
        "exp_W_mean_um": round(float(np.mean(exp_w)), 2),
        "exp_W_std_um":  round(float(np.std(exp_w, ddof=1)), 2) if n_pts > 1 else 0,
        "exp_D_mean_um": round(float(np.mean(exp_d)), 2),
        "exp_D_std_um":  round(float(np.std(exp_d, ddof=1)), 2) if n_pts > 1 else 0,
        "recommended_mcmc": rec,
    }, indent=2)


def tool_calibrate_case(args: dict) -> str:
    global _last_calibration_result
    import calib_tools as ct

    material   = str(args.get("material", "IN718"))
    P          = float(args.get("P_W", 285.0))
    V          = float(args.get("V_mmps", 960.0))
    spot_um    = float(args.get("spot_um", 80.0))
    out_dir    = str(args.get("out_dir", "calib_outputs"))
    n_steps    = int(args.get("n_steps", 7500))
    burn_in    = int(args.get("burn_in", 4000))
    n_ensemble = int(args.get("n_ensemble", 30))
    n_predict  = int(args.get("n_predict", 2000))

    if os.path.isabs(out_dir):
        out_dir = "calib_outputs"

    df = ct.load_dataset(_find_dataset())
    result = ct.run_calibration(
        df=df, material=material, P=P, V=V, spot_um=spot_um,
        out_dir=out_dir, n_steps=n_steps, burn_in=burn_in,
        n_ensemble=n_ensemble, n_predict=n_predict,
        verbose=True, seed=123,
    )

    # Store full result for evaluate_calibration to read
    _last_calibration_result = result

    # Return a SHORT summary to the LLM (not the full giant JSON)
    mcmc = result.get("mcmc", {})
    sim  = result.get("sim_summary", {})
    summary = {
        "case_id":      result.get("case_id"),
        "n_points":     result.get("n_points"),
        "accept_rate":  mcmc.get("accept_rate"),
        "mu_eta":       mcmc.get("mu_eta"),
        "mu_alpha":     mcmc.get("mu_alpha"),
        "std_eta":      mcmc.get("std_eta"),
        "std_alpha":    mcmc.get("std_alpha"),
        "sim_W_mean":   sim.get("W_mean"),
        "sim_D_mean":   sim.get("D_mean"),
        "output_json":  result.get("paths", {}).get("json"),
        "status":       "complete",
        "NEXT_ACTION":  "You MUST now call evaluate_calibration() to assess quality.",
    }
    return json.dumps(summary, indent=2, default=str)


def tool_evaluate_calibration(args: dict) -> str:
    """
    Evaluate MCMC calibration quality.  Reads from the stored result
    of the last calibrate_case call (no arguments needed from LLM).
    """
    import numpy as np
    global _last_calibration_result

    if _last_calibration_result is None:
        return json.dumps({"error": "No calibration result stored. Run calibrate_case first."})

    result = _last_calibration_result
    mcmc   = result.get("mcmc", {})
    sim    = result.get("sim_summary", {})
    n_pts  = result.get("n_points", 0)
    accept = mcmc.get("accept_rate", 0.0)

    issues  = []
    quality = "ACCEPTABLE"
    action  = "accept"

    # Acceptance rate
    if accept < 0.15:
        issues.append(f"Accept rate {accept:.4f} critically low. Chain not converged.")
        quality, action = "POOR", "rerun_more_steps"
    elif accept < 0.20:
        issues.append(f"Accept rate {accept:.4f} marginal.")
        quality, action = "MARGINAL", "rerun_more_steps"
    elif accept > 0.80:
        issues.append(f"Accept rate {accept:.4f} too high -- under-explored posterior.")
        quality, action = "MARGINAL", "warn"
    else:
        issues.append(f"Accept rate {accept:.4f} -- optimal range.")

    # Data sufficiency
    if n_pts < 6:
        issues.append(f"Only {n_pts} points -- wide posterior expected.")

    # Parameter uncertainty
    mu_eta    = mcmc.get("mu_eta", 1)
    mu_alpha  = mcmc.get("mu_alpha", 1)
    std_eta   = mcmc.get("std_eta", 0)
    std_alpha = mcmc.get("std_alpha", 0)
    if mu_eta > 0 and std_eta / mu_eta > 0.20:
        issues.append(f"eta uncertainty {100*std_eta/mu_eta:.1f}% -- high.")
    if mu_alpha > 0 and std_alpha / mu_alpha > 0.15:
        issues.append(f"alpha uncertainty {100*std_alpha/mu_alpha:.1f}%.")

    # Rerun recommendations
    rerun_params = None
    if action == "rerun_more_steps":
        n_eff = mcmc.get("n_effective", 3500)
        rerun_params = {
            "n_steps":    min(20000, n_eff * 4),
            "burn_in":    min(10000, n_eff * 2),
            "n_ensemble": 40,
        }

    return json.dumps({
        "quality": quality, "action": action, "accept_rate": accept,
        "n_data_points": n_pts,
        "sim_W_mean_um": sim.get("W_mean"), "sim_D_mean_um": sim.get("D_mean"),
        "mu_eta": mu_eta, "mu_alpha": mu_alpha,
        "std_eta": std_eta, "std_alpha": std_alpha,
        "issues": issues, "rerun_params": rerun_params,
    }, indent=2)


# ── Tool registry ─────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable[[dict], str]] = {
    "list_cases":           tool_list_cases,
    "get_case_info":        tool_get_case_info,
    "calibrate_case":       tool_calibrate_case,
    "evaluate_calibration": tool_evaluate_calibration,
}

# ── Tool schemas for Ollama ───────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_cases",
            "description": "List available (Material, P, V, Spot, n) cases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {
                        "type": "string",
                        "description": "Optional filter e.g. 'IN718'. Omit for all.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_case_info",
            "description": "Get data size and recommended MCMC settings BEFORE calibrating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {"type": "string",  "description": "e.g. 'IN718'"},
                    "P_W":      {"type": "number",  "description": "Laser power (W)"},
                    "V_mmps":   {"type": "number",  "description": "Scan speed (mm/s)"},
                    "spot_um":  {"type": "number",  "description": "D4sigma spot (um)"},
                },
                "required": ["material", "P_W", "V_mmps", "spot_um"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate_case",
            "description": "Run MCMC calibration. Use settings from get_case_info. Call evaluate_calibration after.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material":   {"type": "string",  "description": "e.g. 'IN718'"},
                    "P_W":        {"type": "number",  "description": "Laser power (W)"},
                    "V_mmps":     {"type": "number",  "description": "Scan speed (mm/s)"},
                    "spot_um":    {"type": "number",  "description": "D4sigma spot (um)"},
                    "out_dir":    {"type": "string",  "description": "Default 'calib_outputs'"},
                    "n_steps":    {"type": "integer", "description": "MCMC steps from get_case_info"},
                    "burn_in":    {"type": "integer", "description": "Burn-in from get_case_info"},
                    "n_ensemble": {"type": "integer", "description": "Ensemble size from get_case_info"},
                },
                "required": ["material", "P_W", "V_mmps", "spot_um"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_calibration",
            "description": "Evaluate last calibration quality. No arguments needed -- reads stored result automatically. Returns: accept / warn / rerun_more_steps.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Agent loop ────────────────────────────────────────────────────────────

def _compute_min_ess(result: dict) -> tuple[int, str]:
    """
    Compute minimum ESS across the 4 MCMC hyperparameters from posterior_chain.
    Returns (min_ess, param_name).  Falls back to (n_effective, 'heuristic').
    """
    import numpy as np
    mcmc_raw = result.get("_mcmc_raw", None)
    if mcmc_raw is not None and "posterior_chain" in mcmc_raw:
        post_chain = np.asarray(mcmc_raw["posterior_chain"])
        param_names = ["mu_eta", "mu_alpha", "std_eta", "std_alpha"]
        ess_vals = {}
        for j, name in enumerate(param_names):
            x = post_chain[:, j].copy()
            x -= x.mean()
            var = np.var(x, ddof=0)
            if var < 1e-15:
                ess_vals[name] = 1
                continue
            n = len(x)
            fft_x = np.fft.fft(x, n=2 * n)
            acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
            tau = 1.0
            for k in range(1, min(500, n // 2)):
                if acf[k] < 0:
                    break
                tau += 2.0 * acf[k]
            ess_vals[name] = max(1, int(n / tau))
        min_name = min(ess_vals, key=ess_vals.get)
        return ess_vals[min_name], min_name, ess_vals
    # fallback
    n_eff = result.get("_mcmc_raw", {}).get("n_effective",
            result.get("mcmc", {}).get("n_effective", 0))
    return n_eff, "heuristic", {}


def run_agent(
    user_prompt: str,
    model: str = "llama3-groq-tool-use:8b",
    max_steps: int = 20,
    min_ess_target: int = 100,
    rerun_n_steps: int = 30000,
    rerun_burn_in: int = 12000,
    rerun_n_ensemble: int = 40,
    debug: bool = False,
) -> str:
    """
    Agent loop with ESS escalation.

    The LLM drives the workflow (list_cases → get_case_info → calibrate_case →
    evaluate_calibration → final report).  However, the rerun decision is made
    by Python directly — we do NOT rely on the 8B model to decide to rerun.

    After evaluate_calibration returns action='rerun_more_steps' OR after
    Python detects min_ess < min_ess_target, the loop injects a
    calibrate_case call with (rerun_n_steps, rerun_burn_in, rerun_n_ensemble)
    without asking the LLM.
    """

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT.strip()},
        {"role": "user",    "content": user_prompt.strip()},
    ]

    calibration_result_str = None
    _rerun_done = False          # only escalate once
    _last_calib_args = {}        # remember case args for rerun

    def dbg(*a):
        if debug:
            print(*a, flush=True)

    for step in range(max_steps):
        dbg(f"\n--- step {step+1}/{max_steps} ---")

        resp = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            stream=False,
            options={"temperature": 0.0},
        )

        msg        = resp.message
        content    = msg.content or ""
        tool_calls = msg.tool_calls or []

        dbg(f"  content: {content[:120]}")
        dbg(f"  tool_calls: {tool_calls}")

        if tool_calls:
            for call in tool_calls:
                fn_name = call.function.name
                fn_args = call.function.arguments or {}

                dbg(f"  >> calling {fn_name}({fn_args})")

                if fn_name not in TOOL_REGISTRY:
                    tool_result = f"ERROR: unknown tool '{fn_name}'"
                else:
                    try:
                        tool_result = TOOL_REGISTRY[fn_name](fn_args)
                    except Exception as e:
                        tool_result = f"ERROR in {fn_name}: {e}"

                dbg(f"  << result: {str(tool_result)[:200]}")

                # Remember case identity whenever calibrate_case is called
                if fn_name == "calibrate_case":
                    calibration_result_str = tool_result
                    _last_calib_args = {
                        "material":   fn_args.get("material",   "IN718"),
                        "P_W":        fn_args.get("P_W",        285.0),
                        "V_mmps":     fn_args.get("V_mmps",     960.0),
                        "spot_um":    fn_args.get("spot_um",    80.0),
                        "out_dir":    fn_args.get("out_dir",    "calib_outputs"),
                    }

                # ── ESS check after evaluate_calibration ──
                if fn_name == "evaluate_calibration" and not _rerun_done:
                    try:
                        eval_result = json.loads(tool_result)
                    except Exception:
                        eval_result = {}

                    # Compute real ESS directly from stored posterior_chain
                    min_ess, min_param, ess_all = _compute_min_ess(
                        _last_calibration_result or {}
                    )

                    needs_rerun = (
                        eval_result.get("action") == "rerun_more_steps"
                        or min_ess < min_ess_target
                    )

                    if needs_rerun:
                        _rerun_done = True
                        dbg(f"\n  [RERUN] "
                            f"min_ess={min_ess} ({min_param}) < {min_ess_target}. "
                            f"ESS breakdown: {ess_all}")
                        print(f"\n{'='*70}")
                        print(f"[Agent] ESS check: min_ess={min_ess} for '{min_param}' "
                              f"— below target of {min_ess_target}.")
                        print(f"[Agent] Forcing rerun: "
                              f"n_steps={rerun_n_steps}, burn_in={rerun_burn_in}, "
                              f"n_ensemble={rerun_n_ensemble}")
                        print(f"{'='*70}\n")

                        # Inject evaluate_calibration result into message history
                        messages.append({"role": "assistant", "content": "",
                                         "tool_calls": tool_calls})
                        messages.append({"role": "tool", "content": tool_result})

                        # Build rerun args
                        rerun_args = dict(_last_calib_args)
                        rerun_args["n_steps"]    = rerun_n_steps
                        rerun_args["burn_in"]    = rerun_burn_in
                        rerun_args["n_ensemble"] = rerun_n_ensemble

                        # Execute rerun directly (no LLM decision needed)
                        dbg(f"  >> [AGENT] calling calibrate_case({rerun_args})")
                        try:
                            rerun_result = TOOL_REGISTRY["calibrate_case"](rerun_args)
                        except Exception as e:
                            rerun_result = f"ERROR in rerun: {e}"
                        calibration_result_str = rerun_result
                        dbg(f"  << [AGENT] result: {str(rerun_result)[:200]}")

                        # Inject rerun as if LLM called it
                        fake_call_msg = {
                            "role": "assistant", "content": "",
                            "tool_calls": [{
                                "function": {
                                    "name": "calibrate_case",
                                    "arguments": rerun_args,
                                }
                            }]
                        }
                        messages.append(fake_call_msg)
                        messages.append({"role": "tool", "content": rerun_result})

                        # Now run evaluate_calibration again automatically
                        dbg(f"  >> [AGENT] calling evaluate_calibration({{}})")
                        try:
                            eval2_result = TOOL_REGISTRY["evaluate_calibration"]({})
                        except Exception as e:
                            eval2_result = f"ERROR in evaluate: {e}"
                        dbg(f"  << [AGENT] eval2: {str(eval2_result)[:200]}")

                        min_ess2, min_param2, ess_all2 = _compute_min_ess(
                            _last_calibration_result or {}
                        )
                        print(f"[Agent] Post-rerun ESS: {ess_all2}")
                        print(f"[Agent] Post-rerun min_ess={min_ess2} for '{min_param2}'\n")

                        messages.append({
                            "role": "assistant", "content": "",
                            "tool_calls": [{"function": {
                                "name": "evaluate_calibration", "arguments": {}
                            }}]
                        })
                        messages.append({"role": "tool", "content": eval2_result})

                        # Ask LLM to produce the final summary now
                        messages.append({
                            "role": "user",
                            "content": (
                                "The rerun with 30000 steps is complete. "
                                "Please provide the final calibration summary including: "
                                "n_data_points, MCMC settings used (initial + rerun), "
                                "calibrated parameters (mu_eta, mu_alpha, std_eta, std_alpha), "
                                "acceptance rate, ESS values, and quality verdict."
                            )
                        })
                        # Skip normal message append below and continue loop
                        continue

                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                messages.append({"role": "tool",      "content": tool_result})

            continue

        messages.append({"role": "assistant", "content": content})
        if content.strip():
            return content

    if calibration_result_str:
        return calibration_result_str
    return "Max steps reached."