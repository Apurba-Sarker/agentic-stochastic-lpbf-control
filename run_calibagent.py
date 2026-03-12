#!/usr/bin/env python3
"""
run_agent.py -- Ollama agent runner for LPBF MCMC calibration.
Requires Ollama running locally with a tool-calling model.

Examples
--------
python run_agent.py "Calibrate IN718 P=285 V=960 spot=80"
python run_agent.py "Calibrate IN625 P=195 V=800 spot=80" --fast
python run_agent.py "What cases are available for IN625?"
python run_agent.py "Calibrate IN718 P=285 V=960 spot=80" --model qwen3:8b --debug


"""
import argparse
from calib_agent.agent import run_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt",      type=str)
    ap.add_argument("--model",     type=str, default="llama3-groq-tool-use:8b")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--fast",      action="store_true",
                    help="Append 'fast=true' to prompt for quick MCMC settings")
    ap.add_argument("--debug",     action="store_true")
    args = ap.parse_args()

    prompt = args.prompt
    if args.fast and "fast" not in prompt.lower():
        prompt += " [use fast MCMC: n_steps=800, burn_in=200, n_ensemble=15]"

    result = run_agent(
        user_prompt=prompt,
        model=args.model,
        max_steps=args.max_steps,
        debug=args.debug,
    )
    print(result)


if __name__ == "__main__":
    main()
