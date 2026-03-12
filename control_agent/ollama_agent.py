"""
control_agent/ollama_agent.py
==============================
Ollama-backed conversational agent for the LPBF analytical inverse controller.

Usage (interactive REPL):
    python control_agent/ollama_agent.py

Usage (single-shot from CLI):
    python control_agent/ollama_agent.py --track 45deg --full-pipeline
    python control_agent/ollama_agent.py --message "Run triangle, no cross-sections"
    python control_agent/ollama_agent.py --track horizontal --power-levels coarse_50w

Usage (import):
    from control_agent.ollama_agent import ControlAgent
    agent = ControlAgent()
    print(agent.chat("Run the full pipeline for horizontal"))

Configuration:
    OLLAMA_MODEL env var (default: "llama3-groq-tool-use:8b")
    OLLAMA_HOST  env var (default: "http://localhost:11434")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

from control_agent.prompts import build_messages
from control_agent.tools import TOOL_REGISTRY, AgentState

# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3-groq-tool-use:8b")
DEFAULT_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")


# ── Tool-call parser ──────────────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(
    r"```tool_call\s*\n(.*?)\n```",
    re.DOTALL,
)


def extract_tool_calls(text: str) -> list[dict]:
    """Extract all ```tool_call ... ``` blocks from an LLM response."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1).strip())
            calls.append(obj)
        except json.JSONDecodeError:
            pass
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove ```tool_call ... ``` blocks from text (leave narrative)."""
    return _TOOL_CALL_RE.sub("", text).strip()


# ── Agent ─────────────────────────────────────────────────────────────────
class ControlAgent:
    """
    Conversational Ollama agent for the LPBF control pipeline.

    Parameters
    ----------
    model  : Ollama model name (e.g. "llama3.2", "mistral")
    host   : Ollama server URL
    verbose: Print tool-execution details to stdout
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str  = DEFAULT_HOST,
        verbose: bool = True,
    ):
        self.model   = model
        self.host    = host
        self.verbose = verbose
        self.state   = AgentState()
        self.history: list[dict] = []  # conversation turns (excluding system+few-shot)

        if not _OLLAMA_AVAILABLE:
            print(
                "[WARNING] 'ollama' Python package not installed.\n"
                "          Install with: pip install ollama\n"
                "          Falling back to direct tool dispatch mode.\n"
            )

    # ── LLM call ──────────────────────────────────────────────────────────
    def _llm(self, user_message: str) -> str:
        """Call Ollama and return the raw assistant text."""
        messages = build_messages(user_message, self.history)

        if not _OLLAMA_AVAILABLE:
            # Fallback: simple keyword-based dispatch (no LLM)
            return self._keyword_dispatch(user_message)

        client   = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.0, "num_predict": 1024},
        )
        return response["message"]["content"]

    # ── Keyword fallback (no LLM) ──────────────────────────────────────────
    def _keyword_dispatch(self, msg: str) -> str:
        """
        Minimal keyword-based dispatcher when Ollama is unavailable.
        Understands: list, load <name>, run, plots, pipeline <name>,
                     power levels <file>, list power levels, clear power.
        """
        m = msg.lower()

        # ── power-level commands ──────────────────────────────────────────
        if "list" in m and ("power" in m or "level" in m):
            return '```tool_call\n{"tool": "list_power_levels", "args": {}}\n```'

        if ("clear" in m or "continuous" in m) and "power" in m:
            return '```tool_call\n{"tool": "load_power_levels", "args": {"file": "none"}}\n```'

        # "load power levels from <file>" / "use power levels <file>"
        for kw in ("power level", "power_level", "power file", "quantis"):
            if kw in m:
                # Try to extract a filename token after the keyword
                parts = msg.split()
                for i, tok in enumerate(parts):
                    if tok.lower() in ("from", "file", "levels", "level"):
                        if i + 1 < len(parts):
                            fname = parts[i + 1].strip("'\",")
                            return (
                                f'```tool_call\n'
                                f'{{"tool": "load_power_levels", "args": {{"file": "{fname}"}}}}\n'
                                f'```'
                            )
                # No filename found — ask
                return (
                    "Please specify the power-levels filename, e.g.:\n"
                    "  'Use power levels from coarse_50w'"
                )

        # ── track listing ─────────────────────────────────────────────────
        if "list" in m and "track" in m:
            return '```tool_call\n{"tool": "list_tracks", "args": {}}\n```'

        # ── pipeline / load / run ─────────────────────────────────────────
        for name in ("horizontal", "45deg", "triangle"):
            if name in m:
                if "pipeline" in m or ("run" in m and "full" in m):
                    xsec  = "no cross" not in m and "no xsec" not in m
                    # Include power levels if they are loaded
                    pl    = ""
                    if self.state.power_levels_source:
                        pl = f', "power_levels_file": "{self.state.power_levels_source}"'
                    return (
                        f'```tool_call\n'
                        f'{{"tool": "run_full_pipeline", "args": {{"track_file": "{name}", '
                        f'"include_xsections": {str(xsec).lower()}{pl}}}}}\n'
                        f'```'
                    )
                if "load" in m:
                    return (
                        f'```tool_call\n'
                        f'{{"tool": "load_track", "args": {{"track_file": "{name}"}}}}\n'
                        f'```'
                    )

        if "controller" in m or "control" in m:
            return '```tool_call\n{"tool": "run_controller", "args": {}}\n```'
        if "plot" in m:
            xsec = "no cross" not in m
            return (
                f'```tool_call\n'
                f'{{"tool": "generate_plots", "args": {{"include_xsections": {str(xsec).lower()}, '
                f'"include_scan_path": true}}}}\n'
                f'```'
            )

        return (
            "I'm not sure what to do. Try:\n"
            "  'Run the full pipeline for [horizontal|45deg|triangle]'\n"
            "  'Load power levels from <filename>'\n"
            "  'List power level files'"
        )

    # ── Execute tools ──────────────────────────────────────────────────────
    def _execute_tools(self, llm_text: str) -> tuple[str, list[str]]:
        """
        Parse and execute all tool_calls found in llm_text.

        Returns (narrative_text, tool_results).
        """
        narrative     = strip_tool_calls(llm_text)
        calls         = extract_tool_calls(llm_text)
        tool_results  = []

        for call in calls:
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})

            if self.verbose:
                print(f"\n[TOOL] {tool_name}({tool_args})")

            fn = TOOL_REGISTRY.get(tool_name)
            if fn is None:
                result = f"ERROR: Unknown tool '{tool_name}'."
            else:
                try:
                    result = fn(tool_args, self.state)
                except Exception as exc:
                    result = f"ERROR running {tool_name}: {exc}"

            if self.verbose:
                print(f"[RESULT]\n{result}\n")

            tool_results.append(result)

        return narrative, tool_results

    # ── Main chat method ───────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        """
        Send a user message, let the LLM decide which tools to call,
        execute them, and return a final human-readable response.
        """
        # LLM turn
        llm_text = self._llm(user_message)

        # Execute tools
        narrative, tool_results = self._execute_tools(llm_text)

        # Build final response
        parts = []
        if narrative:
            parts.append(narrative)
        if tool_results:
            parts.extend(tool_results)
        final_response = "\n\n".join(parts)

        # Update conversation history
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": final_response})

        return final_response

    def reset(self) -> None:
        """Clear conversation history and agent state."""
        self.history = []
        self.state   = AgentState()
        print("Agent state and history cleared.")


# ── CLI ───────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LPBF control agent powered by Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive REPL
  python control_agent/ollama_agent.py

  # One-shot full pipeline
  python control_agent/ollama_agent.py --track 45deg

  # One-shot with quantised power
  python control_agent/ollama_agent.py --track horizontal --power-levels coarse_50w

  # Custom message
  python control_agent/ollama_agent.py --message "Load triangle and run controller only"

  # Use a different model
  OLLAMA_MODEL=mistral python control_agent/ollama_agent.py --track horizontal
""",
    )
    p.add_argument("--track",    metavar="NAME",
                   help="Track name (horizontal | 45deg | triangle) — runs full pipeline")
    p.add_argument("--message",  metavar="MSG",
                   help="Single message to send to the agent")
    p.add_argument("--no-xsections", action="store_true",
                   help="Skip cross-section computation")
    p.add_argument("--power-levels", metavar="FILE", default="",
                   help="Power-levels file to use (bare name or path, "
                        "e.g. 'coarse_50w'). Omit for continuous control.")
    p.add_argument("--model",    default=DEFAULT_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_MODEL})")
    p.add_argument("--host",     default=DEFAULT_HOST,
                   help=f"Ollama server URL (default: {DEFAULT_HOST})")
    p.add_argument("--quiet",    action="store_true",
                   help="Suppress verbose tool output")
    return p


def _repl(agent: ControlAgent) -> None:
    print("=" * 60)
    print("  LPBF Control Agent  (type 'exit' or Ctrl-C to quit)")
    print("  Type 'reset' to clear state and history")
    print("=" * 60)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        response = agent.chat(user_input)
        print(f"\nAgent:\n{response}")


def main() -> None:
    args  = _build_parser().parse_args()
    agent = ControlAgent(
        model=args.model,
        host=args.host,
        verbose=not args.quiet,
    )

    if args.track:
        xsec = not args.no_xsections

        # Load power levels before building the message if requested
        if args.power_levels:
            from control_agent.tools import tool_load_power_levels
            result = tool_load_power_levels({"file": args.power_levels}, agent.state)
            print(f"[Power levels]\n{result}\n")

        msg = f"Run the full pipeline for the {args.track} track."
        if not xsec:
            msg += " Skip cross-sections."
        if args.power_levels:
            msg += f" Use power levels from {args.power_levels}."

        print(f"You: {msg}")
        response = agent.chat(msg)
        print(f"\nAgent:\n{response}")

    elif args.message:
        print(f"You: {args.message}")
        response = agent.chat(args.message)
        print(f"\nAgent:\n{response}")

    else:
        _repl(agent)


if __name__ == "__main__":
    main()