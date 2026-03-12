#!/usr/bin/env python3
"""
run_control.py
==============
Root-level convenience script to launch the LPBF control agent.

Placed at the project root so it can find both control_tools/ and
control_agent/ without any path gymnastics.

Usage examples
--------------
  # Interactive REPL (Ollama):
  python run_control.py

  # Full pipeline for one track type:
  python run_control.py --track 45deg
  python run_control.py --track triangle
  python run_control.py --track horizontal --no-xsections

  # Single message:
  python run_control.py --message "Load the 45-degree track and show me what configs are available"

  # Different Ollama model:
  OLLAMA_MODEL=mistral python run_control.py --track triangle

  # Quiet mode (suppress per-tool verbose output):
  python run_control.py --track 45deg --quiet

Environment variables
---------------------
  OLLAMA_MODEL   Ollama model to use (default: llama3.2)
  OLLAMA_HOST    Ollama server URL   (default: http://localhost:11434)

Track files
-----------
  control_tools/tracks/horizontal.json
  control_tools/tracks/45deg.json
  control_tools/tracks/triangle.json

  You can add your own track JSON file to control_tools/tracks/ with any
  subset of TrackConfig fields and a required "track_type" key.
"""

import sys
from pathlib import Path

# Ensure this directory is on sys.path (it should be when run from root, but
# adding it explicitly makes the script robust to being called from elsewhere).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_agent.ollama_agent import main

if __name__ == "__main__":
    main()
