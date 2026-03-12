# Agentic Stochastic Control of LPBF

This repository contains the code accompanying our paper, **“LLM Agent-guided Stochastic Control of Multi-layer and Multi-track Laser Powder Bed Fusion Process.”** The framework combines two coordinated agents:

- **CalibAgent** for Bayesian stochastic calibration of an analytical LPBF thermal model using experimental melt-pool data
- **ControlAgent** for feedforward laser-power scheduling to maintain near-uniform melt-pool depth across complex scan geometries

The workflow is developed for LPBF melt-pool prediction and control using an analytical thermal model, stochastic parameter inference, and agent-guided execution logic. The paper demonstrates the framework on IN718 across horizontal hatch, 45° hatch, concentric triangular hatch, and multi-layer L-shaped scan paths.

## Repository structure

```text
agentic-stochastic-lpbf-control/
├── calib_agent/
├── calib_tools/
├── control_agent/
├── control_tools/
├── path_temperature_field.py
├── run_calibagent.py
├── run_control.py
└── control_triangle_full_domain.gif