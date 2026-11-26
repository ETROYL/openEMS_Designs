# ETROYL Desktop GUI

The PySide6 desktop application is being developed incrementally around the
existing YAML-driven openEMS design and optimization code.

## Current functionality

- Open, edit, save and validate YAML design files.
- Generate CSXCAD geometry from the current YAML editor contents.
- Open the generated geometry in the external AppCSXCAD viewer.
- Run a standalone openEMS FDTD simulation from the current YAML editor contents.
- Display the simulation process output and allow the user to stop a run.
- Save the exact YAML snapshot alongside each simulation output for reproducibility.
- Run the existing root-level `postprocess.py` from the Results tab.
- Display S-parameter/impedance and NF2FF radiation plots inside the GUI.
- Keep simulation output in a separate, uniquely named folder selected by the user.
- Run the ETROYL discrete optimizer from the Optimization tab with live evaluation progress, best-objective reporting, a live objective/best-so-far convergence plot and cooperative stop requests.
- Save each optimization launch as an immutable YAML snapshot and a timestamped run under `<design>_opt/`.
- Save each run's SQLite evaluation history, individual FDTD runs and `best_design.yaml` together.
- Load the best optimized YAML into the editor without changing the original source-file path.
- CAD preview and simulation use unsaved editor changes; neither overwrites the
  selected source design.

The Optimization tab currently integrates the ETROYL discrete Sobol + Gaussian-process + expected-improvement optimizer. Its controls include the new-evaluation budget, initial Sobol sample count, candidate-pool size, random seed and objective function. Available objectives are matched-bandwidth cost, worst-case S11, S11 magnitude, impedance smoothness and the combined S11 + impedance-smoothness objective. The combined objective exposes independent weights; defaults (1.0 and 0.0053) preserve the original formula. Pause and stop requests are cooperative: an in-progress FDTD evaluation completes before another candidate is launched. A paused run can be resumed against its existing SQLite history and original run settings. Optuna and SciPy are listed as upcoming strategies but are not yet executable. The output structure is `<design>_opt/run_<timestamp>/`, containing `config.yaml`, `run_settings.json`, `evaluations.sqlite`, `runs/` and (when at least one evaluation succeeds) `best_design.yaml`.

The standalone simulation currently reads its frequency range, end criterion
and maximum timestep count from the YAML `simulation` block. It uses PML_8
boundaries and the thread count from the repository's `parallel.py` module.
The Results tab reuses the repository-level `postprocess.py` implementation. It currently processes a single-port design and reports the radiation-efficiency assumption explicitly: 100% for an ideal lossless PEC model, rather than a computed loss efficiency. Multi-port processing and GUI-controlled resource allocation will be added in later increments.

## Install

Activate the Python environment that already has the openEMS and CSXCAD
bindings configured, then install the optional GUI dependencies:

```bash
python3 -m pip install -r requirements-gui.txt

The Optimization tab embeds a live Matplotlib convergence plot beside the run console.
```

The CAD preview also requires the `AppCSXCAD` executable to be installed and
available on `PATH`.

## Launch

Run from the repository root:

```bash
python3 -m gui.app
```

The GUI is the supported entry point for the integrated design, simulation and post-processing workflow.
