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
- CAD preview and simulation use unsaved editor changes; neither overwrites the
  selected source design.

The standalone simulation currently reads its frequency range, end criterion
and maximum timestep count from the YAML `simulation` block. It uses PML_8
boundaries and the thread count from the repository's `parallel.py` module.
The Results tab reuses the repository-level `postprocess.py` implementation. It currently processes a single-port design and reports the radiation-efficiency assumption explicitly: 100% for an ideal lossless PEC model, rather than a computed loss efficiency. Multi-port processing and GUI-controlled resource allocation will be added in later increments.

## Install

Activate the Python environment that already has the openEMS and CSXCAD
bindings configured, then install the optional GUI dependencies:

```bash
python3 -m pip install -r requirements-gui.txt
```

The CAD preview also requires the `AppCSXCAD` executable to be installed and
available on `PATH`.

## Launch

Run from the repository root:

```bash
python3 -m gui.app
```

The GUI is the supported entry point for the integrated design, simulation and post-processing workflow.
