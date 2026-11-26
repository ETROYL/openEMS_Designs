# ETROYL Desktop GUI

The desktop application is being developed incrementally around the existing
YAML-driven openEMS design and optimization code. The current milestone
provides the PySide6 application shell and a YAML open/edit/save/validate
workflow. Optimizer, CAD, simulation and result adapters are intentionally
not connected yet.

## Install

Activate the Python environment that already has the openEMS and CSXCAD
bindings configured, then install the optional GUI dependencies:

```bash
python3 -m pip install -r requirements-gui.txt
```

## Launch

Run from the repository root:

```bash
python3 -m gui.app
```

The existing command-line workflows remain available and unchanged.
