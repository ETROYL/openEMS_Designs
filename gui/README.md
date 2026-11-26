# ETROYL Desktop GUI

The PySide6 desktop application is being developed incrementally around the
existing YAML-driven openEMS design and optimization code.

## Current functionality

- Open, edit, save and validate YAML design files.
- Generate CSXCAD geometry from the current YAML editor contents.
- Open the generated geometry in the external AppCSXCAD viewer.
- CAD preview uses unsaved editor changes and temporary files; it does not
  overwrite the selected design or create output in the design directory.

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

The existing command-line workflows remain available and unchanged.
