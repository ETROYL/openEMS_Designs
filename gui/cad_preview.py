"""Build the currently edited YAML design and open it with AppCSXCAD."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from CSXCAD import ContinuousStructure
from openEMS import openEMS

from design import build_design, load_config


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m gui.cad_preview <design.yaml>", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1]).resolve()
    cfg = load_config(str(config_path))
    simulation = cfg.get("simulation", {})
    fstart = float(simulation.get("fstart", 0.1)) * 1e9
    fstop = float(simulation.get("fstop", 1.0)) * 1e9
    if fstart <= 0 or fstop <= fstart:
        raise ValueError("simulation.fstart/fstop must define a positive increasing frequency range")

    app = shutil.which("AppCSXCAD")
    if app is None:
        raise FileNotFoundError(
            "AppCSXCAD executable was not found on PATH. Install the CSXCAD GUI or add AppCSXCAD to PATH."
        )

    csx = ContinuousStructure()
    fdtd = openEMS(EndCriteria=1e-4)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite((fstart + fstop) / 2.0, fstop - fstart)
    build_design(csx, fdtd, str(config_path), enable_nf2ff=False)

    with tempfile.TemporaryDirectory(prefix="etroyl_cad_") as workdir:
        xml_path = Path(workdir) / "design_preview.xml"
        csx.Write2XML(str(xml_path))
        print(f"Generated preview geometry: {xml_path}", flush=True)
        print("Launching AppCSXCAD…", flush=True)
        return subprocess.run([app, str(xml_path)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
