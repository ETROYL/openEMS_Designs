# openEMS Design & Simulation Automation Framework

A modular, YAML-driven Python framework for modeling, visualizing, simulating, and post-processing 3D electromagnetic structures using [openEMS](https://github.com/thliebig/openEMS-Project) and CSXCAD.

## 📋 Features

- **YAML-based geometry definition**: define materials, polygon/box primitives, and excitation ports in clean YAML configuration files.
- **Adaptive wavelength meshing**: automatically calculates the optimal mesh resolution (≈ λ<sub>min</sub>/20) and air/PML padding (≈ λ<sub>max</sub>/4) from the simulation frequency bounds.
- **Auto-directory management**: results are isolated into dedicated output folders named after the target YAML configuration file.
- **Full-wave post-processing**: automatically computes and plots S<sub>11</sub>, input impedance (Z<sub>in</sub>), VSWR, far-field radiation patterns, peak directivity, realized gain, and total vs. radiation efficiency.

## 🛠️ openEMS Installation (Linux)

### Prerequisites (Debian/Ubuntu)

Install the required build tools, libraries, and Python dependencies:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libhdf5-dev libvtk9-dev \
                    libboost-all-dev libcgal-dev libtinyxml-dev \
                    python3-pip python3-numpy python3-matplotlib python3-h5py
```

### Build & Install openEMS from Source

```bash
# Clone repository with submodules
git clone --recursive https://github.com/thliebig/openEMS-Project.git
cd openEMS-Project

# Build openEMS and CSXCAD
./update_openEMS.sh ~/opt/openEMS --with-CSXCAD-python --with-openEMS-python

# Export paths (add these to your ~/.bashrc)
export PATH=$PATH:~/opt/openEMS/bin
export PYTHONPATH=$PYTHONPATH:~/opt/openEMS/share/CSXCAD/python:~/opt/openEMS/share/openEMS/python
```

Verify the installation:

```bash
python3 -c "import openEMS; import CSXCAD; print('openEMS successfully imported!')"
```

## 🗂️ Project Structure

```text
.
├── design.py          # Core logic: YAML parser, adaptive mesh generator & NF2FF builder
├── postprocess.py     # Post-processing: S11, impedance, gain, patterns & efficiency plots
├── README.md          # Project documentation
├── simulate.py        # FDTD execution script (invokes openEMS engine)
├── view_cad.py        # CAD visualization script (AppCSXCAD integration)
└── designs/
    └── bowtie01.yaml  # Design configuration file (geometry, materials, ports)
```

## 🚀 Quickstart Workflow

Run the full design-to-postprocessing pipeline using a configuration file.

### 1. Build and verify the design grid

Run `design.py` to test configuration parsing and display the adaptively calculated mesh limits:

```bash
python3 design.py --config designs/bowtie01.yaml
```

### 2. Visualize the structure in AppCSXCAD

Inspect geometry, materials, excitation ports, and 3D grid lines interactively:

```bash
python3 view_cad.py --config designs/bowtie01.yaml
```

### 3. Run the FDTD simulation

Execute the openEMS FDTD solver engine. Output files are generated under the design-specific output folder:

```bash
python3 simulate.py --config designs/bowtie01.yaml
```

### 4. Post-process & generate plots

Calculate S-parameters and far-field radiation metrics:

```bash
python3 postprocess.py --config designs/bowtie01.yaml
```

## ⚙️ How the Scripts Work

### `design.py`

- Reads the YAML design file (e.g., `bowtie01.yaml`).
- Dynamically derives simulation cell boundaries and mesh steps:
  - Maximum mesh step: $\Delta l_{max} = \lambda_{min} / 20$ (prevents numerical dispersion at $f_{stop}$).
  - Air padding: $\lambda_{max} / 4$ (prevents reactive near-field corruption from the PML).
- Instantiates metal materials, 2D linear polygons, and 3D box primitives.
- Configures 50 Ω lumped excitation ports.
- Attaches a NF2FF (Near-Field to Far-Field) box for far-field calculations.

### `view_cad.py`

- Calls `design.py` to construct the CSXCAD mesh and geometry structure in memory.
- Launches the AppCSXCAD GUI for interactive 3D inspection before running heavy FDTD computations.

### `simulate.py`

- Prepares the Gaussian excitation pulse spanning $[f_{start}, f_{stop}]$.
- Generates the simulation output directory based on the input configuration filename.
- Writes XML geometry files and executes the openEMS full-wave solver until convergence (e.g., a $10^{-4}$ energy threshold).

### `postprocess.py`

- Reads the port time-domain voltage and current signals, $u(t)$ and $i(t)$.
- Computes frequency-domain S-parameters (S<sub>11</sub>), VSWR, and complex input impedance (Z<sub>in</sub>).
- Invokes `nf2ff.CalcNF2FF()` to compute:
  - 2D polar radiation patterns (E-plane and H-plane).
  - Peak directivity (dBi).
  - Realized gain (dBi), including S<sub>11</sub> mismatch loss.
  - Radiation efficiency ($\eta_{rad}$) vs. total efficiency ($\eta_{tot}$).
- Saves the figure deliverables (`s11_plot.png` and `radiation_plot.png`) inside the generated design directory.

## 📄 Design YAML Reference Example (`bowtie01.yaml`)

```yaml
simulation:
  unit: 1.0     # Metric unit scale (1.0 = meters)
  fstart: 0.2   # Start frequency in GHz
  fstop: 1.2    # Stop frequency in GHz
  n_pts: 1001   # Frequency sweep resolution

materials:
  - name: "PEC_Patch"
    type: "metal"

primitives:
  - type: "polygon"
    material: "PEC_Patch"
    priority: 10
    elevation: 0
    length: 0.0005
    norm_dir: "z"
    points:
      x: [0.001, 0.011, 0.161, 0.161, 0.011, 0.001]
      y: [0.0005, 0.068, 0.083, -0.083, -0.068, -0.0005]

ports:
  - id: 1
    type: "lumped"
    resistance: 50.0
    direction: "x"
    excite: 1
    start: [-0.001, -0.0005, 0]
    stop:  [ 0.001,  0.0005,  0.0005]
```
