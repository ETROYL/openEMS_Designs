# openEMS Parametric Design, Simulation & Global Optimization Automation Framework

A modular, YAML-driven Python framework for modeling, visualizing, simulating, post-processing, and **globally optimizing** 3D electromagnetic structures using [openEMS](https://github.com/thliebig/openEMS-Project) and CSXCAD.

---

## 📋 Key Features

- **Parametric YAML geometry definition**: define materials, primitives (polygons/boxes), and ports using math expressions and bounded optimization variables.
- **Two global optimization engines**:
  - **Optuna TPE sampler** (`optimization.py`): Bayesian optimization with live 2D geometry tracking.
  - **SciPy SHGO + Nelder-Mead** (`scipy_optimization.py`): global simplicial-homology search combined with local simplex refinement, discrete parameter quantization, and evaluation caching.
- **Pluggable cost functions** (`objectives.py`): S<sub>11</sub> minimization, worst-case S<sub>11</sub>, in-band bandwidth, and Z<sub>in</sub> smoothness, plus a weighted combined objective.
- **Live 2D geometry monitor**: real-time Matplotlib view of the geometry of each trial/evaluation (can be disabled for headless runs).
- **Adaptive wavelength meshing**: grid step ≈ λ<sub>min</sub>/20 and PML air padding ≈ λ<sub>max</sub>/4, derived from the simulation frequency bounds.
- **High-accuracy export**: when an optimization finishes, a ready-to-run optimized YAML is written with strict convergence settings ($10^{-4}$ energy decay, 1024 frequency points).
- **Full-wave post-processing**: S<sub>11</sub>, Z<sub>in</sub>, VSWR, far-field patterns, peak directivity, realized gain, and efficiency.

---

## 🛠️ Installation & Environment Setup

### 1. Prerequisites (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake git libhdf5-dev libvtk9-dev \
                    libboost-all-dev libcgal-dev libtinyxml-dev \
                    python3-pip python3-numpy python3-matplotlib python3-h5py
```

Install the Python optimization dependencies:

```bash
pip install optuna scipy pyyaml
```

### 2. Build openEMS from Source

```bash
# Clone repository with submodules
git clone --recursive https://github.com/thliebig/openEMS-Project.git
cd openEMS-Project

# Build openEMS and the CSXCAD Python bindings
./update_openEMS.sh ~/opt/openEMS --with-CSXCAD-python --with-openEMS-python

# Add these to your ~/.bashrc
export PATH=$PATH:~/opt/openEMS/bin
export PYTHONPATH=$PYTHONPATH:~/opt/openEMS/share/CSXCAD/python:~/opt/openEMS/share/openEMS/python
```

Verify the installation:

```bash
python3 -c "import openEMS; import CSXCAD; print('openEMS imported successfully!')"
```

---

## 🗂️ Project Structure

```text
.
├── design.py              # YAML parser, expression evaluator, adaptive meshing & CSX geometry builder
├── objectives.py          # Cost functions: S11, worst-case S11, bandwidth, Zin smoothness, combined
├── optimization.py        # Optuna TPE optimizer with live geometry monitor
├── scipy_optimization.py  # SciPy SHGO + Nelder-Mead optimizer with quantization & caching
├── simulate.py            # Single-run FDTD execution (invokes the openEMS engine)
├── postprocess.py         # S11, Zin, VSWR, far-field patterns, gain & efficiency plots
├── view_cad.py            # Interactive AppCSXCAD 3D geometry viewer
├── README.md              # Project documentation
└── designs/
    ├── bowtie01.yaml          # Static (fixed-geometry) design file
    └── dipole_variable.yaml   # Parametric design file used for optimization (example below)
```

> **Note:** `simulate.py`, `postprocess.py` and `view_cad.py` derive their output folder by stripping the last five characters of the config path, so design files must use the `.yaml` extension. `designs/foo.yaml` → results in `designs/foo/`.

---

## 🚀 Workflow Guide

### 1. Inspect the default geometry in 3D

Before optimizing, check the geometry built from each variable's `default` value:

```bash
python3 view_cad.py --config designs/dipole_variable.yaml
```

You can also run `design.py` on its own to test parsing and display the adaptively calculated mesh limits:

```bash
python3 design.py --config designs/dipole_variable.yaml
```

### 2. Run a global optimization

Choose one of the two engines.

**Method A: Optuna (TPE sampler)**. Good for exploring continuous search spaces with many variables.

```bash
python3 optimization.py --config designs/dipole_variable.yaml --n_trials 200
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `designs/bowtie_variable.yaml` | Path to the parametric YAML design file |
| `--n_trials` | `30` | Number of optimization trials |
| `--no_plot` | off | Disable the live 2D plot window (headless runs) |

**Method B: SciPy (SHGO + Nelder-Mead)**. Deterministic simplicial-homology space partitioning with local simplex refinement.

```bash
python3 scipy_optimization.py --config designs/dipole_variable.yaml --max_evals 80
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `designs/bowtie_variable.yaml` | Path to the parametric YAML design file |
| `--max_evals` | `80` | Maximum simulation budget (hard cap) |
| `--no_plot` | off | Disable the live 2D plot window (headless runs) |

**What you get:**

- **Live monitor**: a 2D XY footprint plot of the current trial's polygons and lumped port, updated before each simulation starts.
- **Per-trial data**: every candidate is simulated in its own folder.
  - Optuna: `designs/<config_name>_opt/trial_000/`, `trial_001/`, …
  - SciPy: `designs/<config_name>_opt_shgo/eval_001/`, `eval_002/`, …
- **Optimal design**: `designs/<config_name>_optimized.yaml`, with all variables resolved to their optimal values and high-accuracy simulation settings applied.

> Trial folders contain full FDTD field/port data, so long runs can use significant disk space. Delete the `_opt` / `_opt_shgo` folders once you have the optimized YAML.

### 3. Run the high-accuracy verification simulation

Run a full-convergence FDTD pass on the generated optimal design:

```bash
python3 simulate.py --config designs/dipole_variable_optimized.yaml
```

### 4. Post-process & analyze performance

```bash
python3 postprocess.py --config designs/dipole_variable_optimized.yaml
```

This computes S<sub>11</sub>, Z<sub>in</sub>, VSWR, peak directivity, radiation patterns, realized gain, and efficiency. Plots (`s11_plot.png` and `radiation_plot.png`) are saved in `designs/dipole_variable_optimized/`.

### Quick reference

```bash
python3 design.py          --config designs/dipole_variable.yaml
python3 optimization.py    --config designs/dipole_variable.yaml --n_trials 200
python3 postprocess.py     --config designs/dipole_variable_optimized.yaml
```

---

## 🎯 Objective Functions (`objectives.py`)

All cost functions take the simulation results (frequency vector, complex S<sub>11</sub>, S<sub>11</sub> in dB, Z<sub>in</sub>) and return a scalar to **minimize**. `f_min` / `f_max` restrict the evaluated band; when left as `None` (the default in both optimizers) the **entire simulated spectrum** is used.

| Function | What it does |
|---|---|
| `minimize_s11_in_band` | Mean + standard deviation of the linear \|S<sub>11</sub>\| over the band. Penalizes both poor and uneven matching. |
| `minimize_max_s11_in_band` | Worst-case S<sub>11</sub> (dB) plus a quadratic penalty for every point above `target_db` (default −10 dB). |
| `maximize_bandwidth_in_band` | Rewards the fraction of the band where S<sub>11</sub> is below `target_db` (default −10 dB). |
| `maximize_impedance_smoothness` | Standard deviation of \|Z<sub>in</sub>\| across the band: flatter impedance gives a lower cost. |
| `multi_objective_combined` | **Used by default.** `1.0 × minimize_s11_in_band + 0.0053 × maximize_impedance_smoothness`. |

### Changing the objective

- **Optuna**: in `objective_wrapper()` inside `optimization.py`, uncomment the cost line you want and comment out the others.
- **SciPy**: in `ScipyObjectiveWrapper.__call__()` inside `scipy_optimization.py`, replace the `multi_objective_combined(...)` call (and its import from `objectives`).
- **New objective**: add a function to `objectives.py` that accepts the results dictionary and returns a float.

Failed simulations return a cost of `1e6`, so the optimizers steer away from them.

---

## ⚙️ How the Optimizers Work

### Optuna TPE (`optimization.py`)

- Reads the `variables` block and calls `suggest_float` / `suggest_int` per variable, honoring `step`.
- Variables with `optimize: false`, or with `start == stop`, are held at their `default`.
- Minimizes the cost over `--n_trials` trials. Each trial resolves the expressions, simulates in a trial folder, and evaluates the objective.
- Prints the best trial, the best objective value, and the cleaned (6-decimal) optimal parameters.

### SciPy SHGO + Nelder-Mead (`scipy_optimization.py`)

- Uses `scipy.optimize.shgo` for global search with Nelder-Mead as the local minimizer (current settings: 64 sampling points, 3 homology iterations, local simplex limit of 30 evaluations).
- **Quantization**: continuous optimizer values are snapped to each variable's `step` grid and clipped to `[start, stop]`, so the geometry stays on realistic increments.
- **Memoization**: identical quantized parameter sets are never re-simulated. Cached results do not count toward the budget.
- **Budget cap**: `--max_evals` limits the number of real FDTD runs. Because SHGO's initial sampling can consume much of a small budget, use a comfortably larger `--max_evals` if you want local refinement to run.

---

## 🧩 How the Core Scripts Work

### `design.py`

- Reads the YAML file and builds a variable dictionary from each variable's `default`, overridden by optimizer-supplied values (`active_vars`).
- Evaluates every string in the config as a math expression using those variables (`math`, `np` and `abs` are available). Results are rounded to 6 decimals (0.1 µm for meter units). Plain strings such as material names are left untouched.
- Computes mesh limits: Δl<sub>max</sub> = λ<sub>min</sub>/20, minimum snap spacing λ<sub>min</sub>/60, air padding λ<sub>max</sub>/4.
- Instantiates metal materials, linear polygons, boxes, and lumped ports with the configured resistance.
- Attaches a NF2FF box only when `enable_nf2ff=True`. The optimizers disable it for speed.

### `view_cad.py`

- Builds the geometry with default variable values and writes a preview XML into the design folder.
- Launches AppCSXCAD for interactive inspection of geometry, ports and mesh lines.

### `simulate.py`

- Builds a Gaussian excitation spanning [f<sub>start</sub>, f<sub>stop</sub>].
- Reads `end_criteria` (default `1e-4`) and `max_timesteps` (default `500000`) from the YAML `simulation` block.
- Uses PML_8 absorbing boundaries on all six sides.
- Writes the geometry XML and runs the full-wave solver into the design-specific output folder.

### `postprocess.py`

- Reads port voltage/current data and computes S<sub>11</sub>, VSWR and complex Z<sub>in</sub>.
- Calls `nf2ff.CalcNF2FF()` to compute E-plane/H-plane patterns at the best-match frequency, plus directivity, realized gain and efficiency across the band.
- Radiation efficiency is taken as 100 % (lossless PEC in air); total efficiency includes the S<sub>11</sub> mismatch loss.
- Saves `s11_plot.png` and `radiation_plot.png` in the design folder.

---

## 🧵 Multithreading Control

openEMS uses OpenMP. The thread count is controlled by `OMP_NUM_THREADS`:

- `optimization.py` does **not** set it, so you can control it from the shell:

  ```bash
  export OMP_NUM_THREADS=16
  python3 optimization.py --config designs/dipole_variable.yaml
  ```

- `simulate.py`, `postprocess.py` and `scipy_optimization.py` currently set it inside the script, which overrides any value exported in the shell. Edit this line to match your CPU:

  ```python
  os.environ["OMP_NUM_THREADS"] = "16"
  ```

---

## 📄 Parametric Design YAML Reference

The example below (`designs/dipole_variable.yaml`) is a generic tapered planar dipole that illustrates the format. Replace it with your own geometry.

```yaml
variables:
  arm_length:                 # Optimized between start and stop, in steps
    type: "float"
    start: 0.15
    stop: 0.30
    step: 0.01
    default: 0.20             # Used by view_cad.py / design.py
  root_width:
    type: "float"
    start: 0.004
    stop: 0.020
    step: 0.002
    default: 0.010
  tip_width:
    type: "float"
    start: 0.010
    stop: 0.080
    step: 0.005
    default: 0.040
  feed_gap:                   # Held fixed during optimization
    type: "float"
    start: 0.002
    stop: 0.002
    default: 0.002
    optimize: false

simulation:
  unit: 1.0                   # Metric unit scale (1.0 = meters)
  fstart: 0.3                 # Start frequency in GHz
  fstop: 1.0                  # Stop frequency in GHz
  n_pts: 301                  # Frequency sweep resolution
  end_criteria: 1e-2          # Loose energy threshold for fast optimization trials
  max_timesteps: 50000        # Caps long runs during optimization
  target_s11_db: -15.0        # Informational design target stored in the config

materials:
  - name: "PEC_Patch"
    type: "metal"

primitives:
  # Right arm (tapered from root_width to tip_width)
  - type: "polygon"
    material: "PEC_Patch"
    priority: 10
    elevation: 0
    length: 0.001
    norm_dir: "z"
    points:
      x: ["feed_gap / 2.0", "arm_length", "arm_length", "feed_gap / 2.0"]
      y: ["root_width / 2.0", "tip_width / 2.0", "-tip_width / 2.0", "-root_width / 2.0"]

  # Left arm (mirror image)
  - type: "polygon"
    material: "PEC_Patch"
    priority: 10
    elevation: 0
    length: 0.001
    norm_dir: "z"
    points:
      x: ["-feed_gap / 2.0", "-arm_length", "-arm_length", "-feed_gap / 2.0"]
      y: ["root_width / 2.0", "tip_width / 2.0", "-tip_width / 2.0", "-root_width / 2.0"]

ports:
  - id: 1
    type: "lumped"
    resistance: 50.0
    direction: "x"
    excite: 1
    start: ["-feed_gap / 2.0", "-root_width / 2.0", 0]
    stop:  ["feed_gap / 2.0",   "root_width / 2.0",  0.001]
```

### Variable options

| Key | Description |
|---|---|
| `type` | `float` (default) or `int` |
| `start` / `stop` | Lower and upper search bounds |
| `step` | Optional discretization step (quantization grid) |
| `default` | Value used for visualization and for variables that are not optimized |
| `optimize` | Set to `false` to keep the variable fixed at `default`. A variable with `start == stop` is also fixed. |

### Expression rules

- Any string value in `points`, port/box `start`/`stop`, etc. is evaluated as a math expression using the variable names (e.g. `"arm_length / 2.0"`).
- Numeric values (not strings) are passed through unchanged.
- Static designs without a `variables` block (such as `bowtie01.yaml`) continue to work with all single-run scripts.

### Optimized output

After an optimization, `designs/<name>_optimized.yaml` contains the fully resolved numeric geometry and the high-accuracy overrides:

```yaml
simulation:
  end_criteria: 1.0e-4
  max_timesteps: 1000000
  n_pts: 1024
  target_s11_db: -40.0
```
