# ETROYL Discrete Surrogate-Assisted Optimizer

## Scope of this first implementation

This branch introduces the discrete parameter-space foundation and a Sobol-based initial sampler. It does not yet launch openEMS simulations or fit a surrogate model. The existing Optuna and SciPy scripts remain untouched.

## Design assumptions

- Typical design has 5–10 active parameters; FDTD trials typically take less than one minute.
- The objective is initially the existing scalar cost to minimize.
- Every optimized float variable must declare a positive YAML step.
- The grid is anchored at start: value(k) = start + k * step.
- If stop is not exactly reachable, the final point is the largest grid value not exceeding stop.
- optimize: false, literal variables and zero-width ranges are fixed dimensions.
- A YAML default is a CAD-preview default, not an optimization bound. It may intentionally lie outside [start, stop]; the supplied bowtie has arm_length default 0.18 while its search lower bound is 0.20.

## Supplied examples: search-space sizes

### Two-disk design

| Variable | Grid points |
|---|---:|
| disk_radius | 41 |
| feed_gap | 5 |
| port_width | 20 |

Total: **4,100** discrete geometries.

### Six-variable bowtie design

| Variable | Grid points |
|---|---:|
| arm_length | 8 (0.20 through 0.34 m) |
| arm_width | 13 (0.10 through 0.34 m) |
| flare_width | 20 |
| flare_length | 16 |
| feed_gap | 5 |
| feed_width | 13 |

Total: **2,163,200** discrete geometries. The requested stop values 0.35 m are not grid points for arm_length or arm_width with a 0.02 m step, so they are not included.

## Planned optimizer loop

1. Generate an initial space-filling set using scrambled Sobol points mapped to integer grid indices.
2. Run each candidate through the existing design builder, FDTD wrapper and objective function.
3. Persist parameter indices, physical values, objective, metrics, simulation settings, status and a stable evaluation fingerprint.
4. Fit an ensemble surrogate (initial candidate: Extra Trees; compare Random Forest during benchmarking).
5. Generate a large pool of legal discrete candidates and rank them using expected improvement plus explicit exploration and diversity.
6. Evaluate a small batch, update the model and repeat until the simulation budget or stopping rule is reached.
7. Re-run the best few geometries using strict convergence and frequency settings.

Acquisition uncertainty must be calibrated against held-out or cross-validated residuals. Raw tree-to-tree spread must not be assumed to be a calibrated posterior standard deviation. Candidate selection should reserve budget for global exploration and avoid near-duplicate geometries within a batch.

## Benchmark protocol

Compare random discrete search, existing Optuna TPE and the new optimizer using identical YAML designs, objective functions, simulation settings and FDTD evaluation budgets. Repeat with multiple random seeds. Report best-so-far objective versus actual unique FDTD runs, wall-clock time, failed runs and repeatability. Do not compare optimizer iteration counts alone.

## Current module status

- optimizer/parameter_space.py: exact Decimal-backed grid definitions, integer-index conversion, fixed variables and grid metadata.
- optimizer/initial_sampling.py: unique scrambled Sobol samples projected onto the discrete grid.
- tests/test_discrete_parameter_space.py: unit tests for both supplied examples and edge cases.
