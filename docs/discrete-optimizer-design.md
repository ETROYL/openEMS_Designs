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
- optimizer/evaluator.py: candidate-to-FDTD adapter with injectable simulation/objective functions, geometry fingerprints, isolated run directories, structured success/failure results and compact S11/impedance metrics.
- optimizer/evaluation_store.py: SQLite history keyed by geometry fingerprint, with duplicate protection, explicit retry/replace, status filtering and best-result lookup.
- tests/test_discrete_parameter_space.py: unit tests for both supplied examples and edge cases.
- tests/test_candidate_evaluator.py: mock-runner tests for parameter mapping, objective evaluation, fingerprints, metrics and failure reporting.
- tests/test_evaluation_store.py: persistence, duplicate protection, explicit retry, failed-run retention and best-result selection.

### Candidate evaluation API

The evaluator can be tested without openEMS by injecting a simulation runner and objective function. On a machine with the repository's openEMS Python dependencies installed, bind it to the current implementation with:

```python
from optimizer import CandidateEvaluator, DiscreteParameterSpace

space = DiscreteParameterSpace.from_config(config)
evaluator = CandidateEvaluator.from_existing_pipeline(
    config_file="designs/your_design.yaml",
    output_dir="designs/your_design_discrete_opt",
    parameter_space=space,
)
result = evaluator.evaluate((3, 2, 1), evaluation_id=1)
print(result.status, result.objective, result.parameters)
```

The evaluator reports failed runs explicitly with a configurable penalty; downstream optimization and reporting must inspect status, rather than treating a penalty as a valid physical result. Repeated attempts receive separate simulation directories, while the geometry/config fingerprint remains stable for later deduplication in the evaluation database.

### Evaluation database API

Store evaluation history in a local SQLite database. The fingerprint is the primary key, so a geometry/configuration already evaluated is not silently simulated and recorded again. Failed evaluations are stored as well.

```python
from optimizer import EvaluationStore

with EvaluationStore("designs/your_design_discrete_opt/evaluations.sqlite") as store:
    # Save once; returns False if this fingerprint is already present.
    inserted = store.save_result(result)

    previous = store.get(result.fingerprint)
    best = store.best()  # lowest objective among successful evaluations
    print(store.count(), store.count("failed"))
```

A retry is deliberately explicit: call `store.save_result(new_result, replace=True)` only when you intend to replace the existing record for that fingerprint. The database is restart-persistent; keep it alongside the optimization output directory and back it up with the simulation results you want to retain.


## Surrogate and acquisition component

`optimizer/surrogate.py` adds a lightweight Gaussian-process regression model
with a Matérn-5/2 ARD kernel and expected improvement (EI) for minimization.
It uses the repository's existing NumPy/SciPy dependency set; no scikit-learn
dependency is introduced.

- Model inputs are integer-grid coordinates normalized by each dimension's
  number of levels. The candidate API rejects non-integer coordinates.
- The model returns a predicted objective mean and uncertainty for each legal
  candidate.
- EI balances exploitation (low predicted objective) and exploration
  (uncertainty), with `xi` controlling the exploration margin.
- `rank_candidates()` ranks a supplied pool of discrete candidates and can
  exclude already-observed coordinates.
- Only successful evaluations should be used to fit the model. Failed runs
  remain in the SQLite history and must be excluded from the training targets.

The current GP uses fixed length scales and signal variance. This is a
transparent first baseline, not a claim that its uncertainty is perfectly
calibrated. Hyperparameter tuning, candidate-pool generation, batch diversity,
and end-to-end orchestration are subsequent steps. Exact GP fitting scales
cubically with the number of training observations, so we will benchmark the
practical training-set size before selecting a refit policy.

Example:

```python
from optimizer import GaussianProcessSurrogate

successful = store.list_results(status="ok")
model = GaussianProcessSurrogate(length_scale=0.25)
model.fit(
    [result.indices for result in successful],
    [result.objective for result in successful],
    levels=space.levels(),
)
ranked = model.rank_candidates(
    candidate_pool,
    best_objective=min(result.objective for result in successful),
    observed={result.indices for result in successful},
)
next_candidate = ranked[0][0]
```

Unit tests for the surrogate use synthetic functions and do not launch FDTD.

### Discrete candidate-pool selection

`optimizer/candidate_selection.py` generates a reproducible Sobol-based pool
of unexplored candidates, with every coordinate represented as an integer
grid index. It excludes previously observed points before the pool is scored.

```python
from optimizer import discrete_candidate_pool, select_next_candidates

observed = {result.indices for result in store.list_results()}
pool = discrete_candidate_pool(
    space, n_candidates=4096, excluded=observed, seed=42
)
ranked = model.rank_candidates(
    pool, best_objective=store.best().objective, observed=observed
)
next_candidate = ranked[0][0]

# Or combine pool generation and ranking:
next_batch = select_next_candidates(
    space, model, best_objective=store.best().objective,
    observed=observed, n_candidates=1, pool_size=4096, seed=42,
)
```

The pool size controls the number of candidates scored by the GP; it is not an
FDTD evaluation budget. The selector only proposes candidates. The later
orchestration layer will calculate stable fingerprints and perform a final
database-level duplicate check immediately before simulation.
