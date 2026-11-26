# Multi-band FDTD simulation

A GUI- and optimizer-independent simulation layer. It runs openEMS
simulations over configured frequency intervals, extracts complex port
responses, and combines them into a broadband response. Both optimization
and accurate simulation profiles use the same execution path.

## Standalone command line

    python3 -m multiband_fdtd \
      --config designs/dipole.yaml \
      --output results/dipole_run \
      --bands 0.1:0.65:101,0.4:1.65:101,1.35:3.0:101 \
      --profile accurate --frequency-range 0.1 3.0 --points 1201 --threads 4

Each interval receives its own YAML snapshot, mesh and output directory.
Complex S11 and impedance are saved per interval; the combined response is
saved as NPZ and CSV, with a broadband magnitude/phase plot. The output
directory must not already exist.

## Python API

    from multiband_fdtd import FrequencyBand, run_multiband

    result = run_multiband(
        config_path="design.yaml",
        output_dir="results/fdtd_run",
        bands=[
            FrequencyBand(0.10, 0.65, 101),
            FrequencyBand(0.40, 1.65, 101),
            FrequencyBand(1.35, 3.00, 101),
        ],
        profile="accurate",
        frequency_range_ghz=(0.10, 3.00),
        n_points=1201,
        threads=4,
        smoothing_sigma_points=0.0,
    )

Profiles provide defaults for energy decay, timestep limit and reconstructed
sample count. Explicit values override profile defaults. The dB energy-decay
target is converted to the linear ratio expected by openEMS. The returned
dictionary provides objective-compatible keys: freq, s11, s11_db and z_in.

Optional Gaussian low-pass smoothing operates on real and imaginary response
components. It is disabled by default and should be chosen conservatively to
avoid suppressing genuine narrow resonances.


## Interval preparation

A bounded set of candidate intervals can be generated before measurement:

    from multiband_fdtd import generate_candidate_bands

    candidates = generate_candidate_bands(
        frequency_range_ghz=(0.1, 3.0),
        interval_widths_ghz=(0.5, 1.0, 1.5),
        boundary_step_ghz=0.25,
        n_pts=41,
        max_candidates=40,
    )

The complete requested range is always retained as a fallback candidate.

Measurements can be collected and a complete interval sequence selected in
one call:

    from multiband_fdtd import characterize_and_select_intervals

    plan = characterize_and_select_intervals(
        config_path="design.yaml",
        output_dir="results/interval_characterization",
        frequency_range_ghz=(0.1, 3.0),
        interval_widths_ghz=(0.5, 1.0, 1.5),
        boundary_step_ghz=0.25,
        n_pts=41,
        max_candidates=40,
        minimum_overlap_ghz=0.1,
        end_criteria_db=-10.0,
        max_timesteps=50000,
        threads=1,
    )

    selected_bands = plan["selected_bands"]

The returned plan includes the generated candidates, measured records,
selected intervals and total measured runtime. The selection settings and
selected sequence are also stored in characterization.json. The selected
intervals can be reused unchanged for subsequent evaluations in a run.

## Automatic interval preparation and simulation

The complete preparation and production sequence is also available through
one API call. Preparation and production settings are independent, and their
outputs are kept in separate subdirectories:

    from multiband_fdtd import run_multiband_auto

    result = run_multiband_auto(
        config_path="design.yaml",
        output_dir="results/automatic_run",
        frequency_range_ghz=(0.1, 3.0),
        profile="accurate",
        interval_widths_ghz=(0.5, 1.0, 1.5),
        boundary_step_ghz=0.25,
        pilot_n_pts=41,
        max_candidates=40,
        minimum_overlap_ghz=0.1,
        pilot_end_criteria_db=-10.0,
        pilot_max_timesteps=50000,
        pilot_threads=1,
        threads=4,
        smoothing_sigma_points=0.0,
    )

The output directory contains `characterization/` and `simulation/`, plus
an `automatic_run.json` summary. The selected interval sequence is measured
using the YAML design's configured default variable values and then reused
unchanged for the production call. The production profile controls its own
convergence and sampling defaults; explicit production overrides can be
provided through `end_criteria_db`, `max_timesteps` and `n_points`.
