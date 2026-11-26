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
