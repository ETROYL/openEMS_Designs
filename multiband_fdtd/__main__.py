"""Command-line entry point for standalone multi-band FDTD simulations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .engine import run_multiband


def _parse_bands(value: str) -> list[tuple[float, float, int]]:
    bands = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                "Bands must be comma-separated fstart:fstop[:n_pts] values."
            )
        try:
            low, high = float(parts[0]), float(parts[1])
            points = int(parts[2]) if len(parts) == 3 else 101
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        bands.append((low, high, points))
    return bands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--bands", required=True, type=_parse_bands,
        help="Comma-separated GHz intervals, e.g. 0.1:0.65:101,0.4:1.65:101,1.35:3:101",
    )
    parser.add_argument("--profile", choices=("optimization", "accurate"),
                        default="accurate")
    parser.add_argument("--frequency-range", nargs=2, type=float,
                        metavar=("FMIN_GHZ", "FMAX_GHZ"))
    parser.add_argument("--points", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--end-criteria-db", type=float)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--smooth-sigma-points", type=float, default=0.0)
    args = parser.parse_args()

    frequency_range = tuple(args.frequency_range) if args.frequency_range else None
    result = run_multiband(
        args.config, args.output, args.bands,
        profile=args.profile,
        frequency_range_ghz=frequency_range,
        n_points=args.points,
        threads=args.threads,
        end_criteria_db=args.end_criteria_db,
        max_timesteps=args.max_timesteps,
        smoothing_sigma_points=args.smooth_sigma_points,
    )

    frequency = result["frequency_ghz"]
    phase = np.rad2deg(np.angle(result["s11"]))
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_mag.plot(frequency, result["s11_db"], linewidth=1.7)
    ax_mag.set_ylabel("|S11| (dB)")
    ax_mag.grid(True, alpha=0.3)
    ax_mag.set_title("Reconstructed multi-band FDTD response")
    ax_phase.plot(frequency, phase, linewidth=1.5)
    ax_phase.set_xlabel("Frequency (GHz)")
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = Path(result["output_dir"]) / "broadband_response.png"
    fig.savefig(plot_path, dpi=180)
    print(f"Total FDTD runtime: {result['total_runtime_seconds']:.2f} s")
    print(f"Broadband response: {result['output_dir']}")
    print(f"Plot: {plot_path}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
