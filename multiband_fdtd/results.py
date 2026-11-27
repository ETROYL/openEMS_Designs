"""Post-process reconstructed broadband response data for the desktop GUI.

This module consumes the standard CSV written by the multi-band simulation
runner. It does not rerun FDTD or inspect individual band recordings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_REQUIRED_COLUMNS = 10


def _threshold_ranges(frequency: np.ndarray, s11_db: np.ndarray,
                      threshold_db: float) -> list[dict[str, float]]:
    """Return contiguous frequency ranges meeting an S11 threshold."""
    mask = s11_db <= threshold_db
    if not np.any(mask):
        return []
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    ranges = []
    for start, stop in zip(starts, stops):
        ranges.append({
            "start_ghz": float(frequency[start]),
            "stop_ghz": float(frequency[stop]),
            "bandwidth_ghz": float(frequency[stop] - frequency[start]),
        })
    return ranges


def analyze_broadband_csv(csv_path: str | Path) -> dict:
    """Load reconstructed response and calculate antenna performance metrics."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Broadband response CSV not found: {csv_path}")
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float,
                             encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read broadband response CSV: {exc}") from exc
    if data.dtype.names is None or data.size < 2:
        raise ValueError("Broadband response CSV must contain at least two data rows.")
    names = data.dtype.names
    required = (
        "frequency_GHz", "s11_smoothed_dB",
        "z_in_smoothed_real", "z_in_smoothed_imag",
    )
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError("Broadband response CSV is missing columns: " + ", ".join(missing))

    frequency = np.atleast_1d(data["frequency_GHz"]).astype(float)
    s11_db = np.atleast_1d(data["s11_smoothed_dB"]).astype(float)
    z_real = np.atleast_1d(data["z_in_smoothed_real"]).astype(float)
    z_imag = np.atleast_1d(data["z_in_smoothed_imag"]).astype(float)
    if not all(np.all(np.isfinite(v)) for v in (frequency, s11_db, z_real, z_imag)):
        raise ValueError("Broadband response contains non-finite values.")
    if np.any(np.diff(frequency) <= 0):
        raise ValueError("Frequency samples must be strictly increasing.")

    minimum_index = int(np.argmin(s11_db))
    s11_linear = 10.0 ** (s11_db / 20.0)
    vswr = (1.0 + s11_linear) / np.maximum(1.0 - s11_linear, 1e-12)

    # Report distinct local minima, ranked by return loss.
    minima = np.flatnonzero(
        (s11_db[1:-1] <= s11_db[:-2]) &
        (s11_db[1:-1] < s11_db[2:])
    ) + 1
    minima = sorted(minima.tolist(), key=lambda i: float(s11_db[i]))[:10]

    result = {
        "sample_count": int(frequency.size),
        "frequency_start_ghz": float(frequency[0]),
        "frequency_stop_ghz": float(frequency[-1]),
        "minimum_s11_db": float(s11_db[minimum_index]),
        "minimum_s11_frequency_ghz": float(frequency[minimum_index]),
        "vswr_at_minimum_s11": float(vswr[minimum_index]),
        "impedance_at_minimum_s11_ohm": {
            "real": float(z_real[minimum_index]),
            "imag": float(z_imag[minimum_index]),
        },
        "bands_below_minus_10_db": _threshold_ranges(frequency, s11_db, -10.0),
        "bands_below_minus_20_db": _threshold_ranges(frequency, s11_db, -20.0),
        "resonances": [
            {"frequency_ghz": float(frequency[i]), "s11_db": float(s11_db[i])}
            for i in minima
        ],
    }
    result["total_bandwidth_below_minus_10_db_ghz"] = float(sum(
        band["bandwidth_ghz"] for band in result["bands_below_minus_10_db"]
    ))
    result["total_bandwidth_below_minus_20_db_ghz"] = float(sum(
        band["bandwidth_ghz"] for band in result["bands_below_minus_20_db"]
    ))
    return result


def create_results_plot(csv_path: str | Path, output_path: str | Path) -> None:
    """Create a readable full-width S11 and impedance report plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float,
                         encoding="utf-8")
    frequency = np.atleast_1d(data["frequency_GHz"])
    s11_db = np.atleast_1d(data["s11_smoothed_dB"])
    z_real = np.atleast_1d(data["z_in_smoothed_real"])
    z_imag = np.atleast_1d(data["z_in_smoothed_imag"])

    figure, (s11_axis, impedance_axis) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True
    )
    s11_axis.plot(frequency, s11_db, linewidth=1.5, label="S11")
    s11_axis.axhline(-10.0, linestyle="--", linewidth=1.0, label="-10 dB")
    s11_axis.axhline(-20.0, linestyle=":", linewidth=1.0, label="-20 dB")
    s11_axis.set_ylabel("S11 (dB)")
    s11_axis.set_title("Broadband antenna response")
    s11_axis.grid(True, alpha=0.3)
    s11_axis.legend(loc="best")
    impedance_axis.plot(frequency, z_real, label="Real(Z)")
    impedance_axis.plot(frequency, z_imag, label="Imag(Z)")
    impedance_axis.set_xlabel("Frequency (GHz)")
    impedance_axis.set_ylabel("Input impedance (Ω)")
    impedance_axis.grid(True, alpha=0.3)
    impedance_axis.legend(loc="best")
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def process_results(csv_path: str | Path, output_dir: str | Path) -> dict:
    """Generate the metrics JSON and detailed plot from reconstructed CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = analyze_broadband_csv(csv_path)
    create_results_plot(csv_path, output_dir / "multiband_results.png")
    metrics_path = output_dir / "multiband_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze reconstructed broadband antenna results.")
    parser.add_argument("--csv", required=True, help="Path to broadband_response.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for report outputs")
    args = parser.parse_args()
    metrics = process_results(args.csv, args.output_dir)
    print(f"Analyzed {metrics['sample_count']} frequency samples.", flush=True)
    print(f"Minimum S11: {metrics['minimum_s11_db']:.2f} dB at "
          f"{metrics['minimum_s11_frequency_ghz']:.6g} GHz", flush=True)
    print(f"-10 dB total bandwidth: "
          f"{metrics['total_bandwidth_below_minus_10_db_ghz']:.4g} GHz", flush=True)
    print(f"-20 dB total bandwidth: "
          f"{metrics['total_bandwidth_below_minus_20_db_ghz']:.4g} GHz", flush=True)
    print(f"Saved report plot: {Path(args.output_dir) / 'multiband_results.png'}", flush=True)
    print(f"Saved metrics: {Path(args.output_dir) / 'multiband_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
