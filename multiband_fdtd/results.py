"""Post-process broadband response and recorded far-field data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_REQUIRED_COLUMNS = (
    "frequency_GHz", "s11_smoothed_dB",
    "z_in_smoothed_real", "z_in_smoothed_imag",
)


def _load_response(csv_path: str | Path) -> dict[str, np.ndarray]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Broadband response CSV not found: {csv_path}")
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float,
                             encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read broadband response CSV: {exc}") from exc
    if data.dtype.names is None:
        raise ValueError("Broadband response CSV must contain a header row.")
    missing = [name for name in _REQUIRED_COLUMNS if name not in data.dtype.names]
    if missing:
        raise ValueError("Broadband response CSV is missing columns: " + ", ".join(missing))
    if data.size < 2:
        raise ValueError("Broadband response CSV must contain at least two data rows.")
    values = {
        "frequency_ghz": np.atleast_1d(data["frequency_GHz"]).astype(float),
        "s11_db": np.atleast_1d(data["s11_smoothed_dB"]).astype(float),
        "z_real": np.atleast_1d(data["z_in_smoothed_real"]).astype(float),
        "z_imag": np.atleast_1d(data["z_in_smoothed_imag"]).astype(float),
    }
    if not all(np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("Broadband response contains non-finite values.")
    if np.any(np.diff(values["frequency_ghz"]) <= 0):
        raise ValueError("Frequency samples must be strictly increasing.")
    return values


def _threshold_ranges(frequency: np.ndarray, s11_db: np.ndarray,
                      threshold_db: float) -> list[dict[str, float]]:
    mask = s11_db <= threshold_db
    if not np.any(mask):
        return []
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return [{
        "start_ghz": float(frequency[start]),
        "stop_ghz": float(frequency[stop]),
        "bandwidth_ghz": float(frequency[stop] - frequency[start]),
    } for start, stop in zip(starts, stops)]


def analyze_broadband_csv(csv_path: str | Path) -> dict:
    """Calculate matching metrics from the reconstructed complex response."""
    data = _load_response(csv_path)
    frequency, s11_db = data["frequency_ghz"], data["s11_db"]
    minimum_index = int(np.argmin(s11_db))
    s11_linear = 10.0 ** (s11_db / 20.0)
    vswr = (1.0 + s11_linear) / np.maximum(1.0 - s11_linear, 1e-12)
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
            "real": float(data["z_real"][minimum_index]),
            "imag": float(data["z_imag"][minimum_index]),
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
    """Create the full-width matching, VSWR and impedance figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_response(csv_path)
    frequency = data["frequency_ghz"]
    s11_linear = 10.0 ** (data["s11_db"] / 20.0)
    vswr = (1.0 + s11_linear) / np.maximum(1.0 - s11_linear, 1e-12)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].plot(frequency, data["s11_db"], linewidth=1.5)
    axes[0, 0].axhline(-10.0, linestyle="--", linewidth=1.0, label="-10 dB")
    axes[0, 0].axhline(-20.0, linestyle=":", linewidth=1.0, label="-20 dB")
    axes[0, 0].set_ylabel("S11 (dB)")
    axes[0, 0].set_title("Return loss")
    axes[0, 0].legend()
    axes[0, 1].plot(frequency, vswr, linewidth=1.5)
    axes[0, 1].axhline(2.0, linestyle="--", linewidth=1.0, label="VSWR 2")
    axes[0, 1].set_ylabel("VSWR")
    axes[0, 1].set_title("Voltage standing-wave ratio")
    axes[0, 1].legend()
    axes[1, 0].plot(frequency, data["z_real"], label="Real(Z)")
    axes[1, 0].set_ylabel("Resistance (Ω)")
    axes[1, 0].set_xlabel("Frequency (GHz)")
    axes[1, 0].set_title("Input resistance")
    axes[1, 0].legend()
    axes[1, 1].plot(frequency, data["z_imag"], label="Imag(Z)")
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1, 1].set_ylabel("Reactance (Ω)")
    axes[1, 1].set_xlabel("Frequency (GHz)")
    axes[1, 1].set_title("Input reactance")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _build_nf2ff_for_band(band_dir: Path):
    """Recreate the NF2FF object using the exact saved band configuration."""
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from design import build_design, load_config

    config_path = band_dir / "design.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Band design snapshot is missing: {config_path}")
    cfg = load_config(str(config_path))
    simulation = cfg.get("simulation") or {}
    fstart = float(simulation["fstart"]) * 1e9
    fstop = float(simulation["fstop"]) * 1e9
    end_criteria = float(simulation.get("end_criteria", 1e-4))
    max_timesteps = int(simulation.get("max_timesteps", 1000000))
    csx = ContinuousStructure()
    fdtd = openEMS(EndCriteria=end_criteria, NrTS=max_timesteps)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(0.5 * (fstart + fstop), fstop - fstart)
    fdtd.SetBoundaryCond(["PML_8"] * 6)
    _port, nf2ff = build_design(
        csx, fdtd, config_file=str(config_path), enable_nf2ff=True
    )
    if nf2ff is None:
        raise RuntimeError(f"Could not recreate NF2FF box for {band_dir}")
    return nf2ff


def calculate_far_field(csv_path: str | Path, simulation_dir: str | Path,
                         output_dir: str | Path) -> dict:
    """Evaluate far-field patterns and antenna metrics from recorded band fields."""
    simulation_dir = Path(simulation_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = simulation_dir / "multiband_run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Multi-band run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("far_field_recorded", False):
        raise RuntimeError(
            "This simulation was not recorded for far-field analysis. "
            "Run a new accurate simulation with far-field recording enabled."
        )
    band_defs = manifest.get("bands") or []
    band_runs = manifest.get("band_results") or []
    if not band_defs or len(band_defs) != len(band_runs):
        raise ValueError("The run manifest does not contain matching band definitions and outputs.")

    bands = []
    for definition, run in zip(band_defs, band_runs):
        band_dir = Path(run["output_dir"])
        if not band_dir.is_absolute():
            band_dir = simulation_dir / band_dir
        metadata_path = band_dir / "band_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Band metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("far_field_recorded", False):
            raise RuntimeError(f"Far-field data was not recorded for {band_dir}")
        bands.append((float(definition["fstart_ghz"]),
                      float(definition["fstop_ghz"]), band_dir))

    def band_for(frequency_ghz: float):
        matches = [item for item in bands
                   if item[0] - 1e-9 <= frequency_ghz <= item[1] + 1e-9]
        if not matches:
            raise ValueError(f"No recorded simulation band covers {frequency_ghz:g} GHz.")
        return matches[0]

    cache: dict[Path, object] = {}
    def nf2ff_for(band_dir: Path):
        if band_dir not in cache:
            print(f"Preparing far-field extraction for {band_dir.name}…", flush=True)
            cache[band_dir] = _build_nf2ff_for_band(band_dir)
        return cache[band_dir]

    response = _load_response(csv_path)
    freq_ghz = response["frequency_ghz"]
    s11_db = response["s11_db"]
    s11_linear = 10.0 ** (s11_db / 20.0)
    minimum_index = int(np.argmin(s11_db))
    resonance_ghz = float(freq_ghz[minimum_index])
    theta = np.linspace(-180.0, 180.0, 361)
    phi = [0.0, 90.0]
    eval_freqs_ghz = np.linspace(float(freq_ghz[0]), float(freq_ghz[-1]), 21)
    directivity, realized_gain, total_efficiency = [], [], []
    pattern_result = None

    for index, frequency_ghz in enumerate(eval_freqs_ghz, start=1):
        band_dir = band_for(float(frequency_ghz))[2]
        nf2ff = nf2ff_for(band_dir)
        result = nf2ff.CalcNF2FF(
            str(band_dir), float(frequency_ghz) * 1e9, theta, phi
        )
        dmax = float(result.Dmax[0])
        dmax_dbi = 10.0 * np.log10(max(dmax, 1e-12))
        s11_mag = float(np.interp(frequency_ghz, freq_ghz, s11_linear))
        mismatch_efficiency = max(0.0, 1.0 - s11_mag ** 2)
        # PEC-only model: radiation efficiency is an explicit idealized assumption.
        realized = dmax_dbi + 10.0 * np.log10(max(mismatch_efficiency, 1e-12))
        directivity.append(dmax_dbi)
        realized_gain.append(realized)
        total_efficiency.append(100.0 * mismatch_efficiency)
        print(f"Far-field {index}/21: {frequency_ghz:.4f} GHz, "
              f"directivity {dmax_dbi:.2f} dBi", flush=True)
    band_dir = band_for(resonance_ghz)[2]
    pattern_result = nf2ff_for(band_dir).CalcNF2FF(
        str(band_dir), resonance_ghz * 1e9, theta, phi
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e_norm = np.asarray(pattern_result.E_norm[0])
    e_db = 20.0 * np.log10(e_norm / max(float(np.max(e_norm)), 1e-15) + 1e-12)
    figure = plt.figure(figsize=(14, 10))
    polar = figure.add_subplot(221, projection="polar")
    theta_rad = np.deg2rad(theta)
    polar.plot(theta_rad, e_db[:, 0], linewidth=1.8, label="E-plane (phi=0°)")
    polar.plot(theta_rad, e_db[:, 1], linestyle="--", linewidth=1.8, label="H-plane (phi=90°)")
    polar.set_rmin(-30)
    polar.set_rmax(0)
    polar.set_title(f"Radiation pattern at {resonance_ghz:.4g} GHz (normalized dB)")
    polar.legend(loc="lower right", fontsize="small")

    eval_x = eval_freqs_ghz
    ax = figure.add_subplot(222)
    ax.plot(eval_x, directivity, "o-", linewidth=1.6)
    ax.set_title("Peak directivity")
    ax.set_ylabel("Directivity (dBi)")
    ax.set_xlabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)

    ax = figure.add_subplot(223)
    ax.plot(eval_x, realized_gain, "s-", linewidth=1.6)
    ax.set_title("Realized gain (ideal PEC model)")
    ax.set_ylabel("Realized gain (dBi)")
    ax.set_xlabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)

    ax = figure.add_subplot(224)
    ax.plot(eval_x, np.full(len(eval_x), 100.0), "--",
            label="Assumed radiation efficiency (PEC)")
    ax.plot(eval_x, total_efficiency, "^-", label="Total efficiency incl. mismatch")
    ax.set_title("Efficiency")
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    figure.tight_layout()
    radiation_plot = output_dir / "radiation_plot.png"
    figure.savefig(radiation_plot, dpi=190, bbox_inches="tight")
    plt.close(figure)

    metrics = {
        "radiation_efficiency_percent": 100.0,
        "radiation_efficiency_status": "assumed_ideal_lossless_PEC",
        "frequency_ghz": [float(x) for x in eval_freqs_ghz],
        "directivity_dbi": [float(x) for x in directivity],
        "realized_gain_dbi": [float(x) for x in realized_gain],
        "total_efficiency_percent": [float(x) for x in total_efficiency],
        "radiation_pattern_frequency_ghz": resonance_ghz,
        "radiation_plot": str(radiation_plot),
    }

    # Separate plot assets let the GUI display each result at full size.
    def save_curve(filename, title, ylabel, values, *, label=None):
        fig, axis = plt.subplots(figsize=(12, 7))
        axis.plot(eval_freqs_ghz, values, linewidth=1.8, label=label)
        axis.set_title(title)
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if label:
            axis.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    pattern_figure = plt.figure(figsize=(10, 8))
    pattern_axis = pattern_figure.add_subplot(111, projection="polar")
    pattern_axis.plot(theta_rad, e_db[:, 0], linewidth=1.8, label="E-plane (phi=0°)")
    pattern_axis.plot(theta_rad, e_db[:, 1], linestyle="--", linewidth=1.8,
                      label="H-plane (phi=90°)")
    pattern_axis.set_rmin(-30)
    pattern_axis.set_rmax(0)
    pattern_axis.set_title(f"Radiation pattern at {resonance_ghz:.4g} GHz")
    pattern_axis.legend(loc="lower right")
    pattern_figure.tight_layout()
    pattern_path = output_dir / "radiation_pattern.png"
    pattern_figure.savefig(pattern_path, dpi=190, bbox_inches="tight")
    plt.close(pattern_figure)

    metrics["radiation_pattern_plot"] = str(pattern_path)
    metrics["directivity_plot"] = save_curve(
        "directivity_plot.png", "Peak directivity", "Directivity (dBi)", directivity
    )
    metrics["realized_gain_plot"] = save_curve(
        "realized_gain_plot.png", "Realized gain (ideal PEC model)",
        "Realized gain (dBi)", realized_gain
    )
    metrics["efficiency_plot"] = save_curve(
        "efficiency_plot.png", "Total efficiency", "Efficiency (%)",
        total_efficiency, label="Total efficiency incl. mismatch"
    )

    metrics_path = output_dir / "far_field_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved radiation and far-field plot: {radiation_plot}", flush=True)
    print(
        "Radiation efficiency is assumed to be 100% for ideal PEC; "
        "conductor and dielectric losses are not modeled.",
        flush=True,
    )
    return metrics


def process_results(csv_path: str | Path, output_dir: str | Path,
                    simulation_dir: str | Path | None = None) -> dict:
    """Generate matching plots and, when recordings exist, far-field results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = analyze_broadband_csv(csv_path)
    create_results_plot(csv_path, output_dir / "multiband_results.png")
    metrics_path = output_dir / "multiband_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    far_field = None
    if simulation_dir is not None:
        far_field = calculate_far_field(csv_path, simulation_dir, output_dir)
        metrics["far_field"] = far_field
    create_result_plot_set(csv_path, output_dir, far_field)
    return metrics




def create_result_plot_set(
    csv_path: str | Path,
    output_dir: str | Path,
    far_field: dict | None = None,
) -> list[dict[str, str]]:
    """Create full-size individual plots and a compact 2x2 overview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    data = _load_response(csv_path)
    frequency = data["frequency_ghz"]
    s11_db = data["s11_db"]
    s11_mag = 10.0 ** (s11_db / 20.0)
    vswr = (1.0 + s11_mag) / np.maximum(1.0 - s11_mag, 1e-12)
    z_real, z_imag = data["z_real"], data["z_imag"]
    analysis = analyze_broadband_csv(csv_path)

    def save_plot(filename, draw, figsize=(12, 7)):
        figure, axis = plt.subplots(figsize=figsize)
        draw(axis)
        axis.grid(True, alpha=0.28)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        return str(path)

    def draw_s11(axis):
        axis.plot(frequency, s11_db, linewidth=1.8)
        axis.axhline(-10, linestyle="--", linewidth=1.0, label="-10 dB")
        axis.axhline(-20, linestyle=":", linewidth=1.0, label="-20 dB")
        axis.set_title("S11 magnitude")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("|S11| (dB)")
        axis.legend()

    def draw_vswr(axis):
        axis.plot(frequency, vswr, linewidth=1.8)
        axis.axhline(2.0, linestyle="--", label="VSWR 2")
        axis.set_title("Voltage standing-wave ratio")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("VSWR")
        axis.legend()

    def draw_bandwidth(axis):
        rows = []
        for threshold, bands in (
            ("-10 dB", analysis["bands_below_minus_10_db"]),
            ("-20 dB", analysis["bands_below_minus_20_db"]),
        ):
            for index, band in enumerate(bands, start=1):
                rows.append((threshold, index, band))
        if not rows:
            axis.text(0.5, 0.5, "No bands below -10 dB", ha="center", va="center",
                      transform=axis.transAxes)
        else:
            labels = [f"{t} band {i}" for t, i, _ in rows]
            widths = [item["bandwidth_ghz"] for _, _, item in rows]
            axis.barh(labels, widths)
        axis.set_title("Matched bandwidth by threshold")
        axis.set_xlabel("Bandwidth (GHz)")

    def draw_impedance(axis):
        axis.plot(frequency, z_real, label="Resistance")
        axis.plot(frequency, z_imag, label="Reactance")
        axis.axhline(0, linestyle=":", linewidth=0.8)
        axis.set_title("Input impedance")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("Impedance (Ω)")
        axis.legend()

    catalog = [
        {"label": "|S11| magnitude (dB)", "filename": "s11_magnitude_db.png",
         "path": save_plot("s11_magnitude_db.png", draw_s11)},
        {"label": "VSWR", "filename": "vswr_plot.png",
         "path": save_plot("vswr_plot.png", draw_vswr)},
        {"label": "Matched bandwidth", "filename": "bandwidth_plot.png",
         "path": save_plot("bandwidth_plot.png", draw_bandwidth)},
        {"label": "Input impedance", "filename": "input_impedance.png",
         "path": save_plot("input_impedance.png", draw_impedance)},
        {"label": "Input resistance", "filename": "input_resistance.png",
         "path": save_plot("input_resistance.png",
                           lambda ax: (ax.plot(frequency, z_real), ax.set_title("Input resistance"),
                                       ax.set_xlabel("Frequency (GHz)"), ax.set_ylabel("Resistance (Ω)")) )},
        {"label": "Input reactance", "filename": "input_reactance.png",
         "path": save_plot("input_reactance.png",
                           lambda ax: (ax.plot(frequency, z_imag), ax.set_title("Input reactance"),
                                       ax.set_xlabel("Frequency (GHz)"), ax.set_ylabel("Reactance (Ω)")) )},
    ]

    far_field = far_field or {}
    for label, key, filename in (
        ("Radiation pattern", "radiation_pattern_plot", "radiation_pattern.png"),
        ("Directivity", "directivity_plot", "directivity_plot.png"),
        ("Realized gain", "realized_gain_plot", "realized_gain_plot.png"),
        ("Total efficiency", "efficiency_plot", "efficiency_plot.png"),
    ):
        path = far_field.get(key)
        if path and Path(path).is_file():
            catalog.append({"label": label, "filename": filename, "path": str(path)})

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    draw_s11(axes[0, 0])
    if far_field.get("frequency_ghz") and far_field.get("total_efficiency_percent"):
        axes[0, 1].plot(far_field["frequency_ghz"],
                        far_field["total_efficiency_percent"], linewidth=1.5)
        axes[0, 1].set_title("Total efficiency")
        axes[0, 1].set_xlabel("Frequency (GHz)")
        axes[0, 1].set_ylabel("Efficiency (%)")
    else:
        draw_vswr(axes[0, 1])
    draw_bandwidth(axes[1, 0])
    draw_impedance(axes[1, 1])
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.suptitle("Integrated simulation overview", fontsize=14)
    figure.tight_layout()
    overview_path = output_dir / "integrated_overview.png"
    figure.savefig(overview_path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    catalog.append({
        "label": "Integrated overview (2×2)",
        "filename": "integrated_overview.png",
        "path": str(overview_path),
    })
    (output_dir / "plot_catalog.json").write_text(
        json.dumps(catalog, indent=2), encoding="utf-8"
    )
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze broadband antenna simulation results.")
    parser.add_argument("--csv", required=True, help="Path to broadband_response.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for report outputs")
    parser.add_argument("--simulation-dir", help="Multi-band simulation directory containing band recordings")
    args = parser.parse_args()
    metrics = process_results(args.csv, args.output_dir, args.simulation_dir)
    print(f"Analyzed {metrics['sample_count']} frequency samples.", flush=True)
    print(f"Minimum S11: {metrics['minimum_s11_db']:.2f} dB at "
          f"{metrics['minimum_s11_frequency_ghz']:.6g} GHz", flush=True)
    print(f"-10 dB total bandwidth: "
          f"{metrics['total_bandwidth_below_minus_10_db_ghz']:.4g} GHz", flush=True)
    print(f"-20 dB total bandwidth: "
          f"{metrics['total_bandwidth_below_minus_20_db_ghz']:.4g} GHz", flush=True)
    if metrics.get("far_field"):
        print("Full far-field analysis completed.", flush=True)
    print(f"Saved matching report: {Path(args.output_dir) / 'multiband_results.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
