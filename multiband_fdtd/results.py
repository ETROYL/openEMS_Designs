"""Post-process broadband response and recorded far-field data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wrapped_phase_degrees(s11_complex: np.ndarray) -> np.ndarray:
    """Return principal S11 phase values in the inclusive [-180°, +180°] range."""
    return np.rad2deg(np.angle(np.asarray(s11_complex, dtype=complex)))


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
    if {"s11_smoothed_real", "s11_smoothed_imag"}.issubset(data.dtype.names):
        values["s11_complex"] = (
            np.atleast_1d(data["s11_smoothed_real"]).astype(float)
            + 1j * np.atleast_1d(data["s11_smoothed_imag"]).astype(float)
        )
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
    port, nf2ff = build_design(
        csx, fdtd, config_file=str(config_path), enable_nf2ff=True
    )
    if isinstance(port, (list, tuple)):
        if len(port) != 1:
            raise NotImplementedError(
                "Far-field power metrics currently require exactly one excited port."
            )
        port = port[0]
    if port is None:
        raise RuntimeError(f"Could not recreate the input port for {band_dir}")
    if nf2ff is None:
        raise RuntimeError(f"Could not recreate NF2FF box for {band_dir}")
    return port, nf2ff


def _calculate_efficiencies(prad: float, p_acc: float, p_inc: float) -> dict[str, float]:
    """Return radiation, mismatch and total efficiencies from simulated powers."""
    values = np.asarray([prad, p_acc, p_inc], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Far-field power values must be finite.")
    prad, p_acc, p_inc = (float(value) for value in values)
    if prad < 0 or p_acc <= 0 or p_inc <= 0:
        raise ValueError(
            "Invalid far-field power balance: Prad must be non-negative and "
            "P_acc/P_inc must be positive."
        )
    return {
        "radiation": prad / p_acc,
        "mismatch": p_acc / p_inc,
        "total": prad / p_inc,
    }


def _power_to_dbi(linear_value: float) -> float:
    """Convert a non-negative linear power/directivity ratio to dBi."""
    if not np.isfinite(linear_value) or linear_value < 0:
        return float("nan")
    return 10.0 * np.log10(max(linear_value, 1e-30))


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
    minimum_index = int(np.argmin(s11_db))
    resonance_ghz = float(freq_ghz[minimum_index])
    theta = np.linspace(-180.0, 180.0, 361)
    phi = [0.0, 90.0]
    eval_freqs_ghz = np.linspace(float(freq_ghz[0]), float(freq_ghz[-1]), 21)
    directivity, gain, realized_gain = [], [], []
    radiation_efficiency, mismatch_efficiency, total_efficiency = [], [], []
    pattern_result = None

    for index, frequency_ghz in enumerate(eval_freqs_ghz, start=1):
        band_dir = band_for(float(frequency_ghz))[2]
        port, nf2ff = nf2ff_for(band_dir)
        frequency_hz = float(frequency_ghz) * 1e9
        result = nf2ff.CalcNF2FF(
            str(band_dir), frequency_hz, theta, phi
        )
        # CalcPort supplies incident, reflected and accepted input powers at
        # the same frequency used by NF2FF. Do not infer accepted power from
        # smoothed S11 data.
        port.CalcPort(str(band_dir), np.asarray([frequency_hz], dtype=float))
        prad = float(np.asarray(result.Prad, dtype=float).reshape(-1)[0])
        p_acc = float(np.asarray(port.P_acc, dtype=float).reshape(-1)[0])
        p_inc = float(np.asarray(port.P_inc, dtype=float).reshape(-1)[0])
        efficiencies = _calculate_efficiencies(prad, p_acc, p_inc)

        dmax = float(np.asarray(result.Dmax, dtype=float).reshape(-1)[0])
        dmax_dbi = _power_to_dbi(dmax)
        gain_dbi = _power_to_dbi(dmax * efficiencies["radiation"])
        realized_dbi = _power_to_dbi(dmax * efficiencies["total"])
        directivity.append(dmax_dbi)
        gain.append(gain_dbi)
        realized_gain.append(realized_dbi)
        radiation_efficiency.append(100.0 * efficiencies["radiation"])
        mismatch_efficiency.append(100.0 * efficiencies["mismatch"])
        total_efficiency.append(100.0 * efficiencies["total"])
        print(
            f"Far-field {index}/21: {frequency_ghz:.4f} GHz, "
            f"directivity {dmax_dbi:.2f} dBi, gain {gain_dbi:.2f} dBi, "
            f"realized gain {realized_dbi:.2f} dBi",
            flush=True,
        )
    band_dir = band_for(resonance_ghz)[2]
    _pattern_port, pattern_nf2ff = nf2ff_for(band_dir)
    pattern_result = pattern_nf2ff.CalcNF2FF(
        str(band_dir), resonance_ghz * 1e9, theta, phi
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _dark_plot_theme()
    e_norm = np.asarray(pattern_result.E_norm[0])
    e_db = 20.0 * np.log10(e_norm / max(float(np.max(e_norm)), 1e-15) + 1e-12)
    theta_rad = np.deg2rad(theta)
    pattern_frequency_ghz = float(resonance_ghz)
    figure = plt.figure(figsize=(10, 8))
    polar = figure.add_subplot(111, projection="polar")
    polar.plot(theta_rad, e_db[:, 0], color="#3995ff", linewidth=1.8,
               label="E-plane (phi=0°)")
    polar.plot(theta_rad, e_db[:, 1], color="#ff5252", linewidth=1.8,
               label="H-plane (phi=90°)")
    polar.set_rmin(-30)
    polar.set_rmax(0)
    polar.set_title(f"Radiation pattern at {pattern_frequency_ghz:.4g} GHz")
    polar.legend(loc="lower right")
    figure.tight_layout()
    pattern_path = output_dir / "radiation_pattern.png"
    figure.savefig(pattern_path, dpi=190, bbox_inches="tight")
    plt.close(figure)

    metrics = {
        "frequency_ghz": [float(x) for x in eval_freqs_ghz],
        "directivity_dbi": [float(x) for x in directivity],
        "gain_dbi": [float(x) for x in gain],
        "realized_gain_dbi": [float(x) for x in realized_gain],
        "radiation_efficiency_percent": [float(x) for x in radiation_efficiency],
        "mismatch_efficiency_percent": [float(x) for x in mismatch_efficiency],
        "total_efficiency_percent": [float(x) for x in total_efficiency],
        "efficiency_status": "calculated_from_nf2ff_and_port_power",
        "radiation_pattern_frequency_ghz": pattern_frequency_ghz,
        "radiation_pattern_theta_deg": [float(x) for x in theta],
        "radiation_pattern_e_plane_db": [float(x) for x in e_db[:, 0]],
        "radiation_pattern_h_plane_db": [float(x) for x in e_db[:, 1]],
        "radiation_plot": str(pattern_path),
        "radiation_pattern_plot": str(pattern_path),
    }

    # Legacy combined image retained for compatibility with existing result folders.
    figure = plt.figure(figsize=(14, 10))
    polar = figure.add_subplot(221, projection="polar")
    polar.plot(theta_rad, e_db[:, 0], color="#3995ff", linewidth=1.8, label="E-plane")
    polar.plot(theta_rad, e_db[:, 1], color="#ff5252", linewidth=1.8, label="H-plane")
    polar.set_rmin(-30)
    polar.set_rmax(0)
    polar.set_title(f"Radiation pattern at {pattern_frequency_ghz:.4g} GHz")
    polar.legend(loc="lower right", fontsize="small")
    ax = figure.add_subplot(222)
    ax.plot(eval_freqs_ghz, directivity, color="#39c6a5", marker="o", linewidth=1.6)
    ax.set_title("Peak directivity")
    ax.set_ylabel("Directivity (dBi)")
    ax.set_xlabel("Frequency (GHz)")
    ax.grid(True, alpha=0.3)
    ax = figure.add_subplot(223)
    ax.plot(eval_freqs_ghz, gain, color="#3995ff", marker="o", linewidth=1.6,
            label="Gain")
    ax.plot(eval_freqs_ghz, realized_gain, color="#ff9f43", marker="s",
            linewidth=1.6, label="Realized gain")
    ax.set_title("Gain and realized gain")
    ax.set_ylabel("Gain (dBi)")
    ax.set_xlabel("Frequency (GHz)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    ax = figure.add_subplot(224)
    ax.plot(eval_freqs_ghz, radiation_efficiency, "--",
            color="#f2f4f7", label="Radiation efficiency")
    ax.plot(eval_freqs_ghz, mismatch_efficiency, color="#3995ff",
            label="Mismatch efficiency")
    ax.plot(eval_freqs_ghz, total_efficiency, color="#b38cff", marker="^",
            label="Total efficiency")
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

    metrics_path = output_dir / "far_field_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved radiation and far-field plot: {radiation_plot}", flush=True)
    print(
        "Radiation, mismatch and total efficiencies were calculated from "
        "NF2FF radiated power and port accepted/incident power.",
        flush=True,
    )
    return metrics


def update_radiation_pattern(
    csv_path: str | Path,
    simulation_dir: str | Path,
    output_dir: str | Path,
    frequency_ghz: float,
) -> dict:
    """Recalculate only the selected far-field cut and refresh the overview."""
    simulation_dir = Path(simulation_dir).resolve()
    output_dir = Path(output_dir).resolve()
    frequency_ghz = float(frequency_ghz)
    if not np.isfinite(frequency_ghz) or frequency_ghz <= 0:
        raise ValueError("Radiation-pattern frequency must be a positive finite value.")

    manifest_path = simulation_dir / "multiband_run.json"
    metrics_path = output_dir / "far_field_metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Far-field run manifest or metrics file is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("far_field_recorded", False):
        raise RuntimeError("Far-field recordings are not available for this simulation.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    band_defs = manifest.get("bands") or []
    band_runs = manifest.get("band_results") or []
    if not band_defs or len(band_defs) != len(band_runs):
        raise ValueError("The run manifest does not contain matching band definitions and outputs.")

    selected_band = None
    for definition, run in zip(band_defs, band_runs):
        start = float(definition["fstart_ghz"])
        stop = float(definition["fstop_ghz"])
        if start - 1e-9 <= frequency_ghz <= stop + 1e-9:
            band_dir = Path(run["output_dir"])
            if not band_dir.is_absolute():
                band_dir = simulation_dir / band_dir
            selected_band = band_dir
            break
    if selected_band is None:
        raise ValueError(
            f"No recorded simulation band covers {frequency_ghz:g} GHz. "
            f"Available range: {band_defs[0]['fstart_ghz']:g}–"
            f"{band_defs[-1]['fstop_ghz']:g} GHz."
        )

    metadata_path = selected_band / "band_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Band metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("far_field_recorded", False):
        raise RuntimeError(f"Far-field data was not recorded for {selected_band}.")

    print(f"Calculating radiation pattern at {frequency_ghz:.6g} GHz…", flush=True)
    _port, nf2ff = _build_nf2ff_for_band(selected_band)
    theta = np.linspace(-180.0, 180.0, 361)
    result = nf2ff.CalcNF2FF(
        str(selected_band), frequency_ghz * 1e9, theta, [0.0, 90.0]
    )
    e_norm = np.asarray(result.E_norm[0])
    e_db = 20.0 * np.log10(
        e_norm / max(float(np.max(e_norm)), 1e-15) + 1e-12
    )
    metrics.update({
        "radiation_pattern_frequency_ghz": frequency_ghz,
        "radiation_pattern_theta_deg": [float(x) for x in theta],
        "radiation_pattern_e_plane_db": [float(x) for x in e_db[:, 0]],
        "radiation_pattern_h_plane_db": [float(x) for x in e_db[:, 1]],
    })
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    create_result_plot_set(csv_path, output_dir, metrics)
    print(f"Updated radiation pattern and 2x4 overview at {frequency_ghz:.6g} GHz.", flush=True)
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




def _dark_plot_theme():
    """Apply the ETROYL charcoal plotting palette."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#171b22",
        "axes.facecolor": "#202630",
        "axes.edgecolor": "#697586",
        "axes.labelcolor": "#e6eaf0",
        "text.color": "#f2f4f7",
        "xtick.color": "#c5ccd6",
        "ytick.color": "#c5ccd6",
        "grid.color": "#596273",
        "grid.alpha": 0.32,
        "legend.facecolor": "#252d38",
        "legend.edgecolor": "#596273",
        "savefig.facecolor": "#171b22",
    })


def _port_reference_ohms(output_dir: Path) -> float | None:
    """Read the first port resistance from the saved simulation design YAML."""
    import yaml

    output_dir = Path(output_dir).resolve()
    candidates = [output_dir / "design.yaml", output_dir.parent / "design.yaml"]

    # Multi-band runs keep the authoritative design snapshot in each band
    # directory; the plot output directory is not necessarily one of them.
    for manifest_path in (
        output_dir / "multiband_run.json",
        output_dir.parent / "multiband_run.json",
    ):
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for run in manifest.get("band_results") or []:
                band_dir = Path(run["output_dir"])
                if not band_dir.is_absolute():
                    band_dir = manifest_path.parent / band_dir
                candidates.append(band_dir / "design.yaml")
        except (OSError, ValueError, TypeError, KeyError):
            continue

    for config_path in candidates:
        if not config_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            ports = config.get("ports") or []
            if ports:
                value = float(ports[0]["resistance"])
                if np.isfinite(value) and value > 0:
                    return value
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
            continue
    return None


def _draw_radiation_pattern(axis, far_field: dict) -> None:
    """Draw the selected E/H-plane cuts from stored far-field samples."""
    theta = np.asarray(far_field.get("radiation_pattern_theta_deg", []), dtype=float)
    e_plane = np.asarray(far_field.get("radiation_pattern_e_plane_db", []), dtype=float)
    h_plane = np.asarray(far_field.get("radiation_pattern_h_plane_db", []), dtype=float)
    if theta.size and e_plane.size == theta.size and h_plane.size == theta.size:
        theta_rad = np.deg2rad(theta)
        axis.plot(theta_rad, e_plane, color="#3995ff", linewidth=1.8, label="E-plane")
        axis.plot(theta_rad, h_plane, color="#ff5252", linewidth=1.8, label="H-plane")
        # PolarAxes uses the y-axis limits for radial limits; set_rlim is not a Matplotlib Axes API.
        axis.set_ylim(-30, 0)
        axis.set_yticks([-30, -20, -10, 0])
        axis.legend(loc="lower right", fontsize=7)
        frequency = far_field.get("radiation_pattern_frequency_ghz")
        if frequency is not None:
            axis.set_title(f"Radiation pattern · {float(frequency):.4g} GHz", pad=12)
    else:
        axis.text(0.5, 0.5, "Far-field pattern unavailable", ha="center",
                  va="center", transform=axis.transAxes)


def create_result_plot_set(
    csv_path: str | Path,
    output_dir: str | Path,
    far_field: dict | None = None,
) -> list[dict[str, str]]:
    """Create the ETROYL 2x4 results overview and individual diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _dark_plot_theme()
    output_dir = Path(output_dir)
    data = _load_response(csv_path)
    frequency = data["frequency_ghz"]
    s11_db = data["s11_db"]
    s11_mag = 10.0 ** (s11_db / 20.0)
    vswr = (1.0 + s11_mag) / np.maximum(1.0 - s11_mag, 1e-12)
    z_real, z_imag = data["z_real"], data["z_imag"]
    analysis = analyze_broadband_csv(csv_path)
    far_field = far_field or {}
    z0 = _port_reference_ohms(output_dir)

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
        axis.plot(frequency, s11_db, color="#3995ff", linewidth=1.8)
        axis.axhline(-10, color="#aeb8c6", linestyle="--", linewidth=1.0, label="-10 dB")
        axis.axhline(-20, color="#aeb8c6", linestyle=":", linewidth=1.0, label="-20 dB")
        axis.set_title("S11 magnitude")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("|S11| (dB)")
        axis.legend()

    def draw_vswr(axis):
        axis.plot(frequency, vswr, color="#ff5252", linewidth=1.8)
        axis.axhline(2.0, color="#f2f4f7", linestyle="--", linewidth=1.1, label="VSWR = 2")
        axis.set_ylim(0, 5)
        axis.set_title("Voltage standing-wave ratio")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("VSWR")
        axis.legend()

    def draw_real_impedance(axis):
        axis.plot(frequency, z_real, color="#3995ff", linewidth=1.8, label="Re(Zin)")
        if z0 is not None:
            axis.axhline(z0, color="#f2f4f7", linestyle="--", linewidth=1.1,
                         label=f"Port impedance = {z0:g} Ω")
            axis.legend()
        axis.set_title("Real input impedance")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("Resistance (Ω)")

    def draw_imag_impedance(axis):
        axis.plot(frequency, z_imag, color="#ff9f43", linewidth=1.8, label="Im(Zin)")
        axis.axhline(0.0, color="#f2f4f7", linestyle="--", linewidth=1.1,
                     label="Zero reference")
        axis.set_title("Imaginary input impedance")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("Reactance (Ω)")
        axis.legend()

    def draw_phase(axis):
        phase_deg = wrapped_phase_degrees(data["s11_complex"])
        axis.plot(frequency, phase_deg, color="#b38cff", linewidth=1.7)
        axis.set_ylim(-180, 180)
        axis.set_yticks([-180, -90, 0, 90, 180])
        axis.set_title("S11 phase (wrapped)")
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel("Phase (degrees)")

    def draw_far_curve(axis, key, title, ylabel, color):
        values = far_field.get(key)
        eval_freqs = far_field.get("frequency_ghz")
        if values and eval_freqs and len(values) == len(eval_freqs):
            axis.plot(eval_freqs, values, color=color, linewidth=1.8)
        else:
            axis.text(0.5, 0.5, "Far-field data unavailable", ha="center",
                      va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel(ylabel)

    catalog = [
        {"label": "|S11| magnitude (dB)", "filename": "s11_magnitude_db.png",
         "path": save_plot("s11_magnitude_db.png", draw_s11)},
        {"label": "VSWR", "filename": "vswr_plot.png",
         "path": save_plot("vswr_plot.png", draw_vswr)},
        {"label": "Real input impedance", "filename": "input_resistance.png",
         "path": save_plot("input_resistance.png", draw_real_impedance)},
        {"label": "Imaginary input impedance", "filename": "input_reactance.png",
         "path": save_plot("input_reactance.png", draw_imag_impedance)},
    ]

    if "s11_complex" in data:
        catalog.append({
            "label": "Phase(S11) · wrapped ±180°",
            "filename": "s11_phase.png",
            "path": save_plot("s11_phase.png", draw_phase),
        })

    if far_field.get("radiation_pattern_theta_deg"):
        catalog.append({
            "label": "Radiation pattern",
            "filename": "radiation_pattern.png",
            "path": save_plot(
                "radiation_pattern.png",
                lambda axis: _draw_radiation_pattern(axis, far_field),
                figsize=(10, 8),
            ),
        })
    for label, key, filename, title, ylabel, color in (
        ("Gain", "gain_dbi", "gain_plot.png",
         "Gain", "Gain (dBi)", "#3995ff"),
        ("Realized gain", "realized_gain_dbi", "realized_gain_plot.png",
         "Realized gain", "Realized gain (dBi)", "#ff9f43"),
        ("Directivity", "directivity_dbi", "directivity_plot.png",
         "Directivity", "Directivity (dBi)", "#39c6a5"),
        ("Radiation efficiency", "radiation_efficiency_percent",
         "radiation_efficiency_plot.png", "Radiation efficiency",
         "Efficiency (%)", "#f2f4f7"),
        ("Mismatch efficiency", "mismatch_efficiency_percent",
         "mismatch_efficiency_plot.png", "Mismatch efficiency",
         "Efficiency (%)", "#3995ff"),
        ("Total efficiency", "total_efficiency_percent", "efficiency_plot.png",
         "Total efficiency", "Efficiency (%)", "#b38cff"),
    ):
        if far_field.get(key):
            catalog.append({
                "label": label, "filename": filename,
                "path": save_plot(
                    filename,
                    lambda ax, k=key, t=title, y=ylabel, c=color:
                        draw_far_curve(ax, k, t, y, c),
                ),
            })

    # The requested overview remains a readable 2x4 matrix. The phase plot is
    # available separately in the plot selector so none of the eight panels
    # has to be compressed or omitted.
    figure = plt.figure(figsize=(20, 10), constrained_layout=True)
    axes = [
        figure.add_subplot(2, 4, 1),
        figure.add_subplot(2, 4, 2),
        figure.add_subplot(2, 4, 3),
        figure.add_subplot(2, 4, 4),
        figure.add_subplot(2, 4, 5, projection="polar"),
        figure.add_subplot(2, 4, 6),
        figure.add_subplot(2, 4, 7),
        figure.add_subplot(2, 4, 8),
    ]
    draw_s11(axes[0])
    draw_vswr(axes[1])
    draw_real_impedance(axes[2])
    draw_imag_impedance(axes[3])
    _draw_radiation_pattern(axes[4], far_field)
    gain_values = far_field.get("gain_dbi")
    realized_values = far_field.get("realized_gain_dbi")
    eval_freqs = far_field.get("frequency_ghz")
    if gain_values and realized_values and eval_freqs and (
        len(gain_values) == len(eval_freqs) == len(realized_values)
    ):
        axes[5].plot(eval_freqs, gain_values, color="#3995ff",
                     linewidth=1.7, label="Gain")
        axes[5].plot(eval_freqs, realized_values, color="#ff9f43",
                     linewidth=1.7, label="Realized gain")
        axes[5].set_title("Gain and realized gain")
        axes[5].set_xlabel("Frequency (GHz)")
        axes[5].set_ylabel("Gain (dBi)")
        axes[5].legend(fontsize=7)
    else:
        draw_far_curve(axes[5], "gain_dbi", "Gain and realized gain",
                       "Gain (dBi)", "#3995ff")
    draw_far_curve(axes[6], "directivity_dbi", "Directivity",
                   "Directivity (dBi)", "#39c6a5")
    draw_far_curve(axes[7], "total_efficiency_percent", "Total efficiency",
                   "Efficiency (%)", "#b38cff")
    axes[7].plot(far_field.get("frequency_ghz", []),
                 far_field.get("radiation_efficiency_percent", []),
                 "--", color="#f2f4f7", linewidth=1.2,
                 label="Radiation efficiency")
    axes[7].plot(far_field.get("frequency_ghz", []),
                 far_field.get("mismatch_efficiency_percent", []),
                 color="#3995ff", linewidth=1.2,
                 label="Mismatch efficiency")
    axes[7].set_ylim(0, 105)
    axes[7].legend(fontsize=6)
    for axis in (axes[0], axes[1], axes[2], axes[3], axes[5], axes[6], axes[7]):
        axis.grid(True, alpha=0.28)
        axis.set_xlim(float(frequency[0]), float(frequency[-1]))
    overview_path = output_dir / "integrated_overview.png"
    figure.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    catalog.insert(0, {
        "label": "Integrated overview (2×4)",
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
    parser.add_argument("--pattern-frequency", type=float,
                        help="Update the radiation pattern at this frequency in GHz")
    parser.add_argument("--pattern-only", action="store_true",
                        help="Update only the selected radiation pattern and integrated overview")
    args = parser.parse_args()
    if args.pattern_only:
        if not args.simulation_dir or args.pattern_frequency is None:
            parser.error("--pattern-only requires --simulation-dir and --pattern-frequency")
        update_radiation_pattern(
            args.csv, args.simulation_dir, args.output_dir, args.pattern_frequency
        )
        return 0
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
