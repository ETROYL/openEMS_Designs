"""Post-process an openEMS simulation for the ETROYL desktop GUI."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Saved YAML snapshot for this simulation")
    parser.add_argument("--sim-dir", required=True, type=Path, help="FDTD output directory")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plot images without opening interactive Matplotlib windows",
    )
    return parser.parse_args()


def find_port_file(directory: Path, prefixes: tuple[str, ...]) -> Path | None:
    """Find a recorded port-voltage file, allowing openEMS naming variants."""
    for prefix in prefixes:
        for extension in ("", ".pdt", ".h5"):
            candidate = directory / f"{prefix}{extension}"
            if candidate.is_file():
                return candidate
    return None


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    sim_dir = args.sim_dir.resolve()

    if args.no_show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#171b22", "axes.facecolor": "#202630",
        "axes.edgecolor": "#697586", "axes.labelcolor": "#e6eaf0",
        "text.color": "#f2f4f7", "xtick.color": "#c5ccd6",
        "ytick.color": "#c5ccd6", "grid.color": "#596273",
        "legend.facecolor": "#252d38", "legend.edgecolor": "#596273",
        "savefig.facecolor": "#171b22",
    })

    if not config_path.is_file():
        raise FileNotFoundError(f"Design snapshot does not exist: {config_path}")
    if not sim_dir.is_dir():
        raise FileNotFoundError(f"Simulation output directory does not exist: {sim_dir}")

    cfg = load_config(str(config_path))
    simulation = cfg.get("simulation") or {}
    ports = cfg.get("ports") or []
    if not ports:
        raise ValueError("The design YAML must define at least one port.")

    fstart = float(simulation["fstart"]) * 1e9
    fstop = float(simulation["fstop"]) * 1e9
    num_pts = int(simulation.get("n_pts", 1024))
    if fstart <= 0 or fstop <= fstart:
        raise ValueError("simulation.fstart/fstop must define a positive increasing frequency range.")
    if num_pts < 2:
        raise ValueError("simulation.n_pts must be at least 2.")

    z0 = float(ports[0]["resistance"])
    if z0 <= 0:
        raise ValueError("The first port resistance must be positive.")

    voltage_file = find_port_file(sim_dir, ("port_ut_1", "port_ut1"))
    if voltage_file is None:
        raise FileNotFoundError(f"Port output files were not found in {sim_dir}.")

    print(f"Design snapshot: {config_path}", flush=True)
    print(f"Simulation data: {sim_dir}", flush=True)
    print(f"Frequency points: {num_pts}", flush=True)
    print("Reconstructing design and calculating port parameters…", flush=True)

    f0 = (fstart + fstop) / 2.0
    fc = fstop - fstart
    csx = ContinuousStructure()
    fdtd = openEMS(EndCriteria=1e-4)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(f0, fc)

    port, nf2ff = build_design(
        csx, fdtd, config_file=str(config_path), enable_nf2ff=True
    )
    if isinstance(port, (list, tuple)):
        if len(port) != 1:
            raise NotImplementedError(
                "This GUI post-processing version currently supports one port. "
                "Multi-port S-parameter processing will be added in a later increment."
            )
        port = port[0]
    if nf2ff is None:
        raise ValueError("The design did not create an NF2FF box; far-field processing is unavailable.")

    freq = np.linspace(fstart, fstop, num_pts)
    freq_ghz = freq / 1e9
    port.CalcPort(str(sim_dir), freq)

    with np.errstate(divide="ignore", invalid="ignore"):
        s11 = port.uf_ref / port.uf_inc
        s11_db = 20 * np.log10(np.abs(s11) + 1e-12)
        z_in = port.uf_tot / port.if_tot
        vswr = (1 + np.abs(s11)) / (1 - np.abs(s11) + 1e-12)

    if not np.all(np.isfinite(s11_db)):
        raise ValueError("S11 calculation produced non-finite values; check the port recordings.")

    # np.angle returns the principal phase in [-pi, +pi]; do not unwrap it.
    s11_phase_deg = np.rad2deg(np.angle(s11))
    fig_phase, ax_phase = plt.subplots(figsize=(12, 6))
    ax_phase.plot(freq_ghz, s11_phase_deg, color="#b38cff", linewidth=1.8)
    ax_phase.set_ylim([-180, 180])
    ax_phase.set_yticks([-180, -90, 0, 90, 180])
    ax_phase.set_xlabel("Frequency (GHz)")
    ax_phase.set_ylabel("Phase(S11) (degrees)")
    ax_phase.set_title("S11 phase (wrapped)")
    ax_phase.grid(True, alpha=0.28)
    fig_phase.tight_layout()
    phase_path = sim_dir / "s11_phase_plot.png"
    fig_phase.savefig(phase_path, dpi=180, bbox_inches="tight")
    plt.close(fig_phase)
    print(f"Saved wrapped S11 phase plot: {phase_path}", flush=True)

    fig1, ((ax1, ax3), (ax2, ax4)) = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True
    )
    ax1.plot(freq_ghz, s11_db, "b-", linewidth=2)
    ax1.axhline(-10, color="k", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("|S11| (dB)")
    ax1.set_ylim([-35, 0])
    ax1.grid(True)
    ax1.set_title("S11 Parameter")

    ax2.plot(freq_ghz, vswr, "r-", linewidth=2)
    ax2.axhline(2, color="k", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("VSWR")
    ax2.grid(True)

    ax3.plot(freq_ghz, z_in.real, "b-", linewidth=2)
    ax3.axhline(z0, color="k", linestyle="--", linewidth=1.5)
    ax3.set_ylabel("Re(Zin) [Ohm]")
    ax3.grid(True)
    ax3.set_title(f"Input impedance (reference {z0:g} Ohm)")

    ax4.plot(freq_ghz, z_in.imag, "r-", linewidth=2)
    ax4.axhline(0, color="k", linestyle="--", linewidth=1.5)
    ax4.set_xlabel("Frequency (GHz)")
    ax4.set_ylabel("Im(Zin) [Ohm]")
    ax4.grid(True)

    fig1.tight_layout()
    plot_path_s11 = sim_dir / "s11_plot.png"
    fig1.savefig(plot_path_s11, dpi=180)
    plt.close(fig1)
    print(f"Saved S-parameter and impedance plot: {plot_path_s11}", flush=True)

    if not np.all(np.isfinite(s11_db)):
        raise ValueError("Cannot determine a valid resonance frequency from S11.")
    f_res_idx = int(np.argmin(s11_db))
    f_res = float(freq[f_res_idx])
    f_res_ghz = float(freq_ghz[f_res_idx])
    print(f"Calculating NF2FF patterns at {f_res_ghz:.4f} GHz…", flush=True)

    theta = np.linspace(-180, 180, 361)
    phi = [0.0, 90.0]
    nf2ff_res = nf2ff.CalcNF2FF(str(sim_dir), f_res, theta, phi)

    eval_freqs = np.linspace(fstart, fstop, 21)
    directivity_list = []
    realized_gain_list = []
    assumed_rad_efficiency_list = []
    total_efficiency_list = []

    for index, f_eval in enumerate(eval_freqs, start=1):
        print(f"NF2FF frequency {index}/21: {f_eval / 1e9:.4f} GHz", flush=True)
        result = nf2ff.CalcNF2FF(str(sim_dir), float(f_eval), theta, phi)
        d_max = float(result.Dmax[0])
        d_max_dbi = 10 * np.log10(max(d_max, 1e-12))

        s11_eval = np.interp(float(f_eval), freq, s11)
        mismatch_efficiency = max(0.0, 1.0 - float(np.abs(s11_eval) ** 2))

        # This is an assumption, not a computed loss model. It is only valid
        # as an idealized estimate for a lossless PEC structure in free space.
        assumed_rad_efficiency = 1.0
        total_efficiency = assumed_rad_efficiency * mismatch_efficiency
        realized_gain_dbi = d_max_dbi + 10 * np.log10(total_efficiency + 1e-12)

        directivity_list.append(d_max_dbi)
        realized_gain_list.append(realized_gain_dbi)
        assumed_rad_efficiency_list.append(100.0 * assumed_rad_efficiency)
        total_efficiency_list.append(100.0 * total_efficiency)

    eval_freqs_ghz = eval_freqs / 1e9
    fig2 = plt.figure(figsize=(11, 8))
    ax_polar = fig2.add_subplot(221, projection="polar")
    e_norm = 20 * np.log10(
        nf2ff_res.E_norm[0] / np.max(nf2ff_res.E_norm[0]) + 1e-12
    )
    theta_rad = np.deg2rad(theta)
    ax_polar.plot(theta_rad, e_norm[:, 0], color="#3995ff", linewidth=2, label="E-plane (phi=0 deg)")
    ax_polar.plot(theta_rad, e_norm[:, 1], color="#ff5252", linewidth=2, label="H-plane (phi=90 deg)")
    ax_polar.set_rmin(-30)
    ax_polar.set_rmax(0)
    ax_polar.set_title(f"Radiation pattern at {f_res_ghz:.2f} GHz (dB)", va="bottom")
    ax_polar.legend(loc="lower right", fontsize="small")

    ax_dir = fig2.add_subplot(222)
    ax_dir.plot(eval_freqs_ghz, directivity_list, "g-o", linewidth=2)
    ax_dir.set_xlabel("Frequency (GHz)")
    ax_dir.set_ylabel("Directivity (dBi)")
    ax_dir.set_title("Peak directivity vs frequency")
    ax_dir.grid(True)

    ax_gain = fig2.add_subplot(223)
    ax_gain.plot(eval_freqs_ghz, realized_gain_list, "m-s", linewidth=2)
    ax_gain.set_xlabel("Frequency (GHz)")
    ax_gain.set_ylabel("Realized gain (dBi)")
    ax_gain.set_title(f"Realized gain (idealized, {z0:g} Ohm port reference)")
    ax_gain.grid(True)

    ax_eff = fig2.add_subplot(224)
    ax_eff.plot(
        eval_freqs_ghz, assumed_rad_efficiency_list, "r--", linewidth=2,
        label="Assumed radiation efficiency (100%)",
    )
    ax_eff.plot(
        eval_freqs_ghz, total_efficiency_list, "c-^", linewidth=2,
        label="Total efficiency incl. mismatch",
    )
    ax_eff.set_xlabel("Frequency (GHz)")
    ax_eff.set_ylabel("Efficiency (%)")
    ax_eff.set_ylim([0, 105])
    ax_eff.set_title("Efficiency estimate (lossless PEC assumption)")
    ax_eff.legend(loc="lower right", fontsize="small")
    ax_eff.grid(True)

    fig2.tight_layout()
    plot_path_rad = sim_dir / "radiation_plot.png"
    fig2.savefig(plot_path_rad, dpi=180)
    plt.close(fig2)
    print(f"Saved radiation and far-field plot: {plot_path_rad}", flush=True)
    print(
        "Note: radiation efficiency is assumed to be 100% (ideal lossless PEC); "
        "it is not calculated from conductor or dielectric losses.",
        flush=True,
    )

    if not args.no_show:
        plt.show()
    print("Post-processing completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
