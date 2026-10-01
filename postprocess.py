import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design, load_config

def find_port_file(directory, prefixes):
    """Finds the port file checking multiple prefix variations and extensions."""
    for prefix in prefixes:
        for ext in ['', '.pdt', '.h5']:
            candidate = os.path.join(directory, f"{prefix}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None

def main():
    parser = argparse.ArgumentParser(description="Generic openEMS Post-Processor")
    parser.add_argument("--config", type=str, default="bowtie.yaml", help="Path to YAML design file")
    args = parser.parse_args()

    config_file = getattr(args, 'config', 'bowtie.yaml')
    cfg = load_config(config_file)
    sim_dir = config_file[0:-5]  # str(cfg.get('simulation', {}).get('path', 'tmp'))
    fstart = (cfg.get('simulation').get('fstart')) * 1e9
    fstop = (cfg.get('simulation', {}).get('fstop', 1)) * 1e9
    num_pts = int(cfg.get('simulation', {}).get('n_pts', 1024))
    Z0 = cfg.get('ports')[0].get("resistance")

    if not os.path.exists(sim_dir):
        print(f"Error: Directory '{sim_dir}' does not exist.")
        return

    u_file = find_port_file(sim_dir, ['port_ut_1', 'port_ut1'])
    if not u_file:
        print(f"Error: Port output files not found in '{sim_dir}'.")
        return

    # Reconstruct CSX & FDTD structure
    f0 = (fstart + fstop) / 2.0
    fc = (fstop - fstart)

    CSX = ContinuousStructure()
    FDTD = openEMS(EndCriteria=1e-4)
    FDTD.SetCSX(CSX)
    FDTD.SetGaussExcite(f0, fc)

    # Build design & link NF2FF box
    port, nf2ff = build_design(CSX, FDTD, config_file)

    freq = np.linspace(fstart, fstop, num_pts)
    freq_ghz = freq / 1e9

    # Compute S-parameters
    port.CalcPort(sim_dir, freq)

    s11 = port.uf_ref / port.uf_inc
    s11_db = 20 * np.log10(np.abs(s11) + 1e-12)
    z_in = Z0 * (1 + s11) / (1 - s11)
    vswr = (1 + abs(s11)) / (1 - abs(s11) + 1e-12)

    # =========================================================================
    # Figure 1: Circuit Parameters (S11, Impedance, VSWR)
    # =========================================================================
    fig1, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    ax1.plot(freq_ghz, s11_db, 'b-', linewidth=2)
    ax1.axhline(-10, color='k', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('|S11| (dB)')
    ax1.set_ylim([-35, 0])
    ax1.grid(True)
    ax1.set_title("Bowtie Antenna S11 Parameter")

    ax2.plot(freq_ghz, vswr, 'r-', linewidth=2)
    ax2.axhline(2, color='k', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('VSWR')
    ax2.grid(True)

    ax3.plot(freq_ghz, z_in.real, 'b-', linewidth=2)
    ax3.axhline(Z0, color='k', linestyle='--', linewidth=1.5)
    ax3.set_ylabel('Re(Z_in) [Ohm]')
    ax3.grid(True)
    ax3.set_title("Bowtie Antenna Impedance")

    ax4.plot(freq_ghz, z_in.imag, 'r-', linewidth=2)
    ax4.axhline(0, color='k', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Im(Z_in) [Ohm]')
    ax4.grid(True)

    plt.tight_layout()
    plot_path_s11 = os.path.join(sim_dir, 's11_plot.png')
    plt.savefig(plot_path_s11, dpi=300)
    print(f"Saved S11 Plot to: {plot_path_s11}")

    # =========================================================================
    # Figure 2: Far-Field Radiation, Directivity, Realized Gain & Efficiency
    # =========================================================================
    f_res_idx = np.argmin(s11_db)
    f_res = freq[f_res_idx]
    f_res_ghz = freq_ghz[f_res_idx]

    print(f"\nCalculating Far-Field patterns at resonance frequency: {f_res_ghz:.3f} GHz...")

    # Far-Field Angular Sweep
    theta = np.linspace(-180, 180, 361)
    phi = [0.0, 90.0]  # E-plane and H-plane

    # Calculate Pattern at Resonance
    nf2ff_res = nf2ff.CalcNF2FF(sim_dir, f_res, theta, phi)

    # Sample frequency points across band
    eval_freqs = np.linspace(fstart, fstop, 21)
    directivity_list = []
    realized_gain_list = []
    rad_efficiency_list = []
    tot_efficiency_list = []

    for f_eval in eval_freqs:
        res = nf2ff.CalcNF2FF(sim_dir, f_eval, theta, phi)
        d_max = res.Dmax[0]
        d_max_dbi = 10 * np.log10(d_max)

        # Get S11 at f_eval
        s11_eval = np.interp(f_eval, freq, s11)
        mismatch_loss = 1.0 - np.abs(s11_eval)**2  # (1 - |S11|^2)

        # Efficiency Calculations
        rad_eff = 1.0  # 100% for lossless PEC in air
        tot_eff = rad_eff * mismatch_loss

        # Gain & Realized Gain
        realized_gain_dbi = d_max_dbi + 10 * np.log10(tot_eff + 1e-12)

        directivity_list.append(d_max_dbi)
        realized_gain_list.append(realized_gain_dbi)
        rad_efficiency_list.append(rad_eff * 100.0)
        tot_efficiency_list.append(tot_eff * 100.0)

    eval_freqs_ghz = eval_freqs / 1e9

    # Generate Figure 2
    fig2 = plt.figure(figsize=(11, 8))

    # Subplot 1: Radiation Pattern
    ax_polar = fig2.add_subplot(221, projection='polar')
    e_norm = 20 * np.log10(nf2ff_res.E_norm[0] / np.max(nf2ff_res.E_norm[0]) + 1e-12)

    theta_rad = np.deg2rad(theta)
    ax_polar.plot(theta_rad, e_norm[:, 0], 'b-', linewidth=2, label='E-plane (phi=0°)')
    ax_polar.plot(theta_rad, e_norm[:, 1], 'r--', linewidth=2, label='H-plane (phi=90°)')
    ax_polar.set_rmin(-30)
    ax_polar.set_rmax(0)
    ax_polar.set_title(f"Radiation Pattern @ {f_res_ghz:.2f} GHz (dB)", va='bottom')
    ax_polar.legend(loc='lower right', fontsize='small')

    # Subplot 2: Peak Directivity
    ax_dir = fig2.add_subplot(222)
    ax_dir.plot(eval_freqs_ghz, directivity_list, 'g-o', linewidth=2)
    ax_dir.set_xlabel('Frequency (GHz)')
    ax_dir.set_ylabel('Directivity (dBi)')
    ax_dir.set_title('Peak Directivity vs Frequency')
    ax_dir.grid(True)

    # Subplot 3: Realized Gain (Including Mismatch Loss)
    ax_gain = fig2.add_subplot(223)
    ax_gain.plot(eval_freqs_ghz, realized_gain_list, 'm-s', linewidth=2)
    ax_gain.set_xlabel('Frequency (GHz)')
    ax_gain.set_ylabel('Realized Gain (dBi)')
    ax_gain.set_title('Realized Gain vs Frequency (50 Ω Matched)')
    ax_gain.grid(True)

    # Subplot 4: Radiation Efficiency & Total Efficiency
    ax_eff = fig2.add_subplot(224)
    ax_eff.plot(eval_freqs_ghz, rad_efficiency_list, 'r--', linewidth=2, label='Radiation Efficiency')
    ax_eff.plot(eval_freqs_ghz, tot_efficiency_list, 'c-^', linewidth=2, label='Total Efficiency (incl. S11)')
    ax_eff.set_xlabel('Frequency (GHz)')
    ax_eff.set_ylabel('Efficiency (%)')
    ax_eff.set_ylim([0, 105])
    ax_eff.set_title('Efficiency vs Frequency')
    ax_eff.legend(loc='lower right', fontsize='small')
    ax_eff.grid(True)

    plt.tight_layout()
    plot_path_rad = os.path.join(sim_dir, 'radiation_plot.png')
    plt.savefig(plot_path_rad, dpi=300)
    print(f"Saved Radiation & Far-Field Plot to: {plot_path_rad}")

    plt.show()

if __name__ == "__main__":
    main()