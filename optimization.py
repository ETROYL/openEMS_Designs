import os
import sys
import argparse
import yaml
import optuna
import numpy as np
import matplotlib.pyplot as plt
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design, load_config
from objectives import (
    minimize_max_s11_in_band,
    maximize_bandwidth_in_band,
    minimize_s11_in_band,
    maximize_impedance_smoothness,
    multi_objective_combined
)

class SuppressStdout:
    def __enter__(self):
        self._null_file = open(os.devnull, 'w')
        self._original_stdout = sys.stdout
        sys.stdout = self._null_file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._null_file.close()

# ---------------------------------------------------------------------
# Live 2D Geometry Plotter
# ---------------------------------------------------------------------
class GeometryMonitor2D:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.fig.canvas.manager.set_window_title("Live Antenna Geometry Monitor")
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)

    def update(self, trial_num, cfg):
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title(f"Trial {trial_num:03d} - Live Geometry Footprint (XY-Plane)", fontsize=11, fontweight='bold')
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        # Plot Primitives (Polygons & Boxes)
        for prim in cfg.get('primitives', []):
            if prim['type'] == 'polygon':
                px = prim['points']['x']
                py = prim['points']['y']
                # Close the polygon loop for display
                px_loop = px + [px[0]]
                py_loop = py + [py[0]]
                
                label = prim.get('material', 'Patch')
                self.ax.plot(px_loop, py_loop, '-o', linewidth=2, markersize=3, label=label)
                self.ax.fill(px_loop, py_loop, alpha=0.3)

        # Plot Port Location
        for p in cfg.get('ports', []):
            p_start = p['start']
            p_stop = p['stop']
            self.ax.plot([p_start[0], p_stop[0]], [p_start[1], p_stop[1]], 'r-X', linewidth=3, markersize=8, label='Lumped Port')

        # Formatting
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)


def run_simulation_instance(config_file, trial_vars, trial_dir):
    os.makedirs(trial_dir, exist_ok=True)
    cfg = load_config(config_file, active_vars=trial_vars)
    
    unit = float(cfg.get('simulation', {}).get('unit', 1.0))
    fstart = float(cfg['simulation']['fstart']) * 1e9
    fstop  = float(cfg['simulation']['fstop'])  * 1e9
    num_pts = int(cfg.get('simulation', {}).get('n_pts', 301))
    
    end_criteria = float(cfg.get('simulation', {}).get('end_criteria', 1e-2))
    max_timesteps = int(cfg.get('simulation', {}).get('max_timesteps', 40000))
    
    f0 = (fstart + fstop) / 2.0
    fc = (fstop - fstart)

    CSX = ContinuousStructure()
    FDTD = openEMS(EndCriteria=end_criteria, NrTS=max_timesteps)
    FDTD.SetCSX(CSX)
    FDTD.SetGaussExcite(f0, fc)

    port, _ = build_design(CSX, FDTD, config_file=config_file, active_vars=trial_vars, enable_nf2ff=False)

    xml_filename = 'geometry.xml'
    xml_path = os.path.join(trial_dir, xml_filename)
    CSX.Write2XML(xml_path)
    
    cwd = os.getcwd()
    try:
        with SuppressStdout():
            FDTD.Run(trial_dir, xml_filename, verbose=0)
    finally:
        os.chdir(cwd)

    freq = np.linspace(fstart, fstop, num_pts)
    port.CalcPort(trial_dir, freq)

    s11 = port.uf_ref / port.uf_inc
    s11_db = 20 * np.log10(np.abs(s11) + 1e-12)
    z_in = port.uf_tot / port.if_tot
    
    return {
        'freq': freq,
        's11': s11,
        's11_db': s11_db,
        'z_in': z_in,
        'port': port,
        'evaluated_cfg': cfg
    }


def objective_wrapper(trial, config_file, base_sim_dir, monitor):
    with open(config_file, 'r') as f:
        raw_cfg = yaml.safe_load(f)

    trial_vars = {}
    for v_name, v_opts in raw_cfg.get('variables', {}).items():
        if isinstance(v_opts, dict):
            if not v_opts.get('optimize', True) or v_opts.get('start') == v_opts.get('stop'):
                trial_vars[v_name] = v_opts.get('default', v_opts.get('start'))
                continue

            v_type = v_opts.get('type', 'float')
            start = v_opts['start']
            stop = v_opts['stop']
            step = v_opts.get('step', None)

            if v_type == 'int':
                trial_vars[v_name] = trial.suggest_int(v_name, int(start), int(stop), step=int(step) if step else 1)
            else:
                trial_vars[v_name] = trial.suggest_float(v_name, float(start), float(stop), step=float(step) if step else None)
        else:
            trial_vars[v_name] = v_opts

    trial_dir = os.path.join(base_sim_dir, f"trial_{trial.number:03d}")
    try:
        # Pre-evaluate config to plot live geometry right before simulation starts
        evaluated_cfg = load_config(config_file, active_vars=trial_vars)
        if monitor:
            monitor.update(trial.number, evaluated_cfg)

        results = run_simulation_instance(config_file, trial_vars, trial_dir)
        
        # Optimize 
        # cost = maximize_impedance_smoothness(results, f_min=None, f_max=None)
        # cost = minimize_s11_in_band(results, f_min=None, f_max=None)
        cost = multi_objective_combined(results, f_min=None, f_max=None)
        return cost
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 1e6


def main():
    parser = argparse.ArgumentParser(description="Generic openEMS Geometry Optimizer with Live Visualizer")
    parser.add_argument("--config", type=str, default="designs/bowtie_variable.yaml", help="Path to YAML design file")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--no_plot", action="store_true", help="Disable live 2D plotting window")
    args = parser.parse_args()

    abs_config_path = os.path.abspath(args.config)
    base_name = os.path.splitext(os.path.basename(abs_config_path))[0]
    sim_dir = os.path.abspath(os.path.join("designs", base_name + "_opt"))

    print(f"Starting Optimization Study for '{args.config}'...")
    print(f"Output directory: {sim_dir}\n")

    monitor = None if args.no_plot else GeometryMonitor2D()

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    
    study.optimize(
        lambda trial: objective_wrapper(trial, abs_config_path, sim_dir, monitor), 
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    if monitor:
        plt.ioff()
        plt.show()

    print("\n" + "="*50)
    print(" OPTIMIZATION COMPLETED ")
    print("="*50)
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Objective Value: {study.best_value:.4f}")

    
    rounded_params = {
        k: round(v, 6) if isinstance(v, float) else v 
        for k, v in study.best_params.items()
    }

    print("Optimal Parameters Found (Cleaned):")
    for k, v in rounded_params.items():
        print(f"  - {k}: {v}")

    HIGH_ACCURACY_SIM_SETTINGS = {
        'end_criteria': 1e-4,     # -40 dB energy decay for full convergence
        'max_timesteps': 1000000, # Allow extended timesteps for sharp resonances
        'n_pts': 1024,            # Higher frequency resolution for plotting
        'target_s11_db': -40.0    # Strict goal target
    }

    best_yaml_path = os.path.join("designs", f"{base_name}_optimized.yaml")
    optimized_cfg = load_config(abs_config_path, active_vars=rounded_params)
    
    if 'simulation' in optimized_cfg:
        optimized_cfg['simulation'].update(HIGH_ACCURACY_SIM_SETTINGS)

    with open(best_yaml_path, 'w') as f:
        yaml.dump(optimized_cfg, f, default_flow_style=False)
    
    print(f"\nSaved optimal design config to: {best_yaml_path}")

if __name__ == "__main__":
    main()