import os
import sys
import argparse
import yaml
import optuna
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

from design import build_design, load_config
from objectives import minimize_s11_in_band, multi_objective_combined

# Suppress standard output during FDTD runs for clean progress output
class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_simulation_instance(config_file, trial_vars, trial_dir):
    """Helper to execute an isolated openEMS simulation trial."""
    os.makedirs(trial_dir, exist_ok=True)
    
    cfg = load_config(config_file, active_vars=trial_vars)
    
    unit = float(cfg.get('simulation', {}).get('unit', 1.0))
    fstart = float(cfg['simulation']['fstart']) * 1e9
    fstop  = float(cfg['simulation']['fstop'])  * 1e9
    num_pts = int(cfg.get('simulation', {}).get('n_pts', 501))
    end_criteria = float(cfg.get('simulation', {}).get('end_criteria', 1e-2))
    
    f0 = (fstart + fstop) / 2.0
    fc = (fstop - fstart)

    CSX = ContinuousStructure()
    FDTD = openEMS(EndCriteria=end_criteria)
    FDTD.SetCSX(CSX)
    FDTD.SetGaussExcite(f0, fc)

    # Disable NF2FF during optimization trials for maximum speed
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

    # Calculate voltage/current and impedance metrics ONLY
    freq = np.linspace(fstart, fstop, num_pts)
    port.CalcPort(trial_dir, freq)

    s11 = port.uf_ref / port.uf_inc
    s11_db = 20 * np.log10(np.abs(s11) + 1e-12)
    
    ports_cfg = cfg.get('ports', [{}])[0]
    Z0 = float(ports_cfg.get('resistance', 50.0))
    z_in = Z0 * (1 + s11) / (1 - s11)

    return {
        'freq': freq,
        's11': s11,
        's11_db': s11_db,
        'z_in': z_in,
        'port': port
    }

def objective_wrapper(trial, config_file, base_sim_dir):
    """Optuna objective function wrapper."""
    with open(config_file, 'r') as f:
        raw_cfg = yaml.safe_load(f)

    # 1. Dynamically sample variables based on YAML definitions
    trial_vars = {}
    for v_name, v_opts in raw_cfg.get('variables', {}).items():
        v_type = v_opts.get('type', 'float')
        start = v_opts['start']
        stop = v_opts['stop']
        step = v_opts.get('step', None)

        if v_type == 'int':
            trial_vars[v_name] = trial.suggest_int(v_name, int(start), int(stop), step=int(step) if step else 1)
        else:
            trial_vars[v_name] = trial.suggest_float(v_name, float(start), float(stop), step=float(step) if step else None)

    # 2. Run simulation in isolated directory
    trial_dir = os.path.join(base_sim_dir, f"trial_{trial.number:03d}")
    try:
        results = run_simulation_instance(config_file, trial_vars, trial_dir)
        
        # 3. Compute cost using objective function
        cost = minimize_s11_in_band(results, f_min=0.3e9, f_max=0.6e9)
        return cost
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 1e6  # Penalty for invalid geometries

def main():
    parser = argparse.ArgumentParser(description="Generic openEMS Geometry Optimizer")
    parser.add_argument("--config", type=str, default="designs/bowtie_variable.yaml", help="Path to YAML design file")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of optimization trials")
    args = parser.parse_args()

    # Convert config path to absolute path to prevent working directory issues
    abs_config_path = os.path.abspath(args.config)

    base_name = os.path.splitext(os.path.basename(abs_config_path))[0]
    sim_dir = os.path.abspath(os.path.join("designs", base_name + "_opt"))

    print(f"Starting Optimization Study for '{args.config}'...")
    print(f"Output directory: {sim_dir}\n")

    # Optuna Bayesian Optimizer Study (TPESampler)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    
    study.optimize(
        lambda trial: objective_wrapper(trial, abs_config_path, sim_dir), 
        n_trials=args.n_trials,
        n_jobs=8,  # Run 4 parallel openEMS processes
        show_progress_bar=True
    )

    print("\n" + "="*50)
    print(" OPTIMIZATION COMPLETED ")
    print("="*50)
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Objective Value: {study.best_value:.4f}")
    print("Optimal Parameters Found:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")

    # Save best configuration to a new YAML
    best_yaml_path = os.path.join("designs", f"{base_name}_optimized.yaml")
    optimized_cfg = load_config(abs_config_path, active_vars=study.best_params)
    
    with open(best_yaml_path, 'w') as f:
        yaml.dump(optimized_cfg, f, default_flow_style=False)
    
    print(f"\nSaved optimal design config to: {best_yaml_path}")

if __name__ == "__main__":
    main()