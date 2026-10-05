import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import shgo
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design, load_config
from objectives import multi_objective_combined

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
        self.fig.canvas.manager.set_window_title("Live Antenna Geometry Monitor (SHGO + Nelder-Mead)")
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)

    def update(self, trial_num, cfg):
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title(f"Evaluation {trial_num:03d} - Live Geometry Footprint (XY-Plane)", fontsize=11, fontweight='bold')
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        # Plot Primitives (Polygons & Boxes)
        for prim in cfg.get('primitives', []):
            if prim['type'] == 'polygon':
                px = prim['points']['x']
                py = prim['points']['y']
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

    os.environ["OMP_NUM_THREADS"] = "16" #SIAVASH check
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


def quantize_variable(val, v_opts):
    """
    Rounds/quantizes continuous float vectors from Scipy optimizer to discrete step sizes
    or integer boundaries defined in YAML config.
    """
    start = float(v_opts['start'])
    stop = float(v_opts['stop'])
    step = v_opts.get('step', None)
    v_type = v_opts.get('type', 'float')

    val = np.clip(val, start, stop)

    if step is not None and step > 0:
        val = start + np.round((val - start) / step) * step
        val = np.clip(val, start, stop)

    if v_type == 'int':
        return int(np.round(val))
    return float(round(val, 6))


class ScipyObjectiveWrapper:
    def __init__(self, config_file, base_sim_dir, monitor, max_evals=80):
        self.config_file = config_file
        self.base_sim_dir = base_sim_dir
        self.monitor = monitor
        self.max_evals = max_evals
        
        self.eval_count = 0
        self.memo_cache = {}
        self.best_cost = float('inf')
        self.best_vars = None

        with open(config_file, 'r') as f:
            self.raw_cfg = yaml.safe_load(f)

        self.var_names = []
        self.bounds = []

        for v_name, v_opts in self.raw_cfg.get('variables', {}).items():
            if isinstance(v_opts, dict):
                if v_opts.get('optimize', True) and v_opts.get('start') != v_opts.get('stop'):
                    self.var_names.append(v_name)
                    self.bounds.append((float(v_opts['start']), float(v_opts['stop'])))

    def __call__(self, x_vec):
        # 1. Enforce Evaluation Budget Cap
        if self.eval_count >= self.max_evals:
            return float(self.best_cost if self.best_cost != float('inf') else 1e6)

        # 2. Map continuous Scipy input vector to quantized YAML variables
        trial_vars = {}
        idx = 0
        for v_name, v_opts in self.raw_cfg.get('variables', {}).items():
            if isinstance(v_opts, dict):
                if v_opts.get('optimize', True) and v_opts.get('start') != v_opts.get('stop'):
                    trial_vars[v_name] = quantize_variable(x_vec[idx], v_opts)
                    idx += 1
                else:
                    trial_vars[v_name] = v_opts.get('default', v_opts.get('start'))
            else:
                trial_vars[v_name] = v_opts

        # 3. Deduplication Memoization (Avoids redundant FDTD runs during Nelder-Mead simplex steps)
        memo_key = tuple(sorted((k, v) for k, v in trial_vars.items()))
        if memo_key in self.memo_cache:
            return self.memo_cache[memo_key]

        self.eval_count += 1
        trial_dir = os.path.join(self.base_sim_dir, f"eval_{self.eval_count:03d}")

        print(f"\n--- [SHGO + Nelder-Mead] Evaluation {self.eval_count}/{self.max_evals} ---")
        for k, v in trial_vars.items():
            print(f"  {k}: {v}")

        try:
            evaluated_cfg = load_config(self.config_file, active_vars=trial_vars)
            if self.monitor:
                self.monitor.update(self.eval_count, evaluated_cfg)

            results = run_simulation_instance(self.config_file, trial_vars, trial_dir)
            cost = float(multi_objective_combined(results, f_min=None, f_max=None))

            self.memo_cache[memo_key] = cost
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_vars = trial_vars.copy()
                print(f"  => New Best Objective Value: {cost:.4f}")

            return cost
        except Exception as e:
            print(f"  Evaluation {self.eval_count} failed with error: {e}")
            self.memo_cache[memo_key] = 1e6
            return 1e6


def main():
    parser = argparse.ArgumentParser(description="SHGO + Nelder-Mead Global Optimizer for openEMS")
    parser.add_argument("--config", type=str, default="designs/bowtie_variable.yaml", help="Path to YAML design file")
    parser.add_argument("--max_evals", type=int, default=80, help="Maximum simulation budget cap")
    parser.add_argument("--no_plot", action="store_true", help="Disable live 2D plotting window")
    args = parser.parse_args()

    abs_config_path = os.path.abspath(args.config)
    base_name = os.path.splitext(os.path.basename(abs_config_path))[0]
    sim_dir = os.path.abspath(os.path.join("designs", base_name + "_opt_shgo"))

    print(f"Starting SHGO + Nelder-Mead Optimization for '{args.config}'...")
    print(f"Output directory: {sim_dir}")
    print(f"Max Evaluation Cap: {args.max_evals}\n")

    monitor = None if args.no_plot else GeometryMonitor2D()
    obj_func = ScipyObjectiveWrapper(abs_config_path, sim_dir, monitor, max_evals=args.max_evals)

    # Scipy SHGO global optimization configuration using Nelder-Mead local minimization
    minimizer_kwargs = {
        'method': 'Nelder-Mead',
        'options': {
            'maxev': 30,          # Local simplex evaluation limit per basin
            'fatol': 1e-2,        # Function value tolerance for convergence
            'xatol': 1e-3         # Parameter space tolerance for grid convergence
        }
    }

    res = shgo(
        obj_func,
        bounds=obj_func.bounds,
        n=64,                      # Sampling points for complex simplicial homology space
        iters=3,                   # Homology construction depth
        minimizer_kwargs=minimizer_kwargs,
        options={'maxev': args.max_evals}
    )

    if monitor:
        plt.ioff()
        plt.show()

    print("\n" + "="*50)
    print(" SHGO + NELDER-MEAD OPTIMIZATION COMPLETED ")
    print("="*50)
    print(f"Total Simulations Executed: {obj_func.eval_count}")
    print(f"Best Objective Value Found: {obj_func.best_cost:.4f}")

    rounded_params = obj_func.best_vars if obj_func.best_vars else {}

    print("\nOptimal Parameters Found (Cleaned):")
    for k, v in rounded_params.items():
        print(f"  - {k}: {v}")

    # High-accuracy simulation criteria for final verification run
    HIGH_ACCURACY_SIM_SETTINGS = {
        'end_criteria': 1e-4,     # -40 dB energy decay for full convergence
        'max_timesteps': 1000000, # Extended timesteps for sharp resonances
        'n_pts': 1024,            # Higher frequency resolution
        'target_s11_db': -40.0
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
