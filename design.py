import os
import yaml
import math
import numpy as np

def evaluate_value(val, var_dict):
    """Recursively evaluates strings containing math expressions using var_dict."""
    if isinstance(val, str):
        # Allow standard math functions in string expressions
        allowed_names = {**var_dict, "math": math, "np": np, "abs": abs}
        try:
            # Round the evaluated float result to 6 decimal places (0.1 µm precision)
            res = float(eval(val, {"__builtins__": None}, allowed_names))
            return round(res, 6)
        except Exception:
            # If it's a plain string like a material name ('PEC_Patch'), return as-is
            return val
    elif isinstance(val, list):
        return [evaluate_value(item, var_dict) for item in val]
    elif isinstance(val, dict):
        return {k: evaluate_value(v, var_dict) for k, v in val.items()}
    return val

def load_config(config_path, active_vars=None):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Extract default variables if present in YAML
    var_dict = {}
    if 'variables' in cfg:
        for v_name, v_opts in cfg['variables'].items():
            if isinstance(v_opts, dict):
                var_dict[v_name] = v_opts.get('default', v_opts.get('start', 0.0))
            else:
                var_dict[v_name] = v_opts

    # Override defaults with active trial variables (from optimizer)
    if active_vars is not None:
        var_dict.update(active_vars)

    # Evaluate expressions if variables exist; otherwise return raw config
    if var_dict:
        return evaluate_value(cfg, var_dict)
    return cfg

def build_design(CSX, FDTD, config_file='./bowtie.yaml', active_vars=None, enable_nf2ff=True):
    cfg = load_config(config_file, active_vars=active_vars)

    unit = float(cfg.get('simulation', {}).get('unit', 1.0))
    fstart = float(cfg['simulation']['fstart']) * 1e9
    fstop  = float(cfg['simulation']['fstop'])  * 1e9
    
    C0 = 299792458.0

    lambda_min = C0 / fstop / unit   
    lambda_max = C0 / fstart / unit  

    dl_max = lambda_min / 20.0
    dl_min = lambda_min / 60.0
    air_padding = lambda_max / 4.0

    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    # 1. Parse Materials
    material_map = {}
    for mat in cfg.get('materials', []):
        if mat['type'] == 'metal':
            material_map[mat['name']] = CSX.AddMetal(mat['name'])

    # 2. Parse Primitives
    x_snaps, y_snaps, z_snaps = [], [], []

    for prim in cfg.get('primitives', []):
        mat_obj = material_map[prim['material']]
        p_type = prim['type']
        prio = prim.get('priority', 10)

        if p_type == 'polygon':
            pts = np.array([prim['points']['x'], prim['points']['y']])
            elev = float(prim['elevation'])
            length = float(prim['length'])
            
            mat_obj.AddLinPoly(
                priority=prio,
                points=pts,
                norm_dir=prim.get('norm_dir', 'z'),
                elevation=elev,
                length=length
            )

            x_snaps.extend(prim['points']['x'])
            y_snaps.extend(prim['points']['y'])
            z_snaps.extend([elev, elev + length])

        elif p_type == 'box':
            p_start = prim['start']
            p_stop = prim['stop']
            mat_obj.AddBox(priority=prio, start=p_start, stop=p_stop)
            
            x_snaps.extend([p_start[0], p_stop[0]])
            y_snaps.extend([p_start[1], p_stop[1]])
            z_snaps.extend([p_start[2], p_stop[2]])

    # 3. Parse Ports
    ports = []
    for p in cfg.get('ports', []):
        if p['type'] == 'lumped':
            port_obj = FDTD.AddLumpedPort(
                p['id'], p['resistance'], p['start'], p['stop'], p['direction'], excite=p['excite'], priority=5)
            ports.append(port_obj)

            px = [p['start'][0], p['stop'][0]]
            py = [p['start'][1], p['stop'][1]]
            pz = [p['start'][2], p['stop'][2]]

            x_snaps.extend([px[0] - dl_min, px[0], px[1], px[1] + dl_min])
            y_snaps.extend([py[0] - dl_min, py[0], py[1], py[1] + dl_min])
            z_snaps.extend([pz[0] - dl_min, pz[0], pz[1], pz[1] + dl_min])

    # 4. Mesh Generation
    x_min, x_max = min(x_snaps) - air_padding, max(x_snaps) + air_padding
    y_min, y_max = min(y_snaps) - air_padding, max(y_snaps) + air_padding
    z_min, z_max = min(z_snaps) - air_padding, max(z_snaps) + air_padding

    mesh.AddLine('x', np.unique(x_snaps))
    mesh.AddLine('y', np.unique(y_snaps))
    mesh.AddLine('z', np.unique(z_snaps))

    mesh.AddLine('x', [x_min, x_max])
    mesh.AddLine('y', [y_min, y_max])
    mesh.AddLine('z', [z_min, z_max])

    mesh.SmoothMeshLines('x', dl_max, ratio=1.3)
    mesh.SmoothMeshLines('y', dl_max, ratio=1.3)
    mesh.SmoothMeshLines('z', dl_max, ratio=1.3)

    # 5. Create NF2FF Recording Box
    nf2ff = None
    if enable_nf2ff:
        nf2ff = FDTD.CreateNF2FFBox()

    port_ret = ports[0] if len(ports) == 1 else ports
    return port_ret, nf2ff