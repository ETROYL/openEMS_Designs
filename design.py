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

def get_simulation_settings(cfg, profile="simulation", overrides=None):
    """Return independent solver settings for standalone or optimization runs."""
    if profile not in ("simulation", "optimization"):
        raise ValueError("profile must be 'simulation' or 'optimization'")

    settings = dict(cfg.get("simulation") or {})
    evaluation = {}
    if profile == "optimization":
        optimization = cfg.get("optimization") or {}
        evaluation = optimization.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            raise ValueError("optimization.evaluation settings must be a mapping")
        settings.update(evaluation)

    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("simulation setting overrides must be a mapping")
        settings.update(overrides)

    # Preserve the YAML fallback exactly when no optimization-specific
    # settings are supplied. Otherwise convert dB unless an explicit
    # legacy linear EndCriteria was provided by the evaluation/GUI layer.
    has_optimization_settings = profile != "optimization" or bool(evaluation)
    explicit_linear_override = (
        "end_criteria" in evaluation or
        (overrides is not None and "end_criteria" in overrides)
    )
    if (
        has_optimization_settings
        and "end_criteria_db" in settings
        and not explicit_linear_override
    ):
        decay_db = float(settings["end_criteria_db"])
        if not math.isfinite(decay_db) or decay_db >= 0:
            raise ValueError("end_criteria_db must be a finite negative dB value")
        settings["end_criteria"] = 10.0 ** (decay_db / 10.0)

    return settings


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

NF2FF_MIN_GRID_LINES = 21


def _add_nf2ff_mesh_support_lines(mesh, bounds, minimum_lines=NF2FF_MIN_GRID_LINES):
    """Add NF2FF support lines by bisecting the largest existing mesh gaps.

    Existing geometry and port coordinates are retained. Unlike uniform
    placement across the whole domain, largest-gap subdivision avoids
    introducing support coordinates almost coincident with existing lines.
    """
    if minimum_lines < 21:
        raise ValueError("NF2FF support requires at least 21 grid lines per axis.")

    for axis, (lower, upper) in zip(("x", "y", "z"), bounds):
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"Invalid {axis}-axis mesh bounds for NF2FF support.")

        lines = np.unique(np.asarray(mesh.GetLines(axis), dtype=float).reshape(-1))
        # Ensure the requested bounds are actual mesh coordinates, not only
        # temporary entries in the subdivision calculation.
        missing_bounds = [
            value for value in (lower, upper)
            if not np.any(lines == value)
        ]
        if missing_bounds:
            mesh.AddLine(axis, missing_bounds)
            lines = np.unique(
                np.asarray(mesh.GetLines(axis), dtype=float).reshape(-1)
            )

        while lines.size < minimum_lines:
            gaps = np.diff(lines)
            if gaps.size == 0 or not np.any(gaps > 0):
                raise RuntimeError(
                    f"Cannot construct NF2FF support lines on the {axis}-axis."
                )
            gap_index = int(np.argmax(gaps))
            midpoint = lines[gap_index] + 0.5 * gaps[gap_index]
            if midpoint <= lines[gap_index] or midpoint >= lines[gap_index + 1]:
                raise RuntimeError(
                    f"Insufficient numerical resolution for NF2FF support "
                    f"on the {axis}-axis."
                )
            mesh.AddLine(axis, [midpoint])
            lines = np.insert(lines, gap_index + 1, midpoint)


def build_design(
    CSX, FDTD, config_file='./bowtie.yaml', active_vars=None,
    enable_nf2ff=True, simulation_profile="simulation", simulation_settings_override=None,
):
    cfg = load_config(config_file, active_vars=active_vars)
    simulation = get_simulation_settings(cfg, simulation_profile, overrides=simulation_settings_override)

    unit = float(cfg.get('simulation', {}).get('unit', 1.0))
    fstart = float(simulation['fstart']) * 1e9
    fstop  = float(simulation['fstop'])  * 1e9
    
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

        elif p_type == 'cylinder':
            # start / stop = centers of the two end faces, radius in design units.
            # A flat disk is simply a very short cylinder (e.g. 1 mm thick along z).
            p_start = [float(v) for v in prim['start']]
            p_stop  = [float(v) for v in prim['stop']]
            r = float(prim['radius'])

            mat_obj.AddCylinder(priority=prio, start=p_start, stop=p_stop, radius=r)

            # Mesh snap lines: along the axis use the end faces,
            # in the two perpendicular directions use center +/- radius.
            diff = [abs(p_stop[i] - p_start[i]) for i in range(3)]
            axis = diff.index(max(diff))   # 0=x, 1=y, 2=z
            snaps = (x_snaps, y_snaps, z_snaps)
            for i in range(3):
                if i == axis:
                    snaps[i].extend([p_start[i], p_stop[i]])
                else:
                    snaps[i].extend([p_start[i] - r, p_start[i] + r])

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

    # openEMS' automatic NF2FF placement excludes the PML cells at both
    # ends of each axis. A thin antenna dimension (usually z) can otherwise
    # contain too few grid lines in higher-frequency sub-bands.
    if enable_nf2ff:
        _add_nf2ff_mesh_support_lines(
            mesh, ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        )

    mesh.SmoothMeshLines('x', dl_max, ratio=1.3)
    mesh.SmoothMeshLines('y', dl_max, ratio=1.3)
    mesh.SmoothMeshLines('z', dl_max, ratio=1.3)

    # 5. Create NF2FF Recording Box
    nf2ff = None
    if enable_nf2ff:
        line_counts = {
            axis: len(np.asarray(mesh.GetLines(axis)).reshape(-1))
            for axis in ("x", "y", "z")
        }
        insufficient = {
            axis: count for axis, count in line_counts.items()
            if count < NF2FF_MIN_GRID_LINES
        }
        if insufficient:
            raise RuntimeError(
                "Mesh does not provide enough lines for the NF2FF box "
                f"with PML_8 boundaries: {insufficient}; "
                f"at least {NF2FF_MIN_GRID_LINES} lines are required per axis."
            )
        nf2ff = FDTD.CreateNF2FFBox()

    port_ret = ports[0] if len(ports) == 1 else ports
    return port_ret, nf2ff
