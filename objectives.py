import numpy as np

def _get_frequency_bounds(freq, f_min=None, f_max=None):
    """Utility to auto-detect full simulation spectrum if f_min/f_max are None."""
    f_start = freq[0] if f_min is None else f_min
    f_stop = freq[-1] if f_max is None else f_max
    return f_start, f_stop

def maximize_bandwidth_in_band(results, f_min=None, f_max=None, target_db=-10.0):
    """
    Maximizes the percentage of the frequency spectrum where S11 < target_db.
    If f_min/f_max are None, evaluates across the entire simulated spectrum.
    """
    freq = results['freq']
    s11_db = results['s11_db']

    f_start, f_stop = _get_frequency_bounds(freq, f_min, f_max)
    mask = (freq >= f_start) & (freq <= f_stop)
    in_band_s11 = s11_db[mask]

    if len(in_band_s11) == 0:
        return 1e6

    # Fraction of points satisfying S11 < target_db (0.0 to 1.0)
    fraction_matched = np.mean(in_band_s11 < target_db)
    
    # We want to minimize cost, so cost decreases as fraction approaches 1.0
    mean_s11 = np.mean(in_band_s11)
    cost = (1.0 - fraction_matched) * 100.0 + mean_s11
    return float(cost)


def minimize_max_s11_in_band(results, f_min=None, f_max=None, target_db=-10.0):
    """
    Minimizes the maximum (worst-case) S11 value across the specified band.
    Evaluates across the full simulation spectrum if f_min/f_max are None.
    Adds a heavy penalty for any frequency points exceeding target_db (-10 dB).
    """
    freq = results['freq']
    s11_db = results['s11_db']

    f_start, f_stop = _get_frequency_bounds(freq, f_min, f_max)
    mask = (freq >= f_start) & (freq <= f_stop)
    in_band_s11 = s11_db[mask]

    if len(in_band_s11) == 0:
        return 1e6

    worst_s11 = np.max(in_band_s11)  # Peak (worst) S11 point
    mean_s11 = np.mean(in_band_s11)   # Average S11 across the spectrum

    # Heavy quadratic penalty for points above target threshold (-10 dB)
    violations = np.maximum(0, in_band_s11 - target_db)
    penalty = np.sum(violations**2)

    # Combined UWB cost: forces the entire spectrum below target_db
    cost = worst_s11 + 2.0 * penalty + 0.2 * mean_s11
    return float(cost)


def minimize_s11_in_band(sim_results, f_min=None, f_max=None):
    """
    Objective 1: Minimize peak |S11| (dB) within a target band.
    Evaluates across the full simulation spectrum if f_min/f_max are None.
    """
    freq = sim_results['freq']
    s11 = sim_results['s11']
    
    f_start, f_stop = _get_frequency_bounds(freq, f_min, f_max)
    band_mask = (freq >= f_start) & (freq <= f_stop)
    if not np.any(band_mask):
        return 1e6
    
    return float(np.mean(abs(s11[band_mask])) + np.std(abs(s11[band_mask])))


def maximize_impedance_smoothness(sim_results, f_min=None, f_max=None):
    """
    Objective 2: Maximizes smoothness of Re(Z_in) by minimizing the 
    variance of its frequency derivative d(Re(Z_in))/df.
    Evaluates across the full simulation spectrum if f_min/f_max are None.
    """
    freq = sim_results['freq']
    z_in = sim_results['z_in']
    
    f_start, f_stop = _get_frequency_bounds(freq, f_min, f_max)
    band_mask = (freq >= f_start) & (freq <= f_stop)
    # r_in = np.real(z_in[band_mask])
    absZin = np.abs(z_in[band_mask])
    f_band = freq[band_mask]
    
    if len(f_band) < 2:
        return 1e6

    # Derivative wrt frequency
    # dr_df = np.gradient(r_in)
    
    # Smoothness cost = variance of derivative
    return int(np.std(absZin))


def multi_objective_combined(sim_results, f_min=None, f_max=None):
    """
    Combined Objective: 70% S11 minimization + 30% Impedance smoothness.
    Evaluates dynamically across the full spectrum if f_min/f_max are None.
    """
    cost_s11 = minimize_s11_in_band(sim_results, f_min=f_min, f_max=f_max)
    cost_smooth = maximize_impedance_smoothness(sim_results, f_min=f_min, f_max=f_max)
    
    # Normalize and weight
    return 1.0 * cost_s11 + 0.0053 * float(cost_smooth)