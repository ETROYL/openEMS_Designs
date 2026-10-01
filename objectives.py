import numpy as np

def minimize_s11_in_band(sim_results, f_min=0.3e9, f_max=0.6e9):
    """Objective 1: Minimize peak |S11| (dB) within a target band."""
    freq = sim_results['freq']
    s11_db = sim_results['s11_db']
    
    band_mask = (freq >= f_min) & (freq <= f_max)
    if not np.any(band_mask):
        return 0.0
    
    # Return peak S11 in dB (lower is better)
    return float(np.max(s11_db[band_mask]))

def maximize_impedance_smoothness(sim_results, f_min=0.3e9, f_max=0.8e9):
    """
    Objective 2: Maximizes smoothness of Re(Z_in) by minimizing the 
    variance of its frequency derivative d(Re(Z_in))/df.
    """
    freq = sim_results['freq']
    z_in = sim_results['z_in']
    
    band_mask = (freq >= f_min) & (freq <= f_max)
    r_in = np.real(z_in[band_mask])
    f_band = freq[band_mask]
    
    # Derivative wrt frequency
    dr_df = np.gradient(r_in, f_band)
    
    # Smoothness cost = variance of derivative
    return float(np.var(dr_df))

def multi_objective_combined(sim_results):
    """
    Combined Objective: 70% S11 minimization + 30% Impedance smoothness.
    """
    cost_s11 = minimize_s11_in_band(sim_results, 0.3e9, 0.6e9)
    cost_smooth = maximize_impedance_smoothness(sim_results, 0.3e9, 0.6e9)
    
    # Normalize and weight
    return 0.7 * cost_s11 + 0.3 * (cost_smooth * 1e6) 
