import numpy as np
import pandas as pd


# write the code, but it should also work for a single time point. 
def IVRMSE(pred, real):

    # Sum over the grid,
    # sum over all predictions over time


    element = np.power(real-pred,2)


import numpy as np

def ivrmse_h(sigma_actual, sigma_predicted, t0, T, h, nt):
    """
    Calculate IVRMSE_h as defined in the formula.
    
    Parameters:
    - sigma_actual: 2D array (T rows, n_t columns) of actual volatilities.
    - sigma_predicted: 2D array (T rows, n_t columns) of predicted volatilities.
    - t0: Starting time index.
    - T: Total time index (exclusive).
    - h: Horizon offset.
    - nt: List of n_t values (number of points at each t).
    
    Returns:
    - IVRMSE_h: Float, the calculated IVRMSE_h value.
    """
    # Validate input lengths
    assert len(nt) == sigma_actual.shape[0] == sigma_predicted.shape[0], "nt and sigma arrays must match in length."
    
    # Initialize numerator and denominator
    numerator = 0
    denominator = 0
    
    # Loop over t from t0 to T-h
    for t in range(t0, T - h):
        n_t = nt[t]  # Number of data points at time t
        denominator += n_t
        errors = sigma_actual[t + h, :n_t] - sigma_predicted[t + h, :n_t]
        numerator += np.sum(errors**2)
    
    # Calculate IVRMSE_h
    ivrmse_h = np.sqrt(numerator / denominator)
    return ivrmse_h

import numpy as np

def r2_oos_h(sigma_actual, sigma_predicted, t0, T, h, nt):
    """
    Calculate R_{oos,h}^2 as defined in the formula.
    
    Parameters:
    - sigma_actual: 2D array (T rows, n_t columns) of actual volatilities.
    - sigma_predicted: 2D array (T rows, n_t columns) of predicted volatilities.
    - t0: Starting time index.
    - T: Total time index (exclusive).
    - h: Horizon offset.
    - nt: List of n_t values (number of points at each t).
    
    Returns:
    - R2_oos_h: Float, the calculated R_{oos,h}^2 value.
    """
    # Validate input lengths
    assert len(nt) == sigma_actual.shape[0] == sigma_predicted.shape[0], "nt and sigma arrays must match in length."
    
    numerator = 0  # Sum of squared prediction errors
    denominator = 0  # Sum of squared deviations from the mean
    
    # Loop over t from t0 to T-h
    for t in range(t0, T - h):
        n_t = nt[t]  # Number of data points at time t
        sigma_t_h = sigma_actual[t + h, :n_t]
        sigma_pred_t_h = sigma_predicted[t + h, :n_t]
        sigma_mean_t_h = np.mean(sigma_t_h)  # Mean of actual volatilities at t+h
        
        # Update numerator and denominator
        numerator += np.sum((sigma_t_h - sigma_pred_t_h) ** 2)
        denominator += np.sum((sigma_t_h - sigma_mean_t_h) ** 2)
    
    # Calculate R2_oos_h
    R2_oos_h = 1 - (numerator / denominator)
    return R2_oos_h


def r_oos(pred, real):
    return np.power(pred-real,2)