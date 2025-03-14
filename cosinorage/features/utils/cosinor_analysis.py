###########################################################################
# Copyright (C) 2024 ETH Zurich
# CosinorAge: Prediction of biological age based on accelerometer data
# using the CosinorAge method proposed by Shim, Fleisch and Barata
# (https://www.nature.com/articles/s41746-024-01111-x)
# 
# Authors: Jacob Leo Oskar Hunecke
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from scipy import optimize


def cosinor_multiday(df: pd.DataFrame) -> pd.DataFrame:
    """
    A parametric approach to study circadian rhythmicity assuming cosinor shape, fitting a model for multiple days.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a Timestamp index and a column 'ENMO' containing minute-level activity data.

    Returns:
    --------
    tuple:
        - dict: Dictionary containing cosinor parameters:
            - MESOR: Midline Estimating Statistic Of Rhythm (rhythm-adjusted mean)
            - amplitude: Half the difference between maximum and minimum values
            - acrophase: Time of peak relative to midnight in radians
            - acrophase_time: Time of peak in hours (0-24)
        - pandas.Series: Fitted values for each timepoint

    Raises:
    -------
    ValueError:
        If DataFrame doesn't have required 'ENMO' column or timestamp index
        If data length is not a multiple of 1440 (minutes in a day)
    """
    # Ensure the DataFrame contains the required columns
    if 'ENMO' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame must have a Timestamp index and an 'ENMO' column.")

    # Ensure the data length is consistent
    total_minutes = len(df)
    dim = 1440  # Number of data points in a day
    if total_minutes % dim != 0:
        raise ValueError("Data length is not a multiple of a day (1440 minutes or adjusted for the window size).")

    time_minutes = np.arange(1, total_minutes + 1)
    df['time'] = time_minutes 

    results = fit_cosinor(df['time'], df['ENMO'], period=1440)
    
    mesor = results['MESOR']
    amplitude = results['amplitude']
    acrophase = results['acrophase']
    fitted_vals_df = results['fitted_values']

    # shifted by 2pi to make it match the gt cosinorage predictions 
    if acrophase > 0:
        acrophase -= 2*np.pi

    acrophase_time = float(-(acrophase+2*np.pi)/(2*np.pi)*24)+24
    
    """
    # Add cosine and sine components
    df['cos'] = np.cos(2 * np.pi * df['time'] / 1440)
    df['sin'] = np.sin(2 * np.pi * df['time'] / 1440)
    
    # Fit cosinor model
    model = ols("ENMO ~ cos + sin", data=df).fit()
    
    # Extract parameters
    mesor = float(model.params['Intercept'])
    beta_cos = model.params['cos']
    beta_sin = model.params['sin']
    amplitude = float(np.sqrt(beta_cos**2 + beta_sin**2))
    acrophase = float(np.arctan2(beta_sin, beta_cos))
    acrophase_time = float(acrophase/(2*np.pi)*24)
    fitted_vals_df = model.fittedvalues
    """

    acrophase_time *= 60

    # Convert the results into a DataFrame
    return {'mesor': mesor, 'amplitude': amplitude, 'acrophase': acrophase, 'acrophase_time': acrophase_time}, fitted_vals_df


def cosinor_model(t, M, A, phi, tau):
    """
    Cosinor model function with counterclockwise acrophase.
    
    Parameters:
    t : array-like
        Time points
    M : float
        MESOR (Midline Statistic of Rhythm)
    A : float
        Amplitude (always positive)
    phi : float
        Acrophase in radians (counterclockwise)
    tau : float
        Period
        
    Returns:
    array-like: Fitted values
    """
    # Note the negative sign before phi for counterclockwise orientation
    return M + A * np.cos(2 * np.pi * t / tau + phi)

def fit_cosinor(time, data, period=24):
    """
    Fit cosinor model to time series data.
    
    Parameters:
    time : array-like
        Time points
    data : array-like
        Observed values
    period : float, optional
        Known period (default=24)
        
    Returns:
    dict: Dictionary containing fitted parameters and statistics
    """
    # Initial parameter guesses
    M_guess = np.mean(data)
    A_guess = np.abs((np.max(data) - np.min(data)) / 2)
    phi_guess = 0
    
    # Define the residual function for optimization
    def residuals(params):
        M, A_signed, phi = params
        # Force amplitude to be positive by taking absolute value
        A = np.abs(A_signed)
        return data - cosinor_model(time, M, A, phi, period)
    
    # Perform least squares optimization
    optimal_params, _ = optimize.leastsq(
        residuals, 
        [M_guess, A_guess, phi_guess]
    )
    
    # Extract fitted parameters
    M_fit, A_signed, phi_fit = optimal_params
    
    # Ensure amplitude is positive
    A_fit = np.abs(A_signed)
    
    # Keep acrophase in radians (can be negative)
    # No normalization to positive values
    
    # Calculate fitted values
    fitted = cosinor_model(time, M_fit, A_fit, phi_fit, period)
    
    # Calculate R-squared
    ss_tot = np.sum((data - np.mean(data))**2)
    ss_res = np.sum((data - fitted)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    results = {
        'MESOR': M_fit,
        'amplitude': A_fit,
        'acrophase': phi_fit,  # in radians, can be negative
        'period': period,
        'r_squared': r_squared,
        'fitted_values': fitted
    }
    
    return results