###########################################################################
# Copyright (C) 2025 ETH Zurich
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

import numpy as np
import pandas as pd
from CosinorPy import cosinor1


def cosinor_multiday(df: pd.DataFrame) -> pd.DataFrame:
    """
    A parametric approach to study circadian rhythmicity assuming cosinor shape, fitting a model for multiple days.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a Timestamp index and a column 'ENMO' containing minute-level activity data.

    Returns
    -------
    tuple
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
    if "enmo" not in df.columns or not pd.api.types.is_datetime64_any_dtype(
        df.index
    ):
        raise ValueError(
            "The DataFrame must have a Timestamp index and an 'ENMO' column."
        )

    # Ensure the data length is consistent
    total_minutes = len(df)
    dim = 1440  # Number of data points in a day
    if total_minutes % dim != 0:
        raise ValueError(
            "Data length is not a multiple of a day (1440 minutes or adjusted for the window size)."
        )

    time_minutes = np.arange(1, total_minutes + 1)
    df["time"] = time_minutes

    results = fit_cosinor(df["time"], df["enmo"], period=1440)

    mesor = results["MESOR"]
    amplitude = results["amplitude"]
    acrophase = results["acrophase"]
    fitted_vals_df = results["fitted_values"]

    # shifted by 2pi to make it match the gt cosinorage predictions
    if acrophase > 0:
        acrophase -= 2 * np.pi

    # Check for invalid acrophase values
    if np.isnan(acrophase) or np.isinf(acrophase):
        acrophase_time = np.nan
    else:
        acrophase_time = float(-(acrophase + 2 * np.pi) / (2 * np.pi) * 24) + 24

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
    return {
        "mesor": mesor,
        "amplitude": amplitude,
        "acrophase": acrophase,
        "acrophase_time": acrophase_time,
    }, fitted_vals_df


def cosinor_model(t, M, A, phi, tau):
    """
    Cosinor model function with counterclockwise acrophase.

    This function implements the standard cosinor model for fitting periodic data.
    The model assumes a cosine function with adjustable amplitude, phase, and period.

    Parameters
    ----------
    t : array-like
        Time points at which to evaluate the model.
    M : float
        MESOR (Midline Statistic of Rhythm) - the rhythm-adjusted mean.
    A : float
        Amplitude - half the peak-to-trough difference (always positive).
    phi : float
        Acrophase in radians (counterclockwise orientation).
    tau : float
        Period of the rhythm in the same units as t.

    Returns
    -------
    array-like
        Fitted values at the specified time points.

    Notes
    -----
    - The model uses counterclockwise acrophase orientation (negative sign before phi)
    - The function implements: M + A * cos(2π * t / τ + φ)
    - Amplitude A should always be positive
    - The acrophase φ represents the time of peak activity

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> # Create time points (24 hours in minutes)
    >>> t = np.arange(0, 1440, 1)  # 0 to 1440 minutes
    >>>
    >>> # Define cosinor parameters
    >>> M = 0.5    # MESOR
    >>> A = 0.3    # Amplitude
    >>> phi = 0    # Acrophase (peak at midnight)
    >>> tau = 1440 # Period (24 hours in minutes)
    >>>
    >>> # Generate fitted values
    >>> fitted = cosinor_model(t, M, A, phi, tau)
    >>> print(f"Peak value: {fitted.max():.3f}")
    >>> print(f"Trough value: {fitted.min():.3f}")
    """
    # Note the negative sign before phi for counterclockwise orientation
    return M + A * np.cos(2 * np.pi * t / tau + phi)


def fit_cosinor(time, data, period=24):
    """
    Fit cosinor model to time series data.

    This function fits a cosinor model to time series data using the CosinorPy library.
    It estimates the MESOR, amplitude, and acrophase parameters that best describe
    the periodic pattern in the data.

    Parameters
    ----------
    time : array-like
        Time points corresponding to the data values.
    data : array-like
        Observed values to fit the cosinor model to.
    period : float, default=24
        Known period of the rhythm in the same units as time.
        For circadian rhythms, typically 24 hours or 1440 minutes.

    Returns
    -------
    dict
        Dictionary containing fitted parameters and statistics:
        - 'MESOR': Midline Estimating Statistic Of Rhythm (rhythm-adjusted mean)
        - 'amplitude': Half the peak-to-trough difference
        - 'acrophase': Time of peak relative to the period in radians
        - 'fitted_values': Array of fitted values at each time point

    Notes
    -----
    - Uses CosinorPy.cosinor1.fit_cosinor for the core fitting algorithm
    - The period is converted to minutes (period * 60) for internal processing
    - The function returns both the fitted parameters and the fitted values
    - The acrophase is returned in radians and represents the timing of peak activity

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # Create sample circadian data
    >>> time = np.arange(0, 1440, 1)  # 24 hours in minutes
    >>> data = 0.5 + 0.3 * np.cos(2 * np.pi * time / 1440 + np.pi/4) + 0.1 * np.random.randn(1440)
    >>>
    >>> # Fit cosinor model
    >>> results = fit_cosinor(time, data, period=24)
    >>> print(f"MESOR: {results['MESOR']:.3f}")
    >>> print(f"Amplitude: {results['amplitude']:.3f}")
    >>> print(f"Acrophase: {results['acrophase']:.3f} radians")
    """

    fit, _, _, statistics = cosinor1.fit_cosinor(
        time, data, period=24 * 60, plot_on=False
    )

    results = {
        "MESOR": statistics["values"][0],
        "amplitude": statistics["values"][1],
        "acrophase": statistics["values"][2],
        "fitted_values": fit.fittedvalues,
    }

    return results
