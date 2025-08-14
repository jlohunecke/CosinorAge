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

from typing import List

import numpy as np
import pandas as pd


def IS(data: pd.Series) -> float:
    r"""Calculate the interdaily stability (IS) for the entire dataset.

    Interdaily stability quantifies the strength of coupling between the
    rest-activity rhythm and environmental zeitgebers. It compares the
    24-hour pattern across days.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column. Should contain multiple days of minute-level data.

    Returns
    -------
    float
        Interdaily stability value ranging from 0 to 1, where:
        - 0 indicates no stability (random activity patterns)
        - 1 indicates perfect stability (identical daily patterns)
        Returns np.nan if insufficient data or calculation fails.

    Notes
    -----
    - Resamples data to hourly resolution for calculation
    - IS = (D * sum((hourly_means - overall_mean)²)) / sum((all_values - overall_mean)²)
    - Higher values indicate more consistent daily activity patterns
    - Requires multiple days of data for meaningful calculation
    - Used in circadian rhythm analysis to assess rhythm stability

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample multi-day activity data
    >>> dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    >>> # Simulate consistent daily pattern
    >>> hours = dates.hour
    >>> enmo = pd.Series(np.sin(hours * np.pi / 12) + 1 + np.random.normal(0, 0.1, 4320), index=dates)
    >>>
    >>> # Calculate interdaily stability
    >>> is_value = IS(enmo)
    >>> print(f"Interdaily Stability: {is_value:.3f}")
    >>> # Higher values indicate more consistent daily patterns
    """
    if len(data) == 0:
        return np.nan

    data_ = data.copy()[["enmo"]]
    data_ = data_.resample("h").mean()
    data_["hour"] = data_.index.hour

    # Calculate key values
    H = 24  # Hours per day
    D = len(pd.unique(data_.index.date))  # Number of days
    z_mean = data_["enmo"].mean()  # Overall mean

    # Calculate hourly means across days
    hourly_means = data_.groupby("hour")["enmo"].mean()

    # Calculate numerator
    numerator = D * np.sum(np.power(hourly_means - z_mean, 2), axis=0)

    # Calculate denominator
    denominator = np.sum(np.power(data_["enmo"] - z_mean, 2), axis=0)

    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.nan

    IS = float(numerator / denominator)

    return IS


def IV(data: pd.Series) -> float:
    r"""Calculate the intradaily variability (IV) for the entire dataset.

    Intradaily variability quantifies the fragmentation of rest-activity patterns
    within each 24-hour period. It is calculated as the ratio of the mean squared
    first derivative to the variance.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column. Should contain multiple days of minute-level data.

    Returns
    -------
    float
        Intradaily variability value, where:
        - Lower values indicate less fragmented activity patterns
        - Higher values indicate more fragmented activity patterns
        Returns np.nan if insufficient data or calculation fails.

    Notes
    -----
    - Resamples data to hourly resolution for calculation
    - IV = (P * sum((z_p - z_{p-1})²)) / ((P-1) * sum((z_p - z_mean)²))
    - Lower values indicate more consolidated rest-activity periods
    - Higher values indicate more fragmented sleep and activity patterns
    - Used in circadian rhythm analysis to assess rhythm fragmentation

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample multi-day activity data
    >>> dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    >>> # Simulate fragmented activity pattern
    >>> hours = dates.hour
    >>> enmo = pd.Series(np.random.uniform(0, 1, 4320), index=dates)  # Random activity
    >>>
    >>> # Calculate intradaily variability
    >>> iv_value = IV(enmo)
    >>> print(f"Intradaily Variability: {iv_value:.3f}")
    >>> # Higher values indicate more fragmented activity patterns
    """
    if len(data) == 0:
        return np.nan

    data_ = data.copy()[["enmo"]]
    P = len(data_)

    # resample to hourly data
    data_ = data_.resample("h").mean()

    # Calculate numerator: P * sum((z_p - z_{p-1})^2)
    first_derivative_squared = np.sum(
        np.power(
            data_[1:].reset_index(drop=True)
            - data_[:-1].reset_index(drop=True),
            2,
        ),
        axis=0,
    )
    numerator = float(P * first_derivative_squared.iloc[0])

    # Calculate denominator: (P-1) * sum((z_p - z_mean)^2)
    deviations_squared = np.sum(np.power(data_ - data_.mean(), 2), axis=0)
    denominator = float((P - 1) * deviations_squared.iloc[0])

    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.nan

    IV = numerator / denominator

    return IV


def M10(data: pd.Series) -> List[float]:
    r"""Calculate the M10 (mean activity during the 10 most active hours)
    and the start time of the 10 most active hours (M10_start) for each day.

    M10 provides information about the most active period during each day,
    which typically corresponds to the main activity phase.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column. Should contain minute-level data for multiple days.

    Returns
    -------
    tuple
        Tuple containing two lists:
        - m10: List of mean activity values during the 10 most active hours for each day
        - m10_start: List of start times (datetime) of the 10 most active hours for each day
        Returns empty lists if insufficient data.

    Notes
    -----
    - Uses rolling 10-hour windows (600 minutes) to find the most active period
    - Calculates mean activity within each window and finds the maximum
    - Returns both the activity level and start time of the most active period
    - Used in circadian rhythm analysis to identify the main activity phase
    - Typically corresponds to daytime activity periods

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample multi-day activity data
    >>> dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    >>> # Simulate activity with peak during day
    >>> hours = dates.hour
    >>> enmo = pd.Series(np.sin(hours * np.pi / 12) + 1 + np.random.normal(0, 0.1, 4320), index=dates)
    >>>
    >>> # Calculate M10 for each day
    >>> m10_values, m10_starts = M10(enmo)
    >>> print(f"M10 values: {m10_values}")
    >>> print(f"M10 start times: {m10_starts}")
    """
    if len(data) == 0:
        return [], []

    data_ = data.copy()[["enmo"]]
    daily_groups = data_.groupby(data_.index.date)

    m10 = []
    m10_start = []
    for date, day_data in daily_groups:
        # calculate the rolling mean over 10-hour windows
        window_size = 600  # 10 hours * 60 minutes
        rolling_means = (
            day_data[::-1]
            .rolling(window=window_size, center=False)
            .mean()[::-1]
            .dropna()
        )

        # Find the window with maximum activity
        max_mean = float(rolling_means.max().iloc[0])
        max_start_idx = rolling_means.idxmax().iloc[0]

        if pd.isna(max_mean) or np.isnan(max_mean) or np.isinf(max_mean):
            m10.append(np.nan)
            m10_start.append(np.nan)
        else:
            m10.append(max_mean)
            m10_start.append(max_start_idx)

    return m10, m10_start


def L5(data: pd.Series) -> List[float]:
    r"""Calculate the L5 (mean activity during the 5 least active hours)
    and the start time of the 5 least active hours (L5_start) for each day.

    L5 provides information about the least active period during each day,
    which typically corresponds to the main rest phase.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column. Should contain minute-level data for multiple days.

    Returns
    -------
    tuple
        Tuple containing two lists:
        - l5: List of mean activity values during the 5 least active hours for each day
        - l5_start: List of start times (datetime) of the 5 least active hours for each day
        Returns empty lists if insufficient data.

    Notes
    -----
    - Uses rolling 5-hour windows (300 minutes) to find the least active period
    - Calculates mean activity within each window and finds the minimum
    - Returns both the activity level and start time of the least active period
    - Used in circadian rhythm analysis to identify the main rest phase
    - Typically corresponds to nighttime sleep periods

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample multi-day activity data
    >>> dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    >>> # Simulate activity with low during night
    >>> hours = dates.hour
    >>> enmo = pd.Series(np.sin(hours * np.pi / 12) + 1 + np.random.normal(0, 0.1, 4320), index=dates)
    >>>
    >>> # Calculate L5 for each day
    >>> l5_values, l5_starts = L5(enmo)
    >>> print(f"L5 values: {l5_values}")
    >>> print(f"L5 start times: {l5_starts}")
    """
    if len(data) == 0:
        return [], []

    data_ = data.copy()[["enmo"]]
    daily_groups = data_.groupby(data_.index.date)

    l5 = []
    l5_start = []
    for date, day_data in daily_groups:
        # calculate the rolling mean over 5-hour windows
        window_size = 300  # 5 hours * 60 minutes
        rolling_means = (
            day_data[::-1]
            .rolling(window=window_size, center=False)
            .mean()[::-1]
            .dropna()
        )

        # Find the window with minimum activity
        min_mean = float(rolling_means.min().iloc[0])
        min_start_idx = rolling_means.idxmin().iloc[0]

        if pd.isna(min_mean) or np.isnan(min_mean) or np.isinf(min_mean):
            l5.append(np.nan)
            l5_start.append(np.nan)
        else:
            l5.append(min_mean)
            l5_start.append(min_start_idx)

    return l5, l5_start


def RA(m10: List[float], l5: List[float]) -> List[float]:
    r"""Calculate the relative amplitude (RA) for each day.

    Relative amplitude is calculated as the difference between the most active
    10-hour period and least active 5-hour period, divided by their sum.
    This provides a normalized measure of the daily activity rhythm strength.

    Parameters
    ----------
    m10 : List[float]
        List of M10 values (mean activity during 10 most active hours) for each day.
        Should be output from the M10() function.
    l5 : List[float]
        List of L5 values (mean activity during 5 least active hours) for each day.
        Should be output from the L5() function.

    Returns
    -------
    List[float]
        List of relative amplitude values for each day, where:
        - Values range from 0 to 1
        - Higher values indicate stronger daily activity rhythms
        - Lower values indicate weaker daily activity rhythms
        Returns empty list if input lists are empty.

    Notes
    -----
    - RA = (M10 - L5) / (M10 + L5)
    - Normalized measure that accounts for overall activity level
    - Higher values indicate more pronounced rest-activity cycles
    - Used in circadian rhythm analysis to assess rhythm strength
    - Requires both M10 and L5 values from the same dataset

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample multi-day activity data
    >>> dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    >>> hours = dates.hour
    >>> enmo = pd.Series(np.sin(hours * np.pi / 12) + 1 + np.random.normal(0, 0.1, 4320), index=dates)
    >>>
    >>> # Calculate M10 and L5 first
    >>> m10_values, m10_starts = M10(enmo)
    >>> l5_values, l5_starts = L5(enmo)
    >>>
    >>> # Calculate relative amplitude
    >>> ra_values = RA(m10_values, l5_values)
    >>> print(f"Relative Amplitude values: {ra_values}")
    >>> # Higher values indicate stronger daily activity rhythms
    """
    if len(m10) == 0 or len(l5) == 0:
        return []

    if len(m10) != len(l5):
        raise ValueError("m10 and l5 must have the same length")

    ra = []
    for i in range(len(m10)):
        denominator = m10[i] + l5[i]
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            ra.append(np.nan)
        else:
            ra.append((m10[i] - l5[i]) / denominator)

    return ra
