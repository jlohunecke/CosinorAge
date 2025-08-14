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

from itertools import tee
from typing import List

import numpy as np
import pandas as pd
from skdh.sleep.endpoints import (NumberWakeBouts, PercentTimeAsleep,
                                  SleepOnsetLatency, TotalSleepTime,
                                  WakeAfterSleepOnset)
from skdh.sleep.sleep_classification import compute_sleep_predictions


def apply_sleep_wake_predictions(
    data: pd.DataFrame, sleep_params: dict
) -> pd.DataFrame:
    """
    Apply sleep-wake prediction to accelerometer data using ENMO values.

    This function uses machine learning algorithms to classify each minute as either
    sleep or wake based on the activity level (ENMO values). The prediction is based
    on the assumption that lower activity levels correspond to sleep periods.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ENMO values in a column named 'ENMO'.
        Should have a datetime index with minute-level resolution.
    sleep_params : dict
        Dictionary containing sleep prediction parameters:
        - 'sleep_ck_sf': Sampling frequency for sleep classification (default: 0.0025)
        - 'sleep_rescore': Whether to rescore sleep predictions (default: True)

    Returns
    -------
    pd.Series
        Series containing sleep predictions with the same index as input data:
        - 1 = sleep
        - 0 = wake

    Raises
    ------
    ValueError
        If 'ENMO' column is not found in DataFrame.

    Notes
    -----
    - Uses skdh.sleep.sleep_classification.compute_sleep_predictions for the core algorithm
    - The function adds a 'sleep' column to the input DataFrame
    - Sleep predictions are based on activity patterns and circadian rhythms
    - The algorithm is trained on large datasets of polysomnography-validated sleep

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample ENMO data
    >>> timestamps = pd.date_range('2023-01-01', periods=1440, freq='min')  # One day
    >>> data = pd.DataFrame({
    ...     'ENMO': np.random.uniform(0, 0.1, 1440)  # Random activity levels
    ... }, index=timestamps)
    >>>
    >>> # Apply sleep-wake predictions
    >>> sleep_params = {'sleep_ck_sf': 0.0025, 'sleep_rescore': True}
    >>> sleep_predictions = apply_sleep_wake_predictions(data, sleep_params)
    >>> print(f"Sleep time: {sleep_predictions.sum()} minutes")
    """
    if "enmo" not in data.columns:
        raise ValueError(f"Column ENMO not found in the DataFrame.")

    data_ = data.copy()
    # make sf higher
    sf = sleep_params.get("sleep_ck_sf", 0.0025)
    rescore = sleep_params.get("sleep_rescore", True)

    result = compute_sleep_predictions(data_["enmo"], sf=sf, rescore=rescore)
    data_["sleep"] = pd.DataFrame(result, columns=["sleep"]).set_index(
        data_.index
    )["sleep"]

    return data_["sleep"]


def WASO(data: pd.DataFrame) -> List[int]:
    """
    Calculate Wake After Sleep Onset (WASO) for each 24-hour cycle.

    WASO represents the total time spent awake after the first sleep onset
    until the final wake time. It's a key metric for sleep quality assessment.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (0=sleep, 1=wake)

    Returns
    -------
    List[int]
        List containing WASO values in minutes for each 24-hour cycle.
        Returns 0 for days where no sleep is detected.

    Notes
    -----
    - Processes data in 24-hour cycles starting at midnight
    - Uses the WakeAfterSleepOnset class from sleep_metrics library
    - Higher WASO values indicate more fragmented sleep
    - Important metric for assessing sleep maintenance
    - Zero is returned for days where no sleep is detected

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample sleep data for one day
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> # Simulate sleep pattern with some wake periods
    >>> sleep_pattern = [0] * 480 + [1] * 30 + [0] * 60 + [1] * 20 + [0] * 810  # Sleep with wake periods
    >>> data = pd.DataFrame({'sleep': sleep_pattern}, index=dates)
    >>>
    >>> # Calculate WASO
    >>> waso_values = WASO(data)
    >>> print(f"Wake After Sleep Onset: {waso_values[0]} minutes")
    """
    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    waso = []
    w = WakeAfterSleepOnset()

    # Group by 24-hour cycle
    for date, day_data in daily_groups:
        # Sort by timestamp within the group
        day_data = day_data.sort_index()
        pred = w.predict(np.array(day_data["sleep"]))
        if pd.isna(pred):
            waso.append(0)
        else:
            waso.append(int(pred))

    return waso


def TST(data: pd.DataFrame) -> List[int]:
    """
    Calculate Total Sleep Time (TST) for each 24-hour cycle.

    TST represents the total time spent in sleep state during the analysis period.
    It's a fundamental metric for sleep quantity assessment.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    List[int]
        List containing total sleep time in minutes for each 24-hour cycle.

    Notes
    -----
    - Processes data in 24-hour cycles starting at midnight
    - Uses the TotalSleepTime class from sleep_metrics library
    - Sleep time is calculated by counting all epochs marked as sleep (1)
    - Key metric for assessing sleep quantity
    - Recommended TST varies by age group (7-9 hours for adults)

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample sleep data for one day
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> # Simulate 8 hours of sleep
    >>> sleep_pattern = [1] * 480 + [0] * 960  # 8 hours sleep, 16 hours wake
    >>> data = pd.DataFrame({'sleep': sleep_pattern}, index=dates)
    >>>
    >>> # Calculate TST
    >>> tst_values = TST(data)
    >>> print(f"Total Sleep Time: {tst_values[0]} minutes ({tst_values[0]/60:.1f} hours)")
    """

    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    tst = []
    t = TotalSleepTime()

    for date, day_data in daily_groups:
        day_data = day_data.sort_index()

        pred = t.predict(np.array(day_data["sleep"]))
        if pd.isna(pred):
            tst.append(0)
        else:
            tst.append(int(pred))

    return tst


def PTA(data: pd.DataFrame) -> List[float]:
    """
    Calculate Percent Time Asleep (PTA) for each 24-hour cycle.

    PTA represents the percentage of time spent asleep relative to the total recording time.
    It provides a normalized measure of sleep quantity.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    List[float]
        List containing percent time asleep (0-1) for each 24-hour cycle.
        Values range from 0 (no sleep) to 1 (100% sleep).

    Notes
    -----
    - Processes data in 24-hour cycles starting at midnight
    - Uses the PercentTimeAsleep class from sleep_metrics library
    - PTA = (number of sleep epochs) / (total number of epochs)
    - Useful for comparing sleep patterns across different recording durations
    - Typical PTA values range from 0.25 to 0.40 (25-40% of day)

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample sleep data for one day
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> # Simulate 8 hours of sleep (33.3% of day)
    >>> sleep_pattern = [1] * 480 + [0] * 960  # 8 hours sleep, 16 hours wake
    >>> data = pd.DataFrame({'sleep': sleep_pattern}, index=dates)
    >>>
    >>> # Calculate PTA
    >>> pta_values = PTA(data)
    >>> print(f"Percent Time Asleep: {pta_values[0]:.3f} ({pta_values[0]*100:.1f}%)")
    """
    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    pta = []
    p = PercentTimeAsleep()

    for date, day_data in daily_groups:
        # Sort by timestamp within the group
        day_data = day_data.sort_index()

        pred = p.predict(np.array(day_data["sleep"]))
        if pd.isna(pred):
            pta.append(0)
        else:
            pta.append(float(pred))

    return pta


def NWB(data: pd.DataFrame) -> List[int]:
    """
    Calculate Number of Wake Bouts (NWB) for each 24-hour cycle.

    NWB represents the count of distinct wake episodes occurring between sleep periods
    during the analysis period. It's a measure of sleep fragmentation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    List[int]
        List containing the number of wake bouts for each 24-hour cycle.
        Higher values indicate more fragmented sleep.

    Notes
    -----
    - Processes data in 24-hour cycles starting at midnight
    - Uses the NumberWakeBouts class from sleep_metrics library
    - A wake bout is defined as a continuous period of wake states between sleep states
    - Higher NWB values indicate more fragmented sleep patterns
    - Important metric for assessing sleep quality and continuity

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample sleep data for one day
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> # Simulate fragmented sleep with multiple wake bouts
    >>> sleep_pattern = [1] * 240 + [0] * 30 + [1] * 120 + [0] * 20 + [1] * 120 + [0] * 910
    >>> data = pd.DataFrame({'sleep': sleep_pattern}, index=dates)
    >>>
    >>> # Calculate NWB
    >>> nwb_values = NWB(data)
    >>> print(f"Number of Wake Bouts: {nwb_values[0]}")
    """

    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    nwb = []
    n = NumberWakeBouts()

    for date, day_data in daily_groups:
        day_data = day_data.sort_index()

        pred = n.predict(np.array(day_data["sleep"]))
        if pd.isna(pred):
            nwb.append(0)
        else:
            nwb.append(int(pred))

    return nwb


def SOL(data: pd.DataFrame) -> List[int]:
    """
    Calculate Sleep Onset Latency (SOL) for each 24-hour cycle.

    SOL represents the time taken to fall asleep, measured from the start of the
    recording period until the first sleep onset. It's a key metric for sleep initiation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    List[int]
        List containing sleep onset latency in minutes for each 24-hour cycle.
        Lower values indicate faster sleep onset.

    Notes
    -----
    - Processes data in 24-hour cycles starting at midnight
    - Uses the SleepOnsetLatency class from sleep_metrics library
    - SOL is calculated as the time from the start of the recording until the first detected sleep episode
    - Lower SOL values indicate better sleep initiation
    - Typical SOL values range from 10-30 minutes for healthy sleepers

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample sleep data for one day
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> # Simulate 30 minutes to fall asleep
    >>> sleep_pattern = [0] * 30 + [1] * 1410  # 30 min wake, then sleep
    >>> data = pd.DataFrame({'sleep': sleep_pattern}, index=dates)
    >>>
    >>> # Calculate SOL
    >>> sol_values = SOL(data)
    >>> print(f"Sleep Onset Latency: {sol_values[0]} minutes")
    """
    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    sol = []
    s = SleepOnsetLatency()

    for date, day_data in daily_groups:
        day_data = day_data.sort_index()
        pred = s.predict(np.array(day_data["sleep"]))
        if pd.isna(pred):
            sol.append(0)
        else:
            sol.append(int(pred))

    return sol


def SRI(data: pd.DataFrame) -> float:
    """
    Calculate Sleep Regularity Index (SRI) for the entire dataset.

    SRI quantifies the day-to-day similarity of sleep-wake patterns. It ranges from -100
    (completely irregular) to +100 (perfectly regular).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    float
        Sleep Regularity Index value ranging from -100 to +100:
        - -100: Completely irregular sleep patterns
        - 0: Random sleep patterns
        - +100: Perfectly regular sleep patterns

    Notes
    -----
    - Requires multiple days of data for meaningful calculation
    - Compares sleep-wake patterns across consecutive days
    - Higher SRI values indicate more consistent sleep schedules
    - Important metric for assessing sleep hygiene and circadian rhythm stability
    - Uses overlapping day pairs to calculate similarity

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample multi-day sleep data
    >>> dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    >>> # Simulate consistent sleep pattern
    >>> sleep_pattern = [1] * 480 + [0] * 960  # 8 hours sleep, 16 hours wake
    >>> sleep_data = sleep_pattern * 3  # Repeat for 3 days
    >>> data = pd.DataFrame({'sleep': sleep_data}, index=dates)
    >>>
    >>> # Calculate SRI
    >>> sri_value = SRI(data)
    >>> print(f"Sleep Regularity Index: {sri_value:.1f}")
    >>> # Higher values indicate more regular sleep patterns
    """

    if data.empty:
        return np.nan

    data_ = data.copy()
    data_ = data_.sort_index()

    N = 1440  # minutes per day
    M = len(pd.unique(data_.index.date))

    if M < 2:  # Need at least 2 days for SRI
        return np.nan

    daily_groups = data_.groupby(data_.index.date)
    sri = 0

    def overlapping_pairs(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    for (date_prev, day_data_prev), (
        date_next,
        day_data_next,
    ) in overlapping_pairs(daily_groups):
        # check for concordance of sleep states between consecutive days
        concordance = (
            day_data_prev["sleep"].reset_index(drop=True)
            == day_data_next["sleep"].reset_index(drop=True)
        ).sum()
        sri += concordance

    denominator = M * (N - 1)
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.nan
    
    sri = float(-100 + 200 / denominator * sri)

    return sri
