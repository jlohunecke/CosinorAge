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

from typing import Optional

import pandas as pd
import pytz

from .calc_enmo import calculate_enmo
from .filtering import filter_consecutive_days, filter_incomplete_days
from .frequency_detection import detect_frequency_from_timestamps
from .galaxy_binary import (calc_weartime, calibrate_accelerometer,
                            detect_wear_periods, remove_noise)


def read_generic_xD_data(
    file_path: str,
    data_type: str,
    meta_dict: dict,
    n_dimensions: int,
    time_format: str = "unix-ms",
    time_column: str = "timestamp",
    time_zone: Optional[str] = None,
    data_columns: Optional[list] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read generic accelerometer or count data from a CSV file.

    This function loads data from a CSV file and standardizes the column names
    for further processing. It supports both 1-dimensional (counts/ENMO) and
    3-dimensional (accelerometer) data formats.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the data.
    meta_dict : dict
        Dictionary to store metadata about the loaded data. Will be populated with:
        - raw_n_datapoints: Number of data points
        - raw_start_datetime: Start timestamp
        - raw_end_datetime: End timestamp
        - sf: Sampling frequency in Hz
        - raw_data_frequency: Sampling frequency as string
        - raw_data_type: Type of data ('Counts' or 'Accelerometer')
        - raw_data_unit: Unit of data ('counts' or 'mg')
    n_dimensions : int
        Number of dimensions in the data. Must be either 1 (for counts/ENMO) or 3 (for accelerometer).
    time_column : str, default='timestamp'
        Name of the timestamp column in the CSV file.
    data_columns : list, optional
        Names of the data columns in the CSV file. If not provided, defaults are:
        - ['counts'] for n_dimensions=1
        - ['x', 'y', 'z'] for n_dimensions=3
    verbose : bool, default=False
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data with standardized column names:
        - For n_dimensions=1: ['ENMO'] (single column)
        - For n_dimensions=3: ['x', 'y', 'z'] (three columns)
        The DataFrame has a datetime index from the timestamp column.

    Raises
    ------
    ValueError
        If n_dimensions is not 1 or 3, or if the number of data_columns doesn't match n_dimensions.

    Examples
    --------
    Load 1-dimensional count data:

    >>> meta_dict = {}
    >>> data = read_generic_xD(
    ...     file_path='data/counts.csv',
    ...     meta_dict=meta_dict,
    ...     n_dimensions=1,
    ...     time_column='time',
    ...     data_columns=['counts']
    ... )
    >>> print(data.columns)
    Index(['ENMO'], dtype='object')

    Load 3-dimensional accelerometer data:

    >>> meta_dict = {}
    >>> data = read_generic_xD(
    ...     file_path='data/accel.csv',
    ...     meta_dict=meta_dict,
    ...     n_dimensions=3,
    ...     time_column='timestamp',
    ...     data_columns=['accel_x', 'accel_y', 'accel_z']
    ... )
    >>> print(data.columns)
    Index(['x', 'y', 'z'], dtype='object')

    Notes
    -----
    The function automatically:
    - Converts timestamps to datetime objects
    - Removes timezone information
    - Fills missing values with 0
    - Sorts data by timestamp
    - Detects sampling frequency from timestamps
    - Populates metadata dictionary with data information
    """

    if n_dimensions not in [1, 3]:
        raise ValueError("n_dimensions must be either 1 or 3")

    if data_columns is not None:
        if n_dimensions != len(data_columns):
            raise ValueError(
                "n_dimensions must be equal to the number of data columns"
            )

    if time_format not in ["unix-ms", "unix-s", "datetime"]:
        raise ValueError(
            "time_format must be either 'unix-ms', 'unix-s' or 'datetime'")

    data = pd.read_csv(file_path)

    if verbose:
        print(f"Read csv file from {file_path}")

    # Set default data_columns if not provided
    if data_columns is None:
        if n_dimensions == 1:
            data_columns = ["counts"]
        elif n_dimensions == 3:
            data_columns = ["x", "y", "z"]
        else:
            raise ValueError("n_dimensions must be either 1 or 3")

    # Rename columns to standard format
    column_mapping = {time_column: "timestamp"}
    if n_dimensions == 1:
        column_mapping[data_columns[0]] = "enmo"
    elif n_dimensions == 3:
        column_mapping[data_columns[0]] = "x"
        column_mapping[data_columns[1]] = "y"
        column_mapping[data_columns[2]] = "z"
    else:
        raise ValueError("n_dimensions must be either 1 or 3")

    data = data.rename(columns=column_mapping)

    if time_zone is not None and time_zone not in pytz.all_timezones:
        raise ValueError(
            "time_zone must be a valid timezone, e.g., 'Europe/Zurich' or 'America/New_York'")

    # convert timestamp to UTC datetime
    if time_format == "unix-s":
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], unit="s", utc=True
        )
    elif time_format == "unix-ms":
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], unit="ms", utc=True
        )
    elif time_format == "datetime":
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], utc=True
        ).dt.tz_convert("UTC")
    else:
        raise ValueError(
            "time_format must be either 'unix-s', 'unix-ms' or 'datetime'")

    # convert datetime to timezone
    if time_zone is not None:
        data["timestamp"] = data["timestamp"].dt.tz_convert(time_zone)

    # drop timezone info (make naive, but keep local time)
    data["timestamp"] = data["timestamp"].dt.tz_localize(None)

    data.set_index("timestamp", inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(f"Loaded {data.shape[0]} Count data records from {file_path}")

    meta_dict["raw_n_datapoints"] = data.shape[0]
    meta_dict["raw_start_datetime"] = data.index.min()
    meta_dict["raw_end_datetime"] = data.index.max()
    meta_dict["sf"] = detect_frequency_from_timestamps(pd.Series(data.index))
    meta_dict["raw_data_frequency"] = f'{meta_dict["sf"]:.3g}Hz'
    meta_dict["raw_data_unit"] = (
        "counts" if data_type == "alternative_count" else "mg" if data_type in ["enmo-mg", "accelerometer-mg"] else "g" if data_type in [
            "enmo-g", "accelerometer-g"] else "ms2" if data_type in ["accelerometer-ms2"] else "unknown"
    )

    return data


def filter_generic_data(
    data: pd.DataFrame,
    data_type: str,
    meta_dict: dict = {},
    verbose: bool = False,
    preprocess_args: dict = {},
) -> pd.DataFrame:
    """
    Filter generic data by removing incomplete days and selecting longest consecutive sequence.

    This function applies data quality filters to ensure only complete and consecutive
    days of data are retained for analysis. It removes incomplete days and selects
    the longest sequence of consecutive days.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with datetime index containing accelerometer or count data.
    data_type : str
        Type of data being processed. Must be one of:
        - 'enmo': ENMO (Euclidean Norm Minus One) data
        - 'accelerometer': Raw accelerometer data (x, y, z)
        - 'alternative_count': Alternative count data
    meta_dict : dict, default={}
        Dictionary to store metadata about the filtering process. Will be updated with:
        - filtered_n_datapoints: Number of data points after filtering
        - filtered_start_datetime: Start timestamp after filtering
        - filtered_end_datetime: End timestamp after filtering
    verbose : bool, default=False
        Whether to print progress information during filtering.
    preprocess_args : dict, default={}
        Additional preprocessing arguments that may affect filtering behavior.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only complete and consecutive days of data.
        The DataFrame maintains the same structure as the input.

    Notes
    -----
    - Removes days that don't have the expected number of data points
    - Selects the longest sequence of consecutive days (minimum 4 days required)
    - Updates metadata with information about the filtered data
    - The function assumes 24-hour periods for day-based filtering

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample data with some incomplete days
    >>> dates = pd.date_range('2023-01-01', periods=10000, freq='min')
    >>> data = pd.DataFrame({'ENMO': np.random.randn(10000)}, index=dates)
    >>>
    >>> # Filter the data
    >>> meta_dict = {}
    >>> filtered_data = filter_generic_data(
    ...     data, data_type='enmo', meta_dict=meta_dict, verbose=True
    ... )
    >>> print(f"Original data points: {len(data)}")
    >>> print(f"Filtered data points: {len(filtered_data)}")
    """
    _data = data.copy()

    # filter out first and last day
    # TODO: only filter out if first or last day are incomplete (floor to only keep seconds and then the first timestamp 
    # needs to be 00:00:00 and for the last day the last timestamp needs to be 23:59:59)
    
    n_old = _data.shape[0]

    # check if first or last day are incomplete
    first_day_start = _data.index.floor("s").date.min()
    last_day_end = _data.index.floor("s").date.max()
    
    # Check if first day starts at midnight (00:00:00)
    first_day_data = _data[_data.index.date == first_day_start]
    if len(first_day_data) > 0 and first_day_data.index.min().time() != pd.Timestamp('00:00:00').time():
        _data = _data.loc[_data.index.date != first_day_start]
    
    # Check if last day ends at 23:59:59
    last_day_data = _data[_data.index.date == last_day_end]
    if len(last_day_data) > 0 and last_day_data.index.max().time() != pd.Timestamp('23:59:59').time():
        _data = _data.loc[_data.index.date != last_day_end]

    """
    _data = _data.loc[
        (_data.index.date != _data.index.date.min())
        & (_data.index.date != _data.index.date.max())
    ]
    """

    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} {data_type} records due to filtering out first and/or last day"
        )

    # filter out sparse days
    required_points_per_day = (
        # required daily coverage (0.5 = 50%)
        preprocess_args.get("required_daily_coverage", 0.5)
        * meta_dict["sf"]  # sampling frequency in Hz (points per second)
        * 60  # seconds per minute
        * 60  # minutes per hour
        * 24  # hours per day
    )
    n_old = _data.shape[0]
    _data = filter_incomplete_days(
        _data,
        data_freq=meta_dict["sf"],
        expected_points_per_day=required_points_per_day,
    )
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} records due to incomplete daily coverage"
        )

    # filter for longest consecutive sequence of days
    n_old = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} records due to filtering for longest consecutive sequence of days"
        )

    return _data


def resample_generic_data(
    data: pd.DataFrame,
    data_type: str,
    meta_dict: dict = {},
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Resample generic data to minute-level resolution.

    This function resamples high-frequency data to minute-level resolution using
    mean aggregation. This is a standard preprocessing step for circadian rhythm analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with datetime index containing high-frequency data.
    data_type : str
        Type of data being processed. Must be one of:
        - 'enmo': ENMO (Euclidean Norm Minus One) data
        - 'accelerometer': Raw accelerometer data (x, y, z)
        - 'alternative_count': Alternative count data
    meta_dict : dict, default={}
        Dictionary to store metadata about the resampling process. Will be updated with:
        - resampled_n_datapoints: Number of data points after resampling
        - resampled_start_datetime: Start timestamp after resampling
        - resampled_end_datetime: End timestamp after resampling
    verbose : bool, default=False
        Whether to print progress information during resampling.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with minute-level resolution. The DataFrame maintains
        the same column structure as the input but with reduced temporal resolution.

    Notes
    -----
    - Uses pandas resample('min').mean() for minute-level aggregation
    - The function assumes the input data has a datetime index
    - All columns are resampled using mean aggregation
    - Updates metadata with information about the resampled data

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample high-frequency data (every 10 seconds)
    >>> dates = pd.date_range('2023-01-01', periods=8640, freq='10S')  # 24 hours
    >>> data = pd.DataFrame({
    ...     'ENMO': np.random.randn(8640),
    ...     'wear': np.ones(8640)
    ... }, index=dates)
    >>>
    >>> # Resample to minute level
    >>> meta_dict = {}
    >>> resampled_data = resample_generic_data(
    ...     data, data_type='enmo', meta_dict=meta_dict, verbose=True
    ... )
    >>> print(f"Original frequency: {len(data)} points")
    >>> print(f"Resampled frequency: {len(resampled_data)} points")
    """
    _data = data.copy()

    # floor index to seconds
    _data.index = _data.index.floor("s")

    # Ensure first day starts at 00:00 and last day ends at 23:59 by extrapolating if needed
    n_old = _data.shape[0]

    # Get the first and last dates and ensure proper day boundaries
    first_datetime = _data.index.min()
    last_datetime = _data.index.max()

    # Create day boundaries: first day starts at 00:00, last day ends at 23:59
    first_day_start = first_datetime.replace(
        hour=0, minute=0, second=0, microsecond=0)
    last_day_end = last_datetime.replace(
        hour=23, minute=59, second=59, microsecond=999999)

    # Check if we need to extrapolate at the beginning
    if first_datetime > first_day_start:
        # Create a complete day range from 00:00 to 23:59
        complete_day_range = pd.date_range(
            start=first_day_start,
            end=last_day_end,
            freq='1min'
        )

        # Reindex the data to include the complete day range and forward fill
        _data = _data.resample("1min").mean()
        _data = _data.reindex(complete_day_range)
        _data = _data.interpolate(method="linear").ffill().bfill()

        if verbose:
            print(
                f"Extrapolated data to ensure first day starts at 00:00 and last day ends at 23:59: {_data.shape[0] - n_old} records added")
    else:
        # Filter to ensure first day starts at 00:00 and last day ends at 23:59
        _data = _data.loc[
            (_data.index >= first_day_start) &
            (_data.index <= last_day_end)
        ]

        if verbose:
            print(
                f"Filtered to ensure first day starts at 00:00 and last day ends at 23:59: {n_old - _data.shape[0]}/{n_old} records removed")

    # Resample to minute level
    n_old = _data.shape[0]
    _data = _data.resample("1min").mean().interpolate(method="linear").bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    return _data


def preprocess_generic_data(
    data: pd.DataFrame,
    data_type: str,
    preprocess_args: dict = {},
    meta_dict: dict = {},
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess generic accelerometer data with calibration, noise removal, and wear detection.

    This function applies a comprehensive preprocessing pipeline to accelerometer data,
    including calibration, noise filtering, and wear detection. The preprocessing steps
    are applied based on the data type and preprocessing arguments.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with datetime index containing accelerometer data.
        For accelerometer data, must have columns ['x', 'y', 'z'].
    data_type : str
        Type of data being processed. Must be one of:
        - 'enmo': ENMO (Euclidean Norm Minus One) data
        - 'accelerometer': Raw accelerometer data (x, y, z)
        - 'alternative_count': Alternative count data
    preprocess_args : dict, default={}
        Dictionary containing preprocessing parameters:
        - 'calibrate': Whether to apply accelerometer calibration (default: False)
        - 'sphere_crit': Sphere fitting criterion for calibration (default: 0.3)
        - 'sd_criteria': Standard deviation criterion for calibration (default: 0.1)
        - 'remove_noise': Whether to apply noise filtering (default: False)
        - 'filter_cutoff': Cutoff frequency for noise filter in Hz (default: 2)
        - 'detect_wear': Whether to apply wear detection (default: False)
        - 'sd_crit': Standard deviation criterion for wear detection (default: 0.013)
        - 'range_crit': Range criterion for wear detection (default: 0.05)
        - 'window_length': Window length for wear detection in seconds (default: 60)
        - 'window_skip': Window skip for wear detection in seconds (default: 30)
    meta_dict : dict, default={}
        Dictionary to store metadata about the preprocessing process.
    verbose : bool, default=False
        Whether to print progress information during preprocessing.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with the same structure as input but with applied
        preprocessing steps. May include additional columns like 'wear' if wear
        detection is enabled.

    Notes
    -----
    - Calibration is only applied to accelerometer data (data_type='accelerometer-mg', 'accelerometer-g', 'accelerometer-ms2')
    - Noise removal uses a Butterworth low-pass filter
    - Wear detection adds a binary 'wear' column (1=worn, 0=not worn)
    - The function skips preprocessing steps that are not enabled in preprocess_args
    - All preprocessing steps are applied in sequence: calibration → noise removal → wear detection

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample accelerometer data
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> data = pd.DataFrame({
    ...     'x': np.random.randn(1440),
    ...     'y': np.random.randn(1440),
    ...     'z': np.random.randn(1440) + 1  # Add gravity component
    ... }, index=dates)
    >>>
    >>> # Apply preprocessing with wear detection
    >>> preprocess_args = {
    ...     'calibrate': True,
    ...     'remove_noise': True,
    ...     'detect_wear': True
    ... }
    >>> meta_dict = {}
    >>> processed_data = preprocess_generic_data(
    ...     data, data_type='accelerometer',
    ...     preprocess_args=preprocess_args, meta_dict=meta_dict, verbose=True
    ... )
    >>> print(f"Processed data shape: {processed_data.shape}")
    >>> print(f"Wear column present: {'wear' in processed_data.columns}")
    """
    _data = data.copy()

    if data_type in ["enmo-mg", "enmo-g", "alternative_count"]:
        # wear detection - not implemented for enmo and alternative_count data yet (current algorithm relies on accelerometer data)
        _data["wear"] = -1

    elif data_type in ["accelerometer-mg", "accelerometer-g", "accelerometer-ms2"]:
        _data[["x_raw", "y_raw", "z_raw"]] = _data[["x", "y", "z"]]

        # recaling of accelerometer data to g
        if data_type == "accelerometer-mg":
            _data[["x", "y", "z"]] = _data[["x", "y", "z"]] / 1000
        elif data_type == "accelerometer-ms2":
            _data[["x", "y", "z"]] = _data[["x", "y", "z"]] / 9.81

        # calibration
        try:
            sphere_crit = preprocess_args.get("autocalib_sphere_crit", 1)
            sd_criter = preprocess_args.get("autocalib_sd_criter", 0.3)
            _data[["x", "y", "z"]] = calibrate_accelerometer(
                _data,
                sphere_crit=sphere_crit,
                sd_criteria=sd_criter,
                meta_dict=meta_dict,
                verbose=verbose,
            )
        except:
            if verbose:
                print("Calibration failed, skipping calibration")

        # noise removal
        try:
            type = preprocess_args.get("filter_type", "highpass")
            cutoff = preprocess_args.get("filter_cutoff", 15)
            _data[["x", "y", "z"]] = remove_noise(
                _data,
                sf=meta_dict["sf"],
                filter_type=type,
                filter_cutoff=cutoff,
                verbose=verbose,
            )
        except:
            if verbose:
                print("Noise removal failed, skipping noise removal")

        # wear detection
        try:
            sd_crit = preprocess_args.get("wear_sd_crit", 0.00013)
            range_crit = preprocess_args.get("wear_range_crit", 0.00067)
            window_length = preprocess_args.get("wear_window_length", 30)
            window_skip = preprocess_args.get("wear_window_skip", 7)
            _data["wear"] = detect_wear_periods(
                _data,
                meta_dict["sf"],
                sd_crit,
                range_crit,
                window_length,
                window_skip,
                meta_dict=meta_dict,
                verbose=verbose,
            )

            # calculate total, wear, and non-wear time
            calc_weartime(
                _data, sf=meta_dict["sf"], meta_dict=meta_dict, verbose=verbose
            )
        except:
            if verbose:
                print("Wear time calculation failed, skipping wear time calculation")

        _data["enmo"] = calculate_enmo(_data, verbose=verbose) * 1000

    else:
        raise ValueError(
            "Data type must be either 'enmo-mg', 'enmo-g', 'accelerometer-mg', 'accelerometer-g', 'accelerometer-ms2' or 'alternative_count'"
        )

    if verbose:
        print(f"Preprocessed {data_type} data")

    return _data
