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

import os
from typing import Union

import pandas as pd
from claid.data_collection.load.load_sensor_data import *

from .calc_enmo import calculate_enmo
from .calibration import calibrate_accelerometer
from .filtering import filter_consecutive_days, filter_incomplete_days
from .frequency_detection import detect_frequency_from_timestamps
from .noise_removal import remove_noise
from .wear_detection import calc_weartime, detect_wear_periods


def read_galaxy_binary_data(
    galaxy_file_dir: str,
    meta_dict: dict,
    time_column: str = "unix_timestamp_in_ms",
    data_columns: Union[list, None] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read accelerometer data from Galaxy Watch binary files.

    Parameters
    ----------
    galaxy_file_dir : str
        Directory containing Galaxy Watch data files
    meta_dict : dict
        Dictionary to store metadata about the loaded data
    time_column : str
        Name of the timestamp column in the binary data
    data_columns : list
        Names of the data columns in the binary data
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        DataFrame containing accelerometer data with columns ['x', 'y', 'z']
    """

    # Set default data_columns if not provided
    if data_columns is None:
        data_columns = ["acceleration_x", "acceleration_y", "acceleration_z"]

    data = pd.DataFrame()

    n_files = 0
    for day_dir in os.listdir(galaxy_file_dir):
        if os.path.isdir(galaxy_file_dir + day_dir):
            for file in os.listdir(galaxy_file_dir + day_dir):
                # only consider binary files
                if file.endswith(".binary") and file.startswith(
                    "acceleration_data"
                ):
                    _temp = acceleration_data_to_dataframe(
                        load_acceleration_data(
                            galaxy_file_dir + day_dir + "/" + file
                        )
                    )
                    data = pd.concat([data, _temp])
                    n_files += 1

    if verbose:
        print(f"Read {n_files} files from {galaxy_file_dir}")

    # Rename columns to standard format
    column_mapping = {time_column: "timestamp"}
    for i, col in enumerate(data_columns):
        if i == 0:
            column_mapping[col] = "x"
        elif i == 1:
            column_mapping[col] = "y"
        elif i == 2:
            column_mapping[col] = "z"

    data = data.rename(columns=column_mapping)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("timestamp", inplace=True)
    data.drop(
        columns=["effective_time_frame", "sensor_body_location"], inplace=True
    )

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(
            f"Loaded {data.shape[0]} accelerometer data records from {galaxy_file_dir}"
        )

    meta_dict["raw_n_datapoints"] = data.shape[0]
    meta_dict["raw_start_datetime"] = data.index.min()
    meta_dict["raw_end_datetime"] = data.index.max()
    meta_dict["sf"] = detect_frequency_from_timestamps(data.index)
    meta_dict["raw_data_frequency"] = f'{meta_dict["sf"]:.3g}Hz'
    meta_dict["raw_data_unit"] = "Custom"

    return data


def filter_galaxy_binary_data(
    data: pd.DataFrame,
    meta_dict: dict = {},
    verbose: bool = False,
    preprocess_args: dict = {},
) -> pd.DataFrame:
    """
    Filter Galaxy Watch accelerometer data by removing incomplete days and selecting longest consecutive sequence.

    Parameters
    ----------
    data : pd.DataFrame
        Raw accelerometer data
    meta_dict : dict
        Dictionary to store metadata about the filtering process
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        Filtered accelerometer data
    """
    _data = data.copy()

    # filter out first and last day
    n_old = _data.shape[0]
    _data = _data.loc[
        (_data.index.date != _data.index.date.min())
        & (_data.index.date != _data.index.date.max())
    ]
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{_data.shape[0]} accelerometer records due to filtering out first and last day"
        )

    # filter out sparse days
    required_points_per_day = (
        preprocess_args.get("required_daily_coverage", 0.5) * 2160000
    )
    n_old = _data.shape[0]
    sf = meta_dict.get("sf", 25)  # Default to 25Hz if not specified
    _data = filter_incomplete_days(
        _data, data_freq=sf, expected_points_per_day=required_points_per_day
    )
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} accelerometer records due to incomplete daily coverage"
        )

    # filter for longest consecutive sequence of days
    old_n = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(
            f"Filtered out {old_n - _data.shape[0]}/{old_n} minute-level accelerometer records due to filtering for longest consecutive sequence of days"
        )

    return _data


def resample_galaxy_binary_data(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Resample Galaxy Watch accelerometer data to a regular interval.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered accelerometer data
    meta_dict : dict
        Dictionary to store metadata about the resampling process
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        Resampled accelerometer data at regular frequency.
    """
    _data = data.copy()

    n_old = _data.shape[0]
    _data = _data.resample("40ms").mean().interpolate(method="linear").bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    return _data


def preprocess_galaxy_binary_data(
    data: pd.DataFrame,
    preprocess_args: dict = {},
    meta_dict: dict = {},
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess Galaxy Watch accelerometer data including rescaling, calibration, noise removal, and wear detection.

    Parameters
    ----------
    data : pd.DataFrame
        Resampled accelerometer data
    preprocess_args : dict
        Dictionary containing preprocessing parameters
    meta_dict : dict
        Dictionary to store metadata about the preprocessing
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        Preprocessed accelerometer data with additional columns for raw values and wear detection
    """
    _data = data.copy()
    _data[["x_raw", "y_raw", "z_raw"]] = _data[["x", "y", "z"]]

    # recaling of accelerometer data according to blog post: https://developer.samsung.com/sdp/blog/en/2025/04/10/understanding-and-converting-galaxy-watch-accelerometer-data
    _data[["x", "y", "z"]] = _data[["x", "y", "z"]] / 4096

    # calibration
    sphere_crit = preprocess_args.get("autocalib_sphere_crit", 1)
    sd_criter = preprocess_args.get("autocalib_sd_criter", 0.3)
    _data[["x", "y", "z"]] = calibrate_accelerometer(
        _data,
        sphere_crit=sphere_crit,
        sd_criteria=sd_criter,
        meta_dict=meta_dict,
        verbose=verbose,
    )

    # noise removal
    type = preprocess_args.get("filter_type", "highpass")
    cutoff = preprocess_args.get("filter_cutoff", 15)
    _data[["x", "y", "z"]] = remove_noise(
        _data,
        sf=meta_dict["sf"],
        filter_type=type,
        filter_cutoff=cutoff,
        verbose=verbose,
    )

    # wear detection
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

    _data["enmo"] = calculate_enmo(_data, verbose=verbose) * 1000

    if verbose:
        print(f"Preprocessed accelerometer data")

    return _data


def acceleration_data_to_dataframe(data) -> pd.DataFrame:
    """
    Convert binary acceleration data to pandas DataFrame.

    This function converts raw binary acceleration data from Samsung Galaxy Watch
    into a structured pandas DataFrame format for further processing.

    Parameters
    ----------
    data : object
        Binary acceleration data object containing samples with the following attributes:
        - acceleration_x: X-axis acceleration value
        - acceleration_y: Y-axis acceleration value
        - acceleration_z: Z-axis acceleration value
        - sensor_body_location: Location of the sensor on the body
        - unix_timestamp_in_ms: Timestamp in milliseconds since Unix epoch
        - effective_time_frame: Effective time frame for the sample

    Returns
    -------
    pd.DataFrame
        DataFrame containing accelerometer data with columns:
        - 'acceleration_x': X-axis acceleration values
        - 'acceleration_y': Y-axis acceleration values
        - 'acceleration_z': Z-axis acceleration values
        - 'sensor_body_location': Sensor location information
        - 'unix_timestamp_in_ms': Timestamps in milliseconds
        - 'effective_time_frame': Effective time frame information

    Notes
    -----
    - This function is used internally by read_galaxy_binary_data
    - The function iterates through all samples in the binary data object
    - Each sample is converted to a dictionary and added to the DataFrame
    - The resulting DataFrame maintains the original data structure from the binary file

    Examples
    --------
    >>> # This function is typically called internally by read_galaxy_binary_data
    >>> # but can be used directly if you have binary data objects:
    >>>
    >>> # Load binary data (example)
    >>> binary_data = load_acceleration_data("path/to/binary/file")
    >>>
    >>> # Convert to DataFrame
    >>> df = acceleration_data_to_dataframe(binary_data)
    >>> print(f"Converted {len(df)} acceleration samples")
    >>> print(f"Columns: {df.columns.tolist()}")
    """
    rows = []
    for sample in data.samples:
        rows.append(
            {
                "acceleration_x": sample.acceleration_x,
                "acceleration_y": sample.acceleration_y,
                "acceleration_z": sample.acceleration_z,
                "sensor_body_location": sample.sensor_body_location,
                "unix_timestamp_in_ms": sample.unix_timestamp_in_ms,
                "effective_time_frame": sample.effective_time_frame,
            }
        )

    return pd.DataFrame(rows)
