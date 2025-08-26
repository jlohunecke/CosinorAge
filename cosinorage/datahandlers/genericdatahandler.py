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

from .datahandler import DataHandler, clock
from .utils.calc_enmo import calculate_minute_level_enmo
from .utils.generic import (filter_generic_data, preprocess_generic_data,
                            read_generic_xD_data, resample_generic_data)


class GenericDataHandler(DataHandler):
    """
    Generic data handler for processing accelerometer and ENMO data from CSV files.

    This class provides a flexible interface for loading and processing various types of
    accelerometer data, including ENMO (Euclidean Norm Minus One), raw accelerometer
    data (x, y, z), and alternative count data. It supports automatic data filtering,
    resampling, preprocessing, and ENMO calculation.



    Attributes
    ----------
    file_path : str
        Path to the CSV file containing the data.
    data_format : str
        Format of the data file.
    data_type : str
        Type of data in the file.
    time_format : str
        Format of timestamps.
    time_column : str
        Name of the timestamp column.
    time_zone : str or None
        Timezone for datetime conversion.
    data_columns : list
        Names of the data columns.
    preprocess_args : dict
        Preprocessing arguments.
    raw_data : pd.DataFrame or None
        Raw data loaded from the file with timestamp index.
    sf_data : pd.DataFrame or None
        Data after filtering and resampling (sensor fusion data).
    ml_data : pd.DataFrame or None
        Minute-level ENMO data calculated from the processed data.
    meta_dict : dict
        Metadata dictionary containing information about the data processing.

    Examples
    --------
    Load ENMO data from a CSV file:

    >>> handler = GenericDataHandler(
    ...     file_path='data/enmo_data.csv',
    ...     data_type='enmo',
    ...     time_column='timestamp',
    ...     data_columns=['enmo']
    ... )
    >>> raw_data = handler.get_raw_data()
    >>> ml_data = handler.get_ml_data()

    Load accelerometer data from a CSV file:

    >>> handler = GenericDataHandler(
    ...     file_path='data/accel_data.csv',
    ...     data_type='accelerometer',
    ...     time_column='time',
    ...     data_columns=['x', 'y', 'z']
    ... )
    >>> raw_data = handler.get_raw_data()
    >>> ml_data = handler.get_ml_data()

    Notes
    -----
    The data processing pipeline includes:
    1. Loading raw data from CSV file
    2. Filtering incomplete days and selecting longest consecutive sequence
    3. Resampling to minute-level data
    4. Preprocessing (wear detection, noise removal, etc.)
    5. Calculating minute-level ENMO values

    The class automatically handles column mapping and timestamp processing.
    """

    def __init__(
        self,
        file_path: str,
        data_format: str = "csv",
        data_type: str = "accelerometer-mg",
        time_format: str = "unix-ms",
        time_column: str = "timestamp",
        time_zone: Optional[str] = None,
        data_columns: Optional[list] = None,
        preprocess_args: dict = {},
        verbose: bool = False,
    ):
        """
        Initialize GenericDataHandler with CSV data file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing the data.
        data_format : str, default='csv'
            Format of the data file. Currently only 'csv' is supported.
        data_type : str, default='accelerometer-mg'
            Type of data in the file. Must be one of:
            - 'enmo-mg', 'enmo-g': ENMO (Euclidean Norm Minus One) data
            - 'accelerometer-mg', 'accelerometer-g', 'accelerometer-ms2': Raw accelerometer data (x, y, z)
            - 'alternative_count': Alternative count data
        time_format : str, default='unix-ms'
            Format of timestamps. Must be one of: 'unix-ms', 'unix-s', 'datetime'.
        time_column : str, default='timestamp'
            Name of the timestamp column in the CSV file.
        time_zone : str, optional
            Timezone for datetime conversion. If None, uses local timezone.
        data_columns : list, optional
            Names of the data columns in the CSV file. If not provided, defaults are:
            - ['enmo'] for data_type='enmo-mg' or 'enmo-g'
            - ['x', 'y', 'z'] for data_type='accelerometer-mg', 'accelerometer-g', or 'accelerometer-ms2'
            - ['counts'] for data_type='alternative_count'
        preprocess_args : dict, default={}
            Additional preprocessing arguments to pass to the filtering and preprocessing functions.
        verbose : bool, default=False
            Whether to print progress information during data loading and processing.
        """

        super().__init__()

        if data_format not in ["csv"]:
            raise ValueError("Data format must be either 'csv'")

        # Handle legacy data types for backward compatibility
        if data_type == "enmo":
            data_type = "enmo-mg"
        elif data_type == "accelerometer":
            data_type = "accelerometer-mg"
        
        if data_type not in ["enmo-mg", "enmo-g", "accelerometer-mg", "accelerometer-g", "accelerometer-ms2", "alternative_count"]:
            raise ValueError(
                "Data type must be either 'enmo-mg', 'enmo-g', 'accelerometer-mg', 'accelerometer-g', 'accelerometer-ms2' or 'alternative_count'"
            )

        if time_format not in ["unix-ms", "unix-s", "datetime"]:
            raise ValueError("time_format must be either 'unix-ms', 'unix-s' or 'datetime'")

        if data_type in ["enmo-mg", "enmo-g"]:
            default_data_columns = ["enmo"]
        elif data_type in ["accelerometer-mg", "accelerometer-g", "accelerometer-ms2"]:
            default_data_columns = ["x", "y", "z"]
        elif data_type == "alternative_count":
            default_data_columns = ["counts"]
        else:
            raise ValueError(
                "Data type must be either 'enmo-mg', 'enmo-g', 'accelerometer-mg', 'accelerometer-g', 'accelerometer-ms2' or 'alternative_count'"
            )

        self.file_path = file_path
        self.data_format = data_format
        self.data_type = data_type
        self.time_format = time_format
        self.time_column = time_column
        self.time_zone = time_zone
        self.data_columns = (
            data_columns if data_columns is not None else default_data_columns
        )
        self.preprocess_args = preprocess_args

        self.meta_dict["datasource"] = "Generic"
        self.meta_dict["data_format"] = "CSV"
        self.meta_dict["time_format"] = time_format
        self.meta_dict["raw_data_type"] = (
            "ENMO"
            if data_type in ["enmo-mg", "enmo-g"]
            else (
                "Accelerometer"
                if data_type in ["accelerometer-mg", "accelerometer-g", "accelerometer-ms2"]
                else (
                    "Alternative Count"
                    if data_type == "alternative_count"
                    else "Unknown"
                )
            )
        )
        self.meta_dict["time_column"] = time_column
        self.meta_dict["time_zone"] = time_zone
        self.meta_dict["data_columns"] = data_columns

        self.__load_data(verbose=verbose)

    @clock
    def __load_data(self, verbose: bool = False):
        if self.data_format == "csv":
            # Determine number of dimensions based on data type
            n_dimensions = 3 if self.data_type in ["accelerometer-mg", "accelerometer-g", "accelerometer-ms2"] else 1

            # Load and process data
            self.raw_data = read_generic_xD_data(
                self.file_path,
                self.data_type,
                meta_dict=self.meta_dict,
                n_dimensions=n_dimensions,
                time_format=self.time_format,
                time_column=self.time_column,
                time_zone=self.time_zone,
                data_columns=self.data_columns,
                verbose=verbose,
            )
            self.sf_data = filter_generic_data(
                self.raw_data,
                self.data_type,
                self.meta_dict,
                verbose=verbose,
                preprocess_args=self.preprocess_args,
            )
            self.ml_data = resample_generic_data(
                self.sf_data, self.data_type, self.meta_dict, verbose=verbose
            )
            self.ml_data = preprocess_generic_data(
                self.ml_data,
                self.data_type,
                preprocess_args=self.preprocess_args,
                meta_dict=self.meta_dict,
                verbose=verbose,
            )
        else:
            raise ValueError("Data format must be either 'csv'")
