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

from .datahandler import DataHandler, clock
from .utils.calc_enmo import calculate_minute_level_enmo
from .utils.galaxy_binary import (filter_galaxy_binary_data,
                                  preprocess_galaxy_binary_data,
                                  read_galaxy_binary_data,
                                  resample_galaxy_binary_data)
from .utils.galaxy_csv import (filter_galaxy_csv_data,
                               preprocess_galaxy_csv_data,
                               read_galaxy_csv_data, resample_galaxy_csv_data)


class GalaxyDataHandler(DataHandler):
    """
    Unified data handler for Samsung Galaxy Watch accelerometer data.

    This class handles loading, filtering, and processing of Galaxy Watch accelerometer data
    in both binary and CSV formats. Currently supports:
    - Binary format with accelerometer data type
    - CSV format with ENMO data type

    Attributes
    ----------
    galaxy_file_path : str
        Path to the Galaxy Watch data file (for CSV) or directory (for binary).
    data_format : str
        Format of the data ('csv' or 'binary').
    data_type : str
        Type of the data ('enmo' or 'accelerometer').
    time_column : str
        Name of the timestamp column.
    data_columns : list
        Names of the data columns.
    preprocess_args : dict
        Arguments for preprocessing.
    """

    def __init__(
        self,
        galaxy_file_path: str,
        data_format: str = "binary",
        data_type: Union[str, None] = None,
        time_column: Union[str, None] = None,
        data_columns: Union[list, None] = None,
        preprocess_args: dict = {},
        verbose: bool = False,
    ):

        super().__init__()

        if data_format not in ["csv", "binary"]:
            raise ValueError("data_format must be either 'csv' or 'binary'")

        # Set default data_type based on data_format if not provided
        if data_type is None:
            if data_format == "csv":
                data_type = "enmo"
            else:  # binary
                data_type = "accelerometer"

        if data_type not in ["enmo", "accelerometer"]:
            raise ValueError(
                "data_type must be either 'enmo' or 'accelerometer'"
            )

        # Set default column names based on data_format and data_type
        if time_column is None:
            if data_format == "csv":
                time_column = "time"  # Only ENMO is supported for CSV
            else:  # binary
                time_column = "unix_timestamp_in_ms"

        if data_columns is None:
            if data_type == "enmo":
                data_columns = ["enmo_mg"]
            else:  # accelerometer (binary only)
                data_columns = [
                    "acceleration_x",
                    "acceleration_y",
                    "acceleration_z",
                ]

        # Validate format-type combinations
        if data_format == "csv" and data_type != "enmo":
            raise ValueError(
                "CSV format currently only supports 'enmo' data_type"
            )
        if data_format == "binary" and data_type != "accelerometer":
            raise ValueError(
                "Binary format currently only supports 'accelerometer' data_type"
            )

        # Validate data_columns based on data_type
        if data_type == "enmo" and len(data_columns) != 1:
            raise ValueError(
                "For 'enmo' data_type, data_columns should contain exactly one column name"
            )
        if data_type == "accelerometer" and len(data_columns) != 3:
            raise ValueError(
                "For 'accelerometer' data_type, data_columns should contain exactly three column names"
            )

        if data_format == "csv":
            if not os.path.isfile(galaxy_file_path):
                raise ValueError(
                    "For CSV format, galaxy_file_path should be a file path. Please also ensure that the file is existing."
                )
        else:  # binary
            if not os.path.isdir(galaxy_file_path):
                raise ValueError(
                    "For binary format, galaxy_file_path should be a directory path. Please also ensure that the directory is existing."
                )

        self.galaxy_file_path = galaxy_file_path
        self.data_format = data_format
        self.data_type = data_type
        self.time_column = time_column
        self.data_columns = data_columns
        self.preprocess_args = preprocess_args

        self.meta_dict["datasource"] = "Samsung Galaxy Smartwatch"
        self.meta_dict["data_format"] = (
            "CSV"
            if data_format == "csv"
            else "Binary" if data_format == "binary" else "Unknown"
        )
        self.meta_dict["raw_data_type"] = (
            "ENMO"
            if data_type == "enmo"
            else "Accelerometer" if data_type == "accelerometer" else "Unknown"
        )
        self.meta_dict["time_column"] = time_column
        self.meta_dict["data_columns"] = data_columns

        self.__load_data(verbose=verbose)

    @clock
    def __load_data(self, verbose: bool = False):
        """
        Internal method to load and process Galaxy Watch data.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print processing information. Defaults to False.
        """

        if self.data_format == "csv" and self.data_type == "enmo":
            # Use CSV processing functions for ENMO data
            self.raw_data = read_galaxy_csv_data(
                self.galaxy_file_path,
                meta_dict=self.meta_dict,
                time_column=self.time_column,
                data_columns=self.data_columns,
                verbose=verbose,
            )
            self.sf_data = filter_galaxy_csv_data(
                self.raw_data,
                meta_dict=self.meta_dict,
                verbose=verbose,
                preprocess_args=self.preprocess_args,
            )
            self.sf_data = resample_galaxy_csv_data(
                self.sf_data, meta_dict=self.meta_dict, verbose=verbose
            )
            self.sf_data = preprocess_galaxy_csv_data(
                self.sf_data,
                preprocess_args=self.preprocess_args,
                meta_dict=self.meta_dict,
                verbose=verbose,
            )
            self.ml_data = calculate_minute_level_enmo(
                self.sf_data, self.meta_dict, verbose=verbose
            )
        elif (
            self.data_format == "binary" and self.data_type == "accelerometer"
        ):
            # Use binary processing functions for accelerometer data
            self.raw_data = read_galaxy_binary_data(
                self.galaxy_file_path,
                meta_dict=self.meta_dict,
                time_column=self.time_column,
                data_columns=self.data_columns,
                verbose=verbose,
            )
            self.sf_data = filter_galaxy_binary_data(
                self.raw_data,
                meta_dict=self.meta_dict,
                verbose=verbose,
                preprocess_args=self.preprocess_args,
            )
            self.sf_data = resample_galaxy_binary_data(
                self.sf_data, meta_dict=self.meta_dict, verbose=verbose
            )
            self.sf_data = preprocess_galaxy_binary_data(
                self.sf_data,
                preprocess_args=self.preprocess_args,
                meta_dict=self.meta_dict,
                verbose=verbose,
            )
            self.ml_data = calculate_minute_level_enmo(
                self.sf_data, self.meta_dict, verbose=verbose
            )
        else:
            # This should not happen due to validation in __init__, but just in case
            raise ValueError(
                f"Unsupported combination: data_format='{self.data_format}', data_type='{self.data_type}'"
            )
