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

from .datahandler import DataHandler, clock
from .utils.nhanes import (filter_and_preprocess_nhanes_data, read_nhanes_data,
                           resample_nhanes_data)


class NHANESDataHandler(DataHandler):
    """
    Data handler for NHANES accelerometer data.

    This class handles loading, filtering, and processing of NHANES accelerometer data.

    Attributes
    ----------
    nhanes_file_dir : str
        Directory containing NHANES data files.
    seqn : str or None
        ID of the person whose data is being loaded.
    """

    def __init__(
        self, nhanes_file_dir: str, seqn: int = None, verbose: bool = False
    ):
        super().__init__()

        if not os.path.isdir(nhanes_file_dir):
            raise ValueError("The input path should be a directory path")

        self.nhanes_file_dir = nhanes_file_dir
        self.seqn = seqn

        self.meta_dict["datasource"] = "nhanes"

        self.__load_data(verbose=verbose)

    @clock
    def __load_data(self, verbose: bool = False):
        """
        Internal method to load and process NHANES data.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print processing information. Defaults to False.
        """

        self.raw_data = read_nhanes_data(
            self.nhanes_file_dir,
            seqn=self.seqn,
            meta_dict=self.meta_dict,
            verbose=verbose,
        )
        self.sf_data = filter_and_preprocess_nhanes_data(
            self.raw_data, meta_dict=self.meta_dict, verbose=verbose
        )
        self.sf_data = resample_nhanes_data(
            self.sf_data, meta_dict=self.meta_dict, verbose=verbose
        )
        self.ml_data = self.sf_data

    def get_ml_data(self):
        """
        Get the minute-level data.
        """
        return self.ml_data
