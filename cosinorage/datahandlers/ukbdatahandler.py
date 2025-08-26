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
from .utils.ukb import filter_ukb_data, read_ukb_data, resample_ukb_data


class UKBDataHandler(DataHandler):
    """
    Data handler for UK Biobank accelerometer data.

    This class handles loading, filtering, and processing of UK Biobank accelerometer data.

    Attributes
    ----------
    qa_file_path : str
        Path to quality assessment file.
    ukb_file_dir : str
        Directory containing UK Biobank data files.
    eid : int
        Participant ID.
    """

    def __init__(
        self,
        qa_file_path: str,
        ukb_file_dir: str,
        eid: int,
        verbose: bool = False,
    ):
        super().__init__()

        if not os.path.isfile(qa_file_path):
            raise ValueError("The QA file path should be a file path")
        if not os.path.isdir(ukb_file_dir):
            raise ValueError(
                "The UKB file directory should be a directory path"
            )

        self.qa_file_path = qa_file_path
        self.ukb_file_dir = ukb_file_dir
        self.eid = eid

        self.meta_dict["datasource"] = "uk-biobank"

        self.__load_data(verbose=verbose)

    @clock
    def __load_data(self, verbose: bool = False):
        """
        Internal method to load and process UK Biobank data.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print processing information. Defaults to False.
        """

        self.raw_data = read_ukb_data(
            self.qa_file_path,
            self.ukb_file_dir,
            self.eid,
            meta_dict=self.meta_dict,
            verbose=verbose,
        )
        self.sf_data = filter_ukb_data(
            self.raw_data, meta_dict=self.meta_dict, verbose=verbose
        )
        self.sf_data = resample_ukb_data(
            self.sf_data, meta_dict=self.meta_dict, verbose=verbose
        )
        self.ml_data = self.sf_data
