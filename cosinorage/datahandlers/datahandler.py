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


import time


def clock(func):
    """
    A decorator that prints the execution time of the decorated function.
    Only prints when verbose=True is passed to the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        # Check if verbose=True was passed to the function
        verbose = kwargs.get('verbose', False)
        if verbose:
            print(f"{func.__name__} executed in {end - start:.2f} seconds")
        
        return result

    return inner


################################## !!!! ##################################
# whenever you implement a new datahandler for a new datasource, check
# the documentation of the source, e.g., the smartwatch, to make sure
# what units the data is in and scale accordingly.
################################## !!!! ##################################


class DataHandler:
    """
    A base class for data handlers that process and store ENMO data at the
    minute level.

    This class provides a common interface for data handlers with methods to load
    data, retrieve processed ENMO values, and save data. The `load_data` and
    `save_data` methods are intended to be overridden by subclasses.

    Attributes:
        datasource (str): The source of the data ('smartwatch', 'nhanes', or 'uk-biobank').
        input_path (str): The path to the input data.
        preprocess (bool): Whether to preprocess the data.
        sf_data (pd.DataFrame): A DataFrame storing accelerometer data.
        acc_freq (int): The frequency of the accelerometer data.
        meta_dict (dict): A dictionary storing metadata.
        ml_data (pd.DataFrame): A DataFrame storing minute-level ENMO values.
    """

    def __init__(self):
        """
        Initializes an empty DataHandler instance with an empty DataFrame
        for storing minute-level ENMO values.

        Args:
            datasource (str): The source of the data ('smartwatch', 'nhanes', or 'uk-biobank').
            input_path (str): The path to the input data.
            preprocess (bool): Whether to preprocess the data.
        """
        self.raw_data = None
        self.sf_data = None
        self.ml_data = None

        self.meta_dict = {}

    def __load_data(self, verbose: bool = False):
        raise NotImplementedError(
            "The load_data method should be implemented by subclasses"
        )

    def save_data(self, output_path: str):
        """
        Save minute-level ENMO data to a specified output path.

        This method is intended to be implemented by subclasses, specifying
        the format and structure for saving data.

        Args:
            output_path (str): The file path where the minute-level ENMO data
                will be saved.
        """
        if self.ml_data is None:
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first."
            )

        self.ml_data.to_csv(output_path, index=False)

    def get_raw_data(self):
        """
        Retrieve the raw data.

        Returns:
            pd.DataFrame: A DataFrame containing the raw data.
        """
        return self.raw_data

    def get_sf_data(self):
        """
        Retrieve the filtered data.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered data.
        """
        try:
            return self.sf_data
        except:
            raise ValueError(
                "No sf_data available."
            )

    def get_ml_data(self):
        """
        Retrieve the minute-level ENMO values.

        Returns:
            pd.DataFrame: A DataFrame containing the minute-level ENMO values.
        """
        if self.ml_data is None:
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first."
            )

        return self.ml_data

    def get_meta_data(self):
        """
        Retrieve the metadata.

        Returns:
            dict: A dictionary containing the metadata.
        """
        return self.meta_dict
