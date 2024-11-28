import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import welch


from .utils.calc_enmo import calculate_enmo, calculate_minute_level_enmo
from .utils.filtering import filter_incomplete_days, filter_consecutive_days
from .utils.smartwatch import read_smartwatch_data, preprocess_smartwatch_data
from .utils.ukb import read_ukb_data, filter_ukb_data, resample_ukb_data
from .utils.nhanes import read_nhanes_data, filter_nhanes_data, resample_nhanes_data
from .utils.galaxy import read_galaxy_data, filter_galaxy_data, resample_galaxy_data, preprocess_galaxy_data
def clock(func):
    """
    A decorator that prints the execution time of the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result

    return inner


class DataLoader:
    """
    A base class for data loaders that process and store ENMO data at the
    minute level.

    This class provides a common interface for data loaders with methods to load
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
        Initializes an empty DataLoader instance with an empty DataFrame
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

    def load_data(self, verbose: bool = False):
        raise NotImplementedError("The load_data method should be implemented by subclasses")

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
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

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
        Retrieve the accelerometer data.

        Returns:
            pd.DataFrame: A DataFrame containing the accelerometer data.
        """
        if self.sf_data is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.sf_data

    def get_ml_data(self):
        """
        Retrieve the minute-level ENMO values.

        Returns:
            pd.DataFrame: A DataFrame containing the minute-level ENMO values.
        """
        if self.ml_data is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.ml_data

    def get_meta_data(self):
        """
        Retrieve the metadata.

        Returns:
            dict: A dictionary containing the metadata.
        """
        return self.meta_dict

    
class NHANESDataLoader(DataLoader):
    def __init__(self, nhanes_file_dir: str, person_id: str = None, verbose: bool = False):
        super().__init__()

        if not os.path.isdir(nhanes_file_dir):
            raise ValueError("The input path should be a directory path")

        self.nhanes_file_dir = nhanes_file_dir
        self.person_id = person_id

        self.meta_dict['datasource'] = 'nhanes'

        self.__load_data(verbose=verbose)
    
    @clock
    def __load_data(self, verbose: bool = False):

        self.raw_data = read_nhanes_data(self.nhanes_file_dir, person_id=self.person_id, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = filter_nhanes_data(self.raw_data, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = resample_nhanes_data(self.sf_data, meta_dict=self.meta_dict, verbose=verbose)
        self.ml_data = self.sf_data


class UKBDataLoader(DataLoader):
    def __init__(self, qa_file_path: str, ukb_file_dir: str, eid: int, verbose: bool = False):
        super().__init__()

        if not os.path.isfile(qa_file_path):
            raise ValueError("The QA file path should be a file path")
        if not os.path.isdir(ukb_file_dir):
            raise ValueError("The UKB file directory should be a directory path")

        self.qa_file_path = qa_file_path
        self.ukb_file_dir = ukb_file_dir
        self.eid = eid

        self.meta_dict['datasource'] = 'uk-biobank'

        self.__load_data(verbose=verbose)

    @clock
    def __load_data(self, verbose: bool = False):
        
        self.raw_data = read_ukb_data(self.qa_file_path, self.ukb_file_dir, self.eid, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = filter_ukb_data(self.raw_data, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = resample_ukb_data(self.sf_data, meta_dict=self.meta_dict, verbose=verbose)
        self.ml_data = self.sf_data


class GalaxyDataLoader(DataLoader):
    def __init__(self, gw_file_dir: str, preprocess: bool = True, preprocess_args: dict = {}, verbose: bool = False):
        super().__init__()

        if not os.path.isdir(gw_file_dir):
            raise ValueError("The Galaxy Watch file directory should be a directory path")

        self.gw_file_dir = gw_file_dir
        self.preprocess = preprocess
        self.preprocess_args = preprocess_args

        self.meta_dict['datasource'] = 'samsung galaxy watch'

        self.__load_data(verbose=verbose)
    
    @clock
    def __load_data(self, verbose: bool = False):

        self.raw_data = read_galaxy_data(self.gw_file_dir, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = filter_galaxy_data(self.raw_data, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = resample_galaxy_data(self.sf_data, meta_dict=self.meta_dict, verbose=verbose)

        if self.preprocess:
            self.sf_data = preprocess_galaxy_data(self.sf_data, preprocess_args=self.preprocess_args, meta_dict=self.meta_dict, verbose=verbose)

        self.sf_data['ENMO'] = calculate_enmo(self.sf_data, verbose=verbose)
        self.ml_data = calculate_minute_level_enmo(self.sf_data, sf=25, verbose=verbose)


class SmartwatchDataLoader(DataLoader):

    def __init__(self, smartwatch_file_dir: str, preprocess: bool = True, preprocess_args: dict = {}):
        super().__init__()
        
        if not os.path.isdir(smartwatch_file_dir):
            raise ValueError("The smartwatch file directory should be a directory path")
        
        self.smartwatch_file_dir = smartwatch_file_dir

        self.preprocess = preprocess
        self.preprocess_args = preprocess_args

        self.meta_dict['datasource'] = 'smartwatch'

    @clock
    def load_data(self, verbose: bool = False):
        # load accelerometer data from csv files into a DataFrame
        self.sf_data = read_smartwatch_data(self.smartwatch_file_dir, meta_dict=self.meta_dict)
        if verbose:
            print(f"Loaded {self.sf_data.shape[0]} accelerometer data records from {self.smartwatch_file_dir}")

        # filter out incomplete days
        n_old = self.sf_data.shape[0]
        self.sf_data = filter_incomplete_days(self.sf_data, self.meta_dict['raw_data_frequency'])
        if verbose:
            print(f"Filtered out {n_old - self.sf_data.shape[0]} accelerometer records due to incomplete daily coverage")

        # if not data left, return
        if self.sf_data.empty:
            self.ml_data = pd.DataFrame()
            return

        # conduct preprocessing if required
        if self.preprocess:
            self.sf_data[['X_raw', 'Y_raw', 'Z_raw']] = self.sf_data[['X', 'Y', 'Z']]
            self.sf_data[['X', 'Y', 'Z', 'wear']] = preprocess_smartwatch_data(self.sf_data[['X', 'Y', 'Z']], self.meta_dict['raw_data_frequency'], self.meta_dict, preprocess_args=self.preprocess_args, verbose=verbose)
            if verbose:
                print(f"Preprocessed accelerometer data")

        # calculate ENMO values at original frequency
        self.sf_data['ENMO'] = calculate_enmo(self.sf_data)*1000
        if verbose:
            print(f"Calculated ENMO for {self.sf_data['ENMO'].shape[0]} accelerometer records")

        # aggregate ENMO values at the minute level in mg
        self.ml_data = calculate_minute_level_enmo(self.sf_data, self.meta_dict['raw_data_frequency'])
        self.ml_data['ENMO'] = self.ml_data['ENMO']
        self.ml_data.index = pd.to_datetime(self.ml_data.index)
        if verbose:
            print(f"Aggregated ENMO values at the minute level leading to {self.ml_data.shape[0]} records")

        self.meta_dict['n_days'] = len(np.unique(self.ml_data.index.date))
