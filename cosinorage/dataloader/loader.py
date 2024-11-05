import pandas as pd
from .utility import concatenate_csv, get_posix_timestamps, filter_incomplete_days
from .enmo import calculate_enmo, calculate_minute_level_enmo


class DataLoader:
    """
    A base class for data loaders that process and store ENMO data at the minute level.

    This class provides a common interface for data loaders with methods to load
    data, retrieve processed ENMO values, and save data. The `load_data` and
    `save_data` methods are intended to be overridden by subclasses.

    Attributes:
        enmo_per_minute (pd.DataFrame): A DataFrame storing minute-level ENMO values.
    """

    def __init__(self):
        """
        Initializes an empty DataLoader instance with an empty DataFrame
        for storing minute-level ENMO values.
        """
        self.enmo_per_minute = pd.DataFrame()

    def load_data(self):
        """
        Load data into the DataLoader instance.

        This method is intended to be implemented by subclasses. It should load data
        and store the minute-level ENMO values in `self.enmo_per_minute`.

        Raises:
            NotImplementedError: This is a placeholder method and must be implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_enmo_per_minute(self):
        """
        Retrieve the minute-level ENMO values.

        Returns:
            pd.DataFrame: A DataFrame containing the minute-level ENMO values.
        """
        return self.enmo_per_minute

    def save_data(self, output_path: str):
        """
        Save minute-level ENMO data to a specified output path.

        This method is intended to be implemented by subclasses, specifying
        the format and structure for saving data.

        Args:
            output_path (str): The file path where the minute-level ENMO data will be saved.

        Raises:
            NotImplementedError: This is a placeholder method and must be implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")


class AccelerometerDataLoader(DataLoader):
    """
    A data loader for processing accelerometer data. This class loads, processes,
    and saves accelerometer data, calculating ENMO (Euclidean Norm Minus One)
    values at the minute level.

    Attributes:
        input_dir_path (str): Path to the directory containing input CSV files.
        data (pd.DataFrame): DataFrame containing raw and processed accelerometer data.
        enmo_per_minute (pd.DataFrame): DataFrame containing ENMO values aggregated at the minute level.
    """

    def __init__(self, input_dir_path: str):
        """
        Initializes the AccelerometerDataLoader with the path to the input data directory.

        Args:
            input_dir_path (str): The path to the directory containing input CSV files.
        """
        super().__init__()
        self.input_dir_path = input_dir_path
        self.data = pd.DataFrame()

    def load_data(self):
        """
        Loads and processes accelerometer data from CSV files in the specified directory.
        This method performs several transformations, including timestamp conversion,
        ENMO calculation, and filtering of incomplete days. It then aggregates ENMO
        values at the minute level and stores the result in `self.enmo_per_minute`.

        Processing steps include:
            1. Concatenating CSV files from the input directory.
            2. Converting timestamps to POSIX format.
            3. Calculating the ENMO metric.
            4. Filtering out incomplete days.
            5. Aggregating ENMO values at the minute level.

        Returns:
            None
        """
        self.data = concatenate_csv(self.input_dir_path)
        self.data["TIMESTAMP"] = get_posix_timestamps(self.data["HEADER_TIMESTAMP"])
        self.data = calculate_enmo(self.data)
        self.data = filter_incomplete_days(self.data)
        self.enmo_per_minute = calculate_minute_level_enmo(self.data)
        self.enmo_per_minute.reset_index(drop=True, inplace=True)

    def save_data(self, output_file_path: str):
        """
        Saves the processed minute-level ENMO data to a CSV file.

        Args:
            output_file_path (str): The file path where the minute-level ENMO data will be saved.

        Returns:
            None
        """
        self.enmo_per_minute.to_csv(output_file_path, index=False)


class ENMODataLoader(DataLoader):
    """
    A data loader for processing ENMO data from a single CSV file. This class
    loads, processes, and saves ENMO (Euclidean Norm Minus One) values at the
    minute level.

    Attributes:
        input_file_path (str): Path to the input CSV file containing ENMO data.
        enmo_per_minute (pd.DataFrame): DataFrame containing processed ENMO values.
    """

    def __init__(self, input_file_path: str):
        """
        Initializes the ENMODataLoader with the path to the input data file.

        Args:
            input_file_path (str): The path to the CSV file containing ENMO data.
        """
        super().__init__()
        self.input_file_path = input_file_path

    def load_data(self):
        """
        Loads and processes ENMO data from the specified CSV file. This method
        performs several transformations, including timestamp conversion, data
        renaming, and filtering of incomplete days. It then stores the processed
        data in `self.enmo_per_minute`.

        Processing steps include:
            1. Loading data from a CSV file.
            2. Selecting 'time' and 'ENMO_t' columns.
            3. Converting timestamps to POSIX format.
            4. Renaming 'ENMO_t' to 'ENMO'.
            5. Dropping unnecessary columns.
            6. Filtering out incomplete days.

        Returns:
            None
        """
        self.enmo_per_minute = pd.read_csv(self.input_file_path)
        self.enmo_per_minute = self.enmo_per_minute[['time', 'ENMO_t']]
        self.enmo_per_minute['TIMESTAMP'] = get_posix_timestamps(self.enmo_per_minute['time'], sample_rate=1 / 60)
        self.enmo_per_minute = self.enmo_per_minute.rename(columns={'ENMO_t': 'ENMO'})
        self.enmo_per_minute = self.enmo_per_minute.drop(columns=['time'])

        self.enmo_per_minute = filter_incomplete_days(self.enmo_per_minute)
        self.enmo_per_minute = self.enmo_per_minute.drop(columns=['DATE'])
        self.enmo_per_minute.reset_index(drop=True, inplace=True)

    def save_data(self, output_file_path: str):
        """
        Saves the processed minute-level ENMO data to a CSV file.

        Args:
            output_file_path (str): The file path where the minute-level ENMO data will be saved.

        Returns:
            None
        """
        self.enmo_per_minute.to_csv(output_file_path, index=False)




