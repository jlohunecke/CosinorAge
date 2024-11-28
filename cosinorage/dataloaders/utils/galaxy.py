import pandas as pd
import os
import numpy as np
from typing import Tuple
from skdh.preprocessing import CountWearDetection, CalibrateAccelerometer, AccelThresholdWearDetection
from scipy.signal import butter, filtfilt
from claid.data_collection.load.load_sensor_data import *

from .smartwatch import preprocess_smartwatch_data
from .filtering import filter_incomplete_days, filter_consecutive_days


def read_galaxy_data(gw_file_dir: str, meta_dict: dict, verbose: bool = False):

    data = pd.DataFrame()

    n_files = 0
    for day_dir in os.listdir(gw_file_dir):
        if os.path.isdir(gw_file_dir + day_dir):
            for file in os.listdir(gw_file_dir + day_dir):
                # only consider binary files
                if file.endswith(".binary") and file.startswith("acceleration_data"):
                    _temp = acceleration_data_to_dataframe(load_acceleration_data(gw_file_dir + day_dir + "/" + file))
                    data = pd.concat([data, _temp])
                    n_files += 1

    if verbose:
        print(f"Read {n_files} files from {gw_file_dir}")

    data = data.rename(columns={'unix_timestamp_in_ms': 'TIMESTAMP', 'acceleration_x': 'X', 'acceleration_y': 'Y', 'acceleration_z': 'Z'})
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms')
    data.set_index('TIMESTAMP', inplace=True)
    data.drop(columns=['effective_time_frame', 'sensor_body_location'], inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(f"Loaded {data.shape[0]} accelerometer data records from {gw_file_dir}")

    meta_dict['raw_n_timesteps'] = data.shape[0]
    meta_dict['raw_n_days'] = len(np.unique(data.index.date))
    meta_dict['raw_start_datetime'] = data.index.min()
    meta_dict['raw_end_datetime'] = data.index.max()
    meta_dict['raw_frequency'] = 'irregular (~25Hz)'
    meta_dict['raw_datatype'] = 'accelerometer'
    meta_dict['raw_unit'] = ''

    return data


def filter_galaxy_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()

    # filter out first and last day
    n_old = _data.shape[0]
    _data = _data.loc[(_data.index.date != _data.index.date.min()) & (_data.index.date != _data.index.date.max())]
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{_data.shape[0]} accelerometer records due to filtering out first and last day")

    # filter out sparse days
    n_old = _data.shape[0]
    _data = filter_incomplete_days(_data, data_freq=25, expected_points_per_day=2000000)
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{n_old} accelerometer records due to incomplete daily coverage")

    # filter for longest consecutive sequence of days
    old_n = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(f"Filtered out {old_n - _data.shape[0]}/{old_n} minute-level accelerometer records due to filtering for longest consecutive sequence of days")

    return _data


def resample_galaxy_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()

    n_old = _data.shape[0]
    _data = _data.resample('40ms').interpolate(method='linear').bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    meta_dict['resampled_n_timestamps'] = _data.shape[0]
    meta_dict['resampled_n_days'] = len(np.unique(_data.index.date))
    meta_dict['resampled_start_datetime'] = _data.index.min()
    meta_dict['resampled_end_datetime'] = _data.index.max()
    meta_dict['resampled_frequency'] = '25Hz'
    meta_dict['resampled_datatype'] = 'accelerometer'
    meta_dict['resampled_unit'] = ''

    return _data


def preprocess_galaxy_data(data: pd.DataFrame, preprocess_args: dict = {}, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()
    _data[['X_raw', 'Y_raw', 'Z_raw']] = _data[['X', 'Y', 'Z']]

    # TODO  scale it down to proper range
    rescale_factor = preprocess_args.get('rescale_factor', 1)
    _data[['X', 'Y', 'Z']] = _data[['X', 'Y', 'Z']] * rescale_factor

    # calibration
    sphere_crit = preprocess_args.get('autocalib_sphere_crit', 1)
    sd_criter = preprocess_args.get('autocalib_sd_criter', 0.3)
    _data[['X', 'Y', 'Z']] = calibrate(_data, sf=25, sphere_crit=sphere_crit, sd_criteria=sd_criter, meta_dict=meta_dict, verbose=verbose)

    # noise removal
    type = preprocess_args.get('filter_type', 'highpass')
    cutoff = preprocess_args.get('filter_cutoff', 15)
    _data[['X', 'Y', 'Z']] = remove_noise(_data, sf=25, filter_type=type, filter_cutoff=cutoff, verbose=verbose)

    # wear detection
    sd_crit = preprocess_args.get('wear_sd_crit', 0.00013)
    range_crit = preprocess_args.get('wear_range_crit', 0.00067)
    window_length = preprocess_args.get('wear_window_length', 30)
    window_skip = preprocess_args.get('wear_window_skip', 7)
    _data['wear'] = detect_wear(_data, 25, sd_crit, range_crit, window_length, window_skip, meta_dict=meta_dict, verbose=verbose)

    # calculate total, wear, and non-wear time
    calc_weartime(_data, sf=25, meta_dict=meta_dict, verbose=verbose)

    if verbose:
        print(f"Preprocessed accelerometer data")

    return _data


def acceleration_data_to_dataframe(data):
    rows = []
    for sample in data.samples:
        rows.append({
            'acceleration_x': sample.acceleration_x,
            'acceleration_y': sample.acceleration_y,
            'acceleration_z': sample.acceleration_z,
            'sensor_body_location': sample.sensor_body_location,
            'unix_timestamp_in_ms': sample.unix_timestamp_in_ms,
            'effective_time_frame': sample.effective_time_frame
        })

    return pd.DataFrame(rows)


def calibrate(data: pd.DataFrame, sf: float, sphere_crit: float, sd_criteria: float, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:

    _data = data.copy()

    time = np.array(_data.index.astype('int64') // 10 ** 9)
    acc = np.array(_data[["X", "Y", "Z"]]).astype(np.float64)

    calibrator = CalibrateAccelerometer(sphere_crit=sphere_crit, sd_criteria=sd_criteria)
    result = calibrator.predict(time=time, accel=acc, fs=sf)

    _data = pd.DataFrame(result['accel'], columns=['X', 'Y', 'Z'])
    _data.set_index(data.index, inplace=True)

    meta_dict.update({'calibration_offset': result['offset']})
    meta_dict.update({'calibration_scale': result['scale']})

    if verbose:
        print('Calibration done')

    return _data[['X', 'Y', 'Z']]


def remove_noise(data: pd.DataFrame, sf: float, filter_type: str = 'lowpass', filter_cutoff: float = 2, verbose: bool = False) -> pd.DataFrame:
    """
    Remove noise from accelerometer data using a Butterworth low-pass filter.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        cutoff (float): Cutoff frequency for the low-pass filter in Hz (default is 2.5).
        fs (float): Sampling frequency of the accelerometer data in Hz (default is 50).
        order (int): Order of the Butterworth filter (default is 2).

    Returns:
        pd.DataFrame: DataFrame with noise removed from the 'X', 'Y', and 'Z' columns.
    """
    if (filter_type == 'bandpass' or filter_type == 'bandstop') and (type(filter_cutoff) != list or len(filter_cutoff) != 2):
        raise ValueError('Bandpass and bandstop filters require a list of two cutoff frequencies.')

    if (filter_type == 'highpass' or filter_type == 'lowpass') and type(filter_cutoff) not in [float, int]:
        raise ValueError('Highpass and lowpass filters require a single cutoff frequency.')

    if data.empty:
        raise ValueError("Dataframe is empty.")

    if not all(col in data.columns for col in ['X', 'Y', 'Z']):
        raise KeyError("Dataframe must contain 'X', 'Y' and 'Z' columns.")

    def butter_lowpass_filter(data, cutoff, sf, btype, order=2):
        # Design Butterworth filter
        nyquist = 0.5 * sf  # Nyquist frequency
        normal_cutoff = np.array(cutoff) / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)

        # Apply filter to data
        return filtfilt(b, a, data)

    _data = data.copy()

    cutoff = filter_cutoff
    _data['X'] = butter_lowpass_filter(_data['X'], cutoff, sf, btype=filter_type)
    _data['Y'] = butter_lowpass_filter(_data['Y'], cutoff, sf, btype=filter_type)
    _data['Z'] = butter_lowpass_filter(_data['Z'], cutoff, sf, btype=filter_type)

    if verbose:
        print('Noise removal done')

    return _data[['X', 'Y', 'Z']]


def detect_wear(data: pd.DataFrame, sf: float, sd_crit:float, range_crit:float, window_length:int, window_skip:int, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()

    time = np.array(_data.index.astype('int64') // 10 ** 9)
    acc = np.array(_data[["X", "Y", "Z"]]).astype(np.float64) / 1000

    #wear_predictor = CountWearDetection()
    wear_predictor = AccelThresholdWearDetection(sd_crit=sd_crit, range_crit=range_crit, window_length=window_length, window_skip=window_skip)
    ranges = wear_predictor.predict(time=time, accel=acc, fs=sf)['wear']

    wear_array = np.zeros(len(data.index))
    for start, end in ranges:
        wear_array[start:end + 1] = 1

    _data['wear'] = pd.DataFrame(wear_array, columns=['wear']).set_index(data.index)

    if verbose:
        print('Wear detection done')

    return _data[['wear']]


def calc_weartime(data: pd.DataFrame, sf: float, meta_dict: dict, verbose: bool) -> Tuple[float, float, float]:
    """
    Calculate total, wear, and non-wear time from accelerometer data.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with a 'wear' column.
        sf (float): Sampling frequency of the accelerometer data in Hz.

    Returns:
        Tuple[float, float, float]: A tuple containing total time, wear time, and non-wear time in seconds.
    """
    _data = data.copy()

    total = float((_data.index[-1] - _data.index[0]).total_seconds())
    wear = float((_data['wear'].sum()) * (1 / sf))
    nonwear = float((total - wear))

    meta_dict.update({'resampled_total_time': total, 'resampled_wear_time': wear, 'resampled_non-wear_time': nonwear})
    if verbose:
        print('Wear time calculated')