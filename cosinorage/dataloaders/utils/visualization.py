import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch

def plot_orig_enmo(acc_loader, resample: str = '15min', wear: bool = True):
    """
    Plot the original ENMO values resampled at a specified interval.

    Args:
        resample (str): The resampling interval (default is '15min').
        wear (bool): Whether to add color bands for wear and non-wear periods (default is True).
    """
    #_data = self.acc_df.resample('5min').mean().reset_index(inplace=False)
    _data = acc_loader.get_sf_data().resample(f'{resample}').mean().reset_index(inplace=False)
    

    plt.figure(figsize=(12, 6))
    plt.plot(_data['TIMESTAMP'], _data['ENMO'], label='ENMO', color='black')

    if wear:
        # Add color bands for wear and non-wear periods
        # add tqdm progress bar

        for i in tqdm(range(len(_data) - 1)):
            if _data['wear'].iloc[i] != 1:
                start_time = _data['TIMESTAMP'].iloc[i]
                end_time = _data['TIMESTAMP'].iloc[i + 1]
                color = 'red'
                plt.axvspan(start_time, end_time, color=color, alpha=0.3)

    plt.show()

def plot_enmo(loader):
    """
    Plot minute-level ENMO values.

    Returns:
        None
    """
    _data = loader.get_ml_data().reset_index(inplace=False)

    plt.figure(figsize=(12, 6))
    plt.plot(_data['TIMESTAMP'], _data['ENMO'], label='ENMO', color='black')

    if 'wear' in _data.columns:
        plt.fill_between(_data['TIMESTAMP'], _data['wear']*1000, color='green', alpha=0.5, label='wear')
        plt.fill_between(_data['TIMESTAMP'], (1-_data['wear'])*1000, color='red', alpha=0.5, label='non-wear')
        plt.legend()
        
    plt.ylim(0, max(_data['ENMO'])*1.25)
    plt.show()

def plot_orig_enmo_freq(acc_loader):
    # convert to frequency domain
    f, Pxx = welch(acc_loader.get_sf_data()['ENMO'], fs=80, nperseg=1024)

    plt.figure(figsize=(20, 5))
    plt.plot(f, Pxx)
    plt.show()