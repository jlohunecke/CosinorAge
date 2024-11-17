import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.dataloaders.utils.smartwatch import *

# Test basic helper functions first
def test_roll_mean():
    # Test case 1: Basic sequence with known means
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = roll_mean(data, window_size=3)
    expected = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 2: Sequence with negative numbers and decimals
    data = np.array([-1.5, 2.7, -3.2, 4.1, -5.9, 6.3, -7.8, 8.2])
    result = roll_mean(data, window_size=4)
    expected = np.array([0.525, -0.575, 0.325, -0.825, 0.2])  # Means of sliding windows
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 3: Constant sequence (should return same value)
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    result = roll_mean(data, window_size=3)
    expected = np.array([5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 4: Different window sizes
    data = np.array([1, 2, 3, 4, 5])
    result_w2 = roll_mean(data, window_size=2)
    result_w3 = roll_mean(data, window_size=3)
    result_w4 = roll_mean(data, window_size=4)
    np.testing.assert_array_almost_equal(result_w2, np.array([1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_array_almost_equal(result_w3, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_almost_equal(result_w4, np.array([2.5, 3.5]))

    # Test case 5: Large numbers
    data = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    result = roll_mean(data, window_size=3)
    expected = np.array([2e6, 3e6, 4e6])
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 6: Sinusoidal data (common in accelerometer signals)
    t = np.linspace(0, 2*np.pi, 100)
    data = np.sin(t)
    result = roll_mean(data, window_size=10)
    # The mean of a sliding window of sine wave should have smaller amplitude
    assert np.max(np.abs(result)) < np.max(np.abs(data))
    assert len(result) == len(data) - 9

    # Test case 7: Random data with known mean
    np.random.seed(42)
    data = np.random.normal(loc=5, scale=1, size=1000)
    result = roll_mean(data, window_size=100)
    # Mean should be close to 5 for large windows
    assert np.abs(np.mean(result) - 5) < 0.1
    assert len(result) == len(data) - 99

    # Test edge cases
    with pytest.raises(ValueError):
        roll_mean(np.array([1, 2]), window_size=3)  # Array smaller than window
    with pytest.raises(ValueError):
        roll_mean(np.array([1, 2, 3]), window_size=0)  # Invalid window size
    with pytest.raises(ValueError):
        roll_mean(np.array([]), window_size=1)  # Empty array


def test_roll_sd():
    # Test case 1: Basic sequence with known standard deviations
    data = np.array([1, 1, 1, 2, 2, 2])
    result = roll_sd(data, window_size=3)
    expected = np.array([0, 0.57735027, 0.57735027, 0])  # SD of sliding windows
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case 2: Sequence with negative numbers and decimals
    data = np.array([-1.5, 2.7, -3.2, 4.1, -5.9, 6.3])
    result = roll_sd(data, window_size=3)
    expected = np.array([3.0369941, 3.8742741, 5.173329, 6.5023073])  # Verified SDs
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case 3: Constant sequence (should return zeros)
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    result = roll_sd(data, window_size=3)
    np.testing.assert_array_equal(result, np.zeros(3))

    # Test case 4: Different window sizes
    data = np.array([1, 2, 3, 4, 5])
    result_w2 = roll_sd(data, window_size=2)
    np.testing.assert_array_almost_equal(result_w2, np.array([0.70710678, 0.70710678, 0.70710678, 0.70710678]))

    # Test case 5: Random data with known properties
    np.random.seed(42)
    data = np.random.normal(loc=5, scale=1, size=1000)
    result = roll_sd(data, window_size=100)
    # SD should be close to 1 for large windows
    assert 0.8 < np.mean(result) < 1.2
    assert len(result) == len(data) - 99

    # Test edge cases
    with pytest.raises(ValueError):
        roll_sd(np.array([1, 2]), window_size=3)  # Array smaller than window
    with pytest.raises(ValueError):
        roll_sd(np.array([1, 2, 3]), window_size=0)  # Invalid window size
    with pytest.raises(ValueError):
        roll_sd(np.array([]), window_size=1)  # Empty array
    

def test_sliding_window():
    # Test case 1: Basic sequence with step_size=1
    arr = np.array([1, 2, 3, 4, 5])
    result = sliding_window(arr, window_size=3, step_size=1)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(result, expected)

    # Test case 2: Different step sizes
    result_step2 = sliding_window(arr, window_size=2, step_size=2)
    expected_step2 = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(result_step2, expected_step2)

    # Test case 3: Window size equals array length
    result_full = sliding_window(arr, window_size=5, step_size=1)
    expected_full = np.array([[1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(result_full, expected_full)

    # Test case 4: Float values
    float_arr = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    result_float = sliding_window(float_arr, window_size=3, step_size=2)
    expected_float = np.array([[1.5, 2.7, 3.2], [3.2, 4.8, 5.1]])
    np.testing.assert_array_equal(result_float, expected_float)

    # Test edge cases
    # Test case 5: 2D array input
    arr_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    with pytest.raises(ValueError):
        sliding_window(arr_2d, window_size=2, step_size=1)

    # Test case 6: Empty array
    with pytest.raises(ValueError):
        sliding_window(np.array([]), window_size=1, step_size=1)

    # Test case 7: Window size larger than array
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=6, step_size=1)

    # Test case 8: Invalid step size
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=3, step_size=0)

    # Test case 9: Negative step size
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=3, step_size=-1)

    # Test case 10: Step size larger than array
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=2, step_size=6)


def test_resample_index():
    # Test case 1: Basic functionality with regular intervals
    index = pd.date_range(start='2024-01-01', periods=100, freq='1s')
    result = resample_index(index, window_samples=30, step_samples=10)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['start', 'end'])
    assert len(result) == (100 - 30) // 10 + 1
    
    # Test case 2: Verify window sizes and steps
    assert (result['end'] - result['start']).iloc[0] == pd.Timedelta(seconds=29)  # 30 samples - 1
    assert (result['start'].iloc[1] - result['start'].iloc[0]) == pd.Timedelta(seconds=10)
    
    # Test case 3: Different frequencies
    index_ms = pd.date_range(start='2024-01-01', periods=100, freq='100ms')
    result_ms = resample_index(index_ms, window_samples=20, step_samples=5)
    assert len(result_ms) == (100 - 20) // 5 + 1
    assert (result_ms['end'] - result_ms['start']).iloc[0] == pd.Timedelta(milliseconds=1900)
    
    # Test case 4: Edge case - window_samples equals length of index
    result_edge = resample_index(index, window_samples=100, step_samples=10)
    assert len(result_edge) == 1
    assert result_edge['start'].iloc[0] == index[0]
    assert result_edge['end'].iloc[0] == index[-1]
    
    # Test edge cases and error conditions
    with pytest.raises(ValueError):
        resample_index(index, window_samples=0, step_samples=10)  # Invalid window size
    
    with pytest.raises(ValueError):
        resample_index(index, window_samples=30, step_samples=0)  # Invalid step size
    
    with pytest.raises(ValueError):
        resample_index(index, window_samples=101, step_samples=10)  # Window larger than index
    
    with pytest.raises(ValueError):
        resample_index(pd.DatetimeIndex([]), window_samples=10, step_samples=5)  # Empty index


def test_calc_weartime():
    # Test case 1: Basic test with 1Hz sampling
    timestamps_1hz = pd.date_range(start='2024-01-01', periods=10, freq='1s')
    df_1hz = pd.DataFrame({
        'wear': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    }, index=timestamps_1hz)
    
    total, wear, nonwear = calc_weartime(df_1hz, sf=1)
    assert total == 9.0  # 9 seconds between first and last timestamp
    assert wear == 7.0   # 7 samples with wear=1
    assert nonwear == 2.0  # total - wear time
    assert abs((total - (wear + nonwear))) < 1e-10  # times should sum to total

    # Test case 2: Test with 2Hz sampling frequency
    timestamps_2hz = pd.date_range(start='2024-01-01', periods=10, freq='500ms')
    df_2hz = pd.DataFrame({
        'wear': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    }, index=timestamps_2hz)
    
    total, wear, nonwear = calc_weartime(df_2hz, sf=2)
    assert total == 4.5  # 4.5 seconds between first and last timestamp
    assert wear == 3.5   # 7 samples * 0.5 seconds each
    assert nonwear == 1.0  # 2 samples * 0.5 seconds each
    assert abs((total - (wear + nonwear))) < 1e-10

    # Test case 3: Empty DataFrame
    empty_df = pd.DataFrame({'wear': []}, index=pd.DatetimeIndex([]))
    with pytest.raises(IndexError):
        calc_weartime(empty_df, sf=1)

    # Test case 4: Single sample
    timestamp_single = pd.date_range(start='2024-01-01', periods=1)
    df_single = pd.DataFrame({'wear': [1]}, index=timestamp_single)
    
    total, wear, nonwear = calc_weartime(df_single, sf=1)
    assert total == 0.0  # No time difference with single sample
    assert wear == 1.0   # One wear sample
    assert nonwear == -1.0  # total - wear


def test_detect_wear():
    # Test 1: Valid input with wear/non-wear periods
    # Create mock data: First 30s movement (wear), last 30s no movement (non-wear)
    # Test parameters
    sampling_freq = 50  # Hz
    duration = 60*60*24  # seconds
    timestamps = pd.date_range(
        start='2024-01-01', 
        periods=duration * sampling_freq, 
        freq=f'{1000/sampling_freq}ms'
    )

    n_samples = len(timestamps)
    mid_point = n_samples // 2
    
    movement_data = np.random.normal(loc=0, scale=0.1, size=(mid_point, 3))
    no_movement_data = np.zeros((n_samples - mid_point, 3))
    acc_data = np.vstack([movement_data, no_movement_data])
    
    df = pd.DataFrame(
        acc_data, 
        columns=['X', 'Y', 'Z'], 
        index=timestamps
    )
    
    result = detect_wear(df, sampling_freq)
    
    # Verify basic properties
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert len(result) > 0
    assert result['wear'].between(0, 1).all()
    
    # Verify wear detection accuracy
    mid_time = timestamps[mid_point]
    wear_period = result.loc[:mid_time]['wear'].mean()
    non_wear_period = result.loc[mid_time:]['wear'].mean()
    assert wear_period > 0.7  # First half should be mostly wear
    assert non_wear_period < 0.3  # Second half should be mostly non-wear

    # Test 2: Valid input with wear/non-wear periods
    # Create mock data: First 30s movement (wear), last 30s no movement (non-wear)
    # Test parameters
    sampling_freq = 50  # Hz
    duration = 60  # seconds
    timestamps = pd.date_range(
        start='2024-01-01', 
        periods=duration * sampling_freq, 
        freq=f'{1000/sampling_freq}ms'
    )
    
    n_samples = len(timestamps)
    mid_point = n_samples // 2
    
    movement_data = np.random.normal(loc=0, scale=0.1, size=(mid_point, 3))
    no_movement_data = np.zeros((n_samples - mid_point, 3))
    acc_data = np.vstack([movement_data, no_movement_data])
    
    df = pd.DataFrame(
        acc_data, 
        columns=['X', 'Y', 'Z'], 
        index=timestamps
    )
    
    with pytest.raises(ValueError):
        detect_wear(df, sampling_freq)
    
    # Test 2: Invalid input - missing columns
    df_missing_cols = pd.DataFrame({
        'X': [1, 2, 3],
        'Y': [1, 2, 3]
    })
    with pytest.raises(ValueError):
        detect_wear(df_missing_cols, 50)
    
    # Test 3: Invalid input - empty DataFrame
    df_empty = pd.DataFrame(columns=['X', 'Y', 'Z'])
    with pytest.raises(ValueError):
        detect_wear(df_empty, 50)


def test_remove_noise():
    # Create sample data
    sample_size = 1000
    time_index = pd.date_range(start='2023-01-01', periods=sample_size, freq='12.5ms')
    
    # Generate noisy sine waves for X, Y, Z
    t = np.linspace(0, 10, sample_size)
    noise = np.random.normal(0, 0.5, sample_size)
    
    data = {
        'X': np.sin(2 * np.pi * 0.5 * t) + noise,
        'Y': np.sin(2 * np.pi * 0.3 * t + np.pi/4) + noise,
        'Z': np.sin(2 * np.pi * 0.7 * t + np.pi/2) + noise
    }
    
    df = pd.DataFrame(data, index=time_index)
    
    # Apply noise removal
    filtered_df = remove_noise(df, sf=80)
    
    # Assertions
    assert isinstance(filtered_df, pd.DataFrame)
    assert filtered_df.shape == df.shape
    assert all(col in filtered_df.columns for col in ['X', 'Y', 'Z'])

    # Check that filtered data has less variance than original
    for col in ['X', 'Y', 'Z']:
        assert filtered_df[col].var() < df[col].var()
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        remove_noise(pd.DataFrame(), sf=80)  # Empty DataFrame
    
    with pytest.raises(KeyError):
        remove_noise(pd.DataFrame({'A': [1, 2, 3]}), sf=80)  # Missing required columns


def test_read_smartwatch_data(tmp_path):
    """Test read_smartwatch_data function with various scenarios."""
    # Test 1: Successful case with valid data
    # Create test data with known frequency (50Hz)
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='20ms')
    data = {
        'HEADER_TIMESTAMP': timestamps,
        'X': np.random.normal(0, 1, 100),
        'Y': np.random.normal(0, 1, 100),
        'Z': np.random.normal(0, 1, 100)
    }
    df = pd.DataFrame(data)
    
    # Create multiple CSV files
    for i in range(3):
        file_path = tmp_path / f"test_{i}.sensor.csv"
        df_slice = df.iloc[i*33:(i+1)*33]  # Split data into 3 files
        df_slice.to_csv(file_path, index=False)
    
    # Test successful reading
    result_df, freq = read_smartwatch_data(tmp_path)
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(freq, float)
    assert not result_df.empty
    assert all(col in result_df.columns for col in ['X', 'Y', 'Z'])
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert result_df.index.is_monotonic_increasing
    assert abs(freq - 50) < 1  # Should be close to 50Hz
    
    # Test 2: Empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    df_empty, freq_empty = read_smartwatch_data(empty_dir)
    assert df_empty.empty
    assert freq_empty is None
    
    # Test 3: Invalid CSV file
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    invalid_file = invalid_dir / "invalid.sensor.csv"
    with open(invalid_file, 'w') as f:
        f.write("invalid,csv,content\n")
    
    df_invalid, freq_invalid = read_smartwatch_data(invalid_dir)
    assert df_invalid.empty
    assert freq_invalid is None
    
    # Test 4: Inconsistent frequency
    inconsistent_dir = tmp_path / "inconsistent"
    inconsistent_dir.mkdir()
    
    # Create data with inconsistent frequency using just 3 timestamps
    timestamps1 = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:02', '2023-01-01 00:00:04'])
    timestamps2 = pd.to_datetime(['2023-01-01 00:00:05', '2023-01-01 00:00:06', '2023-01-01 00:00:07'])
    
    data1 = {
        'HEADER_TIMESTAMP': timestamps1,
        'X': [0.1, 0.2, 0.3],
        'Y': [0.4, 0.5, 0.6],
        'Z': [0.7, 0.8, 0.9]
    }
    data2 = {
        'HEADER_TIMESTAMP': timestamps2,
        'X': [0.1, 0.2, 0.3],
        'Y': [0.4, 0.5, 0.6],
        'Z': [0.7, 0.8, 0.9]
    }
    
    pd.DataFrame(data1).to_csv(inconsistent_dir / "data1.sensor.csv", index=False)
    pd.DataFrame(data2).to_csv(inconsistent_dir / "data2.sensor.csv", index=False)
    
    with pytest.raises(ValueError, match="Inconsistent timestamp frequency detected."):
        read_smartwatch_data(inconsistent_dir)
    
    # Test 5: Nonexistent directory
    df_nonexist, freq_nonexist = read_smartwatch_data("nonexistent_directory")
    assert df_nonexist.empty
    assert freq_nonexist is None


def test_rescore_wear_detection():
    # Create test data with alternating wear/non-wear periods
    test_data = pd.DataFrame({
        'wear': [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],  # 10 periods
        'start': pd.date_range('2024-01-01', periods=10, freq='15min'),
        'end': pd.date_range('2024-01-01', periods=10, freq='15min') + pd.Timedelta('15min')
    })
    
    # Test the function
    result = rescore_wear_detection(test_data)
    
    # Basic assertions
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert len(result) == len(test_data)
    assert all(x in [0, 1] for x in result['wear'])  # Check values are binary
    assert 'block' not in result.columns  # Check temporary 'block' column was removed


def test_auto_calibrate():
    # Create simple test data
    n_samples = 1000
    time_index = pd.date_range('2023-01-01', periods=n_samples, freq='20ms')
    
    # Generate random accelerometer data
    np.random.seed(42)
    data = np.random.normal(0, 1, (n_samples, 3))
    
    # Create input DataFrame
    df = pd.DataFrame(
        data,
        columns=['X', 'Y', 'Z'],
        index=time_index
    )
    
    # Run auto_calibration
    calibrated_df = auto_calibrate(df, sf=50)
    
    # Check that output shape matches input shape
    assert calibrated_df.shape == df.shape
    assert list(calibrated_df.columns) == ['X', 'Y', 'Z']
    assert (calibrated_df.index == df.index).all()


def test_preprocess_smartwatch_data():
    # Create sample data
    n_samples = 600000  # 10 seconds of data at 100Hz
    sf = 80  # sampling frequency in Hz
    
    # Generate timestamps
    timestamps = pd.date_range(
        start='2023-01-01', 
        periods=n_samples, 
        freq=f'{1000/sf}ms'
    )
    
    # Create synthetic accelerometer data with some noise
    data = {
        'X': np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples),
        'Y': np.cos(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples),
        'Z': np.sin(np.linspace(0, 5*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples)
    }
    
    df = pd.DataFrame(data, index=timestamps)
    meta_dict = {}
    
    # Process the data
    result = preprocess_smartwatch_data(df, sf, meta_dict, verbose=False)
    
    # Basic assertions
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['X', 'Y', 'Z', 'wear'])
    assert len(result) == len(df)
    #assert all(result['wear'].isin([0, 1]))  # wear column should only contain 0s and 1s
    assert all(key in meta_dict for key in ['total time', 'wear time', 'non-wear time'])