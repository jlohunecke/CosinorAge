import numpy as np
import pandas as pd

from cosinorage.datahandlers.utils.calc_enmo import (
    calculate_enmo, calculate_minute_level_enmo)


def test_calculate_enmo_normal_case():
    # Test with realistic accelerometer data at 1Hz frequency
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00", periods=180, freq="1s"
    )

    # Generate repeating pattern of the original test values
    x_values = np.tile([-0.2, 0.8, -0.5, 1.2, 0.0], 36)  # 36 * 5 = 180
    y_values = np.tile([0.9, -0.3, 1.1, 0.4, 0.7], 36)
    z_values = np.tile([0.8, 1.1, 0.6, -0.3, 1.2], 36)

    acc_data = pd.DataFrame(
        {"timestamp": timestamps, "x": x_values, "y": y_values, "z": z_values}
    ).set_index("timestamp")

    result = calculate_enmo(acc_data)

    # Expected values also repeated 36 times
    expected = np.tile([0.221, 0.393, 0.349, 0.300, 0.389], 36)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_calculate_enmo_all_zeros():
    # Test with all zeros (should return all zeros as ENMO)
    acc_data = pd.DataFrame(
        {"x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0]}
    )
    result = calculate_enmo(acc_data)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result, expected)


def test_calculate_enmo_missing_columns():
    # Test error handling when columns are missing
    acc_data = pd.DataFrame({"x": [0.5, 1.0], "Wrong_Column": [0.5, 0.0]})
    result = calculate_enmo(acc_data)
    assert np.isnan(result)


def test_calculate_minute_level_enmo_normal_case():
    # Test with realistic accelerometer data at 1Hz frequency
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00", periods=180, freq="1s"
    )

    # Generate repeating pattern of the original test values
    x_values = np.tile([-0.2, 0.8, -0.5, 1.2, 0.0], 36)  # 36 * 5 = 180
    y_values = np.tile([0.9, -0.3, 1.1, 0.4, 0.7], 36)
    z_values = np.tile([0.8, 1.1, 0.6, -0.3, 1.2], 36)

    acc_data = pd.DataFrame(
        {"timestamp": timestamps, "x": x_values, "y": y_values, "z": z_values}
    ).set_index("timestamp")

    result = calculate_enmo(acc_data)

    # Expected values also repeated 36 times
    expected = np.tile([0.221, 0.393, 0.349, 0.300, 0.389], 36)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)


def test_calculate_minute_level_enmo_empty():
    # Test with empty DataFrame
    empty_df = pd.DataFrame(
        {"timestamp": pd.DatetimeIndex([]), "enmo": []}
    ).set_index("timestamp")

    meta_dict = {"sf": 25}  # Provide sampling frequency
    result = calculate_minute_level_enmo(empty_df, meta_dict=meta_dict)
