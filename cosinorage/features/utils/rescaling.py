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

import numpy as np
import pandas as pd


def min_max_scaling_exclude_outliers(data, upper_quantile=0.999):
    """
    Scales the data using min-max scaling to a [0,100] range, excluding outliers based on quantiles.

    This function applies min-max scaling to normalize data to a [0,100] range while
    using robust bounds that exclude extreme outliers. Values above the upper quantile
    threshold are not excluded from the final result but may exceed 100.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Input data to be scaled. Can be either a pandas Series or numpy array of numeric values.
    upper_quantile : float, default=0.999
        Upper quantile threshold for excluding outliers when calculating min/max bounds.
        Defaults to 0.999 (99.9th percentile).

    Returns
    -------
    pd.Series
        Scaled data with values generally ranging from 0 to 100.

    Notes
    -----
    - If input contains all identical values, returns zeros
    - Values above the upper_quantile may exceed 100 in the output
    - Output maintains the same length as input

    Raises
    ------
    ValueError
        If input data is empty.

    Notes
    -----
    - Uses quantile-based outlier detection for robust scaling
    - Applies min-max scaling: (x - min) / (max - min) * 100
    - Handles edge cases like constant data and zero division
    - Preserves outliers in output but uses robust bounds for scaling
    - Useful for normalizing accelerometer data while handling extreme values

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Example with normal data
    >>> data = pd.Series([1, 2, 3, 4, 5])
    >>> scaled = min_max_scaling_exclude_outliers(data)
    >>> print(scaled)
    >>> # Output: [0.0, 25.0, 50.0, 75.0, 100.0]
    >>>
    >>> # Example with outliers
    >>> data_with_outliers = pd.Series([1, 2, 3, 100])
    >>> scaled = min_max_scaling_exclude_outliers(data_with_outliers, upper_quantile=0.75)
    >>> print(scaled)
    >>> # Output: [0.0, 50.0, 100.0, 4950.0] (outlier exceeds 100)
    >>>
    >>> # Example with constant data
    >>> constant_data = pd.Series([5, 5, 5, 5])
    >>> scaled = min_max_scaling_exclude_outliers(constant_data)
    >>> print(scaled)
    >>> # Output: [0.0, 0.0, 0.0, 0.0]
    """
    # Convert to pandas Series if input is numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Check for empty input
    if len(data) == 0:
        raise ValueError("Input data cannot be empty")

    # Handle single value or constant values
    if len(data.unique()) == 1:
        return pd.Series(np.zeros(len(data)))

    # Calculate the upper bound based on quantiles
    upper_bound = data.quantile(upper_quantile)

    # Filter data to exclude outliers
    filtered_data = data[data <= upper_bound]

    # Calculate min and max of the filtered data
    min_val = filtered_data.min()
    max_val = filtered_data.max()

    # Handle zero division case
    if max_val == min_val:
        return pd.Series(np.zeros(len(data)))

    # Apply min-max scaling to [0,100] - outliers may overshoot
    scaled_data = 100 * (data - min_val) / (max_val - min_val)

    return scaled_data
