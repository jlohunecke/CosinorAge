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

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from skdh.preprocessing import CalibrateAccelerometer


def calibrate_accelerometer(
    data: pd.DataFrame,
    sphere_crit: float,
    sd_criteria: float,
    meta_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calibrate accelerometer data using sphere fitting method.

    This function applies accelerometer calibration using the sphere fitting approach
    to correct for sensor bias and scaling errors. The calibration process fits the
    accelerometer data to a unit sphere and applies correction factors.

    Parameters
    ----------
    data : pd.DataFrame
        Raw accelerometer data with datetime index and columns ['x', 'y', 'z'].
        Data should be in g units (1g = 9.81 m/s²).
    sphere_crit : float
        Sphere fitting criterion threshold. Controls the tolerance for sphere fitting.
        Lower values result in stricter calibration requirements.
    sd_criteria : float
        Standard deviation criterion threshold. Controls the tolerance for standard
        deviation of the calibrated data.
    meta_dict : dict, optional
        Dictionary to store calibration parameters and metadata. If None, an empty
        dict will be created. Updated with calibration results including:
        - 'calibration_offset': Offset correction factors
        - 'calibration_scale': Scale correction factors
    verbose : bool, default=False
        Whether to print progress information during calibration.

    Returns
    -------
    pd.DataFrame
        Calibrated accelerometer data with the same structure as input data.
        The calibrated data has corrected bias and scaling errors.

    Notes
    -----
    - The function uses the skdh.preprocessing.CalibrateAccelerometer class
    - Calibration parameters are stored in meta_dict for future reference
    - The function assumes data is sampled at the frequency specified in meta_dict['sf']
    - If no sampling frequency is found in meta_dict, defaults to 25 Hz

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample accelerometer data
    >>> timestamps = pd.date_range('2023-01-01', periods=1000, freq='40ms')
    >>> data = pd.DataFrame({
    ...     'x': np.random.normal(0, 0.1, 1000),
    ...     'y': np.random.normal(0, 0.1, 1000),
    ...     'z': np.random.normal(1, 0.1, 1000)  # Gravity component
    ... }, index=timestamps)
    >>>
    >>> # Calibrate the data
    >>> meta_dict = {'sf': 25}
    >>> calibrated_data = calibrate_accelerometer(
    ...     data, sphere_crit=0.3, sd_criteria=0.1,
    ...     meta_dict=meta_dict, verbose=True
    ... )
    >>> print(f"Calibration offset: {meta_dict.get('calibration_offset')}")
    """
    if meta_dict is None:
        meta_dict = {}

    _data = data.copy()

    time = np.array(_data.index.astype("int64") // 10**9)
    acc = np.array(_data[["x", "y", "z"]]).astype(np.float64)

    calibrator = CalibrateAccelerometer(
        sphere_crit=sphere_crit, min_hours=24, sd_crit=sd_criteria
    )
    sf = meta_dict.get("sf", 25)  # Default to 25Hz if not specified
    result = calibrator.predict(time=time, accel=acc, fs=sf)

    if "accel" in result:
        _data = pd.DataFrame(result["accel"], columns=["x", "y", "z"])
    else:
        _data = pd.DataFrame(acc, columns=["x", "y", "z"])

    _data.set_index(data.index, inplace=True)

    if "offset" in result:
        meta_dict.update({"calibration_offset": result["offset"]})
    if "scale" in result:
        meta_dict.update({"calibration_scale": result["scale"]})

    if verbose:
        print("Calibration done")

    return _data[["x", "y", "z"]]
