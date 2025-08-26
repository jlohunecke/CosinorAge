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

from ..datahandlers import DataHandler
from .utils.cosinor_analysis import *
from .utils.nonparam_analysis import *
from .utils.physical_activity_metrics import *
from .utils.rescaling import *
from .utils.sleep_metrics import *


class WearableFeatures:
    """A class for computing and managing features from wearable accelerometer data.

    This class processes raw ENMO (Euclidean Norm Minus One) data to compute various
    circadian rhythm and physical activity metrics, including cosinor analysis,
    non-parametric measures, activity levels, and sleep metrics.

    Attributes
    ----------
    ml_data : pd.DataFrame
        Minute-level ENMO data with datetime index
    features_args : dict
        Arguments passed to feature computation functions
    feature_dict : dict
        Dictionary containing computed features organized by category
    """

    def __init__(self, handler: DataHandler, features_args: dict = {}):
        """Initialize WearableFeatures with data from a DataHandler.

        Parameters
        ----------
        handler : DataHandler
            DataHandler instance containing ENMO data
        """
        self.ml_data = handler.get_ml_data().copy()
        self.features_args = features_args

        self.feature_dict = {}

        self.__run()

    def __run(self):
        """Compute all available features at once."""

        self.__compute_cosinor_features()
        self.__compute_nonparam_features()
        self.__compute_physical_activity_metrics()
        self.__compute_sleep_metrics()

    def __compute_cosinor_features(self):
        """Compute cosinor analysis features including MESOR, amplitude, and acrophase.

        Updates feature_dict with:
            - MESOR: Midline Estimating Statistic Of Rhythm
            - amplitude: Half the peak-to-trough difference
            - acrophase: Peak time of the rhythm in radians
            - acrophase_time: Peak time of the rhythm in minutes from midnight
        """

        cosinor_dict = {}

        params, fitted = cosinor_multiday(self.ml_data)

        cosinor_dict.update(params)
        self.feature_dict["cosinor"] = cosinor_dict

        self.ml_data["cosinor_fitted"] = fitted

    def __compute_nonparam_features(self):
        """Compute non-parametric features including IV, IS, RA, M10, and L5."""

        nonparam_dict = {}

        nonparam_dict["IS"] = IS(self.ml_data)
        if nonparam_dict["IS"] > 1 or nonparam_dict["IS"] < 0:
            nonparam_dict["IS_flag"] = (
                "invalid IS value - must be between 0 and 1"
            )

        nonparam_dict["IV"] = IV(self.ml_data)  #
        if nonparam_dict["IV"] > 2:
            nonparam_dict["IV_flag"] = (
                "ultradian rhythm or small sample size (due to IV > 2)"
            )
        if nonparam_dict["IV"] < 0:
            nonparam_dict["IV_flag"] = (
                "invalid IV value - must be greater than 0"
            )

        res = M10(self.ml_data)
        nonparam_dict["M10"] = res[0]
        nonparam_dict["M10_start"] = res[1]

        res = L5(self.ml_data)
        nonparam_dict["L5"] = res[0]
        nonparam_dict["L5_start"] = res[1]

        if nonparam_dict["M10"] < nonparam_dict["L5"]:
            nonparam_dict["M10_L5_flag"] = (
                "M10 is less than L5 - check for errors in non-parametric analysis"
            )

        if "M10" in nonparam_dict.keys() and "L5" in nonparam_dict.keys():
            nonparam_dict["RA"] = RA(nonparam_dict["M10"], nonparam_dict["L5"])
            if not all(0 <= ra <= 1 for ra in nonparam_dict["RA"]):
                nonparam_dict["RA_flag"] = (
                    "invalid RA value - must be between 0 and 1"
                )

        self.feature_dict["nonparam"] = nonparam_dict

    def __compute_physical_activity_metrics(self):
        """Compute physical activity metrics including SB, LIPA, and MVPA."""

        physical_activity_dict = {}
        physical_activity_columns = [
            "sedentary",
            "light",
            "moderate",
            "vigorous",
        ]

        res = activity_metrics(self.ml_data, pa_params=self.features_args)
        physical_activity_dict["sedentary"] = res[0]
        physical_activity_dict["light"] = res[1]
        physical_activity_dict["moderate"] = res[2]
        physical_activity_dict["vigorous"] = res[3]
        self.feature_dict["physical_activity"] = physical_activity_dict

    def __compute_sleep_metrics(self):
        """Compute sleep metrics including TST, WASO, PTA, NWB, SOL, and SRI."""

        if "sleep" not in self.ml_data.columns:
            self.ml_data["sleep"] = apply_sleep_wake_predictions(
                self.ml_data, sleep_params=self.features_args
            )

        sleep_dict = {}

        sleep_dict["TST"] = TST(self.ml_data)
        sleep_dict["WASO"] = WASO(self.ml_data)
        sleep_dict["PTA"] = PTA(self.ml_data)
        sleep_dict["NWB"] = NWB(self.ml_data)
        sleep_dict["SOL"] = SOL(self.ml_data)
        sleep_dict["SRI"] = SRI(self.ml_data)

        if sleep_dict["SRI"] < 0:
            sleep_dict["SRI_flag"] = (
                "negative SRI - very low sleep consistency"
            )

        self.feature_dict["sleep"] = sleep_dict

    def get_features(self):
        """Returns the entire feature DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all computed features
        """
        return self.feature_dict

    def get_ml_data(self):
        """Returns the raw ENMO data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing ENMO data with datetime index
        """
        return self.ml_data
