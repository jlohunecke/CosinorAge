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

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..features.utils.cosinor_analysis import cosinor_multiday

# model parameters
model_params_generic = {
    "shape": 0.01462774,
    "rate": -13.36715309,
    "mesor": -0.03204933,
    "amp1": -0.01971357,
    "phi1": -0.01664718,
    "age": 0.10033692,
}

model_params_female = {
    "shape": 0.01294402,
    "rate": -13.28530410,
    "mesor": -0.02569062,
    "amp1": -0.02170987,
    "phi1": -0.13191562,
    "age": 0.08840283,
}

model_params_male = {
    "shape": 0.013878454,
    "rate": -13.016951633,
    "mesor": -0.023988922,
    "amp1": -0.030620390,
    "phi1": 0.008960155,
    "age": 0.101726103,
}

m_n = -1.405276
m_d = 0.01462774
BA_n = -0.01447851
BA_d = 0.112165
BA_i = 133.5989


class CosinorAge:
    """A class to compute biological age predictions using the CosinorAge method.

    This class implements the CosinorAge method proposed by Shim, Fleisch and Barata
    for predicting biological age based on accelerometer data patterns.

    Args:
        records (List[dict]): A list of dictionaries containing accelerometer data records.
            Each record must contain a 'handler' key with a DataHandler object and an 'age' key.
            Optionally, records can include a 'gender' key with values 'male', 'female', or 'unknown'.
    """

    def __init__(self, records: List[dict]):
        self.records = records

        self.model_params_generic = model_params_generic
        self.model_params_female = model_params_female
        self.model_params_male = model_params_male

        self.__compute_cosinor_ages()

    def __compute_cosinor_ages(self):
        """Compute CosinorAge predictions for all records.

        Processes each record to extract cosinor parameters and calculate biological age.
        Updates each record dictionary with the following keys:
            - mesor: The rhythm-adjusted mean
            - amp1: The amplitude of the circadian rhythm
            - phi1: The acrophase (timing) of the circadian rhythm
            - cosinorage: Predicted biological age
            - cosinorage_advance: Difference between predicted and chronological age
        """
        import numpy as np
        import pandas as pd
        
        for record in self.records:
            try:
                result = cosinor_multiday(record["handler"].get_ml_data())[0]

                # Check if cosinor parameters are valid
                mesor = result["mesor"]
                amplitude = result["amplitude"]
                acrophase = result["acrophase"]
                
                # Validate cosinor parameters
                if (pd.isna(mesor) or np.isnan(mesor) or np.isinf(mesor) or
                    pd.isna(amplitude) or np.isnan(amplitude) or np.isinf(amplitude) or
                    pd.isna(acrophase) or np.isnan(acrophase) or np.isinf(acrophase)):
                    
                    # Set invalid values for this record
                    record["mesor"] = None
                    record["amp1"] = None
                    record["phi1"] = None
                    record["cosinorage"] = None
                    record["cosinorage_advance"] = None
                    continue

                record["mesor"] = mesor
                record["amp1"] = amplitude
                record["phi1"] = acrophase

                bm_data = {
                    "mesor": mesor,
                    "amp1": amplitude,
                    "phi1": acrophase,
                    "age": record["age"],
                }

                gender = record.get("gender", "unknown")
                if gender == "female":
                    coef = self.model_params_female
                elif gender == "male":
                    coef = self.model_params_male
                else:
                    coef = self.model_params_generic

                n1 = {key: bm_data[key] * coef[key] for key in bm_data}
                xb = sum(n1.values()) + coef["rate"]
                m_val = 1 - np.exp((m_n * np.exp(xb)) / m_d)
                cosinorage = float(
                    ((np.log(BA_n * np.log(1 - m_val))) / BA_d) + BA_i
                )

                record["cosinorage"] = float(cosinorage)
                record["cosinorage_advance"] = float(
                    record["cosinorage"] - record["age"]
                )
                
            except Exception as e:
                # Set invalid values for this record if any error occurs
                record["mesor"] = None
                record["amp1"] = None
                record["phi1"] = None
                record["cosinorage"] = None
                record["cosinorage_advance"] = None

    def get_predictions(self):
        """Return the processed records with CosinorAge predictions.

        Returns:
            List[dict]: The records list containing the original data and predictions.
        """
        return self.records

    def plot_predictions(self):
        """Generate visualization plots comparing chronological age vs CosinorAge.

        Creates a plot for each record showing:
            - Chronological age and CosinorAge as points on a timeline
            - Color-coded difference between ages (red for advanced, green for younger)
            - Numerical values for both ages
        """
        for record in self.records:
            # Skip records with invalid cosinorage values
            if record["cosinorage"] is None:
                print(f"Skipping plot for record with invalid cosinorage value")
                continue
                
            plt.figure(figsize=(22.5, 2.5))
            plt.hlines(
                y=0,
                xmin=0,
                xmax=min(record["age"], record["cosinorage"]),
                color="grey",
                alpha=0.8,
                linewidth=2,
                zorder=1,
            )

            if record["cosinorage"] > record["age"]:
                color = "red"
            else:
                color = "green"

            plt.hlines(
                y=0,
                xmin=min(record["age"], record["cosinorage"]),
                xmax=max(record["age"], record["cosinorage"]),
                color=color,
                alpha=0.8,
                linewidth=2,
                zorder=1,
            )

            plt.scatter(
                record["cosinorage"],
                0,
                color=color,
                s=100,
                marker="o",
                label="CosinorAge",
            )
            plt.scatter(
                record["age"], 0, color=color, s=100, marker="o", label="Age"
            )

            plt.text(
                record["cosinorage"],
                0.4,
                "CosinorAge",
                fontsize=12,
                color=color,
                alpha=0.8,
                ha="center",
                va="bottom",
                rotation=45,
            )
            plt.text(
                record["age"],
                0.4,
                "Age",
                fontsize=12,
                color=color,
                alpha=0.8,
                ha="center",
                va="bottom",
                rotation=45,
            )
            plt.text(
                record["age"],
                -0.5,
                f"{record['age']:.1f}",
                fontsize=12,
                color=color,
                alpha=0.8,
                ha="center",
                va="top",
                rotation=45,
            )
            plt.text(
                record["cosinorage"],
                -0.5,
                f"{record['cosinorage']:.1f}",
                fontsize=12,
                color=color,
                alpha=0.8,
                ha="center",
                va="top",
                rotation=45,
            )

            plt.xlim(0, max(record["age"], record["cosinorage"]) * 1.25)
            plt.yticks([])
            plt.ylim(-1.5, 2)
