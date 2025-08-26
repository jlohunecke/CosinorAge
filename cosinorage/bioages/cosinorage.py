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
    for predicting biological age based on accelerometer data patterns. The method
    uses cosinor analysis to extract circadian rhythm parameters (MESOR, amplitude, 
    acrophase) from accelerometer data and applies gender-specific regression models
    to predict biological age.

    The CosinorAge method is based on the principle that circadian rhythm patterns
    in physical activity are associated with biological aging. By analyzing the
    daily activity patterns using cosinor analysis, the method can predict whether
    an individual's biological age is advanced or delayed compared to their
    chronological age.

    Attributes
    ----------
    records : List[dict]
        List of dictionaries containing accelerometer data records with computed predictions.
        Each record contains the original data plus computed cosinor parameters and
        biological age predictions.
    model_params_generic : dict
        Model parameters for generic gender classification (used when gender is 'unknown').
    model_params_female : dict
        Model parameters for female gender classification.
    model_params_male : dict
        Model parameters for male gender classification.

    Examples
    --------
    Basic usage with a single participant:

    >>> from cosinorage.bioages import CosinorAge
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Create a data handler with accelerometer data
    >>> handler = GenericDataHandler(
    ...     file_path='data/participant_001.csv',
    ...     data_type='accelerometer-mg',
    ...     time_column='timestamp',
    ...     data_columns=['x', 'y', 'z']
    ... )
    >>>
    >>> # Create a record with age and gender information
    >>> record = {
    ...     'handler': handler,
    ...     'age': 45.5,
    ...     'gender': 'female'
    ... }
    >>>
    >>> # Compute CosinorAge predictions
    >>> cosinor_age = CosinorAge([record])
    >>> predictions = cosinor_age.get_predictions()
    >>>
    >>> # Access the results
    >>> result = predictions[0]
    >>> print(f"Chronological age: {result['age']:.1f}")
    >>> print(f"Predicted biological age: {result['cosinorage']:.1f}")
    >>> print(f"Age advance: {result['cosinorage_advance']:.1f}")
    >>> print(f"MESOR: {result['mesor']:.4f}")
    >>> print(f"Amplitude: {result['amp1']:.4f}")
    >>> print(f"Acrophase: {result['phi1']:.4f}")

    Multiple participants with different genders:

    >>> from cosinorage.datahandlers import GalaxyDataHandler
    >>>
    >>> # Create multiple data handlers
    >>> handlers = []
    >>> for i in range(3):
    ...     handler = GalaxyDataHandler(f'data/participant_{i+1:03d}.csv')
    ...     handlers.append(handler)
    >>>
    >>> # Create records with different ages and genders
    >>> records = [
    ...     {'handler': handlers[0], 'age': 30.2, 'gender': 'male'},
    ...     {'handler': handlers[1], 'age': 45.8, 'gender': 'female'},
    ...     {'handler': handlers[2], 'age': 62.1, 'gender': 'unknown'}
    ... ]
    >>>
    >>> # Compute predictions for all participants
    >>> cosinor_age = CosinorAge(records)
    >>> predictions = cosinor_age.get_predictions()
    >>>
    >>> # Analyze results
    >>> for i, pred in enumerate(predictions):
    ...     print(f"Participant {i+1}:")
    ...     print(f"  Age: {pred['age']:.1f}, Gender: {pred['gender']}")
    ...     print(f"  Biological age: {pred['cosinorage']:.1f}")
    ...     print(f"  Age advance: {pred['cosinorage_advance']:.1f}")
    ...     if pred['cosinorage_advance'] > 0:
    ...         print("  Status: Biologically older than chronological age")
    ...     else:
    ...         print("  Status: Biologically younger than chronological age")

    Notes
    -----
    - The method requires at least 24 hours of continuous accelerometer data
    - Data should be preprocessed to minute-level ENMO values
    - Gender-specific models provide more accurate predictions than the generic model
    - Invalid cosinor parameters (NaN, inf) result in None values for predictions
    - The method automatically handles missing or invalid data gracefully
    - Age advance > 0 indicates biological age is older than chronological age
    - Age advance < 0 indicates biological age is younger than chronological age

    References
    ----------
    Shim, S., Fleisch, E., & Barata, F. (2024). CosinorAge: A novel method for
    predicting biological age from accelerometer data using circadian rhythm
    analysis. npj Digital Medicine, 7(1), 1-12.
    """

    def __init__(self, records: List[dict]):
        """
        Initialize CosinorAge with accelerometer data records.

        This method initializes the CosinorAge calculator and immediately computes
        biological age predictions for all provided records. The computation is
        performed automatically during initialization.

        Parameters
        ----------
        records : List[dict]
            A list of dictionaries containing accelerometer data records.
            Each record must contain:
            - 'handler': A DataHandler object (e.g., GenericDataHandler, GalaxyDataHandler)
              that provides minute-level ENMO data via get_ml_data() method
            - 'age': Chronological age as a float (e.g., 45.5)
            
            Each record may optionally contain:
            - 'gender': Gender classification as string ('male', 'female', or 'unknown')
              If not provided, defaults to 'unknown' and uses the generic model

        Notes
        -----
        - The computation is performed immediately during initialization
        - Each record is processed independently
        - Failed computations (invalid data) result in None values for predictions
        - Gender-specific models are used when gender is 'male' or 'female'
        - Generic model is used when gender is 'unknown' or not provided
        """
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

        This method returns the complete records list with all computed predictions
        and cosinor parameters. Each record contains the original input data plus
        the computed biological age predictions and circadian rhythm parameters.

        Returns
        -------
        List[dict]
            The records list containing the original data and predictions.
            Each record dictionary includes:
            - Original keys: 'handler', 'age', 'gender'
            - Computed cosinor parameters: 'mesor', 'amp1', 'phi1'
            - Biological age predictions: 'cosinorage', 'cosinorage_advance'

        Notes
        -----
        - Returns the same records that were passed to the constructor
        - Each record is updated in-place with computed predictions
        - Failed computations result in None values for prediction fields
        - The method can be called multiple times without recomputation
        """
        return self.records

    def plot_predictions(self):
        """Generate visualization plots comparing chronological age vs CosinorAge.

        This method creates individual plots for each record showing the comparison
        between chronological age and predicted biological age. The plots use a
        timeline visualization with color coding to indicate whether the biological
        age is advanced (red) or delayed (green) compared to chronological age.

        The plots include:
        - Chronological age and CosinorAge as points on a timeline
        - Color-coded line segments (red for advanced, green for younger)
        - Numerical labels showing exact age values
        - Clear visual distinction between the two age measures

        Notes
        -----
        - Creates one plot per record in the dataset
        - Red color indicates biological age > chronological age (advanced aging)
        - Green color indicates biological age < chronological age (delayed aging)
        - Plots are displayed immediately when called
        - Records with invalid predictions (None values) are skipped
        - Each plot shows exact numerical values for both ages
        - The visualization helps quickly identify aging patterns across participants
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
