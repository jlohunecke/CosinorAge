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

"""
Utility functions for wearable feature computation and analysis.

This module provides a comprehensive set of utility functions for processing
accelerometer data and computing various wearable features including:

**Circadian Rhythm Analysis:**
- cosinor_analysis: Parametric analysis using cosinor modeling
- nonparam_analysis: Non-parametric measures (IS, IV, M10, L5, RA)

**Physical Activity Metrics:**
- physical_activity_metrics: Activity level classification and metrics

**Sleep Analysis:**
- sleep_metrics: Sleep-wake prediction and sleep quality metrics

**Data Processing:**
- rescaling: Data normalization and scaling utilities

**Visualization:**
- visualization: Plotting functions for sleep, activity, and cosinor analysis
- dashboard: Interactive dashboard for feature exploration

**Usage:**
    These utilities are typically used internally by the WearableFeatures
    and BulkWearableFeatures classes, but can also be imported directly
    for custom analysis workflows.

**Examples:**
    >>> from cosinorage.features.utils import cosinor_multiday, IS, IV
    >>> # Compute cosinor parameters
    >>> params, fitted = cosinor_multiday(enmo_data)
    >>> # Calculate interdaily stability
    >>> is_value = IS(enmo_data)
    >>> # Calculate intradaily variability
    >>> iv_value = IV(enmo_data)
"""

