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
Tests for BulkWearableFeatures class.
"""

import traceback
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from cosinorage.datahandlers import DataHandler
from cosinorage.features import WearableFeatures
from cosinorage.features.bulk_features import BulkWearableFeatures


class MockDataHandler(DataHandler):
    """Mock DataHandler for testing that works with DataFrames directly."""

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.ml_data = data.copy()
        self.meta_dict = {
            "datasource": "Mock",
            "data_format": "DataFrame",
            "raw_data_type": "ENMO",
        }


class TestBulkWearableFeatures(unittest.TestCase):
    """Test cases for BulkWearableFeatures class."""

    def setUp(self):
        """Set up test data."""
        # Create sample ENMO data for 7 days (to support all features)
        n_days = 7
        n_samples = 60 * 24 * n_days  # 1-min resolution
        timestamps = pd.date_range(
            start="2024-01-01", periods=n_samples, freq="1min"
        )
        time_hours = (timestamps.hour + timestamps.minute / 60).values
        # Circadian pattern: higher activity during the day, lower at night
        circadian_pattern = 0.5 + 0.3 * np.cos(
            2 * np.pi * (time_hours - 14) / 24
        )
        # Add a daily activity burst (simulate wake/sleep)
        activity_burst = 0.2 * (np.sin(2 * np.pi * (time_hours - 8) / 24) > 0)
        noise = np.random.normal(0, 0.05, n_samples)
        enmo = np.maximum(
            0.01, circadian_pattern + activity_burst + noise
        )  # avoid zeros/NaNs

        self.sample_data = pd.DataFrame({"enmo": enmo}, index=timestamps)

        # Create multiple handlers
        self.handlers = []
        for i in range(3):
            # Create slightly different data for each handler
            data = self.sample_data.copy()
            data["enmo"] = data["enmo"] * (0.9 + 0.2 * np.random.random())
            handler = MockDataHandler(data)
            self.handlers.append(handler)
        # Patch apply_sleep_wake_predictions to return a valid binary array
        patcher = patch(
            "cosinorage.features.utils.sleep_metrics.apply_sleep_wake_predictions",
            lambda df, sleep_params=None: (df["enmo"] < 0.4).astype(int),
        )
        self.sleep_patch = patcher.start()
        self.addCleanup(patcher.stop)
        # Debug: Try to compute features for the first handler and print stack trace if it fails
        try:
            WearableFeatures(self.handlers[0])
        except Exception as e:
            print("DEBUG: Exception in WearableFeatures:", e)
            traceback.print_exc()

    def test_initialization(self):
        """Test BulkWearableFeatures initialization."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers,
            features_args={},
            compute_distributions=True,
        )

        self.assertEqual(len(bulk_features.handlers), 3)
        self.assertEqual(len(bulk_features.individual_features), 3)
        self.assertIsInstance(bulk_features.distribution_stats, dict)

    def test_individual_features_computation(self):
        """Test that individual features are computed correctly."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers, compute_distributions=False
        )

        individual_features = bulk_features.get_individual_features()

        # Check that we have features for each handler
        self.assertEqual(len(individual_features), 3)

        # Check that features have expected structure
        for features in individual_features:
            if features is not None:
                self.assertIn("cosinor", features)
                self.assertIn("nonparam", features)
                self.assertIn("physical_activity", features)
                self.assertIn("sleep", features)

    def test_distribution_statistics(self):
        """Test that distribution statistics are computed correctly."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers, compute_distributions=True
        )

        distribution_stats = bulk_features.get_distribution_stats()

        # Check that we have statistics for features
        self.assertGreater(len(distribution_stats), 0)

        # Check that statistics have expected keys
        for feature_name, stats in distribution_stats.items():
            expected_keys = [
                "count",
                "mean",
                "std",
                "min",
                "max",
                "median",
                "q25",
                "q75",
                "iqr",
                "mode",
                "skewness",
            ]

            for key in expected_keys:
                self.assertIn(key, stats)

    def test_summary_dataframe(self):
        """Test that summary DataFrame is created correctly."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers, compute_distributions=True
        )

        summary_df = bulk_features.get_summary_dataframe()

        # Check DataFrame structure
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertGreater(len(summary_df), 0)

        # Check that it has expected columns
        expected_columns = [
            "feature",
            "count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "q25",
            "q75",
            "iqr",
            "mode",
            "skewness",
        ]

        for col in expected_columns:
            self.assertIn(col, summary_df.columns)

    def test_correlation_matrix(self):
        """Test that correlation matrix is computed correctly."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers, compute_distributions=True
        )

        correlation_matrix = bulk_features.get_feature_correlation_matrix()

        # Check that correlation matrix is a DataFrame
        self.assertIsInstance(correlation_matrix, pd.DataFrame)

        # If we have features, correlation matrix should not be empty
        if len(bulk_features.get_distribution_stats()) > 1:
            self.assertGreater(len(correlation_matrix), 0)
            self.assertGreater(len(correlation_matrix.columns), 0)

    def test_failed_handlers_handling(self):
        """Test that failed handlers are handled gracefully."""
        # Create a mock handler that will fail
        mock_handler = Mock()
        mock_handler.get_ml_data.side_effect = Exception("Test error")

        handlers_with_failure = self.handlers + [mock_handler]

        bulk_features = BulkWearableFeatures(
            handlers=handlers_with_failure, compute_distributions=True
        )

        # Check that failed handlers are recorded
        failed_handlers = bulk_features.get_failed_handlers()
        self.assertGreater(len(failed_handlers), 0)

        # Check that individual_features has None for failed handler
        individual_features = bulk_features.get_individual_features()
        self.assertIn(None, individual_features)

    def test_feature_flattening(self):
        """Test that nested features are flattened correctly."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers, compute_distributions=True
        )

        # Get the flattened features DataFrame
        valid_features = [
            f for f in bulk_features.get_individual_features() if f is not None
        ]
        flattened_df = bulk_features._BulkWearableFeatures__flatten_features(
            valid_features
        )

        # Check that DataFrame is created
        self.assertIsInstance(flattened_df, pd.DataFrame)
        self.assertGreater(len(flattened_df), 0)

        # Check that features are flattened (no nested structure)
        for column in flattened_df.columns:
            if column != "handler_index":
                self.assertIn(
                    "_", column
                )  # Should have category_feature format

    def test_statistical_computation(self):
        """Test that statistical measures are computed correctly."""
        bulk_features = BulkWearableFeatures(
            handlers=self.handlers, compute_distributions=True
        )

        # Get the flattened features DataFrame
        valid_features = [
            f for f in bulk_features.get_individual_features() if f is not None
        ]
        flattened_df = bulk_features._BulkWearableFeatures__flatten_features(
            valid_features
        )

        # Compute statistics
        stats = (
            bulk_features._BulkWearableFeatures__compute_feature_statistics(
                flattened_df
            )
        )

        # Check that statistics are reasonable
        for feature_name, feature_stats in stats.items():
            # Check that mean is between min and max (with tolerance for floating point precision)
            self.assertGreaterEqual(
                feature_stats["mean"], feature_stats["min"] - 1e-10
            )
            self.assertLessEqual(
                feature_stats["mean"], feature_stats["max"] + 1e-10
            )

            # Check that std is non-negative
            self.assertGreaterEqual(feature_stats["std"], 0)

            # Check that q25 <= median <= q75 (with tolerance for floating point precision)
            self.assertLessEqual(
                feature_stats["q25"], feature_stats["median"] + 1e-10
            )
            self.assertLessEqual(
                feature_stats["median"], feature_stats["q75"] + 1e-10
            )

    def test_empty_handlers_list(self):
        """Test behavior with empty handlers list."""
        bulk_features = BulkWearableFeatures(
            handlers=[], compute_distributions=True
        )

        self.assertEqual(len(bulk_features.individual_features), 0)
        self.assertEqual(len(bulk_features.distribution_stats), 0)

    def test_single_handler(self):
        """Test behavior with single handler."""
        bulk_features = BulkWearableFeatures(
            handlers=[self.handlers[0]], compute_distributions=True
        )

        self.assertEqual(len(bulk_features.individual_features), 1)
        # With single handler, std should be 0 or NaN
        distribution_stats = bulk_features.get_distribution_stats()
        for feature_stats in distribution_stats.values():
            if feature_stats["count"] > 1:
                self.assertEqual(feature_stats["std"], 0)

    def test_cosinorage_features_with_ground_truth(self):
        """Test BulkWearableFeatures with CosinorAge and ground truth."""
        # Define age and gender information with ground truth
        cosinor_age_inputs = [
            {"age": 25.5, "gender": "female", "gt_cosinor_age": 26.2},
            {"age": 30.2, "gender": "male", "gt_cosinor_age": 31.1},
            {"age": 28.0, "gender": "unknown", "gt_cosinor_age": 27.8}
        ]

        bulk_features = BulkWearableFeatures(
            handlers=self.handlers,
            compute_distributions=True,
            cosinor_age_inputs=cosinor_age_inputs
        )

        # Get individual features
        individual_features = bulk_features.get_individual_features()
        self.assertEqual(len(individual_features), 3)

        # Check that CosinorAge features were added
        for i, features in enumerate(individual_features):
            if features is not None:
                self.assertIn("cosinorage", features)
                cosinorage_feat = features["cosinorage"]
                
                # Check for expected CosinorAge features
                self.assertIn("cosinorage", cosinorage_feat)
                self.assertIn("cosinorage_advance", cosinorage_feat)
                self.assertIn("cosinor_age_prediction_error", cosinorage_feat)
                
                # Check that prediction error is reasonable
                self.assertIsInstance(cosinorage_feat["cosinor_age_prediction_error"], (int, float))

        # Check distribution stats
        stats = bulk_features.get_distribution_stats()
        cosinorage_features = [k for k in stats.keys() if k in ['cosinorage', 'cosinorage_advance', 'cosinor_age_prediction_error']]
        self.assertGreaterEqual(len(cosinorage_features), 3)

        # Check that feature names are correct (not duplicated)
        incorrect_names = [name for name in stats.keys() if name.startswith('cosinorage_cosinorage')]
        self.assertEqual(len(incorrect_names), 0)

    def test_cosinorage_features_without_ground_truth(self):
        """Test BulkWearableFeatures with CosinorAge but without ground truth."""
        # Define age and gender information without ground truth
        cosinor_age_inputs = [
            {"age": 25.5, "gender": "female"},
            {"age": 30.2, "gender": "male"},
            {"age": 28.0, "gender": "unknown"}
        ]

        bulk_features = BulkWearableFeatures(
            handlers=self.handlers,
            compute_distributions=True,
            cosinor_age_inputs=cosinor_age_inputs
        )

        # Get individual features
        individual_features = bulk_features.get_individual_features()
        self.assertEqual(len(individual_features), 3)

        # Check that CosinorAge features were added but no prediction error
        for i, features in enumerate(individual_features):
            if features is not None:
                self.assertIn("cosinorage", features)
                cosinorage_feat = features["cosinorage"]
                
                # Check for expected CosinorAge features
                self.assertIn("cosinorage", cosinorage_feat)
                self.assertIn("cosinorage_advance", cosinorage_feat)
                
                # Check that prediction error is NOT present
                self.assertNotIn("cosinor_age_prediction_error", cosinorage_feat)

        # Check distribution stats
        stats = bulk_features.get_distribution_stats()
        cosinorage_features = [k for k in stats.keys() if k in ['cosinorage', 'cosinorage_advance']]
        self.assertGreaterEqual(len(cosinorage_features), 2)

    def test_cosinorage_features_validation(self):
        """Test validation of cosinor_age_inputs parameter."""
        # Test with mismatched length
        cosinor_age_inputs = [
            {"age": 25.5, "gender": "female"},
            {"age": 30.2, "gender": "male"}
        ]  # Only 2 inputs for 3 handlers

        with self.assertRaises(ValueError):
            BulkWearableFeatures(
                handlers=self.handlers,
                cosinor_age_inputs=cosinor_age_inputs
            )

        # Test with missing age key
        cosinor_age_inputs = [
            {"gender": "female"},  # Missing age
            {"age": 30.2, "gender": "male"},
            {"age": 28.0, "gender": "unknown"}
        ]

        with self.assertRaises(ValueError):
            BulkWearableFeatures(
                handlers=self.handlers,
                cosinor_age_inputs=cosinor_age_inputs
            )

    def test_cosinorage_features_feature_names(self):
        """Test that CosinorAge feature names are correct and not duplicated."""
        cosinor_age_inputs = [
            {"age": 25.5, "gender": "female", "gt_cosinor_age": 26.2},
            {"age": 30.2, "gender": "male", "gt_cosinor_age": 31.1},
            {"age": 28.0, "gender": "unknown", "gt_cosinor_age": 27.8}
        ]

        bulk_features = BulkWearableFeatures(
            handlers=self.handlers,
            compute_distributions=True,
            cosinor_age_inputs=cosinor_age_inputs
        )

        # Get distribution stats
        stats = bulk_features.get_distribution_stats()
        
        # Check for correct feature names
        expected_cosinorage_features = ['cosinorage', 'cosinorage_advance', 'cosinor_age_prediction_error']
        for feature in expected_cosinorage_features:
            self.assertIn(feature, stats)
        
        # Check for incorrect duplicated names
        incorrect_names = [name for name in stats.keys() if name.startswith('cosinorage_cosinorage')]
        self.assertEqual(len(incorrect_names), 0)

        # Get summary dataframe
        summary_df = bulk_features.get_summary_dataframe()
        cosinorage_rows = summary_df[summary_df['feature'].isin(expected_cosinorage_features)]
        self.assertEqual(len(cosinorage_rows), 3)

        # Get correlation matrix
        corr_matrix = bulk_features.get_feature_correlation_matrix()
        cosinorage_cols = [col for col in corr_matrix.columns if col in expected_cosinorage_features]
        self.assertEqual(len(cosinorage_cols), 3)


if __name__ == "__main__":
    unittest.main()
