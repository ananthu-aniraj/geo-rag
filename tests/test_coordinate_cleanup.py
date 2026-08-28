import os
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.processing.cleanup_coordinate_anomalies import main
from src.utils.io import save_dataframe


class TestCoordinateCleanup(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_cleanup"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_parquet = os.path.join(self.test_dir, "test_anomalies.parquet")
        self.output_parquet = os.path.join(
            self.test_dir, "test_anomalies_clean.parquet"
        )
        self.input_csv = os.path.join(self.test_dir, "test_anomalies.csv")
        self.output_csv = os.path.join(self.test_dir, "test_anomalies_clean.csv")

        # Create a mock dataset:
        # - 15 points at locked latitude 34.0 (an anomaly: >10 count, span > 1.0)
        # - 3 points at latitude 45.0 (normal)
        lats = [34.0] * 15 + [45.0] * 3
        lons = list(np.linspace(-100, -50, 15)) + [-70.0, -70.1, -70.2]
        platforms = ["flickr"] * 18

        df = pd.DataFrame(
            {
                "Photo_ID": [str(i) for i in range(18)],
                "Platform": platforms,
                "Latitude": lats,
                "Longitude": lons,
                "Image_URL": [f"url{i}" for i in range(18)],
            }
        )
        save_dataframe(df, self.input_parquet)
        df.to_csv(self.input_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cleanup_coordinate_anomalies(self):
        test_args = [
            "cleanup_coordinate_anomalies",
            "--input",
            self.input_parquet,
            "--csv",
            self.input_csv,
            "--output",
            self.output_parquet,
            "--output_csv",
            self.output_csv,
        ]

        with patch.object(sys, "argv", test_args):
            main()

        self.assertTrue(os.path.exists(self.output_parquet))
        self.assertTrue(os.path.exists(self.output_csv))

        # Load output and verify
        df_clean = pd.read_parquet(self.output_parquet)
        # The 15 locked-latitude points at 34.0 should have been purged!
        # The 3 points at 45.0 should be kept.
        self.assertEqual(len(df_clean), 3)
        self.assertTrue((df_clean["Latitude"] == 45.0).all())

        # Verify CSV output
        df_clean_csv = pd.read_csv(self.output_csv)
        self.assertEqual(len(df_clean_csv), 3)
        self.assertTrue((df_clean_csv["Latitude"] == 45.0).all())


if __name__ == "__main__":
    unittest.main()
