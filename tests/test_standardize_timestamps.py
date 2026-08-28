import os
import shutil
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from src.processing.standardize_timestamps import main
from src.utils.io import save_dataframe


class TestStandardizeTimestamps(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_scratch_timestamp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_parquet = os.path.join(self.test_dir, "test_timestamps.parquet")
        self.output_parquet = os.path.join(self.test_dir, "test_timestamps_std.parquet")

        # Create a mock dataset with various timestamps and coordinates
        df = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4", "5"],
                "Platform": ["flickr"] * 5,
                # 1. Numeric seconds: 1717156800 (June 1, 2024) -> 2024-05-31 UTC
                # 2. Numeric milliseconds: 1717156800000 -> 2024-05-31 UTC
                # 3. ISO string: "2024-06-01T13:30:00Z"
                # 4. Colon-formatted date: "2024:12:01 22:30:00"
                # 5. Invalid/Null timestamp
                "Captured_At": [
                    "1717156800",
                    "1717156800000",
                    "2024-06-01T13:30:00Z",
                    "2024:12:01 22:30:00",
                    None,
                ],
                # Latitudes:
                # 1. North Temperate (May -> Spring)
                # 2. South Temperate (May -> Autumn)
                # 3. North Tropical (June -> Wet Season)
                # 4. North Temperate (Dec -> Winter, hour 22 -> Night)
                # 5. North Temperate
                "Latitude": [45.0, -45.0, 10.0, 45.0, 45.0],
                "Longitude": [-75.0, 140.0, -10.0, -75.0, -75.0],
                "Image_URL": [f"url{i}" for i in range(1, 6)],
            }
        )
        save_dataframe(df, self.input_parquet)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_standardization(self):
        test_args = [
            "standardize_timestamps",
            "--input",
            self.input_parquet,
            "--output",
            self.output_parquet,
        ]

        with patch.object(sys, "argv", test_args):
            main()

        self.assertTrue(os.path.exists(self.output_parquet))
        df_out = pd.read_parquet(self.output_parquet)

        # Assertions
        # 1. Captured_At should be formatted as ISO 8601 strings
        self.assertEqual(
            df_out["Captured_At"].iloc[0], "2024-05-31T12:00:00Z"
        )  # 1717156800 UTC
        self.assertEqual(
            df_out["Captured_At"].iloc[1], "2024-05-31T12:00:00Z"
        )  # 1717156800000 UTC
        self.assertEqual(df_out["Captured_At"].iloc[2], "2024-06-01T13:30:00Z")
        self.assertEqual(df_out["Captured_At"].iloc[3], "2024-12-01T22:30:00Z")
        self.assertTrue(pd.isna(df_out["Captured_At"].iloc[4]))

        # 2. Season classifications
        # Row 0: Month 5, Lat 45 -> Spring
        self.assertEqual(df_out["Season"].iloc[0], "Spring")
        # Row 1: Month 5, Lat -45 -> Autumn
        self.assertEqual(df_out["Season"].iloc[1], "Autumn")
        # Row 2: Month 6, Lat 10 -> Wet Season (Tropical fallback for month 6)
        self.assertEqual(df_out["Season"].iloc[2], "Wet Season")
        # Row 3: Month 12, Lat 45 -> Winter
        self.assertEqual(df_out["Season"].iloc[3], "Winter")

        # 3. Time of Day classifications
        # Hour 12: Afternoon (12:00:00)
        self.assertEqual(df_out["Time_Of_Day"].iloc[0], "Afternoon")
        # Hour 13: Afternoon (13:30:00)
        self.assertEqual(df_out["Time_Of_Day"].iloc[2], "Afternoon")
        # Hour 22: Night (22:30:00)
        self.assertEqual(df_out["Time_Of_Day"].iloc[3], "Night")


if __name__ == "__main__":
    unittest.main()
