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

    def test_cleanup_mapillary_sequence_expansion(self):
        # Create dataset with Mapillary images in Africa:
        # - 15 points at locked lat 10.0 (count > 10, span > 1.0) with photo IDs 'm_bad_0' .. 'm_bad_14'
        # - 3 drifting points in the SAME sequence: lats 10.05, 10.06, 10.07 with photo IDs 'm_drift_1', 'm_drift_2', 'm_drift_3'
        # - 4 normal points in a GOOD sequence: lats 20.0 with photo IDs 'm_good_1' .. 'm_good_4'
        bad_ids = [f"m_bad_{i}" for i in range(15)]
        bad_lats = [10.0] * 15
        bad_lons = list(np.linspace(10.0, 25.0, 15))

        drift_ids = ["m_drift_1", "m_drift_2", "m_drift_3"]
        drift_lats = [10.05, 10.06, 10.07]
        drift_lons = [15.0, 16.0, 17.0]

        good_ids = [f"m_good_{i}" for i in range(4)]
        good_lats = [20.0] * 4
        good_lons = [5.0, 5.1, 5.2, 5.3]

        all_pids = bad_ids + drift_ids + good_ids
        all_lats = bad_lats + drift_lats + good_lats
        all_lons = bad_lons + drift_lons + good_lons

        df_m = pd.DataFrame(
            {
                "Photo_ID": all_pids,
                "Platform": ["mapillary"] * len(all_pids),
                "Latitude": all_lats,
                "Longitude": all_lons,
                "continent": ["Africa"] * len(all_pids),
                "Image_URL": [f"url_{p}" for p in all_pids],
            }
        )

        in_pq = os.path.join(self.test_dir, "mapillary_test.parquet")
        out_pq = os.path.join(self.test_dir, "mapillary_test_clean.parquet")
        save_dataframe(df_m, in_pq)

        test_args = [
            "cleanup_coordinate_anomalies",
            "--input",
            in_pq,
            "--output",
            out_pq,
            "--platform",
            "mapillary",
            "--continent",
            "africa",
        ]

        # Sequence mapping mock: bad points map to seq_glitch
        mock_seq_map = {pid: "seq_glitch" for pid in bad_ids}
        # Sequence all images mock: seq_glitch contains bad_ids + drift_ids
        mock_all_seq_ids = set(bad_ids + drift_ids)

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.processing.cleanup_coordinate_anomalies.get_mapillary_token",
                return_value="mock_token",
            ):
                with patch(
                    "src.processing.cleanup_coordinate_anomalies.fetch_mapillary_sequences",
                    return_value=mock_seq_map,
                ):
                    with patch(
                        "src.processing.cleanup_coordinate_anomalies.fetch_sequence_all_image_ids",
                        return_value=mock_all_seq_ids,
                    ):
                        main()

        df_cleaned = pd.read_parquet(out_pq)
        # All 15 locked points AND the 3 drifting points from seq_glitch should be purged!
        # Only the 4 points from the good sequence should remain.
        self.assertEqual(len(df_cleaned), 4)
        self.assertEqual(sorted(df_cleaned["Photo_ID"].tolist()), sorted(good_ids))

    def test_cleanup_mapillary_missing_token_fallback(self):
        bad_ids = [f"m_bad_{i}" for i in range(15)]
        bad_lats = [10.0] * 15
        bad_lons = list(np.linspace(10.0, 25.0, 15))

        good_ids = [f"m_good_{i}" for i in range(4)]
        good_lats = [20.0] * 4
        good_lons = [5.0, 5.1, 5.2, 5.3]

        all_pids = bad_ids + good_ids
        all_lats = bad_lats + good_lats
        all_lons = bad_lons + good_lons

        df_m = pd.DataFrame(
            {
                "Photo_ID": all_pids,
                "Platform": ["mapillary"] * len(all_pids),
                "Latitude": all_lats,
                "Longitude": all_lons,
                "continent": ["Africa"] * len(all_pids),
                "Image_URL": [f"url_{p}" for p in all_pids],
            }
        )

        in_pq = os.path.join(self.test_dir, "mapillary_fallback.parquet")
        out_pq = os.path.join(self.test_dir, "mapillary_fallback_clean.parquet")
        save_dataframe(df_m, in_pq)

        test_args = [
            "cleanup_coordinate_anomalies",
            "--input",
            in_pq,
            "--output",
            out_pq,
            "--platform",
            "mapillary",
            "--continent",
            "africa",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "src.processing.cleanup_coordinate_anomalies.get_mapillary_token",
                return_value="",
            ):
                main()

        df_cleaned = pd.read_parquet(out_pq)
        # The 15 locked points should still be purged via point-level fallback!
        self.assertEqual(len(df_cleaned), 4)
        self.assertEqual(sorted(df_cleaned["Photo_ID"].tolist()), sorted(good_ids))


if __name__ == "__main__":
    unittest.main()
