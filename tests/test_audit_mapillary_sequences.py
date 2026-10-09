import json
import os
import tempfile
import unittest

import pandas as pd

from src.processing.audit_mapillary_sequences import audit_and_purge_dataset
from src.utils.mapillary_trajectory_validator import (
    MapillaryTrajectoryValidator,
)


class TestAuditMapillarySequences(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.parquet_path = os.path.join(self.tmpdir.name, "test_input.parquet")
        self.output_path = os.path.join(self.tmpdir.name, "test_output.parquet")
        self.report_path = os.path.join(self.tmpdir.name, "report.json")
        self.db_path = os.path.join(self.tmpdir.name, "test_cache.db")

        # 4 records:
        # Photo 101, 102: Mapillary in Africa (seq_bad - stationary jitter)
        # Photo 103: Mapillary in Africa (seq_good - moving)
        # Photo 104: Flickr in Africa
        # Photo 105: Mapillary in Europe
        df = pd.DataFrame(
            {
                "Photo_ID": ["101", "102", "103", "104", "105"],
                "Platform": [
                    "mapillary",
                    "mapillary",
                    "mapillary",
                    "flickr",
                    "mapillary",
                ],
                "Latitude": [1.0, 1.0001, 2.0, 3.0, 48.8],
                "Longitude": [36.0, 36.0001, 37.0, 38.0, 2.3],
                "Image_URL": [
                    "mapillary://101",
                    "mapillary://102",
                    "mapillary://103",
                    "https://flickr.com/104",
                    "mapillary://105",
                ],
                "continent": ["Africa", "Africa", "Africa", "Africa", "Europe"],
            }
        )
        df.to_parquet(self.parquet_path, index=False)

        # Pre-seed the validator cache
        validator = MapillaryTrajectoryValidator(
            token="dummy_token", db_path=self.db_path
        )
        with validator._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO photo_sequences (photo_id, sequence_id) VALUES (?, ?)",
                ("101", "seq_bad"),
            )
            cursor.execute(
                "INSERT INTO photo_sequences (photo_id, sequence_id) VALUES (?, ?)",
                ("102", "seq_bad"),
            )
            cursor.execute(
                "INSERT INTO photo_sequences (photo_id, sequence_id) VALUES (?, ?)",
                ("103", "seq_good"),
            )
            cursor.execute(
                "INSERT INTO photo_sequences (photo_id, sequence_id) VALUES (?, ?)",
                ("105", "seq_euro"),
            )

            # Insert cached verdicts
            # seq_bad: invalid (stationary jitter)
            cursor.execute(
                """
                INSERT INTO sequences (
                    sequence_id, is_valid, reason, frame_count, total_distance_m,
                    net_displacement_m, tortuosity, max_speed_kmh, p95_speed_kmh,
                    bounding_radius_m, duration_seconds, track_geojson, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "seq_bad",
                    0,
                    "Stationary camera GPS jitter",
                    25,
                    300.0,
                    12.0,
                    25.0,
                    10.0,
                    8.0,
                    15.0,
                    100.0,
                    None,
                    1600000000.0,
                ),
            )
            # seq_good: valid
            cursor.execute(
                """
                INSERT INTO sequences (
                    sequence_id, is_valid, reason, frame_count, total_distance_m,
                    net_displacement_m, tortuosity, max_speed_kmh, p95_speed_kmh,
                    bounding_radius_m, duration_seconds, track_geojson, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "seq_good",
                    1,
                    "Valid moving trajectory",
                    30,
                    1000.0,
                    950.0,
                    1.05,
                    45.0,
                    42.0,
                    500.0,
                    80.0,
                    None,
                    1600000000.0,
                ),
            )
            conn.commit()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dry_run_audit(self):
        res = audit_and_purge_dataset(
            input_parquet=self.parquet_path,
            output_parquet=self.output_path,
            continent="Africa",
            platform="mapillary",
            token="dummy_token",
            db_path=self.db_path,
            dry_run=True,
            report_json=self.report_path,
        )

        self.assertEqual(res["status"], "dry_run")
        self.assertEqual(res["purged_photos"], 2)
        self.assertEqual(res["invalid_sequences"], 1)

        # Output parquet should NOT have been created in dry run
        self.assertFalse(os.path.exists(self.output_path))
        # Report JSON should exist
        self.assertTrue(os.path.exists(self.report_path))
        with open(self.report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["photos_to_purge_count"], 2)
            self.assertIn("seq_bad", data["invalid_sequences"])

    def test_purge_dataset(self):
        res = audit_and_purge_dataset(
            input_parquet=self.parquet_path,
            output_parquet=self.output_path,
            continent="Africa",
            platform="mapillary",
            token="dummy_token",
            db_path=self.db_path,
            dry_run=False,
            report_json=self.report_path,
        )

        self.assertEqual(res["status"], "purged")
        self.assertEqual(res["purged_photos"], 2)
        self.assertEqual(res["total_records_remaining"], 3)

        # Output parquet must exist
        self.assertTrue(os.path.exists(self.output_path))
        out_df = pd.read_parquet(self.output_path)
        self.assertEqual(len(out_df), 3)

        # Photo 101 and 102 should be gone
        pids = out_df["Photo_ID"].tolist()
        self.assertNotIn("101", pids)
        self.assertNotIn("102", pids)
        # 103 (valid Mapillary), 104 (Flickr Africa), 105 (Mapillary Europe) must survive
        self.assertIn("103", pids)
        self.assertIn("104", pids)
        self.assertIn("105", pids)


if __name__ == "__main__":
    unittest.main()
