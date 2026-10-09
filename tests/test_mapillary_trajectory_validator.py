import os
import random
import tempfile
import unittest

from src.utils.mapillary_trajectory_validator import (
    MapillaryTrajectoryValidator,
    TrajectoryMetrics,
    haversine_distance,
)


class TestMapillaryTrajectoryValidator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test_cache.db")
        self.validator = MapillaryTrajectoryValidator(
            token="test_mock_token", db_path=self.db_path
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_haversine_distance(self):
        # 1 degree of longitude at the equator is approx 111,195 - 111,320 meters
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        self.assertGreater(dist, 111000.0)
        self.assertLess(dist, 112000.0)

        # Distance between identical points is 0
        self.assertAlmostEqual(haversine_distance(10.0, 20.0, 10.0, 20.0), 0.0)

    def test_valid_moving_trajectory(self):
        # 30 frames moving along a straight road at ~36 km/h (10 m/s)
        t0 = 1600000000000
        points = []
        for i in range(30):
            lat = 10.0
            lon = 20.0 + i * (10.0 / 111320.0)
            points.append(
                {
                    "photo_id": str(i),
                    "captured_at": t0 + i * 1000,
                    "lat": lat,
                    "lon": lon,
                }
            )

        res = self.validator.validate_sequence("seq_moving", trajectory_points=points)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.sequence_id, "seq_moving")
        self.assertIsNotNone(res.metrics)
        self.assertAlmostEqual(res.metrics.tortuosity, 1.0, places=1)
        self.assertAlmostEqual(res.metrics.max_speed_kmh, 36.0, delta=2.0)
        self.assertIsNotNone(res.geojson_track)
        self.assertEqual(res.geojson_track["type"], "Feature")
        self.assertEqual(res.geojson_track["geometry"]["type"], "LineString")

    def test_stationary_camera_jitter_detected(self):
        # 30 frames trapped in ~15m radius with GPS noise (wandering in random directions)
        t0 = 1600000000000
        points = []
        random.seed(42)
        for i in range(30):
            lat = 10.0 + random.uniform(-0.0001, 0.0001)
            lon = 20.0 + random.uniform(-0.0001, 0.0001)
            points.append(
                {
                    "photo_id": str(i),
                    "captured_at": t0 + i * 1000,
                    "lat": lat,
                    "lon": lon,
                }
            )

        res = self.validator.validate_sequence("seq_jitter", trajectory_points=points)
        self.assertFalse(res.is_valid)
        self.assertIn("Stationary camera GPS jitter", res.reason)
        self.assertGreater(res.metrics.tortuosity, 15.0)
        self.assertLess(res.metrics.bounding_radius_m, 35.0)

    def test_severe_tortuosity_random_walk(self):
        metrics = TrajectoryMetrics(
            frame_count=20,
            total_distance_m=350.0,
            net_displacement_m=10.0,
            tortuosity=35.0,
            max_speed_kmh=15.0,
            p95_speed_kmh=12.0,
            bounding_radius_m=45.0,
            duration_seconds=120.0,
        )
        is_valid, reason = self.validator.evaluate_trajectory_metrics(metrics)
        self.assertFalse(is_valid)
        self.assertIn("Severe random-walk tortuosity", reason)

    def test_impossible_speed_spike(self):
        metrics = TrajectoryMetrics(
            frame_count=5,
            total_distance_m=1000.0,
            net_displacement_m=900.0,
            tortuosity=1.1,
            max_speed_kmh=350.0,  # Impossible ground capture speed
            p95_speed_kmh=300.0,
            bounding_radius_m=500.0,
            duration_seconds=10.0,
        )
        is_valid, reason = self.validator.evaluate_trajectory_metrics(metrics)
        self.assertFalse(is_valid)
        self.assertIn("Impossible speed teleportation", reason)

    def test_sqlite_cache_persistence(self):
        # 1. Evaluate and cache synthetic sequence
        t0 = 1600000000000
        points = [
            {"photo_id": "1", "captured_at": t0, "lat": 1.0, "lon": 1.0},
            {"photo_id": "2", "captured_at": t0 + 1000, "lat": 1.0001, "lon": 1.0},
        ]
        res1 = self.validator.validate_sequence("seq_cached", trajectory_points=points)
        self.assertTrue(res1.is_valid)

        # 2. Retrieve without passing trajectory points
        res2 = self.validator.validate_sequence("seq_cached")
        self.assertTrue(res2.is_valid)
        self.assertEqual(res2.sequence_id, "seq_cached")
        self.assertEqual(res2.metrics.frame_count, 2)
        self.assertEqual(res2.metrics.total_distance_m, res1.metrics.total_distance_m)

    def test_photo_sequences_cache(self):
        # Manually seed photo_sequences into the validator DB
        with self.validator._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO photo_sequences (photo_id, sequence_id) VALUES (?, ?)",
                ("photo_abc", "seq_123"),
            )
            conn.commit()

        resolved = self.validator.resolve_photo_sequences(["photo_abc"])
        self.assertEqual(resolved.get("photo_abc"), "seq_123")


if __name__ == "__main__":
    unittest.main()
