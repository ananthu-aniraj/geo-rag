import os
import shutil
import tempfile
import unittest

from src.visualization.visualize_retrieval import (
    generate_retrieval_html,
    haversine_km,
    select_balanced_samples,
)


class TestVisualizeRetrieval(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_haversine_km(self):
        # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ~ 343 km
        dist = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        self.assertIsNotNone(dist)
        self.assertAlmostEqual(dist, 343.5, delta=10.0)

        # Identical coordinates
        self.assertEqual(haversine_km(45.0, 10.0, 45.0, 10.0), 0.0)

        # None or invalid input
        self.assertIsNone(haversine_km(None, 10.0, 45.0, 10.0))
        self.assertIsNone(haversine_km(45.0, None, 45.0, 10.0))
        self.assertIsNone(haversine_km(float("nan"), 10.0, 45.0, 10.0))

    def test_select_balanced_samples(self):
        queries = []
        for i in range(50):
            queries.append(
                {
                    "query_id": f"q_{i}",
                    "ground_truth": "ClassA" if i < 25 else "ClassB",
                    "p1": 1.0 if i % 2 == 0 else 0.0,
                }
            )

        sampled = select_balanced_samples(queries, max_samples=10, seed=42)
        self.assertEqual(len(sampled), 10)
        # Verify both classes are present
        classes = {q["ground_truth"] for q in sampled}
        self.assertIn("ClassA", classes)
        self.assertIn("ClassB", classes)

    def test_generate_retrieval_html(self):
        out_html = os.path.join(self.test_dir, "test_viz.html")
        mock_queries = [
            {
                "query_id": "flickr_101",
                "query_url": "https://example.com/img1.jpg",
                "query_platform": "flickr",
                "query_lat": 46.5,
                "query_lon": 8.0,
                "ground_truth": "Alpine North",
                "p1": 1.0,
                "p5": 0.8,
                "p10": 0.7,
                "ap": 0.85,
                "retrieved": [
                    {
                        "rank": 1,
                        "id": "mapillary_999",
                        "url": "https://example.com/r1.jpg",
                        "platform": "mapillary",
                        "lat": 46.6,
                        "lon": 8.1,
                        "distance_km": 13.5,
                        "predicted_label": "Alpine North",
                        "similarity": 0.89,
                        "is_match": True,
                    },
                    {
                        "rank": 2,
                        "id": "flickr_888",
                        "url": "https://example.com/r2.jpg",
                        "platform": "flickr",
                        "lat": 52.0,
                        "lon": 13.0,
                        "distance_km": 680.0,
                        "predicted_label": "Continental",
                        "similarity": 0.75,
                        "is_match": False,
                    },
                ],
            }
        ]

        metrics = {
            "p@1": 50.0,
            "p@5": 45.0,
            "p@10": 40.0,
            "map@10": 42.0,
            "mrr@10": 55.0,
        }

        generated_path = generate_retrieval_html(
            out_html,
            "Test Representation Benchmark",
            "test-model",
            metrics,
            mock_queries,
            max_samples=10,
        )

        self.assertTrue(os.path.exists(generated_path))
        with open(generated_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Test Representation Benchmark", content)
        self.assertIn("test-model", content)
        self.assertIn("Alpine North", content)
        self.assertIn("flickr_101", content)
        self.assertIn("50.0%", content)
        self.assertIn('id="query-data"', content)
        self.assertNotIn("{{QUERY_DATA_JSON}}", content)
        self.assertNotIn("{{TITLE}}", content)


if __name__ == "__main__":
    unittest.main()
