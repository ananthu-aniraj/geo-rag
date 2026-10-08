import os
import tempfile
import unittest

import pandas as pd

from src.utils.clean_descriptions import clean_layout_artifacts, clean_parquet_file


class TestCleanDescriptions(unittest.TestCase):
    def test_clean_layout_artifacts_standalone_opening(self):
        text = (
            "The input image is a vertical stack of four real-world photographs. "
            "The dominant cover across the foreground is water."
        )
        cleaned = clean_layout_artifacts(text)
        self.assertNotIn("vertical stack", cleaned.lower())
        self.assertTrue(
            cleaned.startswith("The dominant cover across the foreground is water.")
        )

    def test_clean_layout_artifacts_opening_clause_depicting(self):
        text = (
            "The input image is a vertical stack of four real-world photographs, "
            "all depicting scenes affected by flooding. The dominant cover is water."
        )
        cleaned = clean_layout_artifacts(text)
        self.assertNotIn("vertical stack", cleaned.lower())
        self.assertIn("scenes affected by flooding", cleaned)
        self.assertTrue(cleaned.startswith("Representative views depict"))

    def test_clean_layout_artifacts_opening_clause_appearing(self):
        text = (
            "The input image is a vertical stack of four real-world photographs, "
            "all appearing to be taken at night from a street. Buildings line both sides."
        )
        cleaned = clean_layout_artifacts(text)
        self.assertNotIn("vertical stack", cleaned.lower())
        self.assertTrue(
            cleaned.startswith("Representative views appear to be taken at night")
        )

    def test_clean_layout_artifacts_in_sentence_phrases(self):
        text = (
            "The visual evidence across the four stacked photographs overwhelmingly depicts a dense urban area. "
            "In the bottom frame, parked cars are visible. The primary feature across the vertical stack "
            "is a paved roadway."
        )
        cleaned = clean_layout_artifacts(text)
        self.assertNotIn("stacked photographs", cleaned.lower())
        self.assertNotIn("vertical stack", cleaned.lower())
        self.assertNotIn("bottom frame", cleaned.lower())
        self.assertIn("across the scene", cleaned)

    def test_clean_layout_artifacts_noop_on_clean_text(self):
        text = (
            "A dense broadleaved forest with a natural river winding through a valley."
        )
        self.assertEqual(clean_layout_artifacts(text), text)
        self.assertIsNone(clean_layout_artifacts(None))
        self.assertEqual(clean_layout_artifacts(""), "")

    def test_clean_parquet_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.parquet")
            df = pd.DataFrame(
                {
                    "cluster_id": [1, 2],
                    "visual_description": [
                        "The input image is a vertical stack of four real-world photographs, each depicting farmland.",
                        "Clean visual description of a desert.",
                    ],
                    "cluster_description": [
                        "Across the four stacked photographs, cropland dominates.",
                        "Pure sand dunes and sparse vegetation.",
                    ],
                }
            )
            df.to_parquet(file_path, index=False)

            clean_parquet_file(file_path)

            res_df = pd.read_parquet(file_path)
            self.assertEqual(len(res_df), 2)
            self.assertNotIn(
                "vertical stack", res_df["visual_description"].iloc[0].lower()
            )
            self.assertNotIn(
                "stacked photographs", res_df["cluster_description"].iloc[0].lower()
            )
            self.assertEqual(
                res_df["visual_description"].iloc[1],
                "Clean visual description of a desert.",
            )


if __name__ == "__main__":
    unittest.main()
