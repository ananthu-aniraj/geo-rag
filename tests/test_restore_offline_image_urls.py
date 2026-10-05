import os
import shutil
import tempfile
import unittest

import pandas as pd

from src.utils.restore_offline_image_urls import restore_image_urls


class TestRestoreOfflineImageUrls(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.offline_file = os.path.join(self.test_dir, "offline.parquet")
        self.online_file = os.path.join(self.test_dir, "online.parquet")
        self.output_file = os.path.join(self.test_dir, "restored.parquet")

        # Online dataset
        self.df_online = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4", "5"],
                "Platform": [
                    "flickr",
                    "mapillary",
                    "kartaview",
                    "inaturalist",
                    "wikimedia",
                ],
                "photo_key": [
                    "flickr_1",
                    "mapillary_2",
                    "kartaview_3",
                    "inaturalist_4",
                    "wikimedia_5",
                ],
                "Image_URL": [
                    "https://live.staticflickr.com/1.jpg",
                    "mapillary://2",
                    "kartaview://3",
                    "https://inaturalist.org/4.jpg",
                    "/server/raw/gldv2/5.jpg",
                ],
            }
        )
        self.df_online.to_parquet(self.online_file, compression="zstd")

        # Offline dataset (corrupted Image_URL pointing to local Image_Location)
        self.df_offline = pd.DataFrame(
            {
                "Photo_ID": ["1", "2", "3", "4", "5", "6"],
                "Platform": [
                    "flickr",
                    "mapillary",
                    "kartaview",
                    "inaturalist",
                    "wikimedia",
                    "flickr",
                ],
                "photo_key": [
                    "flickr_1",
                    "mapillary_2",
                    "kartaview_3",
                    "inaturalist_4",
                    "wikimedia_5",
                    "flickr_6",
                ],
                "Image_URL": [
                    "./images/flickr/1.jpg",
                    "./images/mapillary/2.jpg",
                    "./images/kartaview/3.jpg",
                    "./images/inaturalist/4.jpg",
                    "./images/wikimedia/5.jpg",
                    "./images/flickr/6.jpg",  # Not in online dataset
                ],
                "Image_Location": [
                    "./images/flickr/1.jpg",
                    "./images/mapillary/2.jpg",
                    "./images/kartaview/3.jpg",
                    "./images/inaturalist/4.jpg",
                    "./images/wikimedia/5.jpg",
                    "./images/flickr/6.jpg",
                ],
                "file_name": ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg"],
            }
        )
        self.df_offline.to_parquet(self.offline_file, compression="zstd")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_restore_image_urls_target_platforms_only(self):
        restore_image_urls(
            offline_path=self.offline_file,
            online_path=self.online_file,
            output_path=self.output_file,
            key_col="photo_key",
            target_platforms=["flickr", "mapillary", "kartaview", "inaturalist"],
        )

        df_res = pd.read_parquet(self.output_file)
        self.assertEqual(len(df_res), 6)

        # Target platforms must have restored online URLs
        self.assertEqual(
            df_res.loc[df_res["photo_key"] == "flickr_1", "Image_URL"].iloc[0],
            "https://live.staticflickr.com/1.jpg",
        )
        self.assertEqual(
            df_res.loc[df_res["photo_key"] == "mapillary_2", "Image_URL"].iloc[0],
            "mapillary://2",
        )
        self.assertEqual(
            df_res.loc[df_res["photo_key"] == "kartaview_3", "Image_URL"].iloc[0],
            "kartaview://3",
        )
        self.assertEqual(
            df_res.loc[df_res["photo_key"] == "inaturalist_4", "Image_URL"].iloc[0],
            "https://inaturalist.org/4.jpg",
        )

        # Non-target platform (wikimedia) must remain untouched
        self.assertEqual(
            df_res.loc[df_res["photo_key"] == "wikimedia_5", "Image_URL"].iloc[0],
            "./images/wikimedia/5.jpg",
        )

        # Record missing from online dataset (flickr_6) must remain untouched
        self.assertEqual(
            df_res.loc[df_res["photo_key"] == "flickr_6", "Image_URL"].iloc[0],
            "./images/flickr/6.jpg",
        )

        # Image_Location and file_name must remain completely intact
        for i in range(1, 7):
            self.assertTrue(
                df_res.loc[df_res["Photo_ID"] == str(i), "Image_Location"]
                .iloc[0]
                .startswith("./images/")
            )
            self.assertEqual(
                df_res.loc[df_res["Photo_ID"] == str(i), "file_name"].iloc[0],
                f"{i}.jpg",
            )


if __name__ == "__main__":
    unittest.main()
