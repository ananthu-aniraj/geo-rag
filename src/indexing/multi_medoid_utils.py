import numpy as np
import pandas as pd
from PIL import Image


def create_letterboxed_cell(img, target_w=512, target_h=256, bg_color=(40, 40, 40)):
    """
    Resizes a PIL Image to fit inside target_w x target_h keeping aspect ratio,
    padding any empty background space with bg_color (default dark-gray).
    """
    # Create dark background image
    cell = Image.new("RGB", (target_w, target_h), color=bg_color)

    w, h = img.size
    ratio = min(target_w / w, target_h / h)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
    resized_img = img.resize((new_w, new_h), resample)

    # Paste centered in cell
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    cell.paste(resized_img, (paste_x, paste_y))
    return cell


def stitch_cells_vertically(cells, target_w=512, target_h=256):
    """
    Stitches a list of cell images vertically into a single PIL Image.
    """
    if not cells:
        return None
    total_h = len(cells) * target_h
    collage = Image.new("RGB", (target_w, total_h))
    for idx, cell in enumerate(cells):
        collage.paste(cell, (0, idx * target_h))
    return collage


def sample_diverse_medoids(embeddings_norm, indices, centroid_norm, df, n_medoids=4):
    """
    Samples up to n_medoids representative images for a cluster,
    prioritizing geographical coordinate and image URL diversity.
    """
    if len(indices) == 0:
        return []

    # Extract NumPy arrays/lists once up-front to avoid extremely slow Pandas .iloc lookups inside loops
    if isinstance(df, dict):
        lats = df.get("Latitude", np.zeros(len(df)))
        lons = df.get("Longitude", np.zeros(len(df)))
        urls = df.get("Image_URL", [""] * len(df))
    else:
        lats = df["Latitude"].values if "Latitude" in df.columns else np.zeros(len(df))
        lons = (
            df["Longitude"].values if "Longitude" in df.columns else np.zeros(len(df))
        )
        urls = df["Image_URL"].values if "Image_URL" in df.columns else [""] * len(df)

    if embeddings_norm.shape[0] == len(indices):
        cluster_embs = embeddings_norm
    else:
        cluster_embs = embeddings_norm[indices]

    sims = np.dot(cluster_embs, centroid_norm)

    # Sort indices by cosine similarity to centroid descending
    sorted_order = np.argsort(sims)[::-1]
    sorted_indices = indices[sorted_order]

    selected = [sorted_indices[0]]

    # Try to select diverse candidates
    for idx in sorted_indices[1:]:
        if len(selected) >= n_medoids:
            break

        lat = lats[idx]
        lon = lons[idx]
        url = urls[idx]

        is_diverse = True
        for sel_idx in selected:
            sel_lat = lats[sel_idx]
            sel_lon = lons[sel_idx]
            sel_url = urls[sel_idx]

            # Distance comparison to prevent near-duplicate coordinates
            lat_diff = abs(lat - sel_lat)
            lon_diff = abs(lon - sel_lon)
            if lat_diff < 1e-4 and lon_diff < 1e-4:
                is_diverse = False
                break

            if url == sel_url:
                is_diverse = False
                break

        if is_diverse:
            selected.append(idx)

    # Fallback to closest available items if we couldn't fill the quota
    for idx in sorted_indices:
        if len(selected) >= n_medoids:
            break
        if idx not in selected:
            selected.append(idx)

    return selected


def aggregate_medoid_metadata(indices, df):
    """
    Aggregates coordinates (bounding box + centroid), country, continent,
    season, time of day, and Koppen climates across multiple medoid indices.
    """
    if not indices:
        return {
            "location": "Unknown",
            "country": "Unknown",
            "continent": "Unknown",
            "season": "Unknown",
            "time_of_day": "Unknown",
            "koppen_code": "Unknown",
            "koppen_desc": "Unknown",
        }

    items = df.iloc[indices]

    # coordinates & bounding box
    lats = items["Latitude"].dropna().tolist() if "Latitude" in items.columns else []
    lons = items["Longitude"].dropna().tolist() if "Longitude" in items.columns else []

    if lats and lons:
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        mean_lat = sum(lats) / len(lats)
        mean_lon = sum(lons) / len(lons)
        if len(lats) > 1:
            location = f"Bounding Box: ({min_lat:.5f}, {min_lon:.5f}) to ({max_lat:.5f}, {max_lon:.5f}) | Centroid: ({mean_lat:.5f}, {mean_lon:.5f})"
        else:
            location = f"Lat {mean_lat:.5f}, Lon {mean_lon:.5f}"
    else:
        location = "Unknown"

    def get_unique_list(col_name):
        if col_name not in items.columns:
            return "Unknown"
        vals = items[col_name].dropna().astype(str).tolist()
        unique_vals = []
        for v in vals:
            v_strip = v.strip()
            if v_strip and v_strip.lower() not in ["unknown", "nan", "n/a", "none"]:
                if v_strip not in unique_vals:
                    unique_vals.append(v_strip)
        return ", ".join(unique_vals) if unique_vals else "Unknown"

    country = get_unique_list("country")
    continent = get_unique_list("continent")
    season = get_unique_list("Season")
    time_of_day = get_unique_list("Time_Of_Day")
    koppen_code = get_unique_list("Koppen_Code")
    koppen_desc = get_unique_list("Koppen_Desc")

    return {
        "location": location,
        "country": country,
        "continent": continent,
        "season": season,
        "time_of_day": time_of_day,
        "koppen_code": koppen_code,
        "koppen_desc": koppen_desc,
    }


class DataFrameRowWrapper:
    """
    High-performance wrapper around Pandas DataFrame to provide O(1) row access
    via .iat indexer, avoiding the memory copying and CPU overhead of .iloc or .to_dict().
    """

    def __init__(self, df):
        self.df = df
        self.col_map = {col: i for i, col in enumerate(df.columns)}

    def get_row(self, idx):
        return DataFrameRow(self.df, idx, self.col_map)

    def __getitem__(self, idx):
        return DataFrameRow(self.df, idx, self.col_map)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for i in range(len(self.df)):
            yield DataFrameRow(self.df, i, self.col_map)


class DataFrameRow:
    def __init__(self, df, idx, col_map):
        self.df = df
        self.idx = idx
        self.col_map = col_map

    def get(self, key, default=None):
        col_idx = self.col_map.get(key)
        if col_idx is not None:
            val = self.df.iat[self.idx, col_idx]
            if (
                val is None
                or (isinstance(val, float) and np.isnan(val))
                or pd.isna(val)
            ):
                return default
            return val
        return default

    def __getitem__(self, key):
        col_idx = self.col_map.get(key)
        if col_idx is None:
            raise KeyError(key)

        val = self.df.iat[self.idx, col_idx]
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            raise KeyError(key)
        return val

    def __contains__(self, key):
        return key in self.col_map
