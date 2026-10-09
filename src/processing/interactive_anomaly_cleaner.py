"""Interactive Manual Geospatial Anomaly Cleaner.

Enables interactive inspection and removal of H3 cell anomalies on a continent-scoped map,
focusing on Mapillary coverage, with on-click sample photo visualization, multi-platform
targeting, staged removal rules, and a single-step streaming Parquet purge.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set

import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.utils.credentials import get_mapillary_token
from src.utils.io import get_parquet_writer
from src.utils.mapillary_trajectory_validator import MapillaryTrajectoryValidator
from src.visualization.map_utils import get_carto_tile_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Continent Data Index & Inverted Lookup
# ==============================================================================
class ContinentDatasetIndex:
    """In-memory spatial index of a single continent from a Parquet dataset."""

    def __init__(
        self,
        parquet_path: str,
        continent: str,
        h3_res: int = 4,
        rules_path: str = "manual_h3_removals.json",
    ):
        self.parquet_path = parquet_path
        self.continent = continent
        self.h3_res = h3_res
        self.rules_path = rules_path
        self.mapillary_token = get_mapillary_token()
        self.thumb_cache: Dict[str, str] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.validator: Optional[MapillaryTrajectoryValidator] = None

        self.df: pd.DataFrame = pd.DataFrame()
        self.cell_indices: Dict[str, List[int]] = {}
        self.cell_stats: Dict[str, Dict[str, Any]] = {}
        self.total_records = 0

        self.load_and_index()
        self.load_rules()

    def load_and_index(self) -> None:
        logger.info(
            "Loading Parquet index for continent: '%s' from %s...",
            self.continent,
            self.parquet_path,
        )
        t0 = time.time()
        pf = pq.ParquetFile(self.parquet_path)
        avail_cols = pf.schema_arrow.names

        wanted_cols = [
            "Photo_ID",
            "Platform",
            "Latitude",
            "Longitude",
            "Image_URL",
            "Captured_At",
            "H3_Cell",
            "Season",
            "Time_Of_Day",
            "Koppen_Code",
            "Koppen_Desc",
            "continent",
            "country",
        ]
        load_cols = [c for c in wanted_cols if c in avail_cols]

        tbl = pf.read(columns=load_cols)
        df_all = tbl.to_pandas()

        cont_col = "continent" if "continent" in df_all.columns else "Continent"
        if cont_col in df_all.columns:
            mask = df_all[cont_col].astype(str).str.lower() == self.continent.lower()
            self.df = df_all[mask].copy().reset_index(drop=True)
        else:
            logger.warning("No continent column found. Loading entire dataset.")
            self.df = df_all.reset_index(drop=True)

        self.total_records = len(self.df)
        logger.info(
            "Loaded %s records for continent '%s' in %.2fs.",
            f"{self.total_records:,}",
            self.continent,
            time.time() - t0,
        )

        if self.total_records == 0:
            logger.error("No records found for continent '%s'.", self.continent)
            return

        # Standardize strings
        self.df["Platform_Clean"] = (
            self.df["Platform"].fillna("").astype(str).str.lower()
        )

        # Vectorized H3 parent coarsening
        t_h3 = time.time()
        unique_cells = self.df["H3_Cell"].dropna().unique()
        cell_map = {c: h3.cell_to_parent(c, self.h3_res) for c in unique_cells}
        self.df["H3_parent"] = self.df["H3_Cell"].map(cell_map)
        logger.info(
            "Mapped to H3 resolution %d in %.2fs. Unique parent cells: %d",
            self.h3_res,
            time.time() - t_h3,
            self.df["H3_parent"].nunique(),
        )

        # Build inverted index mapping cell -> row indices
        self.cell_indices = self.df.groupby("H3_parent", observed=False).indices

        # Precompute summary statistics per cell
        self.precompute_cell_stats()

    def precompute_cell_stats(self) -> None:
        logger.info("Precomputing cell statistics...")
        t0 = time.time()
        self.cell_stats = {}

        has_country = "country" in self.df.columns
        has_koppen = "Koppen_Code" in self.df.columns
        has_koppen_desc = "Koppen_Desc" in self.df.columns

        grouped = self.df.groupby("H3_parent")
        for cell, group in grouped:
            if not isinstance(cell, str) or not cell:
                continue

            total = len(group)
            m_count = int((group["Platform_Clean"] == "mapillary").sum())
            other_count = total - m_count

            # Top countries
            countries = []
            if has_country:
                top_c = group["country"].dropna().value_counts().head(3)
                countries = list(top_c.items())

            # Top Koppen
            koppen = []
            if has_koppen:
                top_k = group["Koppen_Code"].dropna().value_counts().head(2)
                for code, cnt in top_k.items():
                    desc = ""
                    if has_koppen_desc:
                        sub = group[group["Koppen_Code"] == code]
                        if not sub.empty:
                            desc = str(sub["Koppen_Desc"].dropna().iloc[0] or "")
                    koppen.append((code, desc, cnt))

            lat_mean = (
                float(group["Latitude"].mean()) if "Latitude" in group.columns else 0.0
            )
            lon_mean = (
                float(group["Longitude"].mean())
                if "Longitude" in group.columns
                else 0.0
            )

            self.cell_stats[cell] = {
                "total": total,
                "mapillary": m_count,
                "other": other_count,
                "countries": countries,
                "koppen": koppen,
                "centroid": [lat_mean, lon_mean],
            }
        logger.info(
            "Precomputed stats for %d cells in %.2fs.",
            len(self.cell_stats),
            time.time() - t0,
        )

    def load_rules(self) -> None:
        if os.path.exists(self.rules_path):
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.rules = {
                            item["cell"]: item for item in data if "cell" in item
                        }
                    elif isinstance(data, dict):
                        self.rules = data
                logger.info(
                    "Loaded %d staged rules from %s.",
                    len(self.rules),
                    self.rules_path,
                )
            except Exception as e:
                logger.error("Failed to load rules from %s: %s", self.rules_path, e)

    def save_rules(self) -> None:
        try:
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(list(self.rules.values()), f, indent=2)
            logger.info("Saved %d rules to %s.", len(self.rules), self.rules_path)
        except Exception as e:
            logger.error("Failed to save rules to %s: %s", self.rules_path, e)

    def fetch_mapillary_thumbnails(self, photo_ids: List[str]) -> Dict[str, str]:
        """Fetch Mapillary thumbnails in a single batch query, caching results."""
        if not self.mapillary_token or not photo_ids:
            return {}

        needed = [pid for pid in photo_ids if pid not in self.thumb_cache]
        if not needed:
            return {
                pid: self.thumb_cache[pid]
                for pid in photo_ids
                if pid in self.thumb_cache
            }

        # Query up to 50 IDs at once
        headers = {"Authorization": f"OAuth {self.mapillary_token}"}
        for i in range(0, len(needed), 50):
            chunk = needed[i : i + 50]
            url = f"https://graph.mapillary.com/?ids={','.join(chunk)}&fields=id,thumb_256_url"
            try:
                res = requests.get(url, headers=headers, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    for pid, val in data.items():
                        if isinstance(val, dict) and "thumb_256_url" in val:
                            self.thumb_cache[str(pid)] = val["thumb_256_url"]
            except Exception as e:
                logger.debug("Mapillary thumbnail query error: %s", e)

        return {
            pid: self.thumb_cache[pid] for pid in photo_ids if pid in self.thumb_cache
        }

    def get_cell_samples(self, cell: str, limit: int = 16) -> Dict[str, Any]:
        """Retrieve detailed cell metadata and sample photos."""
        if cell not in self.cell_stats:
            return {"error": "Cell not found"}

        stats = self.cell_stats[cell]
        indices = self.cell_indices.get(cell, [])
        if len(indices) == 0:
            return {**stats, "cell": cell, "samples": []}

        sub = self.df.iloc[indices]

        # Prioritize Mapillary samples first, then others
        mapillary_sub = sub[sub["Platform_Clean"] == "mapillary"]
        other_sub = sub[sub["Platform_Clean"] != "mapillary"]

        m_limit = min(limit, len(mapillary_sub))
        o_limit = min(limit - m_limit, len(other_sub))

        selected = pd.concat([mapillary_sub.head(m_limit), other_sub.head(o_limit)])

        # Batch-fetch Mapillary thumbnail URLs
        m_pids = [
            str(p).strip().removesuffix(".0")
            for p in selected[selected["Platform_Clean"] == "mapillary"][
                "Photo_ID"
            ].dropna()
        ]
        thumbs = self.fetch_mapillary_thumbnails(m_pids)

        samples = []
        for _, row in selected.iterrows():
            pid = str(row.get("Photo_ID", "")).strip().removesuffix(".0")
            plat = str(row.get("Platform_Clean", "unknown"))
            raw_url = str(row.get("Image_URL", ""))

            thumb_url = None
            if plat == "mapillary":
                thumb_url = thumbs.get(pid)
            elif raw_url.startswith("http"):
                thumb_url = raw_url

            samples.append(
                {
                    "photo_id": pid,
                    "platform": plat,
                    "captured_at": str(row.get("Captured_At", "N/A")),
                    "thumb_url": thumb_url,
                    "mapillary_web_url": (
                        f"https://www.mapillary.com/app/?pKey={pid}"
                        if plat == "mapillary"
                        else raw_url
                    ),
                    "lat": float(row.get("Latitude", 0.0)),
                    "lon": float(row.get("Longitude", 0.0)),
                }
            )

        return {
            "cell": cell,
            "total": stats["total"],
            "mapillary": stats["mapillary"],
            "other": stats["other"],
            "centroid": stats["centroid"],
            "countries": stats["countries"],
            "koppen": stats["koppen"],
            "samples": samples,
            "rule": self.rules.get(cell),
        }


# ==============================================================================
# 2. Mapillary Sequence Expansion Helpers (Targeted)
# ==============================================================================
def resolve_violating_mapillary_sequences(
    df: pd.DataFrame,
    cell_indices: Dict[str, List[int]],
    flagged_cells: Set[str],
    token: str,
) -> Set[str]:
    """Retrieves all sequence photo IDs for flagged Mapillary cells."""
    if not token or not flagged_cells:
        return set()

    flagged_pids = set()
    for cell in flagged_cells:
        idxs = cell_indices.get(cell, [])
        if not idxs:
            continue
        sub = df.iloc[idxs]
        m_pids = sub[sub["Platform_Clean"] == "mapillary"]["Photo_ID"].dropna()
        flagged_pids.update([str(p).strip().removesuffix(".0") for p in m_pids])

    if not flagged_pids:
        return set()

    logger.info(
        "Resolving sequence IDs for %d flagged Mapillary photos...",
        len(flagged_pids),
    )
    headers = {"Authorization": f"OAuth {token}"}
    pids_list = list(flagged_pids)
    sequences = set()

    for i in range(0, len(pids_list), 100):
        chunk = pids_list[i : i + 100]
        url = f"https://graph.mapillary.com/?ids={','.join(chunk)}&fields=id,sequence"
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                for _, info in r.json().items():
                    if isinstance(info, dict) and "sequence" in info:
                        sequences.add(str(info["sequence"]))
        except Exception as e:
            logger.warning("Error fetching sequence batch: %s", e)

    logger.info(
        "Found %d distinct sequence IDs. Fetching full sequence photos...",
        len(sequences),
    )
    expanded_image_ids = set()
    for seq_id in sequences:
        url = f"https://graph.mapillary.com/image_ids?sequence_id={seq_id}&limit=2000"
        while url:
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.status_code == 200:
                    d = r.json()
                    for item in d.get("data", []):
                        if "id" in item:
                            expanded_image_ids.add(
                                str(item["id"]).strip().removesuffix(".0")
                            )
                    url = d.get("paging", {}).get("next")
                else:
                    break
            except Exception:
                break

    logger.info("Expanded to %d total sequence photo IDs.", len(expanded_image_ids))
    return expanded_image_ids


# ==============================================================================
# 3. Streaming Parquet Purge Execution
# ==============================================================================
def execute_parquet_purge(
    input_parquet: str,
    output_parquet: str,
    rules: List[Dict[str, Any]],
    df_index: Optional[ContinentDatasetIndex] = None,
) -> Dict[str, Any]:
    """Executes a single-step streaming filter over the input Parquet file."""
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    if not rules:
        logger.info("No rules specified. Outputting copy of input.")
        return {"removed_rows": 0, "status": "no_rules"}

    t0 = time.time()
    pf = pq.ParquetFile(input_parquet)
    num_row_groups = pf.num_row_groups
    schema = pf.schema_arrow

    flagged_all_cells = {r["cell"] for r in rules if r.get("platform") == "all"}
    flagged_mapillary_cells = {
        r["cell"] for r in rules if r.get("platform") == "mapillary"
    }
    seq_expanded_cells = {r["cell"] for r in rules if r.get("expand_sequence")}

    # Detect resolution from rules
    sample_cell = rules[0]["cell"]
    rule_res = h3.get_resolution(sample_cell)

    violating_sequence_pids = set()
    if seq_expanded_cells and df_index and df_index.mapillary_token:
        violating_sequence_pids = resolve_violating_mapillary_sequences(
            df_index.df,
            df_index.cell_indices,
            seq_expanded_cells,
            df_index.mapillary_token,
        )

    has_photo_id = "Photo_ID" in schema.names or "photo_id" in schema.names
    photo_id_col = "Photo_ID" if "Photo_ID" in schema.names else "photo_id"

    temp_output = output_parquet + ".tmp_clean"
    removed_count = 0
    total_processed = 0

    logger.info(
        "Starting streaming purge: %d cells flagged (%d 'all', %d 'mapillary', %d seq-expanded)...",
        len(rules),
        len(flagged_all_cells),
        len(flagged_mapillary_cells),
        len(violating_sequence_pids),
    )

    with get_parquet_writer(temp_output, schema) as writer:
        for rg in range(num_row_groups):
            tbl_rg = pf.read_row_group(rg)
            total_processed += len(tbl_rg)

            rg_h3 = tbl_rg["H3_Cell"].to_numpy().astype(str)
            rg_plat = tbl_rg["Platform"].to_numpy().astype(str)
            rg_plat_lower = np.char.lower(rg_plat)

            # Vectorized parent cell mapping for row group
            unq_h3, inv_h3 = np.unique(rg_h3, return_inverse=True)
            parents_unq = np.array([h3.cell_to_parent(c, rule_res) for c in unq_h3])
            rg_parents = parents_unq[inv_h3]

            keep_mask = np.ones(len(tbl_rg), dtype=bool)

            # 1. Purge all in cell
            if flagged_all_cells:
                in_all_cells = np.isin(rg_parents, list(flagged_all_cells))
                keep_mask[in_all_cells] = False

            # 2. Purge Mapillary in cell
            if flagged_mapillary_cells:
                in_map_cells = np.isin(rg_parents, list(flagged_mapillary_cells))
                is_mapillary = rg_plat_lower == "mapillary"
                keep_mask[in_map_cells & is_mapillary] = False

            # 3. Purge violating sequence IDs
            if violating_sequence_pids and has_photo_id:
                rg_pids = tbl_rg[photo_id_col].to_numpy().astype(str)
                clean_pids = np.array([p.removesuffix(".0") for p in rg_pids])
                in_seq = np.isin(clean_pids, list(violating_sequence_pids))
                keep_mask[in_seq & (rg_plat_lower == "mapillary")] = False

            filtered_tbl = tbl_rg.filter(pa.array(keep_mask))
            writer.write_table(filtered_tbl)
            removed_count += len(tbl_rg) - len(filtered_tbl)

    os.replace(temp_output, output_parquet)
    duration = time.time() - t0
    logger.info(
        "Purge completed in %.2fs. Removed %d rows out of %d (%.2f%%). Output saved to %s.",
        duration,
        removed_count,
        total_processed,
        (removed_count / max(1, total_processed)) * 100,
        output_parquet,
    )

    return {
        "initial_rows": total_processed,
        "removed_rows": removed_count,
        "remaining_rows": total_processed - removed_count,
        "duration_seconds": round(duration, 2),
        "output_parquet": output_parquet,
    }


# ==============================================================================
# 4. Frontend Single-Page HTML App
# ==============================================================================
def generate_html_dashboard(
    continent: str,
    h3_res: int,
    carto_tile_url: str,
    carto_attribution: str,
) -> str:
    """Returns the single-page Leaflet dashboard with full styling and interactivity."""
    template_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "visualization",
        "templates",
        "interactive_anomaly_cleaner.html",
    )
    with open(template_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    return (
        html_template.replace("{{CONTINENT}}", continent)
        .replace("{{H3_RES}}", str(h3_res))
        .replace("{{CARTO_TILE_URL}}", carto_tile_url)
        .replace("{{CARTO_ATTRIBUTION}}", carto_attribution)
    )


# ==============================================================================
# 5. Standard Library HTTP Server Handler
# ==============================================================================
class AnomalyCleanerHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for the interactive anomaly cleaner."""

    dataset_index: ContinentDatasetIndex
    output_parquet: str
    carto_tile_url: str
    carto_attribution: str

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(
            "%s - - [%s] %s",
            self.client_address[0],
            self.log_date_time_string(),
            format % args,
        )

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_str: str, status: int = 200) -> None:
        payload = html_str.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("", "/"):
            html = generate_html_dashboard(
                continent=self.dataset_index.continent,
                h3_res=self.dataset_index.h3_res,
                carto_tile_url=self.carto_tile_url,
                carto_attribution=self.carto_attribution,
            )
            self._send_html(html)
        elif path == "/api/cells":
            features = []
            for cell, stats in self.dataset_index.cell_stats.items():
                try:
                    coords = h3.cell_to_boundary(cell)
                    lngs = [c[1] for c in coords]
                    if max(lngs) - min(lngs) > 180:
                        coords = [
                            (lat, lng + 360 if lng < 0 else lng) for lat, lng in coords
                        ]
                    geojson_coords = [[lng, lat] for lat, lng in coords]
                    geojson_coords.append(geojson_coords[0])

                    top_country = stats["countries"][0][0] if stats["countries"] else ""

                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [geojson_coords],
                            },
                            "properties": {
                                "cell": cell,
                                "total": stats["total"],
                                "mapillary": stats["mapillary"],
                                "other": stats["other"],
                                "top_country": top_country,
                            },
                        }
                    )
                except Exception:
                    continue
            self._send_json({"type": "FeatureCollection", "features": features})
        elif path.startswith("/api/cell/"):
            cell_id = path.removeprefix("/api/cell/").strip()
            if cell_id in self.dataset_index.cell_stats:
                data = self.dataset_index.get_cell_samples(cell_id)
                self._send_json(data)
            else:
                self._send_json({"error": "Cell not found"}, status=404)
        elif path.startswith("/api/sequence/track/"):
            photo_id = path.removeprefix("/api/sequence/track/").strip()
            if not self.dataset_index.validator:
                self.dataset_index.validator = MapillaryTrajectoryValidator(
                    token=self.dataset_index.mapillary_token
                )
            result = self.dataset_index.validator.validate_photo_id(photo_id)
            self._send_json(result.to_dict())
        elif path == "/api/rules":
            self._send_json(list(self.dataset_index.rules.values()))
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_len > 0:
            try:
                body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            except Exception:
                body = {}

        if path == "/api/rules/add":
            cell = body.get("cell")
            if not cell:
                self._send_json({"error": "Missing cell"}, status=400)
                return
            rule = {
                "cell": cell,
                "platform": body.get("platform", "mapillary"),
                "expand_sequence": bool(body.get("expand_sequence", False)),
            }
            self.dataset_index.rules[cell] = rule
            self.dataset_index.save_rules()
            self._send_json({"status": "ok", "rule": rule})
        elif path == "/api/rules/remove":
            cell = body.get("cell")
            if cell in self.dataset_index.rules:
                del self.dataset_index.rules[cell]
                self.dataset_index.save_rules()
            self._send_json({"status": "ok"})
        elif path == "/api/rules/clear":
            self.dataset_index.rules.clear()
            self.dataset_index.save_rules()
            self._send_json({"status": "ok"})
        elif path == "/api/purge":
            res = execute_parquet_purge(
                self.dataset_index.parquet_path,
                self.output_parquet,
                list(self.dataset_index.rules.values()),
                df_index=self.dataset_index,
            )
            self._send_json(res)
        else:
            self._send_json({"error": "Not found"}, status=404)


def create_server(
    dataset_index: ContinentDatasetIndex,
    output_parquet: str,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> http.server.ThreadingHTTPServer:
    """Creates a ThreadingHTTPServer configured with the AnomalyCleanerHandler."""
    tile_url, attribution = get_carto_tile_config("light_all")

    handler_cls = type(
        "ConfiguredAnomalyCleanerHandler",
        (AnomalyCleanerHandler,),
        {
            "dataset_index": dataset_index,
            "output_parquet": output_parquet,
            "carto_tile_url": tile_url,
            "carto_attribution": attribution,
        },
    )

    return http.server.ThreadingHTTPServer((host, port), handler_cls)


# ==============================================================================
# 6. Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Interactive Manual Geospatial Anomaly Cleaner for Mapillary."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="full_pipeline_output/geo_space_deduplicated.parquet",
        help="Path to input Parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to cleaned output Parquet dataset (defaults to input path).",
    )
    parser.add_argument(
        "--continent",
        type=str,
        default="Africa",
        help="Target continent to inspect (e.g. 'Africa', 'Europe', 'Asia', 'North America').",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=4,
        choices=[3, 4, 5],
        help="H3 resolution for aggregation grid (default: 4, ~177km edge).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve the interactive web dashboard (default: 8080).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--rules-file",
        type=str,
        default="manual_h3_removals.json",
        help="Path to JSON file to save and load staged removal rules.",
    )
    parser.add_argument(
        "--batch-purge",
        action="store_true",
        help="Non-interactive mode: apply rules from --rules-file directly to --input -> --output.",
    )
    args = parser.parse_args()

    output_path = args.output if args.output else args.input

    if args.batch_purge:
        if not os.path.exists(args.rules_file):
            print(f"Error: Rules file not found: {args.rules_file}")
            return
        with open(args.rules_file, "r", encoding="utf-8") as f:
            rules = json.load(f)
        print(f"Applying {len(rules)} rules from {args.rules_file} to {args.input}...")
        res = execute_parquet_purge(args.input, output_path, rules)
        print(f"Purge complete: {res}")
        return

    # Interactive Web Dashboard Mode
    dataset_index = ContinentDatasetIndex(
        parquet_path=args.input,
        continent=args.continent,
        h3_res=args.res,
        rules_path=args.rules_file,
    )

    server = create_server(dataset_index, output_path, host=args.host, port=args.port)

    print(
        "================================================================================"
    )
    print("🗺️  GEO-RAG INTERACTIVE ANOMALY CLEANER")
    print(f" -> Continent: {args.continent}")
    print(f" -> Input:     {args.input}")
    print(f" -> Output:    {output_path}")
    print(f" -> Grid Res:  H3 Res {args.res}")
    print(f" -> Server:    http://{args.host}:{args.port}")
    print(
        "================================================================================"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
