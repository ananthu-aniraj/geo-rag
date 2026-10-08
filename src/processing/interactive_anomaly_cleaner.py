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
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Geo-RAG: Interactive Anomaly Cleaner ({continent})</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; height: 100vh; overflow: hidden; display: flex; flex-direction: column; }}

        /* Top Navigation Bar */
        #topbar {{ height: 56px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; z-index: 1000; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
        .brand {{ font-size: 16px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background: #334155; color: #cbd5e1; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-left: 6px; }}
        .badge-continent {{ background: #0284c7; color: white; }}
        .nav-actions {{ display: flex; align-items: center; gap: 12px; }}

        button {{ cursor: pointer; border: none; border-radius: 6px; padding: 7px 14px; font-size: 13px; font-weight: 600; transition: all 0.15s ease; }}
        .btn-primary {{ background: #0284c7; color: white; }}
        .btn-primary:hover {{ background: #0369a1; }}
        .btn-danger {{ background: #dc2626; color: white; }}
        .btn-danger:hover {{ background: #b91c1c; }}
        .btn-warning {{ background: #d97706; color: white; }}
        .btn-warning:hover {{ background: #b45309; }}
        .btn-secondary {{ background: #334155; color: #e2e8f0; }}
        .btn-secondary:hover {{ background: #475569; }}
        .btn-outline {{ background: transparent; border: 1px solid #475569; color: #cbd5e1; }}
        .btn-outline:hover {{ background: #334155; }}

        /* Main Layout */
        #main {{ display: flex; flex: 1; position: relative; height: calc(100vh - 56px); }}
        #map {{ flex: 1; height: 100%; width: 100%; background: #0f172a; }}

        /* Side Inspection Drawer */
        #drawer {{ width: 440px; background: #1e293b; border-left: 1px solid #334155; display: flex; flex-direction: column; z-index: 1000; box-shadow: -4px 0 16px rgba(0,0,0,0.3); transform: translateX(100%); transition: transform 0.25s ease-in-out; position: absolute; right: 0; top: 0; bottom: 0; }}
        #drawer.open {{ transform: translateX(0); }}
        .drawer-header {{ padding: 16px 20px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
        .drawer-header h2 {{ font-size: 16px; font-weight: 700; color: #f1f5f9; }}
        .drawer-body {{ flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }}
        .close-btn {{ background: transparent; color: #94a3b8; font-size: 20px; padding: 4px 8px; }}
        .close-btn:hover {{ color: white; }}

        /* Stats Cards */
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .stat-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; margin-bottom: 4px; }}
        .stat-value {{ font-size: 20px; font-weight: 700; color: #38bdf8; }}
        .stat-value.danger {{ color: #f87171; }}

        /* Action Buttons Area */
        .action-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 10px; }}
        .action-box h3 {{ font-size: 13px; font-weight: 700; color: #cbd5e1; }}
        .checkbox-label {{ display: flex; align-items: center; gap: 8px; font-size: 12px; color: #94a3b8; cursor: pointer; }}

        /* Metadata List */
        .meta-list {{ display: flex; flex-direction: column; gap: 6px; font-size: 13px; }}
        .meta-item {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #283548; }}
        .meta-item span:first-child {{ color: #94a3b8; }}
        .meta-item span:last-child {{ font-weight: 600; color: #e2e8f0; }}

        /* Photo Grid */
        .gallery-header {{ font-size: 13px; font-weight: 700; color: #cbd5e1; margin-top: 6px; display: flex; justify-content: space-between; }}
        .photo-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .photo-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; overflow: hidden; display: flex; flex-direction: column; }}
        .photo-thumb {{ width: 100%; height: 110px; object-fit: cover; background: #1e293b; display: block; }}
        .photo-no-thumb {{ width: 100%; height: 110px; display: flex; align-items: center; justify-content: center; background: #1e293b; color: #64748b; font-size: 11px; }}
        .photo-meta {{ padding: 6px 8px; font-size: 11px; display: flex; flex-direction: column; gap: 2px; }}
        .photo-link {{ color: #38bdf8; text-decoration: none; font-weight: 600; }}
        .photo-link:hover {{ text-decoration: underline; }}

        /* Staged Removals Modal */
        #modal {{ position: fixed; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); z-index: 2000; display: none; align-items: center; justify-content: center; }}
        #modal.open {{ display: flex; }}
        .modal-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; width: 600px; max-height: 80vh; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        .modal-header {{ padding: 16px 20px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
        .modal-body {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .modal-footer {{ padding: 14px 20px; border-top: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}

        .rule-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .rule-table th, .rule-table td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #334155; }}
        .rule-table th {{ color: #94a3b8; text-transform: uppercase; }}

        /* Legend */
        .legend {{ background: #1e293b; border: 1px solid #334155; padding: 10px 14px; border-radius: 8px; color: #cbd5e1; font-size: 12px; line-height: 1.4; position: absolute; bottom: 20px; left: 20px; z-index: 999; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .legend-bar {{ width: 140px; height: 10px; background: linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #f03b20, #bd0026); border-radius: 2px; margin: 4px 0; }}
        .legend-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; }}
    </style>
</head>
<body>
    <div id="topbar">
        <div class="brand">
            <span>🗺️ Geo-RAG Anomaly Cleaner</span>
            <span class="badge badge-continent">{continent}</span>
            <span class="badge">H3 Res {h3_res}</span>
            <span id="badge-total" class="badge">Loading...</span>
        </div>
        <div class="nav-actions">
            <button id="btn-view-staged" class="btn-secondary" onclick="openStagedModal()">
                🗑️ Staged Removals (<span id="staged-count">0</span>)
            </button>
            <button class="btn-danger" onclick="triggerPurge()">
                🚀 Execute Purge
            </button>
        </div>
    </div>

    <div id="main">
        <div id="map"></div>
        <div class="legend">
            <div><strong>Mapillary Density (Log10)</strong></div>
            <div class="legend-bar"></div>
            <div class="legend-labels">
                <span>1</span>
                <span>100</span>
                <span>10k+</span>
            </div>
            <div style="margin-top: 6px; display: flex; align-items: center; gap: 6px;">
                <span style="display:inline-block; width:12px; height:12px; background:#dc2626; border:1px dashed #ef4444; border-radius:2px;"></span>
                <span>Flagged for removal</span>
            </div>
        </div>

        <!-- Inspection Drawer -->
        <div id="drawer">
            <div class="drawer-header">
                <div>
                    <h2 id="drawer-cell">Cell Details</h2>
                    <span id="drawer-coords" style="font-size: 11px; color: #94a3b8;"></span>
                </div>
                <button class="close-btn" onclick="closeDrawer()">✕</button>
            </div>
            <div class="drawer-body">
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-label">Mapillary Images</div>
                        <div id="drawer-mapillary" class="stat-value danger">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Images</div>
                        <div id="drawer-total" class="stat-value">0</div>
                    </div>
                </div>

                <div class="action-box">
                    <h3>Removals & Actions</h3>
                    <div style="display: flex; gap: 8px;">
                        <button id="btn-purge-mapillary" class="btn-danger" style="flex: 1;" onclick="stageRemoval('mapillary')">
                            🚫 Purge Mapillary
                        </button>
                        <button id="btn-purge-all" class="btn-warning" style="flex: 1;" onclick="stageRemoval('all')">
                            🗑️ Purge Cell
                        </button>
                    </div>
                    <label class="checkbox-label">
                        <input type="checkbox" id="chk-expand-seq">
                        <span>Expand to full Mapillary track sequence(s)</span>
                    </label>
                    <button id="btn-unstage" class="btn-outline" style="display: none;" onclick="unstageCurrentCell()">
                        ↩️ Remove Flag / Keep
                    </button>
                </div>

                <div class="action-box">
                    <h3>Generic Metadata</h3>
                    <div class="meta-list">
                        <div class="meta-item">
                            <span>Top Countries</span>
                            <span id="drawer-countries">N/A</span>
                        </div>
                        <div class="meta-item">
                            <span>Köppen Climate</span>
                            <span id="drawer-koppen">N/A</span>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="gallery-header">
                        <span>Representative Photo Samples</span>
                        <span id="drawer-sample-count" style="color: #94a3b8; font-size: 11px;"></span>
                    </div>
                    <div id="drawer-photos" class="photo-grid" style="margin-top: 10px;">
                        <!-- Sample photo cards loaded here -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal for Staged Removals -->
    <div id="modal">
        <div class="modal-box">
            <div class="modal-header">
                <h2>Staged Anomaly Removals</h2>
                <button class="close-btn" onclick="closeStagedModal()">✕</button>
            </div>
            <div class="modal-body">
                <table class="rule-table">
                    <thead>
                        <tr>
                            <th>H3 Cell</th>
                            <th>Target</th>
                            <th>Seq Expand</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="staged-table-body">
                        <!-- Staged rules -->
                    </tbody>
                </table>
            </div>
            <div class="modal-footer">
                <button class="btn-outline" onclick="exportRulesJson()">📥 Export Rules JSON</button>
                <button class="btn-danger" onclick="triggerPurge()">🚀 Execute Purge Now</button>
            </div>
        </div>
    </div>

    <script>
        let map;
        let geojsonLayer;
        let cellDataMap = {{}};
        let stagedRules = {{}};
        let currentCellId = null;

        // Color mapping for log density
        function getColor(count) {{
            if (!count || count <= 0) return '#334155';
            const log = Math.log10(count);
            return log > 3.5 ? '#bd0026' :
                   log > 2.5 ? '#f03b20' :
                   log > 1.8 ? '#fd8d3c' :
                   log > 1.0 ? '#fecc5c' :
                               '#ffffb2';
        }}

        function initMap() {{
            map = L.map('map', {{
                preferCanvas: true,
                zoomControl: true,
                attributionControl: true
            }}).setView([20, 0], 3);

            L.tileLayer('{carto_tile_url}', {{
                attribution: '{carto_attribution}',
                maxZoom: 18
            }}).addTo(map);

            loadGeoJson();
            loadRules();
        }}

        async function loadGeoJson() {{
            try {{
                const res = await fetch('/api/cells');
                const data = await res.json();

                document.getElementById('badge-total').innerText = `${{data.features.length.toLocaleString()}} cells`;

                geojsonLayer = L.geoJSON(data, {{
                    style: styleFeature,
                    onEachFeature: onEachFeature
                }}).addTo(map);

                if (data.features.length > 0) {{
                    map.fitBounds(geojsonLayer.getBounds(), {{ padding: [20, 20] }});
                }}
            }} catch (err) {{
                console.error("Failed to load cells:", err);
            }}
        }}

        function styleFeature(feature) {{
            const cell = feature.properties.cell;
            const isStaged = !!stagedRules[cell];
            const mCount = feature.properties.mapillary;

            if (isStaged) {{
                return {{
                    fillColor: '#dc2626',
                    fillOpacity: 0.85,
                    color: '#ef4444',
                    weight: 2.5,
                    dashArray: '3, 3'
                }};
            }}

            return {{
                fillColor: getColor(mCount),
                fillOpacity: 0.65,
                color: '#475569',
                weight: 1
            }};
        }}

        function onEachFeature(feature, layer) {{
            const p = feature.properties;
            cellDataMap[p.cell] = p;

            layer.bindTooltip(`
                <strong>Cell:</strong> ${{p.cell}}<br/>
                <strong>Mapillary:</strong> ${{p.mapillary.toLocaleString()}}<br/>
                <strong>Total:</strong> ${{p.total.toLocaleString()}}<br/>
                <strong>Country:</strong> ${{p.top_country || 'N/A'}}
            `, {{ sticky: true }});

            layer.on('click', () => selectCell(p.cell));
        }}

        async function selectCell(cellId) {{
            currentCellId = cellId;
            document.getElementById('drawer').classList.add('open');
            document.getElementById('drawer-cell').innerText = cellId;
            document.getElementById('drawer-photos').innerHTML = '<div style="color:#94a3b8; font-size:12px;">Loading samples...</div>';

            try {{
                const res = await fetch(`/api/cell/${{cellId}}`);
                const data = await res.json();

                document.getElementById('drawer-mapillary').innerText = data.mapillary.toLocaleString();
                document.getElementById('drawer-total').innerText = data.total.toLocaleString();
                document.getElementById('drawer-coords').innerText = `Centroid: ${{data.centroid[0].toFixed(3)}}, ${{data.centroid[1].toFixed(3)}}`;

                const cStr = data.countries.map(c => `${{c[0]}} (${{c[1]}})`).join(', ') || 'N/A';
                document.getElementById('drawer-countries').innerText = cStr;

                const kStr = data.koppen.map(k => `${{k[0]}} - ${{k[1]}} (${{k[2]}})`).join('; ') || 'N/A';
                document.getElementById('drawer-koppen').innerText = kStr;

                // Staged state
                const isStaged = !!stagedRules[cellId];
                document.getElementById('btn-unstage').style.display = isStaged ? 'block' : 'none';
                if (isStaged) {{
                    document.getElementById('chk-expand-seq').checked = !!stagedRules[cellId].expand_sequence;
                }}

                // Render Photos
                document.getElementById('drawer-sample-count').innerText = `${{data.samples.length}} photos`;
                const photosEl = document.getElementById('drawer-photos');
                photosEl.innerHTML = '';

                data.samples.forEach(s => {{
                    const card = document.createElement('div');
                    card.className = 'photo-card';

                    const imgHtml = s.thumb_url
                        ? `<img class="photo-thumb" src="${{s.thumb_url}}" alt="${{s.photo_id}}" loading="lazy" />`
                        : `<div class="photo-no-thumb">No Thumbnail</div>`;

                    card.innerHTML = `
                        ${{imgHtml}}
                        <div class="photo-meta">
                            <a class="photo-link" href="${{s.mapillary_web_url}}" target="_blank">ID: ${{s.photo_id}} ↗</a>
                            <span style="color:#94a3b8;">${{s.platform}} · ${{s.captured_at}}</span>
                        </div>
                    `;
                    photosEl.appendChild(card);
                }});
            }} catch (err) {{
                console.error("Failed to load cell details:", err);
            }}
        }}

        async function stageRemoval(platform) {{
            if (!currentCellId) return;
            const expandSeq = document.getElementById('chk-expand-seq').checked;

            const res = await fetch('/api/rules/add', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    cell: currentCellId,
                    platform: platform,
                    expand_sequence: expandSeq
                }})
            }});

            if (res.ok) {{
                stagedRules[currentCellId] = {{ cell: currentCellId, platform: platform, expand_sequence: expandSeq }};
                updateStagedCount();
                geojsonLayer.eachLayer(layer => {{
                    if (layer.feature.properties.cell === currentCellId) {{
                        layer.setStyle(styleFeature(layer.feature));
                    }}
                }});
                document.getElementById('btn-unstage').style.display = 'block';
            }}
        }}

        async function unstageCurrentCell() {{
            if (!currentCellId) return;
            const res = await fetch('/api/rules/remove', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ cell: currentCellId }})
            }});

            if (res.ok) {{
                delete stagedRules[currentCellId];
                updateStagedCount();
                geojsonLayer.eachLayer(layer => {{
                    if (layer.feature.properties.cell === currentCellId) {{
                        layer.setStyle(styleFeature(layer.feature));
                    }}
                }});
                document.getElementById('btn-unstage').style.display = 'none';
            }}
        }}

        async function loadRules() {{
            const res = await fetch('/api/rules');
            const data = await res.json();
            stagedRules = {{}};
            data.forEach(r => {{ stagedRules[r.cell] = r; }});
            updateStagedCount();
        }}

        function updateStagedCount() {{
            const count = Object.keys(stagedRules).length;
            document.getElementById('staged-count').innerText = count;
        }}

        function openStagedModal() {{
            const tbody = document.getElementById('staged-table-body');
            tbody.innerHTML = '';

            Object.values(stagedRules).forEach(r => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${{r.cell}}</code></td>
                    <td><span class="badge">${{r.platform}}</span></td>
                    <td>${{r.expand_sequence ? '✅ Yes' : 'No'}}</td>
                    <td><button class="close-btn" onclick="removeRule('${{r.cell}}')">✕</button></td>
                `;
                tbody.appendChild(tr);
            }});

            document.getElementById('modal').classList.add('open');
        }}

        async function removeRule(cell) {{
            await fetch('/api/rules/remove', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ cell }})
            }});
            delete stagedRules[cell];
            updateStagedCount();
            openStagedModal();
            if (geojsonLayer) {{
                geojsonLayer.eachLayer(layer => {{
                    if (layer.feature.properties.cell === cell) {{
                        layer.setStyle(styleFeature(layer.feature));
                    }}
                }});
            }}
        }}

        function closeStagedModal() {{
            document.getElementById('modal').classList.remove('open');
        }}

        function closeDrawer() {{
            document.getElementById('drawer').classList.remove('open');
            currentCellId = null;
        }}

        function exportRulesJson() {{
            const blob = new Blob([JSON.stringify(Object.values(stagedRules), null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'manual_h3_removals.json';
            a.click();
        }}

        async function triggerPurge() {{
            const count = Object.keys(stagedRules).length;
            if (count === 0) {{
                alert("No cells are staged for removal.");
                return;
            }}

            if (!confirm(`Are you sure you want to execute a streaming purge for ${{count}} staged cell(s)?`)) {{
                return;
            }}

            closeStagedModal();
            alert("Executing purge in the background. Check server console for streaming progress.");
            const res = await fetch('/api/purge', {{ method: 'POST' }});
            const data = await res.json();
            alert(`✅ Purge Complete!\nRemoved: ${{data.removed_rows.toLocaleString()}} rows\nDuration: ${{data.duration_seconds}}s\nSaved to: ${{data.output_parquet}}`);
        }}

        window.onload = initMap;
    </script>
</body>
</html>
"""


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
