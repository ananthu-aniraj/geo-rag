"""Continent-Scoped H3 Spatial-Semantic Map with Representative Photos.

Generates an interactive Folium map focused on a single continent at H3 resolution 4
(configurable), displaying land-cover / cluster distributions per hexagon alongside
representative photos, Mapillary thumbnails, and SSH tunneling / prefix-remapping
support for offline datasets.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional

import branca.colormap as cm
import folium
import h3
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.credentials import get_mapillary_token
from src.utils.io import (
    load_dataframe,
    load_dataset_with_clusters,
    resolve_offline_image_path,
)

try:
    from src.visualization.map_utils import (
        create_carto_tile_layer,
        create_esri_satellite_layer,
        create_folium_map,
    )
except ImportError:
    from map_utils import (
        create_carto_tile_layer,
        create_esri_satellite_layer,
        create_folium_map,
    )

MAPILLARY_TOKEN = get_mapillary_token()

SUPPORTED_CONTINENTS = [
    "Africa",
    "Europe",
    "Asia",
    "North America",
    "South America",
    "Oceania",
    "Antarctica",
]


def normalize_continent_name(name: str) -> str:
    """Normalize input continent name to match dataset title casing."""
    name_clean = name.strip().lower()
    for cont in SUPPORTED_CONTINENTS:
        if cont.lower() == name_clean:
            return cont
    return name.strip().title()


def get_clean_boundary(cell: str) -> List[List[float]]:
    """Get hexagon boundary coordinates, properly adjusting for antimeridian crossing."""
    coords = h3.cell_to_boundary(cell)
    lngs = [c[1] for c in coords]
    if max(lngs) - min(lngs) > 180:
        coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
    return coords


def generate_continent_map(
    continent: str,
    index_path: str,
    dataset_path: Optional[str] = None,
    cleaned_dataset_path: Optional[str] = None,
    output_html: str = "continent_h3_semantic_map.html",
    h3_res: int = 4,
    min_count: int = 10,
    samples_per_cell: int = 4,
    image_root_dir: Optional[List[str]] = None,
    find_prefix: Optional[str] = None,
    replace_prefix: Optional[str] = None,
    tiles: str = "CartoDB Positron",
) -> None:
    continent_canon = normalize_continent_name(continent)
    print("\n========================================================")
    print(f" Generating H3 Spatial-Semantic Map for: {continent_canon}")
    print(f" Target H3 Resolution: {h3_res} | Min Images/Cell: {min_count}")
    print("========================================================")

    if not os.path.exists(index_path):
        print(f"❌ Error: H3 semantic index file not found at '{index_path}'.")
        return

    # 1. Load Pre-Aggregated Index filtered by Continent and Resolution
    print(f"Loading semantic index from: {index_path}...")
    t0 = time.time()
    try:
        table_idx = pq.read_table(
            index_path,
            filters=[("resolution", "==", h3_res)],
        )
        df_idx = table_idx.to_pandas()
    except Exception as e:
        print(f"Warning: Pushdown filter failed ({e}), loading full index...")
        df_idx = load_dataframe(index_path)
        df_idx = df_idx[df_idx["resolution"] == h3_res]

    if "continent" in df_idx.columns:
        df_idx = df_idx[
            df_idx["continent"].astype(str).str.lower() == continent_canon.lower()
        ]

    if df_idx.empty:
        print(
            f"❌ Error: No records found for continent '{continent_canon}' at resolution {h3_res}."
        )
        return

    print(
        f" -> Found {len(df_idx):,} index records for {continent_canon} in {time.time() - t0:.2f}s."
    )

    # Calculate global cell totals to set colormap
    cell_totals = df_idx.groupby("query_cell")["image_count"].sum().reset_index()
    cell_totals = cell_totals[cell_totals["image_count"] >= min_count]

    if cell_totals.empty:
        print(
            f"❌ Error: No cells found with at least {min_count} images for {continent_canon}."
        )
        return

    active_cells = set(cell_totals["query_cell"].values)
    print(
        f" -> Active H3 Res {h3_res} cells (>= {min_count} images): {len(active_cells):,}"
    )

    # Set up logarithmic colormap
    counts = cell_totals["image_count"].values
    min_log = float(np.log10(max(1, min(counts))))
    max_log = float(np.log10(max(counts)))
    if min_log == max_log:
        max_log += 0.1

    colormap = cm.linear.YlOrRd_09.scale(min_log, max_log)
    colormap.caption = f"{continent_canon} Image Density (Log10 scale, H3 Res {h3_res})"

    # 2. Extract Sample Photos from the Dataset (if available)
    samples_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    auto_detected_base = find_prefix or ""

    if dataset_path and os.path.exists(dataset_path):
        print(f"\nSampling representative photos from dataset: {dataset_path}...")
        t_data = time.time()
        pf_data = pq.ParquetFile(dataset_path)
        data_cols = pf_data.schema_arrow.names

        wanted_base = [
            "Photo_ID",
            "Platform",
            "Image_URL",
            "H3_Cell",
            "Latitude",
            "Longitude",
            "Captured_At",
            "continent",
        ]

        df_photos = pd.DataFrame()

        # Check if dataset_path is a decoupled sidecar (lacks H3_Cell / coordinates)
        if "H3_Cell" not in data_cols:
            base_cand = cleaned_dataset_path
            if not base_cand:
                ds_dir = os.path.dirname(os.path.abspath(dataset_path))
                candidates = glob.glob(
                    os.path.join(ds_dir, "*cleaned*.parquet")
                ) or glob.glob(os.path.join(ds_dir, "*deduplicated*.parquet"))
                if candidates:
                    base_cand = candidates[0]

            if base_cand and os.path.exists(base_cand):
                print(f" -> Found companion base dataset for coordinates: {base_cand}")
                try:
                    # Fast zero-copy append: both files share exact row order
                    tbl_clean = pq.read_table(base_cand, columns=wanted_base)
                    side_cols = [
                        c for c in ["cluster_label", "cluster_id"] if c in data_cols
                    ]
                    tbl_side = pq.read_table(dataset_path, columns=side_cols)
                    if len(tbl_clean) == len(tbl_side):
                        tbl_joined = tbl_clean
                        for col_name in tbl_side.column_names:
                            tbl_joined = tbl_joined.append_column(
                                col_name, tbl_side[col_name]
                            )
                        mask = pc.equal(
                            pc.utf8_lower(tbl_joined["continent"]),
                            continent_canon.lower(),
                        )
                        df_photos = tbl_joined.filter(mask).to_pandas()
                    else:
                        raise ValueError(
                            f"Row count mismatch: clean={len(tbl_clean)} vs side={len(tbl_side)}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Fast append failed ({e}), loading via load_dataset_with_clusters..."
                    )
                    df_photos = load_dataset_with_clusters(
                        dataset_path, columns=wanted_base + ["cluster_label"]
                    )
                    if "continent" in df_photos.columns:
                        df_photos = df_photos[
                            df_photos["continent"].astype(str).str.lower()
                            == continent_canon.lower()
                        ]
            else:
                print(
                    " -> Sidecar detected without companion base file; loading via load_dataset_with_clusters..."
                )
                df_photos = load_dataset_with_clusters(
                    dataset_path, columns=wanted_base + ["cluster_label"]
                )
                if "continent" in df_photos.columns:
                    df_photos = df_photos[
                        df_photos["continent"].astype(str).str.lower()
                        == continent_canon.lower()
                    ]
        else:
            # dataset_path has H3_Cell / coordinates directly
            read_cols = [c for c in wanted_base + ["cluster_label"] if c in data_cols]
            try:
                tbl_data = pq.read_table(
                    dataset_path,
                    columns=read_cols,
                    filters=[("continent", "==", continent_canon)],
                )
                df_photos = tbl_data.to_pandas()
            except Exception:
                tbl_data = pq.read_table(dataset_path, columns=read_cols)
                df_photos = tbl_data.to_pandas()
                if "continent" in df_photos.columns:
                    df_photos = df_photos[
                        df_photos["continent"].astype(str).str.lower()
                        == continent_canon.lower()
                    ]

        print(
            f" -> Loaded {len(df_photos):,} continent photo records in {time.time() - t_data:.2f}s."
        )

        if not df_photos.empty and "H3_Cell" in df_photos.columns:
            # Map H3 cells to target resolution parent
            unique_h3 = df_photos["H3_Cell"].dropna().unique()
            cell_map = {c: h3.cell_to_parent(c, h3_res) for c in unique_h3}
            df_photos["query_cell"] = df_photos["H3_Cell"].map(cell_map)

            # Filter to active cells only
            df_photos = df_photos[df_photos["query_cell"].isin(active_cells)]

            # Group photos by cell and sample up to samples_per_cell
            grouped = df_photos.groupby("query_cell")
            has_clusters = (
                "cluster_label" in df_photos.columns
                and df_photos["cluster_label"].notna().any()
            )
            if has_clusters:
                print(
                    " -> Performing stratified sampling across distinct clusters per cell..."
                )
            else:
                print(
                    " -> Performing uniform photo sampling per cell (no cluster labels present)..."
                )

            for cell, group in tqdm(grouped, desc="Sampling photos per cell"):
                samples = []
                selected_rows = []

                if has_clusters:
                    # Stratified sampling: pick 1 photo from each dominant cluster in this cell
                    cluster_groups = group.groupby("cluster_label", observed=True)
                    sorted_clusters = sorted(
                        cluster_groups, key=lambda x: len(x[1]), reverse=True
                    )
                    for _, c_sub in sorted_clusters:
                        if len(selected_rows) >= samples_per_cell:
                            break
                        selected_rows.append(c_sub.iloc[0])

                    # If fewer distinct clusters than samples_per_cell, fill remainder from largest clusters
                    if len(selected_rows) < samples_per_cell:
                        chosen_ids = {str(r.get("Photo_ID", "")) for r in selected_rows}
                        rem = (
                            group[~group["Photo_ID"].astype(str).isin(chosen_ids)]
                            if "Photo_ID" in group.columns
                            else group
                        )
                        for _, rem_row in rem.head(
                            samples_per_cell - len(selected_rows)
                        ).iterrows():
                            selected_rows.append(rem_row)
                else:
                    for _, row in group.head(samples_per_cell).iterrows():
                        selected_rows.append(row)

                for row in selected_rows:
                    pid = str(row.get("Photo_ID", "")).strip()
                    if pid.endswith(".0"):
                        pid = pid[:-2]

                    plat = str(row.get("Platform", "Unknown")).strip()
                    img_url = str(row.get("Image_URL", "")).strip()
                    cap = str(row.get("Captured_At", "N/A")).strip()
                    cl_lbl = str(row.get("cluster_label", "")).strip()
                    if not cl_lbl or cl_lbl.lower() in ("nan", "none"):
                        cl_lbl = plat if plat != "Unknown" else "Photo"
                    lat_val = (
                        float(row["Latitude"]) if pd.notna(row.get("Latitude")) else 0.0
                    )
                    lon_val = (
                        float(row["Longitude"])
                        if pd.notna(row.get("Longitude"))
                        else 0.0
                    )

                    # Resolve local offline path if image_root_dir is supplied
                    resolved_url = img_url
                    if image_root_dir:
                        resolved_path = resolve_offline_image_path(
                            img_url,
                            image_root_dir,
                            photo_id=pid,
                            platform=plat,
                        )
                        if resolved_path:
                            resolved_url = "file://" + os.path.abspath(resolved_path)

                    if not auto_detected_base and resolved_url.startswith("file://"):
                        last_slash = resolved_url.rfind("/")
                        if last_slash != -1:
                            auto_detected_base = resolved_url[: last_slash + 1]

                    mapillary_web_url = (
                        f"https://www.mapillary.com/app/?pKey={pid}"
                        if plat.lower() == "mapillary" and pid
                        else resolved_url
                    )

                    samples.append(
                        {
                            "photo_id": pid,
                            "platform": plat,
                            "url": resolved_url,
                            "mapillary_web_url": mapillary_web_url,
                            "cluster_label": cl_lbl,
                            "captured_at": cap,
                            "lat": lat_val,
                            "lon": lon_val,
                        }
                    )
                samples_by_cell[cell] = samples

    # 3. Calculate Map Viewport Centroid and Bounding Box
    print("\nComputing geographical extent and centering...")
    all_lats, all_lons = [], []
    for cell in active_cells:
        lat, lon = h3.cell_to_latlng(cell)
        all_lats.append(lat)
        all_lons.append(lon)

    center_lat = float(np.mean(all_lats)) if all_lats else 0.0
    center_lon = float(np.mean(all_lons)) if all_lons else 0.0
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)

    m = create_folium_map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles=tiles,
    )

    # Add alternative basemaps for contrast and inspection flexibility
    if "dark" in tiles.lower():
        create_carto_tile_layer("light_all", name="CartoDB Positron (Light)").add_to(m)
    else:
        create_carto_tile_layer("dark_all", name="CartoDB Dark_Matter (Dark)").add_to(m)
    create_carto_tile_layer(
        "rastertiles/voyager", name="CartoDB Voyager (Color)"
    ).add_to(m)
    create_esri_satellite_layer().add_to(m)

    # 4. Build Season/All Observations Layers with Popups & Tooltips
    unique_seasons = (
        sorted(df_idx["Season"].unique()) if "Season" in df_idx.columns else ["Unknown"]
    )

    # Style function mapping density to color
    def style_fn(feature):
        log_val = feature["properties"]["log_count"]
        fill_color = colormap(log_val)
        return {
            "fillColor": fill_color,
            "color": fill_color,
            "weight": 1.5,
            "fillOpacity": 0.65,
        }

    # Custom JS tooltip & popup builder
    js_popup_builder = folium.utilities.JsCode(
        f"""
        function onEachFeature(feature, layer) {{
            var props = feature.properties;
            var countFormatted = Number(props.count).toLocaleString();

            // 1. Hover Tooltip
            var tooltipHtml = '<div style="font-family: Arial, sans-serif; min-width: 260px; max-width: 320px; font-size: 11.5px; padding: 4px;">' +
                              '<b style="font-size: 13px; color: #1e293b; display: block; margin-bottom: 3px;">📍 Cell: ' + props.cell + ' (Res {h3_res})</b>' +
                              '<b>📊 Total Images:</b> ' + countFormatted + '<br/>' +
                              '<hr style="margin: 4px 0; border: 0; border-top: 1px solid #cbd5e1;"/>' +
                              '<b style="color: #0f766e; text-transform: uppercase; font-size: 9.5px; display: block; margin-bottom: 3px;">Dominant Categories:</b>' +
                              '<ul style="margin: 0; padding-left: 14px; list-style-type: square; color: #334155;">';

            var clusters = [];
            try {{ clusters = JSON.parse(props.clusters); }} catch(e) {{}}

            for (var i = 0; i < Math.min(clusters.length, 4); i++) {{
                var pct = Number(clusters[i][1]).toFixed(1);
                var cCount = Number(clusters[i][2]).toLocaleString();
                var label = clusters[i][0];
                tooltipHtml += '<li style="margin-bottom: 3px;"><b>' + label + '</b>: ' + pct + '% (' + cCount + ')</li>';
            }}
            tooltipHtml += '</ul></div>';

            layer.bindTooltip(tooltipHtml, {{
                sticky: true,
                direction: "auto",
                opacity: 0.95
            }});

            // 2. On-Click Modal Popup with Photo Gallery
            var popupHtml = '<div class="h3-popup-card" style="font-family: Arial, sans-serif; width: 340px; max-width: 85vw; padding: 2px;">' +
                            '<div style="font-size: 14px; font-weight: bold; color: #0f172a; margin-bottom: 2px;">📍 H3 Cell: ' + props.cell + ' <span style="font-size: 11px; color: #64748b; font-weight: normal;">(Res {h3_res})</span></div>' +
                            '<div style="font-size: 12px; color: #475569; margin-bottom: 6px;"><b>Total Images:</b> ' + countFormatted + ' &nbsp;|&nbsp; <b>Region:</b> ' + props.continent + '</div>' +
                            '<div style="margin-bottom: 6px;">';

            for (var j = 0; j < Math.min(clusters.length, 3); j++) {{
                var clabel = clusters[j][0];
                var cpct = Number(clusters[j][1]).toFixed(0);
                popupHtml += '<span style="display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 1px 6px; font-size: 10px; color: #334155; margin-right: 4px; margin-bottom: 3px;">' +
                             '<b>' + clabel + '</b> (' + cpct + '%)</span>';
            }}
            popupHtml += '</div><hr style="margin: 6px 0; border: 0; border-top: 1px solid #e2e8f0;"/>';

            var samples = [];
            try {{ samples = JSON.parse(props.samples); }} catch(e) {{}}

            if (samples.length > 0) {{
                popupHtml += '<div style="font-size: 11px; font-weight: bold; color: #334155; margin-bottom: 6px;">📸 Representative Photos:</div>' +
                             '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px;">';

                for (var s = 0; s < samples.length; s++) {{
                    var smp = samples[s];
                    var imgUrl = smp.url;
                    var webUrl = smp.mapillary_web_url || imgUrl;
                    var tag = smp.cluster_label || smp.platform || 'Photo';

                    popupHtml += '<div style="position: relative; border-radius: 6px; overflow: hidden; background: #1e293b; aspect-ratio: 4/3; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">' +
                                 '<a href="' + webUrl + '" target="_blank" data-orig-href="' + webUrl + '" style="display: block; width: 100%; height: 100%;">' +
                                 '<img src="' + imgUrl + '" data-orig-src="' + imgUrl + '" onerror="handleImageError(this, \\'' + smp.photo_id + '\\', \\'' + smp.platform + '\\')" style="width: 100%; height: 100%; object-fit: cover;" alt="' + tag + '"/>' +
                                 '</a>' +
                                 '<div style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(15,23,42,0.8); color: #f8fafc; font-size: 9px; padding: 2px 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">' + tag + '</div>' +
                                 '</div>';
                }}
                popupHtml += '</div>';
            }} else {{
                popupHtml += '<div style="font-size: 11px; color: #94a3b8; font-style: italic;">No preview images loaded for this cell.</div>';
            }}

            popupHtml += '</div>';

            layer.bindPopup(popupHtml, {{
                maxWidth: 380,
                className: "h3-continent-popup"
            }});
        }}
        """
    )

    for season in unique_seasons:
        df_season = (
            df_idx[df_idx["Season"] == season] if "Season" in df_idx.columns else df_idx
        )
        s_totals = df_season.groupby("query_cell")["image_count"].sum().reset_index()
        s_totals = s_totals[s_totals["image_count"] >= min_count]

        if s_totals.empty:
            continue

        print(f"Building map layer for {season} ({len(s_totals):,} cells)...")

        # Aggregate top 5 clusters per cell
        agg_cols = [
            c
            for c in ["query_cell", "cluster_label", "cluster_description"]
            if c in df_season.columns
        ]
        df_cell_agg = (
            df_season.groupby(agg_cols, observed=True)["image_count"]
            .sum()
            .reset_index()
        )
        df_cell_agg = df_cell_agg.sort_values(
            by=["query_cell", "image_count"], ascending=[True, False]
        )
        df_top5 = df_cell_agg.groupby("query_cell").head(5)

        cell_clusters_dict = {}
        for cell, group in df_top5.groupby("query_cell"):
            cell_clusters_dict[cell] = group.to_dict("records")

        features = []
        for _, item in tqdm(
            s_totals.iterrows(), total=len(s_totals), desc=f"Hexagons ({season})"
        ):
            cell = item["query_cell"]
            tot = item["image_count"]

            try:
                boundary = get_clean_boundary(cell)
                geojson_coords = [[lng, lat] for lat, lng in boundary]
                geojson_coords.append(geojson_coords[0])

                top_clusters = cell_clusters_dict.get(cell, [])
                cluster_list = []
                for row in top_clusters:
                    pct = float((row["image_count"] / tot) * 100)
                    desc = str(row.get("cluster_description", "") or "")
                    cluster_list.append(
                        [row["cluster_label"], pct, int(row["image_count"]), desc[:160]]
                    )

                cell_samples = samples_by_cell.get(cell, [])

                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [geojson_coords],
                        },
                        "properties": {
                            "cell": cell,
                            "count": int(tot),
                            "log_count": float(np.log10(tot)),
                            "clusters": json.dumps(cluster_list),
                            "samples": json.dumps(cell_samples),
                            "continent": continent_canon,
                        },
                    }
                )
            except Exception:
                continue

        geojson_data = {"type": "FeatureCollection", "features": features}
        is_default = (
            (season == "Summer")
            or (len(unique_seasons) == 1)
            or (season == unique_seasons[0] and "Summer" not in unique_seasons)
        )
        layer_name = (
            f"🍂 Season: {season}" if season != "Unknown" else "📍 All Observations"
        )
        fg = folium.FeatureGroup(name=layer_name, show=is_default)
        folium.GeoJson(
            geojson_data,
            style_function=style_fn,
            on_each_feature=js_popup_builder,
        ).add_to(fg)
        fg.add_to(m)

    m.add_child(colormap)
    folium.LayerControl(collapsed=False).add_to(m)

    # 5. Inject Offline SSH Tunneling / Image Prefix Remapping Handler
    js_template_path = os.path.join(
        os.path.dirname(__file__), "templates", "image_error_handler.js"
    )
    if os.path.exists(js_template_path):
        with open(js_template_path, "r", encoding="utf-8") as f:
            js_code = (
                f.read()
                .replace("__MAPILLARY_TOKEN__", MAPILLARY_TOKEN or "")
                .replace("{{MAPILLARY_TOKEN}}", MAPILLARY_TOKEN or "")
                .replace("__MAP_NAME__", m.get_name())
                .replace("{{MAP_NAME}}", m.get_name())
                .replace("__AUTO_DETECTED_BASE__", auto_detected_base)
                .replace("{{AUTO_DETECTED_BASE}}", auto_detected_base)
                .replace("__DEFAULT_REPLACE_PREFIX__", replace_prefix or "")
                .replace("{{DEFAULT_REPLACE_PREFIX}}", replace_prefix or "")
            )
        js_header = f"<script>\n{js_code}\n</script>"
        m.get_root().html.add_child(folium.Element(js_header))

    # Fit bounds to continent
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # 6. Save Output
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    m.save(output_html)
    print(f"\n✅ Interactive continent spatial-semantic map saved to: {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive continent-scoped H3 spatial-semantic map with representative photo galleries and SSH tunneling support."
    )
    parser.add_argument(
        "--continent",
        type=str,
        required=True,
        help="Target continent to visualize (e.g. Africa, Europe, Asia, North America, South America, Oceania, Antarctica).",
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to the pre-aggregated H3 semantic index Parquet file (e.g. geo_space_...h3_semantic_index.parquet).",
    )
    parser.add_argument(
        "--dataset",
        "--input",
        "-i",
        dest="dataset",
        type=str,
        required=True,
        help="Path to Parquet dataset (clustered sidecar, merged dataset, or cleaned dataset).",
    )
    parser.add_argument(
        "--cleaned_dataset",
        type=str,
        default=None,
        help="Optional path to base cleaned parquet if --dataset is a clustered sidecar stored separately.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the output HTML map (default: <continent>_h3_semantic_map_res<res>.html).",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=4,
        choices=list(range(3, 9)),
        help="H3 resolution for hexagon aggregation (default: 4, ~177km edge length).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Minimum number of images in a cell to display on the map (default: 10).",
    )
    parser.add_argument(
        "--samples_per_cell",
        type=int,
        default=4,
        help="Number of representative preview photos to show per polygon popup (default: 4).",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        nargs="+",
        default=None,
        help="Optional root directories containing local images (for offline datasets).",
    )
    parser.add_argument(
        "--find_prefix",
        type=str,
        default=None,
        help="Optional local path prefix to replace for SSH tunneling (auto-detected if omitted).",
    )
    parser.add_argument(
        "--replace_prefix",
        type=str,
        default=None,
        help="Optional replacement URL prefix for SSH tunneling (e.g. http://localhost:8000/).",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="CartoDB Positron",
        help="Basemap style (default: 'CartoDB Positron').",
    )
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"❌ Error: H3 semantic index file not found at '{args.index}'.")
        return

    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file not found at '{args.dataset}'.")
        return

    if args.cleaned_dataset and not os.path.exists(args.cleaned_dataset):
        print(
            f"❌ Error: Cleaned base dataset file not found at '{args.cleaned_dataset}'."
        )
        return

    output_html = args.output
    if not output_html:
        cont_slug = args.continent.strip().lower().replace(" ", "_")
        output_html = f"{cont_slug}_h3_semantic_map_res{args.res}.html"

    generate_continent_map(
        continent=args.continent,
        index_path=args.index,
        dataset_path=args.dataset,
        cleaned_dataset_path=args.cleaned_dataset,
        output_html=output_html,
        h3_res=args.res,
        min_count=args.min_count,
        samples_per_cell=args.samples_per_cell,
        image_root_dir=args.image_root_dir,
        find_prefix=args.find_prefix,
        replace_prefix=args.replace_prefix,
        tiles=args.tiles,
    )


if __name__ == "__main__":
    main()
