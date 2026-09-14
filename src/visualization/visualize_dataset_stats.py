"""
Dataset Visualization Suite for Geo-RAG.

Provides static multi-panel dashboards, dedicated camera-trap grouped platform
distribution figures, and multi-layered interactive Folium/H3 web maps.
"""

import argparse
import os
import time
from typing import Any

import branca.colormap as cm
import folium
import h3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CAMERA_TRAP_PLATFORMS = {
    "iwildcam",
    "wildobs",
    "wildlife_insights",
    "wildlifeinsights",
    "snapshotusa",
    "snapshot_usa",
}


def is_camera_trap_platform(platform_name: Any) -> bool:
    """Checks whether a platform name corresponds to a camera trap platform."""
    if not platform_name or pd.isna(platform_name):
        return False
    norm = str(platform_name).strip().lower().replace("-", "_")
    compact = norm.replace("_", "").replace(" ", "")
    return norm in CAMERA_TRAP_PLATFORMS or compact in {
        "iwildcam",
        "wildobs",
        "wildlifeinsights",
        "snapshotusa",
    }


def get_grouped_platform_series(platform_series: pd.Series) -> pd.Series:
    """Groups camera trap platforms (iwildcam, wildobs, wildlife_insights, snapshotusa)
    into a single 'Camera Traps' category while preserving other platforms.
    """
    return platform_series.apply(
        lambda p: "Camera Traps"
        if is_camera_trap_platform(p)
        else ("Unknown" if pd.isna(p) or not str(p).strip() else str(p).strip())
    )


def generate_plots(
    df_filtered: pd.DataFrame,
    is_global: bool,
    location_name: str,
    output_path: str,
    group_camera_traps: bool = False,
) -> None:
    """Generate a multi-panel plot for visualization."""
    if len(df_filtered) == 0:
        print("Warning: No records found. Skipping plot generation.")
        return

    print("Generating statistics plots...")
    _ = time.time()

    # Set modern style
    sns.set_theme(style="whitegrid")

    has_koppen = (
        "Koppen_Code" in df_filtered.columns
        and df_filtered["Koppen_Code"].notna().any()
    )
    has_parent = (
        "parent_cluster_label" in df_filtered.columns
        and df_filtered["parent_cluster_label"].notna().any()
    )
    use_3x2 = has_koppen or has_parent

    if use_3x2:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig.suptitle(
        f"Dataset Statistics: {location_name}", fontsize=18, fontweight="bold", y=0.98
    )

    # --- Subplot 1: Spatial Breakdown ---
    ax1 = axes[0, 0]
    plotted_counts = None
    if is_global:
        # Global: Continent breakdown
        plotted_counts = df_filtered["continent"].value_counts()
        sns.barplot(
            x=plotted_counts.values,
            y=plotted_counts.index,
            ax=ax1,
            palette="viridis",
            hue=plotted_counts.index,
            legend=False,
        )
        ax1.set_title("Distribution by Continent", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Image Count")
    elif location_name.lower() in [
        "africa",
        "europe",
        "asia",
        "north america",
        "south america",
        "oceania",
        "antarctica",
        "australia",
    ]:
        # Continent: Country breakdown
        counts = df_filtered["country"].value_counts()
        plotted_counts = counts.head(10)
        sns.barplot(
            x=plotted_counts.values,
            y=plotted_counts.index,
            ax=ax1,
            palette="viridis",
            hue=plotted_counts.index,
            legend=False,
        )
        ax1.set_title(
            f"Top 10 Countries in {location_name}", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("Image Count")
    else:
        # Country/City: H3 cell breakdown
        if "H3_Cell" in df_filtered.columns:
            counts = df_filtered["H3_Cell"].value_counts()
            plotted_counts = counts.head(10)
            sns.barplot(
                x=plotted_counts.values,
                y=plotted_counts.index,
                ax=ax1,
                palette="viridis",
                hue=plotted_counts.index,
                legend=False,
            )
            ax1.set_title(
                "Top 10 H3 Cells (Resolution 11)", fontsize=14, fontweight="bold"
            )
            ax1.set_xlabel("Image Count")
        else:
            ax1.text(0.5, 0.5, "H3_Cell column not available", ha="center", va="center")
            ax1.set_title(
                "Spatial Breakdown (Unavailable)", fontsize=14, fontweight="bold"
            )

    # Add text labels for horizontal bars in Subplot 1
    if plotted_counts is not None and len(plotted_counts) > 0:
        max_val = plotted_counts.values.max() if len(plotted_counts) > 0 else 1
        for i, v in enumerate(plotted_counts.values):
            # Print label slightly offset to the right of the bar
            ax1.text(
                v + (max_val * 0.01),
                i,
                f" {v:,}",
                va="center",
                fontweight="bold",
                fontsize=9,
            )
        # Expand x limit slightly to prevent label clipping
        ax1.set_xlim(0, max_val * 1.15)

    # --- Subplot 2: Platform Breakdown ---
    ax2 = axes[0, 1]
    if "Platform" in df_filtered.columns:
        platform_data = (
            get_grouped_platform_series(df_filtered["Platform"])
            if group_camera_traps
            else df_filtered["Platform"]
        )
        counts = platform_data.value_counts()
        sns.barplot(
            x=counts.index,
            y=counts.values,
            ax=ax2,
            palette="muted",
            hue=counts.index,
            legend=False,
        )
        title_suffix = " (Camera Traps Grouped)" if group_camera_traps else ""
        ax2.set_title(
            f"Distribution by Platform{title_suffix}", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Image Count")
        ax2.tick_params(axis="x", rotation=25)
        for label in ax2.get_xticklabels():
            label.set_ha("right")
        max_val = counts.values.max() if len(counts) > 0 else 1
        for i, v in enumerate(counts.values):
            ax2.text(
                i,
                v + (max_val * 0.01),
                f"{v:,}",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        ax2.set_ylim(0, max_val * 1.18)
    else:
        ax2.text(0.5, 0.5, "Platform column not available", ha="center", va="center")
        ax2.set_title(
            "Platform Breakdown (Unavailable)", fontsize=14, fontweight="bold"
        )

    # --- Subplot 3: Time of Day Distribution ---
    ax3 = axes[1, 0]
    if "Time_Of_Day" in df_filtered.columns:
        tod_order = ["Dawn", "Morning", "Afternoon", "Dusk", "Night", "Unknown"]
        counts = df_filtered["Time_Of_Day"].value_counts()
        existing_order = [x for x in tod_order if x in counts.index]
        counts = counts.reindex(existing_order)

        sns.barplot(
            x=counts.index,
            y=counts.values,
            ax=ax3,
            palette="magma",
            hue=counts.index,
            legend=False,
        )
        ax3.set_title("Distribution by Time of Day", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Image Count")
        ax3.tick_params(axis="x", rotation=20)
        for label in ax3.get_xticklabels():
            label.set_ha("right")
        max_val = counts.values.max() if len(counts) > 0 else 1
        for i, v in enumerate(counts.values):
            ax3.text(
                i,
                v + (max_val * 0.01),
                f"{v:,}",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        ax3.set_ylim(0, max_val * 1.18)
    else:
        ax3.text(0.5, 0.5, "Time of Day not available", ha="center", va="center")
        ax3.set_title(
            "Time of Day Distribution (Unavailable)", fontsize=14, fontweight="bold"
        )

    # --- Subplot 4: Season Distribution ---
    ax4 = axes[1, 1]
    if "Season" in df_filtered.columns:
        season_order = [
            "Spring",
            "Summer",
            "Autumn",
            "Winter",
            "Wet Season",
            "Dry Season",
            "Unknown",
        ]
        counts = df_filtered["Season"].value_counts()
        existing_order = [x for x in season_order if x in counts.index]
        counts = counts.reindex(existing_order)

        sns.barplot(
            x=counts.index,
            y=counts.values,
            ax=ax4,
            palette="coolwarm",
            hue=counts.index,
            legend=False,
        )
        ax4.set_title("Distribution by Season", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Image Count")
        ax4.tick_params(axis="x", rotation=25)
        for label in ax4.get_xticklabels():
            label.set_ha("right")
        max_val = counts.values.max() if len(counts) > 0 else 1
        for i, v in enumerate(counts.values):
            ax4.text(
                i,
                v + (max_val * 0.01),
                f"{v:,}",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        ax4.set_ylim(0, max_val * 1.18)
    else:
        ax4.text(0.5, 0.5, "Season not available", ha="center", va="center")
        ax4.set_title(
            "Season Distribution (Unavailable)", fontsize=14, fontweight="bold"
        )

    if use_3x2:
        # --- Subplot 5: Koppen-Geiger Climate Distribution ---
        ax5 = axes[2, 0]
        if "Koppen_Code" in df_filtered.columns:
            valid_koppen = df_filtered[
                df_filtered["Koppen_Code"].notna() & (df_filtered["Koppen_Code"] != "")
            ]
            if len(valid_koppen) > 0:
                counts = valid_koppen["Koppen_Code"].value_counts()
                # If there are too many climate codes, plot top 12 to maintain readability
                plotted_counts = counts.head(12) if len(counts) > 12 else counts
                sns.barplot(
                    x=plotted_counts.values,
                    y=plotted_counts.index,
                    ax=ax5,
                    palette="tab10",
                    hue=plotted_counts.index,
                    legend=False,
                )
                ax5.set_title(
                    "Distribution by Köppen Climate Code",
                    fontsize=14,
                    fontweight="bold",
                )
                ax5.set_xlabel("Image Count")
                max_val = plotted_counts.values.max() if len(plotted_counts) > 0 else 1
                for i, v in enumerate(plotted_counts.values):
                    ax5.text(
                        v + (max_val * 0.01),
                        i,
                        f" {v:,}",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                    )
                ax5.set_xlim(0, max_val * 1.18)
            else:
                ax5.text(
                    0.5, 0.5, "No valid Koppen codes found", ha="center", va="center"
                )
                ax5.set_title(
                    "Koppen Climate Distribution", fontsize=14, fontweight="bold"
                )
        else:
            ax5.text(0.5, 0.5, "Koppen Code not available", ha="center", va="center")
            ax5.set_title(
                "Koppen Climate Distribution (Unavailable)",
                fontsize=14,
                fontweight="bold",
            )

        # --- Subplot 6: Top Semantic Parent Cluster Labels ---
        ax6 = axes[2, 1]
        if "parent_cluster_label" in df_filtered.columns:
            counts = df_filtered["parent_cluster_label"].value_counts()
            plotted_counts = counts.head(10)
            if len(plotted_counts) > 0:
                sns.barplot(
                    x=plotted_counts.values,
                    y=plotted_counts.index,
                    ax=ax6,
                    palette="rocket",
                    hue=plotted_counts.index,
                    legend=False,
                )
                ax6.set_title(
                    "Top 10 Semantic Parent Categories", fontsize=14, fontweight="bold"
                )
                ax6.set_xlabel("Image Count")
                max_val = plotted_counts.values.max() if len(plotted_counts) > 0 else 1
                for i, v in enumerate(plotted_counts.values):
                    ax6.text(
                        v + (max_val * 0.01),
                        i,
                        f" {v:,}",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                    )
                ax6.set_xlim(0, max_val * 1.15)
            else:
                ax6.text(
                    0.5, 0.5, "No parent cluster labels found", ha="center", va="center"
                )
                ax6.set_title("Parent Cluster Labels", fontsize=14, fontweight="bold")
        else:
            ax6.text(
                0.5, 0.5, "Parent Cluster Label not available", ha="center", va="center"
            )
            ax6.set_title(
                "Parent Cluster Labels (Unavailable)", fontsize=14, fontweight="bold"
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f" -> Saved statistics plot to: {os.path.abspath(output_path)}")
    plt.close()


def generate_grouped_platform_plot(
    df_filtered: pd.DataFrame, location_name: str, output_path: str
) -> None:
    """Generates a separate, dedicated visualization of platform distribution
    where camera trap platforms (iwildcam, wildobs, wildlife_insights, snapshotusa)
    are grouped into a single 'Camera Traps' category, alongside a detailed
    composition breakdown of the camera trap platforms.
    """
    if len(df_filtered) == 0:
        print("Warning: No records found. Skipping grouped platform plot.")
        return

    if "Platform" not in df_filtered.columns:
        print("Warning: 'Platform' column not found. Skipping grouped platform plot.")
        return

    print("Generating grouped platform plot...")
    sns.set_theme(style="whitegrid")

    grouped_series = get_grouped_platform_series(df_filtered["Platform"])
    grouped_counts = grouped_series.value_counts()
    total_count = len(df_filtered)

    camera_trap_mask = df_filtered["Platform"].apply(is_camera_trap_platform)
    ct_counts = df_filtered.loc[camera_trap_mask, "Platform"].value_counts()
    has_camera_traps = len(ct_counts) > 0

    if has_camera_traps:
        fig, (ax_macro, ax_sub) = plt.subplots(
            1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1.0]}
        )
        fig.suptitle(
            f"Platform Distribution & Camera Trap Breakdown: {location_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
    else:
        fig, ax_macro = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(
            f"Platform Distribution (Camera Traps Grouped): {location_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        ax_sub = None

    # --- Panel 1: Macro Platforms with Camera Traps Grouped ---
    sns.barplot(
        x=grouped_counts.index,
        y=grouped_counts.values,
        ax=ax_macro,
        palette="muted",
        hue=grouped_counts.index,
        legend=False,
    )
    ax_macro.set_title(
        "Platform Breakdown (Camera Traps Grouped)",
        fontsize=13,
        fontweight="bold",
    )
    ax_macro.set_ylabel("Image Count")
    ax_macro.tick_params(axis="x", rotation=25)
    for label in ax_macro.get_xticklabels():
        label.set_ha("right")

    max_macro = grouped_counts.values.max() if len(grouped_counts) > 0 else 1
    for i, v in enumerate(grouped_counts.values):
        pct = (v / total_count) * 100 if total_count > 0 else 0
        ax_macro.text(
            i,
            v + (max_macro * 0.015),
            f"{v:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax_macro.set_ylim(0, max_macro * 1.22)

    # --- Panel 2: Camera Trap Sub-Breakdown ---
    if ax_sub is not None:
        ct_total = ct_counts.sum()
        sns.barplot(
            x=ct_counts.index,
            y=ct_counts.values,
            ax=ax_sub,
            palette="crest",
            hue=ct_counts.index,
            legend=False,
        )
        pct_of_all = (ct_total / total_count) * 100 if total_count > 0 else 0
        ax_sub.set_title(
            f"Camera Trap Composition ({ct_total:,} images | {pct_of_all:.1f}% of total)",
            fontsize=13,
            fontweight="bold",
        )
        ax_sub.set_ylabel("Image Count")
        ax_sub.tick_params(axis="x", rotation=25)
        for label in ax_sub.get_xticklabels():
            label.set_ha("right")

        max_sub = ct_counts.values.max() if len(ct_counts) > 0 else 1
        for i, v in enumerate(ct_counts.values):
            pct_ct = (v / ct_total) * 100 if ct_total > 0 else 0
            ax_sub.text(
                i,
                v + (max_sub * 0.015),
                f"{v:,}\n({pct_ct:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
        ax_sub.set_ylim(0, max_sub * 1.22)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f" -> Saved grouped platform plot to: {os.path.abspath(output_path)}")
    plt.close()


def generate_interactive_map(
    df_filtered: pd.DataFrame, location_name: str, output_html_path: str
) -> None:
    """Generate a multi-layered interactive Folium map centered on the filtered data."""
    if len(df_filtered) == 0:
        print("Warning: No records found. Skipping map generation.")
        return

    print("Generating interactive H3 map...")
    _ = time.time()

    is_global = location_name.lower() == "global dataset"

    # 1. Determine bounding box
    min_lat, max_lat = df_filtered["Latitude"].min(), df_filtered["Latitude"].max()
    min_lon, max_lon = df_filtered["Longitude"].min(), df_filtered["Longitude"].max()

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    # 2. Determine target resolution dynamically
    if lat_span > 100 or lon_span > 100:
        target_res = 4
    elif lat_span > 30 or lon_span > 30:
        target_res = 5
    elif lat_span > 5 or lon_span > 5:
        target_res = 6
    elif lat_span > 1 or lon_span > 1:
        target_res = 7
    else:
        target_res = 8

    print(
        f" -> Selected H3 resolution {target_res} based on latitude span ({lat_span:.2f}°) and longitude span ({lon_span:.2f}°)"
    )

    # 3. Compute parent H3 cells at target resolution
    if "H3_Cell" in df_filtered.columns:
        unique_res11 = df_filtered["H3_Cell"].unique()
        res11_to_parent = {
            c: h3.cell_to_parent(c, target_res)
            if h3.get_resolution(c) != target_res
            else c
            for c in unique_res11
        }
        df_filtered["map_h3"] = df_filtered["H3_Cell"].map(res11_to_parent)
    else:
        df_filtered["map_h3"] = df_filtered.apply(
            lambda r: h3.latlng_to_cell(r["Latitude"], r["Longitude"], target_res),
            axis=1,
        )

    # Pre-aggregate breakdowns per H3 cell using fast groupby-unstack
    platform_counts = pd.DataFrame()
    tod_counts = pd.DataFrame()
    season_counts = pd.DataFrame()
    koppen_counts = pd.DataFrame()

    if "Platform" in df_filtered.columns:
        platform_counts = (
            df_filtered.groupby(["map_h3", "Platform"]).size().unstack(fill_value=0)
        )
    if "Time_Of_Day" in df_filtered.columns:
        tod_counts = (
            df_filtered.groupby(["map_h3", "Time_Of_Day"]).size().unstack(fill_value=0)
        )
    if "Season" in df_filtered.columns:
        season_counts = (
            df_filtered.groupby(["map_h3", "Season"]).size().unstack(fill_value=0)
        )
    if "Koppen_Code" in df_filtered.columns:
        koppen_counts = (
            df_filtered.groupby(["map_h3", "Koppen_Code"]).size().unstack(fill_value=0)
        )

    # 4. Initialize Folium Map
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB Positron"
    )
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    def get_clean_boundary(cell):
        coords = h3.cell_to_boundary(cell)
        lngs = [c[1] for c in coords]
        if max(lngs) - min(lngs) > 180:
            coords = [(lat, lng + 360 if lng < 0 else lng) for lat, lng in coords]
        return coords

    def add_layer(category_col, value, layer_name, theme):
        if category_col:
            df_sub = df_filtered[
                df_filtered[category_col].astype(str).str.lower() == str(value).lower()
            ]
        else:
            df_sub = df_filtered

        if len(df_sub) == 0:
            return

        counts = df_sub["map_h3"].value_counts()

        # Optimization: Apply a dynamic min_count filter to avoid map bloat from sparse/noisy cells
        cell_budget = 4000
        if len(counts) > cell_budget:
            cutoff_quantile = 1.0 - (cell_budget / len(counts))
            min_count_thresh = max(2, int(counts.quantile(cutoff_quantile)))
            counts = counts[counts >= min_count_thresh]

        if len(counts) == 0:
            return

        min_c = counts.min()
        max_c = counts.max()

        if theme == "viridis":
            colors = ["#440154", "#31688e", "#35b779", "#fde725"]
        elif theme == "magma":
            colors = ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"]
        elif theme == "blue":
            colors = ["#eff3ff", "#bdd7e7", "#6baed6", "#2171b5"]
        elif theme == "green":
            colors = ["#edf8e9", "#bae4b3", "#74c476", "#238b45"]
        elif theme == "orange":
            colors = ["#feedde", "#fdbe85", "#fd8d3c", "#d94701"]
        elif theme == "purple":
            colors = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3"]
        else:
            colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"]

        if min_c == max_c:
            colormap = cm.LinearColormap(
                [colors[0], colors[-1]], vmin=min_c, vmax=max_c + 1
            )
        else:
            colormap = cm.LinearColormap(colors, vmin=min_c, vmax=max_c)

        colormap.caption = f"Image Count: {layer_name}"

        layer = folium.FeatureGroup(name=layer_name, show=False)

        features = []
        for cell, count in counts.items():
            boundary = get_clean_boundary(cell)

            properties = {
                "cell": cell,
                "count": int(count),
            }

            if not category_col:
                if not platform_counts.empty and cell in platform_counts.index:
                    p_row = platform_counts.loc[cell]
                    properties["platforms"] = " | ".join(
                        [
                            f"{plat}: {c}"
                            for plat, c in p_row.items()
                            if c > 0 and plat != "Unknown"
                        ]
                    )
                    if not properties["platforms"]:
                        properties["platforms"] = "Unknown"
                else:
                    properties["platforms"] = "N/A"

                if not tod_counts.empty and cell in tod_counts.index:
                    t_row = tod_counts.loc[cell]
                    properties["time_of_day"] = " | ".join(
                        [
                            f"{tod}: {c}"
                            for tod, c in t_row.items()
                            if c > 0 and tod != "Unknown"
                        ]
                    )
                    if not properties["time_of_day"]:
                        properties["time_of_day"] = "Unknown"
                else:
                    properties["time_of_day"] = "N/A"

                if not season_counts.empty and cell in season_counts.index:
                    s_row = season_counts.loc[cell]
                    properties["seasons"] = " | ".join(
                        [
                            f"{season}: {c}"
                            for season, c in s_row.items()
                            if c > 0 and season != "Unknown"
                        ]
                    )
                    if not properties["seasons"]:
                        properties["seasons"] = "Unknown"
                else:
                    properties["seasons"] = "N/A"

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[lng, lat] for lat, lng in boundary]],
                    },
                    "properties": properties,
                }
            )

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
        }

        def style_function(feature):
            count = feature["properties"]["count"]
            color = colormap(count)
            return {
                "fillColor": color,
                "color": "#333333",
                "weight": 0.5,
                "fillOpacity": 0.7,
            }

        def highlight_function(feature):
            return {
                "weight": 2.5,
                "color": "#ffffff",
                "fillOpacity": 0.9,
            }

        if not category_col:
            tooltip_fields = ["cell", "count", "platforms", "time_of_day", "seasons"]
            tooltip_aliases = [
                "H3 Cell:",
                "Total Count:",
                "Platforms:",
                "Time of Day:",
                "Seasons:",
            ]
        else:
            tooltip_fields = ["cell", "count"]
            tooltip_aliases = ["H3 Cell:", "Count:"]

        g = folium.GeoJson(
            geojson_data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=True,
            ),
        )
        g.add_to(layer)
        layer.add_to(m)

    # Base Overview Layer (All Data)
    add_layer(None, None, "Total Density Overview", "viridis")

    # The overview layer is shown by default
    for child in m._children.values():
        if (
            isinstance(child, folium.FeatureGroup)
            and child.layer_name == "Total Density Overview"
        ):
            child.show = True
            break

    # Add dynamic layers only if filtered dataset is reasonably sized
    if not is_global or len(df_filtered) <= 300000:
        # Platform Layers
        if "Platform" in df_filtered.columns:
            platforms = df_filtered["Platform"].dropna().unique()
            themes = ["blue", "green", "orange", "purple", "red"]
            for idx, plat in enumerate(platforms):
                theme = themes[idx % len(themes)]
                add_layer("Platform", plat, f"Platform: {plat}", theme)

        # Time of Day Layers
        if "Time_Of_Day" in df_filtered.columns:
            tods = df_filtered["Time_Of_Day"].dropna().unique()
            themes_tod = {
                "dawn": "purple",
                "morning": "green",
                "afternoon": "red",
                "dusk": "orange",
                "night": "magma",
            }
            for tod in tods:
                if tod == "Unknown":
                    continue
                theme = themes_tod.get(tod.lower(), "red")
                add_layer("Time_Of_Day", tod, f"Time: {tod}", theme)

        # Season Layers
        if "Season" in df_filtered.columns:
            seasons = df_filtered["Season"].dropna().unique()
            themes_season = {
                "spring": "green",
                "summer": "red",
                "autumn": "orange",
                "winter": "blue",
                "wet season": "blue",
                "dry season": "red",
            }
            for season in seasons:
                if season == "Unknown":
                    continue
                theme = themes_season.get(season.lower(), "red")
                add_layer("Season", season, f"Season: {season}", theme)

        # Koppen Climate Layers
        if "Koppen_Code" in df_filtered.columns:
            koppens = df_filtered["Koppen_Code"].dropna().unique()
            for kop in koppens:
                if kop == "Unknown" or kop == "":
                    continue
                first_char = kop[0].upper()
                theme = "red"
                if first_char == "A":
                    theme = "viridis"
                elif first_char == "B":
                    theme = "orange"
                elif first_char == "C":
                    theme = "green"
                elif first_char == "D":
                    theme = "blue"
                elif first_char == "E":
                    theme = "purple"
                add_layer("Koppen_Code", kop, f"Climate: {kop}", theme)

        # Layer Control
        folium.LayerControl(collapsed=False).add_to(m)

    # Ensure directory exists and save
    output_dir = os.path.dirname(os.path.abspath(output_html_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    m.save(output_html_path)
    print(f" -> Saved interactive map to: {os.path.abspath(output_html_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset visualization figures and interactive maps."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input Parquet or CSV dataset.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="Global Dataset",
        help="Display title of the dataset / region.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="dataset_stats.png",
        help="Path to save the generated multi-panel dashboard plot (.png).",
    )
    parser.add_argument(
        "--output_grouped_plot",
        type=str,
        default=None,
        help="Path to save the separate grouped camera trap platforms plot (.png).",
    )
    parser.add_argument(
        "--group_camera_traps",
        action="store_true",
        help="If set, also group camera trap platforms into a single category in the main dashboard.",
    )
    parser.add_argument(
        "--output_map",
        type=str,
        default=None,
        help="Optional path to save an interactive HTML map.",
    )

    args = parser.parse_args()

    # Load dataset
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, low_memory=False)

    location_name = args.location
    is_global = "global" in location_name.lower()

    # Generate main multi-panel plots
    generate_plots(
        df,
        is_global,
        location_name,
        args.output_plot,
        group_camera_traps=args.group_camera_traps,
    )

    # Generate separate grouped platform plot
    if "Platform" in df.columns:
        grouped_path = args.output_grouped_plot
        if not grouped_path:
            base_p, ext_p = os.path.splitext(args.output_plot)
            grouped_path = f"{base_p}_grouped_platforms{ext_p or '.png'}"
        generate_grouped_platform_plot(df, location_name, grouped_path)

    # Optional map generation
    if args.output_map:
        generate_interactive_map(df, location_name, args.output_map)


if __name__ == "__main__":
    main()
