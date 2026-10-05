import argparse
import os
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_PLATFORMS = ["flickr", "mapillary", "kartaview", "inaturalist"]


def restore_image_urls(
    offline_path: str,
    online_path: str,
    output_path: str = None,
    key_col: str = "photo_key",
    target_platforms: list = None,
):
    """
    Restores the original online Image_URL values into an offline Parquet dataset
    from the canonical online dataset.

    Restricted specifically to online scraped platforms (default: flickr, mapillary, kartaview, inaturalist).
    Other platforms (offline datasets like wikimedia/landmarks, snapshotusa, iwildcam, etc.)
    remain completely untouched.

    Only updates Image_URL for records present in the online dataset.
    Preserves all other columns (including Image_Location), exact schemas, row ordering,
    and ZSTD compression.
    """
    if target_platforms is None:
        target_platforms = DEFAULT_PLATFORMS

    target_platforms_set = {p.strip().lower() for p in target_platforms if p}

    if not os.path.exists(offline_path):
        print(f"Error: Offline dataset not found: {offline_path}")
        sys.exit(1)
    if not os.path.exists(online_path):
        print(f"Error: Online dataset not found: {online_path}")
        sys.exit(1)

    in_place = (output_path is None) or (
        os.path.abspath(output_path) == os.path.abspath(offline_path)
    )
    target_output = offline_path if in_place else output_path
    tmp_output = target_output + ".tmp_restore"

    print(f"Step 1: Building lookup dictionary from online dataset: {online_path}...")
    print(f" -> Target platforms restricted to: {sorted(target_platforms_set)}")
    t0 = time.time()

    online_table = pq.read_table(
        online_path, columns=[key_col, "Platform", "Image_URL"]
    )
    keys_on = online_table[key_col].to_pandas()
    plats_on = online_table["Platform"].to_pandas().astype(str).str.strip().str.lower()
    urls_on = online_table["Image_URL"].to_pandas()

    # Filter lookup strictly to target platforms
    mask_on = plats_on.isin(target_platforms_set)
    filtered_keys = keys_on[mask_on]
    filtered_urls = urls_on[mask_on]

    url_lookup = dict(zip(filtered_keys, filtered_urls))
    print(
        f" -> Indexed {len(url_lookup):,} online URL mappings for target platforms in {time.time() - t0:.2f}s."
    )

    print(f"\nStep 2: Processing offline dataset: {offline_path}...")
    pf = pq.ParquetFile(offline_path)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.num_row_groups
    schema = pf.schema_arrow
    url_field_idx = schema.get_field_index("Image_URL")
    url_field_type = schema.field(url_field_idx).type

    print(f" -> Offline rows: {total_rows:,} across {num_row_groups} row groups.")
    print(f" -> Writing temporary file to: {tmp_output}")

    total_updated = 0
    total_kept = 0
    platform_update_counts = {p: 0 for p in target_platforms_set}
    t_start = time.time()

    writer = pq.ParquetWriter(tmp_output, schema=schema, compression="zstd")

    try:
        for i in range(num_row_groups):
            t_rg = time.time()
            rg = pf.read_row_group(i)
            keys = rg.column(key_col).to_pylist()
            plats = rg.column("Platform").to_pylist()
            old_urls = rg.column("Image_URL").to_pylist()

            new_urls = []
            updated_rg = 0
            kept_rg = 0

            for k, plat, orig in zip(keys, plats, old_urls):
                plat_clean = str(plat).strip().lower() if plat else ""
                if plat_clean in target_platforms_set and k in url_lookup:
                    restored_url = url_lookup[k]
                    new_urls.append(restored_url)
                    if orig != restored_url:
                        updated_rg += 1
                        platform_update_counts[plat_clean] = (
                            platform_update_counts.get(plat_clean, 0) + 1
                        )
                    else:
                        kept_rg += 1
                else:
                    new_urls.append(orig)
                    kept_rg += 1

            total_updated += updated_rg
            total_kept += kept_rg

            new_url_arr = pa.array(new_urls, type=url_field_type)
            new_rg = rg.set_column(url_field_idx, "Image_URL", new_url_arr)
            writer.write_table(new_rg)

            print(
                f"    Row group {i+1}/{num_row_groups} ({len(rg):,} rows): "
                f"updated {updated_rg:,} URLs in {time.time() - t_rg:.2f}s."
            )
    finally:
        writer.close()

    print("\nStep 3: Verifying output integrity...")
    pf_out = pq.ParquetFile(tmp_output)
    if pf_out.metadata.num_rows != total_rows:
        print(
            f"❌ Error: Row count mismatch! Original: {total_rows:,}, Output: {pf_out.metadata.num_rows:,}"
        )
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        sys.exit(1)

    print(
        f" -> Verification passed: {pf_out.metadata.num_rows:,} rows successfully verified."
    )
    print(f" -> Total URLs restored from online dataset: {total_updated:,}")
    print(f" -> Total URLs kept as-is (offline datasets or unchanged): {total_kept:,}")
    print(" -> Per-platform updates:")
    for p, c in sorted(platform_update_counts.items()):
        print(f"    * {p}: {c:,} URLs restored")

    # Preserve file permissions if replacing in-place
    try:
        shutil.copymode(offline_path, tmp_output)
    except Exception:
        pass

    os.replace(tmp_output, target_output)
    print(
        f"\n🎉 Successfully restored Image_URLs and saved to: {target_output} (Elapsed: {time.time() - t_start:.2f}s)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Restore original online Image_URL values into offline Parquet dataset from the online master dataset."
    )
    parser.add_argument(
        "--offline",
        type=str,
        required=True,
        help="Path to the offline Parquet dataset (e.g. geo_space_cleaned_offline.parquet).",
    )
    parser.add_argument(
        "--online",
        type=str,
        required=True,
        help="Path to the canonical online Parquet dataset (e.g. geo_space_cleaned.parquet).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path (defaults to updating --offline in-place).",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="photo_key",
        help="Column name to match rows across datasets (default: photo_key).",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=DEFAULT_PLATFORMS,
        help="Target platforms to restore URLs for (default: flickr mapillary kartaview inaturalist).",
    )
    args = parser.parse_args()

    restore_image_urls(
        offline_path=os.path.expanduser(args.offline),
        online_path=os.path.expanduser(args.online),
        output_path=os.path.expanduser(args.output) if args.output else None,
        key_col=args.key,
        target_platforms=args.platforms,
    )


if __name__ == "__main__":
    main()
