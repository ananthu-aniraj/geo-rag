import argparse
import os
import sys
import time
import zipfile

import pandas as pd


def parse_associated_media(url):
    """Cleans and formats iNaturalist image URLs to medium resolution."""
    if not url or pd.isna(url):
        return None
    url = str(url).strip()
    if url.startswith("http") and (
        ".jpg" in url.lower() or ".jpeg" in url.lower() or ".png" in url.lower()
    ):
        if "inaturalist" in url.lower():
            if "/original." in url:
                url = url.replace("/original.", "/medium.")
            elif "/large." in url:
                url = url.replace("/large.", "/medium.")
        return url
    return None


def build_media_mapping(zip_file, multimedia_file):
    """
    Reads multimedia.txt from zip and builds a map of gbifID -> first valid image URL.
    Keep RAM footprint minimal by storing only matching IDs.
    """
    media_map = {}
    print(f"Building image URL mapping from '{multimedia_file}' inside ZIP...")
    start_time = time.time()

    # We only need gbifID and identifier (the URL)
    cols_to_use = ["gbifID", "identifier", "type"]

    with zip_file.open(multimedia_file) as f:
        # Stream read multimedia file in chunks of 500k rows
        chunks = pd.read_csv(
            f,
            sep="\t",
            usecols=cols_to_use,
            chunksize=500000,
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
        )

        idx = 0
        for chunk in chunks:
            # Filter for still images with valid URLs
            chunk = chunk[chunk["type"].str.lower() == "stillimage"].dropna(
                subset=["gbifID", "identifier"]
            )

            for _, row in chunk.iterrows():
                gbif_id = row["gbifID"]
                # Only store the first image URL we encounter for each observation
                if gbif_id not in media_map:
                    img_url = parse_associated_media(row["identifier"])
                    if img_url:
                        media_map[gbif_id] = img_url

            idx += 1
            print(
                f"  Processed {idx * 500000:,} multimedia rows... (Mapped {len(media_map):,} unique observation images)"
            )

    print(
        f"Mapping built in {time.time() - start_time:.2f} seconds. Total observations mapped: {len(media_map):,}"
    )
    return media_map


def process_dwca_zip(zip_path, output_dir, chunk_size=100000, max_records=None):
    """
    Parses Darwin Core Archive ZIP by joining occurrence.txt and multimedia.txt on the fly.
    """
    print(f"Opening GBIF DwC-A ZIP archive: {zip_path}...")
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        namelist = z.namelist()
        multimedia_file = next(
            (f for f in namelist if f.endswith("multimedia.txt")), None
        )
        occurrence_file = next(
            (f for f in namelist if f.endswith("occurrence.txt")), None
        )

        if not multimedia_file or not occurrence_file:
            print(
                "Error: Could not find 'multimedia.txt' or 'occurrence.txt' inside the DwC-A ZIP."
            )
            print(f"Files found in ZIP: {namelist[:10]}")
            sys.exit(1)

        # Step 1: Build the media mapping (gbifID -> image_url)
        media_map = build_media_mapping(z, multimedia_file)

        # Step 2: Stream occurrences and join with media
        print(f"\nStreaming occurrences from '{occurrence_file}'...")

        occurrence_cols = [
            "gbifID",
            "decimalLatitude",
            "decimalLongitude",
            "species",
            "vernacularName",
            "eventDate",
        ]

        total_processed = 0
        chunk_idx = 0

        with z.open(occurrence_file) as f:
            chunks = pd.read_csv(
                f,
                sep="\t",
                usecols=occurrence_cols,
                chunksize=chunk_size,
                dtype=str,
                low_memory=False,
                on_bad_lines="skip",
            )

            for df_chunk in chunks:
                chunk_start = time.time()
                print(f"Processing occurrence chunk {chunk_idx + 1}...")

                # Filter out rows missing coordinates
                df_chunk = df_chunk.dropna(
                    subset=["decimalLatitude", "decimalLongitude"]
                )

                if df_chunk.empty:
                    continue

                parsed_records = []
                for _, row in df_chunk.iterrows():
                    gbif_id = row["gbifID"]

                    # Look up image URL in the pre-built mapping
                    img_url = media_map.get(gbif_id)
                    if not img_url:
                        continue

                    parsed_records.append(
                        {
                            "Photo_ID": gbif_id,
                            "Platform": "iNaturalist",
                            "Latitude": float(row["decimalLatitude"]),
                            "Longitude": float(row["decimalLongitude"]),
                            "Image_URL": img_url,
                            "Scientific_Name": str(row["species"])
                            if not pd.isna(row["species"])
                            else "Unknown",
                            "Common_Name": str(row["vernacularName"])
                            if not pd.isna(row["vernacularName"])
                            else "Unknown",
                            "Date_Observed": str(row["eventDate"])
                            if not pd.isna(row["eventDate"])
                            else "",
                        }
                    )

                if parsed_records:
                    df_out = pd.DataFrame(parsed_records)
                    out_file = os.path.join(
                        output_dir, f"inaturalist_parsed_chunk_{chunk_idx}.csv"
                    )
                    df_out.to_csv(out_file, index=False)

                    total_processed += len(df_out)
                    print(
                        f"  -> Extracted {len(df_out)} valid observations. Saved to: {out_file}"
                    )
                    print(
                        f"  -> Chunk processed in {time.time() - chunk_start:.2f} seconds."
                    )

                chunk_idx += 1

                if max_records and total_processed >= max_records:
                    print(f"Reached limit of {max_records} records.")
                    break

    print("\n" + "=" * 50)
    print("DwC-A Parsing Complete!")
    print(
        f"Processed: {total_processed} observations in {time.time() - start_time:.2f} seconds."
    )
    print(f"Output files saved to: {output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Parse GBIF Darwin Core Archive (DwC-A) ZIP file directly."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the downloaded GBIF DwC-A ZIP file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="inaturalist_parsed_chunks",
        help="Output directory where parsed CSV chunks will be written.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of rows to stream per iteration.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop parsing after extracting this many records.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: DwC-A ZIP file '{args.input}' not found.")
        sys.exit(1)

    process_dwca_zip(args.input, args.out_dir, args.chunk_size, args.limit)


if __name__ == "__main__":
    main()
