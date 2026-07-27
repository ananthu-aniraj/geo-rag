import argparse
import os
import sys
import time
import zipfile

import pandas as pd


def parse_associated_media(media_str):
    """Extracts the first valid image URL from GBIF's associatedMedia string."""
    if not media_str or pd.isna(media_str):
        return None
    urls = [u.strip() for u in str(media_str).split('|')]
    for url in urls:
        if url.startswith('http') and ('.jpg' in url.lower() or '.jpeg' in url.lower() or '.png' in url.lower()):
            if 'inaturalist' in url.lower():
                if '/original.' in url:
                    url = url.replace('/original.', '/medium.')
                elif '/large.' in url:
                    url = url.replace('/large.', '/medium.')
            return url
    return None

def process_gbif_zip(zip_path, output_dir, chunk_size=100000, max_records=None):
    """
    Streams and parses the occurrence file directly from the GBIF ZIP archive.
    """
    print(f"Opening GBIF ZIP archive: {zip_path}...")
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We only read the columns we need to save memory and parsing overhead
    cols_to_use = [
        'gbifID', 
        'decimalLatitude', 
        'decimalLongitude', 
        'associatedMedia', 
        'species', 
        'vernacularName',
        'eventDate'
    ]
    
    total_processed = 0
    chunk_idx = 0
    
    # Open ZIP and stream occurrence file directly without extracting to disk
    with zipfile.ZipFile(zip_path) as z:
        # Find any text or csv file in the zip
        namelist = z.namelist()
        occurrence_file = next((f for f in namelist if f.endswith('.txt') or f.endswith('.csv')), None)
        
        if not occurrence_file:
            print("Error: Could not find any occurrence data file (.txt or .csv) inside the ZIP archive.")
            print(f"Files found in ZIP: {namelist}")
            sys.exit(1)
            
        print(f"Streaming data from '{occurrence_file}' inside ZIP...")
        
        with z.open(occurrence_file) as f:
            chunks = pd.read_csv(
                f, 
                sep='\t', 
                usecols=cols_to_use, 
                chunksize=chunk_size, 
                dtype=str, 
                low_memory=False,
                on_bad_lines='skip'
            )
            
            for df_chunk in chunks:
                chunk_start = time.time()
                print(f"\nProcessing chunk {chunk_idx + 1}...")
                
                # Drop rows missing coordinates or media URLs
                df_chunk = df_chunk.dropna(subset=['decimalLatitude', 'decimalLongitude', 'associatedMedia'])
                
                if df_chunk.empty:
                    continue
                    
                parsed_records = []
                for _, row in df_chunk.iterrows():
                    img_url = parse_associated_media(row['associatedMedia'])
                    if not img_url:
                        continue
                        
                    parsed_records.append({
                        "Photo_ID": str(row['gbifID']),
                        "Platform": "iNaturalist",
                        "Latitude": float(row['decimalLatitude']),
                        "Longitude": float(row['decimalLongitude']),
                        "Image_URL": img_url,
                        "Scientific_Name": str(row['species']) if not pd.isna(row['species']) else "Unknown",
                        "Common_Name": str(row['vernacularName']) if not pd.isna(row['vernacularName']) else "Unknown",
                        "Date_Observed": str(row['eventDate']) if not pd.isna(row['eventDate']) else ""
                    })
                    
                if parsed_records:
                    df_out = pd.DataFrame(parsed_records)
                    out_file = os.path.join(output_dir, f"inaturalist_parsed_chunk_{chunk_idx}.csv")
                    df_out.to_csv(out_file, index=False)
                    
                    total_processed += len(df_out)
                    print(f"  -> Extracted {len(df_out)} valid observations. Saved to: {out_file}")
                    print(f"  -> Chunk processed in {time.time() - chunk_start:.2f} seconds.")
                    
                chunk_idx += 1
                
                if max_records and total_processed >= max_records:
                    print(f"Reached limit of {max_records} records.")
                    break
                    
    print("\n" + "="*50)
    print("Parsing Complete!")
    print(f"Processed: {total_processed} observations in {time.time() - start_time:.2f} seconds.")
    print(f"Output files saved to: {output_dir}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Parse GBIF iNaturalist Occurrence ZIP file directly.")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the downloaded GBIF ZIP file.")
    parser.add_argument("--out_dir", type=str, default="inaturalist_parsed_chunks", 
                        help="Output directory where parsed CSV chunks will be written.")
    parser.add_argument("--chunk_size", type=int, default=100000, 
                        help="Number of rows to stream per iteration.")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Stop parsing after extracting this many records.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: ZIP file '{args.input}' not found.")
        sys.exit(1)
        
    process_gbif_zip(args.input, args.out_dir, args.chunk_size, args.limit)

if __name__ == '__main__':
    main()
