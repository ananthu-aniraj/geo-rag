import os
import argparse
import pandas as pd
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Standalone script to standardize the Captured_At timestamp column in Parquet, CSV, or Pickle databases.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (.parquet, .csv, or .pkl).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the standardized file (omitted to overwrite in-place).")
    args = parser.parse_args()

    out_path = args.output if args.output else args.input

    print(f"Loading dataset: {args.input}")
    is_pkl = args.input.endswith('.pkl')
    is_csv = args.input.endswith('.csv')

    if is_pkl:
        with open(args.input, 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
    elif is_csv:
        df = pd.read_csv(args.input)
    else:
        df = pd.read_parquet(args.input)

    # 1. Normalise columns to Captured_At if present under other names
    col_map = {
        'captured_at': 'Captured_At',
        'Date_Observed': 'Captured_At',
        'observed_on_string': 'Captured_At',
        'datetime_local': 'Captured_At'
    }
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns and 'Captured_At' not in df.columns}
    if rename_dict:
        print(f"Renaming timestamp columns to standard 'Captured_At': {rename_dict}")
        df = df.rename(columns=rename_dict)

    if 'Captured_At' not in df.columns:
        print("[WARNING] No timestamp column found. Creating an empty 'Captured_At' column.")
        df['Captured_At'] = None

    # 2. Standardize timestamps (Vectorized)
    print("Standardizing timestamps...")
    initial_nulls = df['Captured_At'].isna().sum()

    ts_raw = df['Captured_At']
    
    # Initialize unified parsed Datetime series
    dt_col = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
    
    # Identify Unix epoch numeric timestamps vs. string representations
    numeric_ts = pd.to_numeric(ts_raw, errors='coerce')
    is_numeric = numeric_ts.notna() & (numeric_ts > 1e8)
    
    # Parse Unix numeric timestamps (Flickr uses seconds, Mapillary uses milliseconds)
    if is_numeric.any():
        is_ms_mask = is_numeric & (numeric_ts > 5e10)
        is_s_mask = is_numeric & ~is_ms_mask
        
        if is_ms_mask.any():
            dt_col.loc[is_ms_mask] = pd.to_datetime(numeric_ts[is_ms_mask], unit='ms', utc=True, errors='coerce')
        if is_s_mask.any():
            dt_col.loc[is_s_mask] = pd.to_datetime(numeric_ts[is_s_mask], unit='s', utc=True, errors='coerce')
            
    # Parse string representations (Flickr EXIF space-colons, ISO strings, etc.)
    is_string = ts_raw.notna() & ~is_numeric
    if is_string.any():
        str_vals = ts_raw[is_string].astype(str).str.strip()
        
        # Clean Flickr EXIF datetime strings like "2023:05:14 18:22:10" to standard "2023-05-14 18:22:10"
        has_colon_date = str_vals.str.match(r'^\d{4}:\d{2}:\d{2}')
        if has_colon_date.any():
            str_vals.loc[has_colon_date] = str_vals[has_colon_date].str.replace(':', '-', n=2)
            
        dt_col.loc[is_string] = pd.to_datetime(str_vals, errors='coerce', utc=True)

    # Filter years to [1, 9999] range to avoid Python strftime out-of-range limitations
    valid_years = pd.Series(True, index=df.index)
    try:
        years = dt_col.dt.year
        valid_years = (years >= 1) & (years <= 9999)
    except Exception:
        pass

    valid_mask = dt_col.notna() & valid_years

    # Create new Series with None as default
    standardized = pd.Series([None] * len(df), index=df.index, dtype=object)
    standardized[valid_mask] = dt_col[valid_mask].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Fallback to original strings if parsing failed but was not null/empty
    invalid_mask = ~valid_mask & df['Captured_At'].notna() & (df['Captured_At'] != '')
    standardized[invalid_mask] = ts_series[invalid_mask]

    df['Captured_At'] = standardized
    final_nulls = df['Captured_At'].isna().sum()
    print(f" -> Timestamp standardization complete. Null timestamps: {initial_nulls} -> {final_nulls}")

    # 3. Add dynamic season classification (Vectorized)
    print("Classifying local seasons based on latitude and month...")
    
    # Extract month and numeric latitude
    months = dt_col.dt.month
    lats = pd.to_numeric(df['Latitude'], errors='coerce')
    
    # Initialize Series
    seasons = pd.Series(['Unknown'] * len(df), index=df.index, dtype=object)
    valid_season_mask = months.notna() & lats.notna()

    # Tropical Zone (-23.5 <= lat <= 23.5)
    tropical = valid_season_mask & (lats >= -23.5) & (lats <= 23.5)
    wet_months = months.isin([6, 7, 8, 9])
    seasons[tropical & wet_months] = 'Wet Season'
    seasons[tropical & ~wet_months] = 'Dry Season'

    # Northern Temperate/Polar (lat > 23.5)
    north = valid_season_mask & (lats > 23.5)
    seasons[north & months.isin([12, 1, 2])] = 'Winter'
    seasons[north & months.isin([3, 4, 5])] = 'Spring'
    seasons[north & months.isin([6, 7, 8])] = 'Summer'
    seasons[north & months.isin([9, 10, 11])] = 'Autumn'

    # Southern Temperate/Polar (lat < -23.5)
    south = valid_season_mask & (lats < -23.5)
    seasons[south & months.isin([12, 1, 2])] = 'Summer'
    seasons[south & months.isin([3, 4, 5])] = 'Autumn'
    seasons[south & months.isin([6, 7, 8])] = 'Winter'
    seasons[south & months.isin([9, 10, 11])] = 'Spring'

    df['Season'] = seasons
    print(" -> Local seasons classified. Column 'Season' added.")

    # 3. Save output
    print(f"Saving standardized dataset to: {out_path}")
    if out_path.endswith('.pkl'):
        # Save as pickle list of dicts if original was a pickle
        records = df.to_dict('records')
        with open(out_path, 'wb') as f:
            pickle.dump(records, f)
    elif out_path.endswith('.csv'):
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    print("Standardization success!")


if __name__ == "__main__":
    main()
