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
        df = pd.read_csv(args.input, dtype=str)
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
    
    # Initialize output Series with None as default
    standardized = pd.Series([None] * len(df), index=df.index, dtype=object)
    
    # Identify Unix epoch numeric timestamps vs. string representations
    numeric_ts = pd.to_numeric(ts_raw, errors='coerce')
    is_numeric = numeric_ts.notna() & (numeric_ts > 1e8)
    
    # Parse Unix numeric timestamps
    if is_numeric.any():
        is_ms_mask = is_numeric & (numeric_ts > 5e10)
        is_s_mask = is_numeric & ~is_ms_mask
        
        if is_ms_mask.any():
            parsed_ms = pd.to_datetime(numeric_ts[is_ms_mask], unit='ms', utc=True, errors='coerce')
            try:
                valid_ms = (parsed_ms.dt.year >= 1) & (parsed_ms.dt.year <= 9999)
            except Exception:
                valid_ms = pd.Series(True, index=parsed_ms.index)
            
            formatted_ms = pd.Series([None] * len(parsed_ms), index=parsed_ms.index, dtype=object)
            if valid_ms.any():
                formatted_ms[valid_ms] = parsed_ms[valid_ms].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            standardized.loc[is_ms_mask] = formatted_ms
            
        if is_s_mask.any():
            parsed_s = pd.to_datetime(numeric_ts[is_s_mask], unit='s', utc=True, errors='coerce')
            try:
                valid_s = (parsed_s.dt.year >= 1) & (parsed_s.dt.year <= 9999)
            except Exception:
                valid_s = pd.Series(True, index=parsed_s.index)
                
            formatted_s = pd.Series([None] * len(parsed_s), index=parsed_s.index, dtype=object)
            if valid_s.any():
                formatted_s[valid_s] = parsed_s[valid_s].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            standardized.loc[is_s_mask] = formatted_s
            
    # Parse string representations
    is_string = ts_raw.notna() & ~is_numeric
    if is_string.any():
        str_vals = ts_raw[is_string].astype(str).str.strip()
        has_colon_date = str_vals.str.match(r'^\d{4}:\d{2}:\d{2}')
        if has_colon_date.any():
            str_vals.loc[has_colon_date] = str_vals[has_colon_date].str.replace(':', '-', n=2)
            
        parsed_str = pd.to_datetime(str_vals, errors='coerce', utc=True, format='mixed')
        try:
            valid_str = (parsed_str.dt.year >= 1) & (parsed_str.dt.year <= 9999)
        except Exception:
            valid_str = pd.Series(True, index=parsed_str.index)
            
        formatted_str = pd.Series([None] * len(parsed_str), index=parsed_str.index, dtype=object)
        if valid_str.any():
            formatted_str[valid_str] = parsed_str[valid_str].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        standardized.loc[is_string] = formatted_str

    # Fallback to original strings if parsing failed but was not null/empty
    ts_series = ts_raw.astype(str).str.strip()
    is_parsed = standardized.notna()
    invalid_mask = ~is_parsed & ts_raw.notna() & (ts_raw != '')
    standardized[invalid_mask] = ts_series[invalid_mask]

    df['Captured_At'] = standardized
    final_nulls = df['Captured_At'].isna().sum()
    print(f" -> Timestamp standardization complete. Null timestamps: {initial_nulls} -> {final_nulls}")

    # 3. Add dynamic season classification (Vectorized)
    print("Classifying local seasons based on latitude and month...")
    
    # Extract month from standardized ISO 8601 string (e.g. "YYYY-MM-DD...")
    months = pd.to_numeric(standardized.str[5:7], errors='coerce')
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

    # 4. Add dynamic time of day classification (Vectorized)
    print("Classifying time of day based on hour...")
    hours = pd.to_numeric(standardized.str[11:13], errors='coerce')
    time_of_days = pd.Series(['Unknown'] * len(df), index=df.index, dtype=object)
    valid_hour_mask = hours.notna()

    time_of_days[valid_hour_mask & (hours >= 5) & (hours < 8)] = 'Dawn'
    time_of_days[valid_hour_mask & (hours >= 8) & (hours < 12)] = 'Morning'
    time_of_days[valid_hour_mask & (hours >= 12) & (hours < 17)] = 'Afternoon'
    time_of_days[valid_hour_mask & (hours >= 17) & (hours < 20)] = 'Dusk'
    time_of_days[valid_hour_mask & ((hours >= 20) | (hours < 5))] = 'Night'

    df['Time_Of_Day'] = time_of_days
    print(" -> Time of day classified. Column 'Time_Of_Day' added.")

    # 5. Save output
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
