import os
import argparse
import pandas as pd
import pickle


def standardize_timestamp(ts):
    """Standardizes various timestamp formats from Flickr, Mapillary, and iNaturalist to ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)."""
    if pd.isna(ts) or not ts:
        return None
    ts_str = str(ts).strip()
    try:
        # pd.to_datetime handles timezone-aware, custom strings, and timestamps gracefully
        dt = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(dt):
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        pass
    return ts_str


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

    # 2. Standardize timestamps
    print("Standardizing timestamps...")
    initial_nulls = df['Captured_At'].isna().sum()
    df['Captured_At'] = df['Captured_At'].apply(standardize_timestamp)
    final_nulls = df['Captured_At'].isna().sum()
    print(f" -> Timestamp standardization complete. Null timestamps: {initial_nulls} -> {final_nulls}")

    # 3. Add dynamic season classification
    print("Classifying local seasons based on latitude and month...")

    def get_local_season(row):
        lat = row.get('Latitude')
        captured_at = row.get('Captured_At')
        if pd.isna(captured_at) or not captured_at or pd.isna(lat):
            return 'Unknown'
        try:
            # Captured_At is now standardized as YYYY-MM-DDTHH:MM:SSZ
            dt = pd.to_datetime(captured_at)
            month = dt.month
            lat_val = float(lat)
        except Exception:
            return 'Unknown'

        # Tropical Zone (-23.5 to 23.5 degrees) -> Wet/Dry Season
        if -23.5 <= lat_val <= 23.5:
            return 'Wet Season' if month in [6, 7, 8, 9] else 'Dry Season'

        # Northern Temperate/Polar
        if lat_val > 23.5:
            if month in [12, 1, 2]: return 'Winter'
            if month in [3, 4, 5]: return 'Spring'
            if month in [6, 7, 8]: return 'Summer'
            return 'Autumn'

        # Southern Temperate/Polar
        if lat_val < -23.5:
            if month in [12, 1, 2]: return 'Summer'
            if month in [3, 4, 5]: return 'Autumn'
            if month in [6, 7, 8]: return 'Winter'
            return 'Spring'

        return 'Unknown'

    df['Season'] = df.apply(get_local_season, axis=1)
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
