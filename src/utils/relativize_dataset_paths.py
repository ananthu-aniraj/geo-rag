#!/usr/bin/env python3
import argparse
import os
import sys

from src.utils.io import load_dataframe, save_dataframe


def parse_mappings(mapping_strings):
    """Parses list of 'old:new' strings into a list of (old, new) tuples."""
    mappings = []
    if not mapping_strings:
        return mappings

    for item in mapping_strings:
        if ":" not in item:
            print(
                f"Error: Invalid mapping format '{item}'. Expected format is 'old_prefix:new_prefix'.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Split on the first colon to allow for colons in Windows paths if needed
        parts = item.split(":", 1)
        mappings.append((parts[0], parts[1]))
    return mappings


def make_path_relative(path_str, mappings):
    """
    Strips absolute prefixes based on custom mappings, returning the original path
    if no mappings match.
    """
    if not isinstance(path_str, str):
        return path_str

    # Apply user-defined mappings
    for old_prefix, new_prefix in mappings:
        if old_prefix in path_str:
            # Replace the absolute prefix with the relative folder name
            relative_part = path_str.split(old_prefix, 1)[-1].lstrip("/")
            return os.path.normpath(os.path.join(new_prefix, relative_part))

    return path_str


def main():
    parser = argparse.ArgumentParser(
        description="Convert absolute dataset image paths to relative paths for public release."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input Parquet database.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path where the cleaned output Parquet file will be saved.",
    )
    parser.add_argument(
        "--replace-prefix",
        "-r",
        action="append",
        default=[],
        help="Map absolute paths to relative ones (format: 'old_prefix:new_prefix'). Can be specified multiple times.",
    )
    parser.add_argument(
        "--columns",
        "-c",
        nargs="+",
        default=["Image_URL"],
        help="List of columns to check and convert to relative paths.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Parse and validate the old:new mapping arguments
    mappings = parse_mappings(args.replace_prefix)

    print(f"Loading database from {args.input}...")
    df = load_dataframe(args.input)

    modified_cols = []
    for col in args.columns:
        if col in df.columns:
            print(f"Relativizing paths in column '{col}'...")
            df[col] = df[col].apply(lambda x: make_path_relative(x, mappings))
            modified_cols.append(col)

    if not modified_cols:
        print(
            "Warning: None of the target path columns were found in the dataset schema."
        )

    print(f"Saving cleaned dataset to {args.output}...")
    save_dataframe(df, args.output)
    print("✅ Path relativization complete!")


if __name__ == "__main__":
    main()
