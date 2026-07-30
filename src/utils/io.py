import os
import sys
import pickle
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

def load_dataframe(file_path, **kwargs):
    """
    Loads a dataframe from CSV, Parquet, or Pickle files dynamically based on extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif ext == '.csv':
        # Default low_memory=False for safety on mixed-type data
        if 'low_memory' not in kwargs:
            kwargs['low_memory'] = False
        return pd.read_csv(file_path, **kwargs)
    elif ext in ('.pkl', '.pickle'):
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for loading dataframe.")


def save_dataframe(df, file_path, index=False, **kwargs):
    """
    Saves a dataframe to CSV, Parquet, or Pickle with optimal compression default (Zstd for Parquet).
    """
    # Ensure output parent directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.parquet':
        # Default to high-performance zstd compression
        if 'compression' not in kwargs:
            kwargs['compression'] = 'zstd'
        df.to_parquet(file_path, index=index, **kwargs)
    elif ext == '.csv':
        df.to_csv(file_path, index=index, **kwargs)
    elif ext in ('.pkl', '.pickle'):
        df.to_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for saving dataframe.")


def get_parquet_writer(file_path, schema, **kwargs):
    """
    Returns a PyArrow ParquetWriter configured with Zstandard compression.
    """
    if pq is None:
        raise ImportError("pyarrow is required to write Parquet tables via stream.")
    
    # Ensure parent directory exists
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if 'compression' not in kwargs:
        kwargs['compression'] = 'zstd'
        
    return pq.ParquetWriter(file_path, schema, **kwargs)
