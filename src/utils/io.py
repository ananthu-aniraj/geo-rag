"""Backward-compatible facade for Geo-RAG I/O helpers.

Implementation lives in focused modules. Keep this namespace stable for the
many existing Python importers and scripts that still import from ``io``.
"""

import gc
import glob
import hashlib
import os
import re
import time
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.utils import (
    clusters as _clusters,
    embeddings as _embeddings,
    image_download as _image_download,
    table_io as _table_io,
    wildlife_insights as _wildlife_insights,
)
from src.utils.companion_paths import get_core_base_name
from src.utils.credentials import get_wildlife_insights_token
from src.utils.offline_images import (
    KNOWN_PLACEHOLDER_MD5_HASHES,
    build_offline_image_index,
    is_valid_image_file,
    resolve_offline_image_path,
)
from src.utils.table_io import get_parquet_writer, load_dataframe


def save_dataframe(
    df,
    file_path,
    index=False,
    representation_type=None,
    precision=None,
    model_name=None,
    **kwargs,
):
    """Save a dataframe, including decoupled embedding companions when present."""
    return _table_io._save_dataframe(
        df,
        file_path,
        index=index,
        representation_type=representation_type,
        precision=precision,
        model_name=model_name,
        core_name_resolver=get_core_base_name,
        **kwargs,
    )


def load_dataset_with_clusters(
    parquet_path,
    k_clusters=50000,
    columns=None,
    representation_type=None,
    model_name=None,
    precision=None,
    **kwargs,
):
    """Load metadata and cluster assignments from a Parquet dataset."""
    return _clusters._load_dataset_with_clusters(
        parquet_path,
        k_clusters=k_clusters,
        columns=columns,
        representation_type=representation_type,
        model_name=model_name,
        precision=precision,
        _load_dataframe=load_dataframe,
        _get_core_base_name=get_core_base_name,
        **kwargs,
    )


def load_embeddings(
    parquet_path,
    column="embedding",
    representation_type="cls",
    model_name=None,
    precision=None,
):
    """Load embeddings from inline columns or companion NumPy files."""
    return _embeddings._load_embeddings(
        parquet_path,
        column=column,
        representation_type=representation_type,
        model_name=model_name,
        precision=precision,
        core_name_resolver=get_core_base_name,
    )


_http_session = None


def _get_http_session():
    """Return the legacy module's pooled HTTP session."""
    global _http_session
    if _http_session is None:
        _http_session = _image_download._create_http_session(
            session_factory=requests.Session,
            adapter_class=HTTPAdapter,
            retry_class=Retry,
        )
    return _http_session


def resolve_wildlife_insights_url(url, cookie=None, token=None, photo_id=None):
    """Resolve a Wildlife Insights viewer URL to its signed image URL."""
    return _wildlife_insights._resolve_wildlife_insights_url(
        url,
        cookie=cookie,
        token=token,
        photo_id=photo_id,
        token_getter=get_wildlife_insights_token,
    )


def download_image(
    url,
    mapillary_token,
    photo_id=None,
    platform=None,
    offline_dirs=None,
    image_size=448,
    max_retries=3,
):
    """Load an image locally when possible, otherwise download it with retries."""
    return _image_download._download_image_impl(
        url,
        mapillary_token,
        photo_id=photo_id,
        platform=platform,
        offline_dirs=offline_dirs,
        image_size=image_size,
        max_retries=max_retries,
        offline_resolver=resolve_offline_image_path,
        session_getter=_get_http_session,
        wildlife_url_resolver=resolve_wildlife_insights_url,
        image_validator=is_valid_image_file,
    )


__all__ = [
    "gc",
    "glob",
    "hashlib",
    "os",
    "re",
    "time",
    "BytesIO",
    "np",
    "pd",
    "pa",
    "pc",
    "pq",
    "requests",
    "Image",
    "HTTPAdapter",
    "Retry",
    "get_core_base_name",
    "load_dataframe",
    "save_dataframe",
    "get_parquet_writer",
    "load_dataset_with_clusters",
    "load_embeddings",
    "KNOWN_PLACEHOLDER_MD5_HASHES",
    "is_valid_image_file",
    "build_offline_image_index",
    "resolve_offline_image_path",
    "get_wildlife_insights_token",
    "resolve_wildlife_insights_url",
    "download_image",
]
