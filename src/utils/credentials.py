"""
Centralized Credentials and Environment Management for Geo-RAG.

Provides discovery and accessors for API keys, access tokens, and session credentials:
- MAPILLARY_TOKEN (Mapillary Graph API)
- CARTO_API_KEY (Carto basemaps & tiles)
- FLICKR_API_KEY (Flickr API)
- HF_TOKEN / HUGGING_FACE_HUB_TOKEN (HuggingFace Hub)
- WILDOBS_API_KEY / WILDOBSR_API_KEY (WildObs camera trap API)
- WILDLIFE_INSIGHTS_COOKIE & WILDLIFE_INSIGHTS_TOKEN (Wildlife Insights camera trap API)

Supports automatic discovery of .env via:
1. Explicit path or search directory
2. python-dotenv (if installed)
3. Directory tree walk-up search from current working directory and repo root
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


def clean_credential_value(val: Optional[str]) -> str:
    """Strips surrounding whitespace, quotes, and newlines from a credential string."""
    if not val:
        return ""
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        val = val[1:-1].strip()
    return val


def find_env_file(
    search_dir: Optional[Union[Path, str]] = None,
) -> Optional[Path]:
    """Finds the nearest .env file.

    Search precedence:
    1. search_dir (walk-up search if provided as a path or directory)
    2. Directory tree walk-up from Path.cwd() and this module's location
    3. python-dotenv find_dotenv() if installed
    """
    if search_dir is not None:
        target = Path(search_dir).resolve()
        if target.is_file() and target.name == ".env":
            return target
        curr = target if target.is_dir() else target.parent
        visited = set()
        while curr not in visited and curr != curr.parent:
            visited.add(curr)
            env_file = curr / ".env"
            if env_file.is_file():
                return env_file
            curr = curr.parent
        return None

    # Walk-up discovery from cwd and current file parent
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    visited = set()
    for root in search_roots:
        curr = root.resolve()
        while curr not in visited and curr != curr.parent:
            visited.add(curr)
            env_file = curr / ".env"
            if env_file.is_file():
                return env_file
            curr = curr.parent

    # Try python-dotenv discovery as fallback
    try:
        from dotenv import find_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path and Path(dotenv_path).is_file():
            return Path(dotenv_path)
    except ImportError:
        pass

    return None


def load_env(
    search_dir: Optional[Union[Path, str]] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Discovers and parses a .env file, injecting variables into os.environ.

    Args:
        search_dir: Optional directory or file path to search for .env.
        override: If True, overwrite already-set os.environ entries.

    Returns:
        Dictionary of parsed key-value pairs from the .env file.
    """
    env_file = find_env_file(search_dir)
    parsed: Dict[str, str] = {}
    if env_file is None or not env_file.is_file():
        return parsed

    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v_clean = clean_credential_value(v)
                parsed[k] = v_clean
                if k not in os.environ or override or not os.environ[k].strip():
                    os.environ[k] = v_clean
    except Exception:
        pass

    # Also trigger load_dotenv if available
    try:
        from dotenv import load_dotenv

        load_dotenv(str(env_file), override=override)
    except ImportError:
        pass

    return parsed


def get_credential(
    key_name: str,
    default: str = "",
    search_dir: Optional[Union[Path, str]] = None,
) -> str:
    """Retrieves a credential from os.environ or .env file.

    Args:
        key_name: Environment variable name (e.g. 'MAPILLARY_TOKEN').
        default: Fallback string if key is not found or empty.
        search_dir: Optional search path for .env file.

    Returns:
        The cleaned credential string or default.
    """
    val = os.environ.get(key_name)
    cleaned = clean_credential_value(val)
    if cleaned:
        return cleaned

    # Try loading from .env
    load_env(search_dir=search_dir)
    val = os.environ.get(key_name)
    cleaned = clean_credential_value(val)
    if cleaned:
        return cleaned

    return default


def get_mapillary_token(search_dir: Optional[Union[Path, str]] = None) -> str:
    """Retrieves Mapillary Graph API access token from environment or .env."""
    return get_credential("MAPILLARY_TOKEN", search_dir=search_dir)


def get_carto_api_key(search_dir: Optional[Union[Path, str]] = None) -> str:
    """Retrieves CARTO basemap API key from environment or .env."""
    return get_credential("CARTO_API_KEY", search_dir=search_dir)


def get_flickr_api_key(search_dir: Optional[Union[Path, str]] = None) -> str:
    """Retrieves Flickr API key from environment or .env."""
    return get_credential("FLICKR_API_KEY", search_dir=search_dir)


def get_hf_token(search_dir: Optional[Union[Path, str]] = None) -> str:
    """Retrieves HuggingFace Hub token from environment or .env."""
    token = get_credential("HF_TOKEN", search_dir=search_dir)
    if not token:
        token = get_credential("HUGGING_FACE_HUB_TOKEN", search_dir=search_dir)
    return token


def get_wildobs_api_key(search_dir: Optional[Union[Path, str]] = None) -> str:
    """Retrieves WildObs API key from environment or .env."""
    token = get_credential("WILDOBS_API_KEY", search_dir=search_dir)
    if not token:
        token = get_credential("WILDOBSR_API_KEY", search_dir=search_dir)
    return token


def get_wildlife_insights_credentials(
    search_dir: Optional[Union[Path, str]] = None,
) -> Tuple[str, str]:
    """Retrieves Wildlife Insights session cookie and JWT token.

    Returns:
        Tuple of (cookie, token).
    """
    cookie = get_credential("WILDLIFE_INSIGHTS_COOKIE", search_dir=search_dir)
    token = get_credential("WILDLIFE_INSIGHTS_TOKEN", search_dir=search_dir)
    return cookie, token


__all__ = [
    "clean_credential_value",
    "find_env_file",
    "load_env",
    "get_credential",
    "get_mapillary_token",
    "get_carto_api_key",
    "get_flickr_api_key",
    "get_hf_token",
    "get_wildobs_api_key",
    "get_wildlife_insights_credentials",
]
