"""Image validation and lookup for local/offline dataset roots."""

import hashlib
import os

KNOWN_PLACEHOLDER_MD5_HASHES = {
    "6da31bd91c242aa98de0ddae4ad89e3c",  # Wildlife Insights 1600x1200 "Human in frame" PNG placeholder (8,535 bytes)
}


def is_valid_image_file(filepath_or_bytes, reject_placeholders=True):
    """Checks if a file path or raw bytes contains valid image data (JPEG, PNG, GIF, WebP, TIFF)."""
    full_bytes = None
    file_path = None
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        data = filepath_or_bytes[:32]
        full_bytes = filepath_or_bytes
    elif isinstance(filepath_or_bytes, (str, os.PathLike)):
        file_path = filepath_or_bytes
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 100:
            return False
        try:
            with open(file_path, "rb") as f:
                data = f.read(32)
        except Exception:
            return False
    else:
        return False

    if not data or len(data) < 4:
        return False

    # Check for known censor/placeholder images
    if reject_placeholders:
        try:
            h = None
            if full_bytes is not None:
                h = hashlib.md5(full_bytes).hexdigest()
            elif file_path is not None:
                if os.path.getsize(file_path) == 8535:
                    with open(file_path, "rb") as f:
                        h = hashlib.md5(f.read()).hexdigest()
            if h and h in KNOWN_PLACEHOLDER_MD5_HASHES:
                return False
        except Exception:
            pass

    # Check magic bytes for standard image formats
    if data.startswith(b"\xff\xd8\xff"):  # JPEG
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):  # GIF
        return True
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":  # WebP
        return True
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):  # TIFF
        return True

    return False


def build_offline_image_index(image_root_dirs, extensions=None):
    """
    Recursively scans image_root_dirs to build a fast in-memory lookup index
    supporting arbitrary nested directory structures.

    Returns a dict mapping keys (exact filenames, stems, relative subpaths,
    and platform-qualified stems) to absolute file paths on disk.
    """
    if not image_root_dirs:
        return {}

    dirs = [image_root_dirs] if isinstance(image_root_dirs, str) else image_root_dirs
    if extensions is None:
        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tif",
            ".tiff",
            ".gif",
            ".bmp",
        }

    index = {}

    for d in dirs:
        if not d or not os.path.exists(d):
            continue
        abs_d = os.path.abspath(d)
        for root, _, files in os.walk(abs_d):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in extensions:
                    continue
                full_path = os.path.join(root, fn)
                stem = os.path.splitext(fn)[0]
                rel_path = os.path.relpath(full_path, abs_d)

                # Store direct filename and stem
                for k in [fn, fn.lower(), stem, stem.lower()]:
                    if k not in index:
                        index[k] = full_path

                # Store relative subpath
                rel_clean = rel_path.replace("\\", "/")
                for k in [rel_clean, rel_clean.lower()]:
                    if k not in index:
                        index[k] = full_path

                # Subpath components (e.g. "site_A/deployment_1/photo_999.JPG")
                parts = rel_clean.split("/")
                if len(parts) > 1:
                    sub_key = "/".join(parts[-2:])
                    if sub_key not in index:
                        index[sub_key] = full_path
                    if sub_key.lower() not in index:
                        index[sub_key.lower()] = full_path

                    stem_sub_key = f"{parts[-2]}/{stem}"
                    if stem_sub_key not in index:
                        index[stem_sub_key] = full_path
                    if stem_sub_key.lower() not in index:
                        index[stem_sub_key.lower()] = full_path

    return index


def resolve_offline_image_path(
    url, image_root_dirs, photo_id=None, platform=None, image_index=None
):
    """
    Resolves an image URL/ID to a local path on disk by checking an optional
    precomputed recursive image_index, flat files, train/ folders, and nested
    directory structures.
    Returns the absolute path if found, otherwise None.
    """
    # 0. Check precomputed recursive image_index first if provided
    if image_index is not None:
        # A. Try platform + photo_id
        if photo_id:
            photo_str = str(photo_id).strip()
            if photo_str.endswith(".0"):
                photo_str = photo_str[:-2]

            # Fast direct ID check in index
            if photo_str in image_index:
                return image_index[photo_str]
            photo_str_lower = photo_str.lower()
            if photo_str_lower in image_index:
                return image_index[photo_str_lower]

            if platform:
                plat_str = str(platform).strip().lower()
                plat_key = f"{plat_str}/{photo_str_lower}"
                if plat_key in image_index:
                    return image_index[plat_key]
                for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                    candidate = f"{plat_key}{ext.lower()}"
                    if candidate in image_index:
                        return image_index[candidate]

            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"]:
                candidate = f"{photo_str}{ext}"
                if candidate in image_index:
                    return image_index[candidate]
                if candidate.lower() in image_index:
                    return image_index[candidate.lower()]

        # B. Try url
        if url and isinstance(url, str):
            clean_url = url.split("://")[1] if "://" in url else url
            clean_url = clean_url.strip()
            if clean_url in image_index:
                return image_index[clean_url]
            clean_url_lower = clean_url.lower()
            if clean_url_lower in image_index:
                return image_index[clean_url_lower]

            url_no_params = clean_url.split("?")[0].split("#")[0]
            if url_no_params != clean_url:
                if url_no_params in image_index:
                    return image_index[url_no_params]
                if url_no_params.lower() in image_index:
                    return image_index[url_no_params.lower()]

            basename = os.path.basename(url_no_params)
            if basename in image_index:
                return image_index[basename]
            basename_lower = basename.lower()
            if basename_lower in image_index:
                return image_index[basename_lower]

            stem = os.path.splitext(basename)[0]
            if stem in image_index:
                return image_index[stem]
            stem_lower = stem.lower()
            if stem_lower in image_index:
                return image_index[stem_lower]

            parts = url_no_params.replace("\\", "/").split("/")
            if len(parts) > 1:
                sub_key = "/".join(parts[-2:])
                if sub_key in image_index:
                    return image_index[sub_key]
                if sub_key.lower() in image_index:
                    return image_index[sub_key.lower()]

        # When precomputed image_index is provided, image_root_dirs have been fully indexed.
        # Avoid expensive fall-through disk checks on non-matching records.
        return None

    if not image_root_dirs:
        env_dirs = os.environ.get("IMAGE_ROOT_DIR") or os.environ.get(
            "OFFLINE_DATASET_DIRS"
        )
        if env_dirs:
            image_root_dirs = env_dirs
        else:
            local_defaults = ["./images", "./data", "images", "data"]
            image_root_dirs = [d for d in local_defaults if os.path.isdir(d)]
            if not image_root_dirs and url and os.path.isfile(url):
                return os.path.abspath(url)

    if not image_root_dirs:
        return None

    if isinstance(image_root_dirs, (str, bytes, os.PathLike)):
        image_root_dirs = [image_root_dirs]

    dirs = []
    for item in image_root_dirs:
        if not item:
            continue
        if isinstance(item, str):
            item = os.path.expanduser(item.strip())
            if " " in item and not os.path.exists(item):
                for part in item.split():
                    p = os.path.expanduser(part.strip())
                    if p and os.path.exists(p):
                        dirs.append(os.path.abspath(p))
                continue
        if os.path.exists(item):
            dirs.append(os.path.abspath(item))

    if not dirs:
        return None

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]
    plat_str = str(platform).strip().lower() if platform else ""
    photo_str = str(photo_id).strip() if photo_id is not None else ""
    if photo_str.endswith(".0"):
        photo_str = photo_str[:-2]

    # Clean and extract relative candidate paths from url
    rel_candidates = []
    if url and isinstance(url, str):
        clean_u = url.split("://")[1] if "://" in url else url
        clean_u = clean_u.split("?")[0].split("#")[0].replace("\\", "/")
        rel_u = clean_u.lstrip("/")
        if rel_u:
            rel_candidates.append(rel_u)
        if plat_str and rel_u.lower().startswith(f"{plat_str}/"):
            rel_candidates.append(rel_u[len(plat_str) + 1 :])
        parts = rel_u.split("/")
        if len(parts) > 1:
            rel_candidates.append("/".join(parts[-2:]))
            if len(parts) > 2:
                rel_candidates.append("/".join(parts[-3:]))
        bname = os.path.basename(rel_u)
        if bname and bname not in rel_candidates:
            rel_candidates.append(bname)

    for d in dirs:
        candidate_subdirs = [d]
        standard_subdirs = ["images", "img", "photos", "train"]
        if plat_str:
            standard_subdirs.extend(
                [
                    plat_str,
                    f"{plat_str}_scraped",
                    f"{plat_str}_scraped/images",
                    f"{plat_str}/images",
                    f"images/{plat_str}",
                    f"{plat_str}_data",
                    f"{plat_str}_subset",
                ]
            )
        for sub in standard_subdirs:
            sub_path = os.path.join(d, sub)
            if os.path.isdir(sub_path):
                candidate_subdirs.append(sub_path)

        for sub_dir in candidate_subdirs:
            # 1. Match against relative URL candidates and case variations
            for rel_c in rel_candidates:
                p = os.path.join(sub_dir, rel_c)
                if os.path.isfile(p):
                    return os.path.abspath(p)
                if "." in rel_c:
                    stem, ext = os.path.splitext(rel_c)
                    for e in [ext.lower(), ext.upper()]:
                        if e != ext:
                            p_alt = os.path.join(sub_dir, f"{stem}{e}")
                            if os.path.isfile(p_alt):
                                return os.path.abspath(p_alt)
                else:
                    for ext in extensions:
                        p_ext = os.path.join(sub_dir, f"{rel_c}{ext}")
                        if os.path.isfile(p_ext):
                            return os.path.abspath(p_ext)

            # 2. Match against photo_id
            if photo_str:
                for ext in extensions:
                    p_pid = os.path.join(sub_dir, f"{photo_str}{ext}")
                    if os.path.isfile(p_pid):
                        return os.path.abspath(p_pid)
                if len(photo_str) == 16:
                    p_nested = os.path.join(
                        sub_dir,
                        photo_str[0],
                        photo_str[1],
                        photo_str[2],
                        f"{photo_str}.jpg",
                    )
                    if os.path.isfile(p_nested):
                        return os.path.abspath(p_nested)

    return None
