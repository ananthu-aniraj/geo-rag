"""Base-name helpers shared by the table, cluster and embedding loaders."""


def get_core_base_name(base_name):
    """Recursively strips common pipeline suffixes to find the base prefix (e.g. geo_space)."""
    if "_clustered_k_" in base_name:
        base_name = base_name.split("_clustered_k_")[0]
    suffixes = [
        "_filtered",
        "_cleaned",
        "_deduplicated",
        "_clustered",
        "_offline",
        "_float16",
        "_float32",
        "_cls",
        "_avg_patch",
        "_cls_avg_patch",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                changed = True
    return base_name
