"""
Centralized Configuration Loader for Geo-RAG.

Supports hierarchical configuration loading:
1. Base configuration (tracked in Git, e.g. config/evaluation/params_offline.yaml)
2. Machine-specific local overrides (gitignored):
   - Global config/local.yaml (one file for all folders: evaluation, pipeline, scrapers)
   - Sibling folder local.yaml (e.g. config/evaluation/local.yaml)
   - Sibling file [name].local.yaml (e.g. config/evaluation/params_offline.local.yaml)
"""

import copy
import os
import sys
from typing import Any, Dict, Optional

import yaml


def find_repo_root(start_path: Optional[str] = None) -> str:
    """Finds repository root directory by walking up to find .git or config/ directory."""
    curr = os.path.abspath(start_path or os.getcwd())
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".git")) or os.path.isdir(
            os.path.join(curr, "config")
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(start_path or os.getcwd())


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges override dictionary into base dictionary.
    Modifies base in-place and returns it.
    """
    if not isinstance(override, dict):
        return override

    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _extract_section_for_file(
    local_cfg: Dict[str, Any],
    rel_path: str,
    base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extracts relevant override keys from local_cfg for a given config file.
    Supports both nested category schemas:
      evaluation:
        lucas: ...
        params_offline: ...
    and flat top-level schemas:
      lucas: ...
      params_offline: ...
    """
    merged_override: Dict[str, Any] = {}
    norm_rel = rel_path.replace("\\", "/").strip("/")
    parts = norm_rel.split("/")

    subfolder = parts[0] if len(parts) > 1 else ""
    filename = parts[-1]
    stem = os.path.splitext(filename)[0]

    # 1. Match subfolder in local_cfg (e.g. "evaluation", "pipeline", "scrapers")
    if subfolder and subfolder in local_cfg and isinstance(local_cfg[subfolder], dict):
        sub_dict = local_cfg[subfolder]
        # Match exact file stem (e.g. evaluation.params_offline, scrapers.flickr_scraper)
        if stem in sub_dict and isinstance(sub_dict[stem], dict):
            deep_merge(merged_override, sub_dict[stem])
        # Match top-level keys of base_cfg directly inside subfolder (e.g. evaluation.lucas)
        for base_key in base_cfg.keys():
            if base_key in sub_dict and isinstance(
                sub_dict[base_key], (dict, list, int, float, str, bool)
            ):
                merged_override[base_key] = copy.deepcopy(sub_dict[base_key])
        # If base_cfg has a single dict key equal to subfolder (e.g. pipeline: { ... })
        if len(base_cfg) == 1:
            sole_key = next(iter(base_cfg.keys()))
            if sole_key == subfolder and isinstance(base_cfg[sole_key], dict):
                for k, v in sub_dict.items():
                    if k != stem and k in base_cfg[sole_key]:
                        if sole_key not in merged_override:
                            merged_override[sole_key] = {}
                        merged_override[sole_key][k] = copy.deepcopy(v)

    # 2. Match file stem directly at root of local_cfg (e.g. params_offline: ..., flickr_scraper: ...)
    if stem in local_cfg and isinstance(local_cfg[stem], dict):
        deep_merge(merged_override, local_cfg[stem])

    # 3. Match top-level keys of base_cfg directly at root of local_cfg (e.g. lucas: ..., pipeline: ...)
    for base_key in base_cfg.keys():
        if base_key in local_cfg:
            merged_override[base_key] = copy.deepcopy(local_cfg[base_key])

    # 4. If base_cfg has a single dict key (e.g. "scraper", "pipeline", "eval") and merged_override
    # has flat keys matching base_cfg[sole_key], wrap them under sole_key
    if len(base_cfg) == 1:
        sole_key = next(iter(base_cfg.keys()))
        if isinstance(base_cfg[sole_key], dict) and sole_key not in merged_override:
            sole_sub = base_cfg[sole_key]
            if any(k in sole_sub for k in merged_override.keys()):
                merged_override = {sole_key: merged_override}

    return merged_override


def load_config(
    config_path: str,
    local_override_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads base YAML configuration and applies any active local overrides.

    Override discovery order:
    1. config/local.yaml (central repository-wide local override)
    2. Sibling local.yaml (e.g. config/evaluation/local.yaml)
    3. Sibling [stem].local.yaml (e.g. config/evaluation/params_offline.local.yaml)
    4. Explicit local_override_path argument
    """
    root = repo_root or find_repo_root()

    # Resolve config path
    resolved_path = config_path
    if not os.path.isabs(resolved_path):
        candidate = os.path.join(root, config_path)
        if os.path.exists(candidate):
            resolved_path = candidate

    base_cfg: Dict[str, Any] = {}
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    base_cfg = copy.deepcopy(loaded)
        except Exception as e:
            print(
                f"Warning: Failed to load base config '{resolved_path}': {e}",
                file=sys.stderr,
            )

    # Compute relative path under config directory
    rel_to_root = os.path.relpath(resolved_path, root)
    if rel_to_root.startswith("config" + os.sep):
        rel_config_path = os.path.relpath(resolved_path, os.path.join(root, "config"))
    else:
        rel_config_path = rel_to_root

    # 1. Check global config/local.yaml
    global_local_path = os.path.join(root, "config", "local.yaml")
    if os.path.exists(global_local_path):
        try:
            with open(global_local_path, "r", encoding="utf-8") as f:
                local_cfg = yaml.safe_load(f)
            if isinstance(local_cfg, dict):
                overrides = _extract_section_for_file(
                    local_cfg, rel_config_path, base_cfg
                )
                if overrides:
                    deep_merge(base_cfg, overrides)
        except Exception as e:
            print(
                f"Warning: Failed to load local override '{global_local_path}': {e}",
                file=sys.stderr,
            )

    # 2. Check sibling directory local.yaml (e.g. config/evaluation/local.yaml)
    dir_local_path = os.path.join(os.path.dirname(resolved_path), "local.yaml")
    if os.path.exists(dir_local_path) and os.path.abspath(
        dir_local_path
    ) != os.path.abspath(global_local_path):
        try:
            with open(dir_local_path, "r", encoding="utf-8") as f:
                dir_cfg = yaml.safe_load(f)
            if isinstance(dir_cfg, dict):
                stem = os.path.splitext(os.path.basename(resolved_path))[0]
                if stem in dir_cfg and isinstance(dir_cfg[stem], dict):
                    deep_merge(base_cfg, dir_cfg[stem])
                else:
                    for k in base_cfg.keys():
                        if k in dir_cfg:
                            base_cfg[k] = copy.deepcopy(dir_cfg[k])
        except Exception as e:
            print(
                f"Warning: Failed to load dir local override '{dir_local_path}': {e}",
                file=sys.stderr,
            )

    # 3. Check sibling file [stem].local.yaml (e.g. params_offline.local.yaml)
    file_local_path = resolved_path.replace(".yaml", ".local.yaml").replace(
        ".yml", ".local.yml"
    )
    if os.path.exists(file_local_path):
        try:
            with open(file_local_path, "r", encoding="utf-8") as f:
                f_cfg = yaml.safe_load(f)
            if isinstance(f_cfg, dict):
                deep_merge(base_cfg, f_cfg)
        except Exception as e:
            print(
                f"Warning: Failed to load file local override '{file_local_path}': {e}",
                file=sys.stderr,
            )

    # 4. Explicit override path
    if local_override_path and os.path.exists(local_override_path):
        try:
            with open(local_override_path, "r", encoding="utf-8") as f:
                exp_cfg = yaml.safe_load(f)
            if isinstance(exp_cfg, dict):
                deep_merge(base_cfg, exp_cfg)
        except Exception as e:
            print(
                f"Warning: Failed to load explicit local override '{local_override_path}': {e}",
                file=sys.stderr,
            )

    return base_cfg


def get_param(config_path: str, *keys: str, default: Any = None) -> Any:
    """
    Retrieves a nested configuration value with local overrides applied.
    Example: get_param("config/evaluation/params_offline.yaml", "lucas", "img_dir")
    """
    cfg = load_config(config_path)
    curr: Any = cfg
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return curr


def format_for_shell(val: Any) -> str:
    """Formats Python primitives cleanly for bash consumption."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        return " ".join(str(item) for item in val)
    return str(val)


def main():
    """
    CLI interface for shell scripts and terminal querying.
    Usage:
      python -m src.utils.config get <config_path> <key1> [<key2> ...]
      python -m src.utils.config show <config_path>
    """
    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.utils.config [get <config_path> <keys...> | show <config_path>]"
        )
        sys.exit(1)

    action = sys.argv[1]
    config_path = sys.argv[2]

    if action == "get":
        keys = sys.argv[3:]
        val = get_param(config_path, *keys)
        print(format_for_shell(val))
    elif action == "show":
        cfg = load_config(config_path)
        print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
