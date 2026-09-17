#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Source the .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

YAML_PATH="config/scrapers/wikiloc_scraper.yaml"

if [ ! -f "$YAML_PATH" ]; then
    echo "❌ Error: Configuration file '$YAML_PATH' not found."
    exit 1
fi

echo "============================================================"
echo "Wikiloc Remote Region Scraper Runner"
echo "============================================================"
echo "Config: $YAML_PATH"

python3 -c "
import subprocess, sys, os
from src.utils.config import load_config

loaded_cfg = load_config('$YAML_PATH')
cfg = loaded_cfg.get('wikiloc_scraper') or loaded_cfg.get('scraper', {})

cmd = [sys.executable, 'src/scrapers/scrape_wikiloc.py']

regions = cfg.get('regions', [])
if isinstance(regions, str):
    regions = [regions]
if regions:
    cmd.extend(['--regions'] + [str(r) for r in regions])

urls = cfg.get('urls', [])
if isinstance(urls, str):
    urls = [urls]
if urls:
    cmd.extend(['--urls'] + [str(u) for u in urls])

activity = cfg.get('activity')
if activity:
    cmd.extend(['--activity', str(activity)])

max_pages = cfg.get('max_pages_per_region')
if max_pages is not None:
    cmd.extend(['--max_pages_per_region', str(max_pages)])

max_trails = cfg.get('max_trails')
if max_trails is not None:
    cmd.extend(['--max_trails', str(max_trails)])

max_photos = cfg.get('max_photos_per_trail')
if max_photos is not None:
    cmd.extend(['--max_photos_per_trail', str(max_photos)])

delay = cfg.get('delay')
if delay is not None:
    cmd.extend(['--delay', str(delay)])

output = cfg.get('output')
if output:
    cmd.extend(['--output', str(output)])

if cfg.get('no_csv', False):
    cmd.append('--no_csv')

if cfg.get('download_images', True):
    cmd.append('--download_images')
    image_dir = cfg.get('image_dir')
    if image_dir:
        cmd.extend(['--image_dir', str(image_dir)])
else:
    cmd.append('--no_download_images')

threads = cfg.get('threads')
if threads is not None:
    cmd.extend(['--threads', str(threads)])

if not cfg.get('headless', True):
    cmd.append('--no-headless')

# Forward any additional command-line arguments passed to this script
cmd.extend(sys.argv[1:])

print(f'Executing: {\" \".join(cmd)}\n')
ret = subprocess.run(cmd)
sys.exit(ret.returncode)
" "$@"
