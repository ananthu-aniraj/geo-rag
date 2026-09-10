#!/bin/bash

# Enforce execution from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Source the .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$WILDOBS_API_KEY" ]; then
    echo "❌ Error: WILDOBS_API_KEY is not set in your environment or .env file."
    exit 1
fi

YAML_PATH="config/scrapers/wildobs_scraper.yaml"

if [ ! -f "$YAML_PATH" ]; then
    echo "❌ Error: Configuration file '$YAML_PATH' not found."
    exit 1
fi

echo "============================================================"
echo "WildObs Camera Trap Scraper Runner"
echo "============================================================"
echo "Config: $YAML_PATH"

python3 -c "
import yaml, subprocess, sys, os

with open('$YAML_PATH') as f:
    cfg = yaml.safe_load(f).get('scraper', {})

cmd = [sys.executable, 'src/scrapers/scrape_wildobs.py']

project_ids = cfg.get('project_ids', [])
if project_ids:
    cmd.extend(['--project_ids'] + [str(p) for p in project_ids])
elif cfg.get('all_open', True):
    cmd.append('--all_open')

max_per_cam = cfg.get('max_images_per_camera')
if max_per_cam is not None:
    cmd.extend(['--max_images_per_camera', str(max_per_cam)])

samples_per_bin = cfg.get('samples_per_bin')
if samples_per_bin is not None:
    cmd.extend(['--samples_per_bin', str(samples_per_bin)])

tod_mode = cfg.get('tod_mode')
if tod_mode:
    cmd.extend(['--tod_mode', str(tod_mode)])

day_start = cfg.get('day_start')
if day_start is not None:
    cmd.extend(['--day_start', str(day_start)])

day_end = cfg.get('day_end')
if day_end is not None:
    cmd.extend(['--day_end', str(day_end)])

if cfg.get('include_blanks', False):
    cmd.append('--include_blanks')

taxa = cfg.get('taxa', [])
if taxa:
    cmd.extend(['--taxa'] + [str(t) for t in taxa])

output = cfg.get('output')
if output:
    cmd.extend(['--output', str(output)])

if cfg.get('no_csv', False):
    cmd.append('--no_csv')

if cfg.get('download_images', False):
    cmd.append('--download_images')
    image_dir = cfg.get('image_dir')
    if image_dir:
        cmd.extend(['--image_dir', str(image_dir)])

threads = cfg.get('threads')
if threads is not None:
    cmd.extend(['--threads', str(threads)])

# Forward any additional command-line arguments passed to this script
cmd.extend(sys.argv[1:])

print(f'Executing: {\" \".join(cmd)}\n')
ret = subprocess.run(cmd)
sys.exit(ret.returncode)
" "$@"
