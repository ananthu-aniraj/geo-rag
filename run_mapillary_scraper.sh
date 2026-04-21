#!/bin/bash

# Configuration
TOTAL_CHUNKS=10000
SCRIPT_NAME="mapillary_scraper.py"
BASE_DIR="/home/ananthu/DATA/data_ananthu/mapillary_scrape"

echo "Starting Mapillary grid search for $TOTAL_CHUNKS chunks..."

# Loop from 0 up to (TOTAL_CHUNKS - 1)
for (( i=0; i<$TOTAL_CHUNKS; i++ ))
do
    echo "========================================"
    echo "Processing chunk $i / $((TOTAL_CHUNKS - 1))"
    echo "========================================"
    
    # Run the Python script
    python3 "$SCRIPT_NAME" --chunk "$i" --total_chunks "$TOTAL_CHUNKS" --base_dir $BASE_DIR
    
    # Safety Check: If the Python script crashes completely, stop the Bash loop
    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: Script failed on chunk $i. Halting execution."
        exit 1
    fi
done

echo "All $TOTAL_CHUNKS chunks completed successfully!"