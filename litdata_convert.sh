#!/bin/bash

# Script to convert TCGA dataset to LitData format and integrate with training

# Set paths
TCGA_DIR="/data/TCGA"
LITDATA_DIR="/data/litTCGA"
BADDATA_FILE="baddata.txt"

# Number of tiles per magnification level
# Adjust based on your needs and available storage
TILES_PER_MAG=100

# Number of workers for parallel processing
NUM_WORKERS=8

echo "========================================="
echo "TCGA to LitData Conversion Script"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Input directory: $TCGA_DIR"
echo "  Output directory: $LITDATA_DIR"
echo "  Tiles per magnification: $TILES_PER_MAG"
echo "  Number of workers: $NUM_WORKERS"
echo "  Exclusion file: $BADDATA_FILE"
echo ""

# Check if input directory exists
if [ ! -d "$TCGA_DIR" ]; then
    echo "Error: Input directory $TCGA_DIR does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$LITDATA_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi