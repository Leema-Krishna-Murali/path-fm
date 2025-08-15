#!/bin/bash
BUCKET_NAME="tcga-omezarr"
MANIFEST_FILE="manifest.txt"

# --- Validation ---
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to continue."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    exit 1
fi

if [ -z "$R2_ENDPOINT_URL" ]; then
    echo "Error: R2_ENDPOINT_URL is not set. Please export it first."
    exit 1
fi

# --- Stage 1: Get all top-level prefixes ---
echo "Fetching the list of all top-level directories..."
PREFIXES=$(aws s3api list-objects-v2 \
    --bucket "$BUCKET_NAME" \
    --delimiter "/" \
    --query "CommonPrefixes[].Prefix" \
    --output text \
    --endpoint-url "$R2_ENDPOINT_URL")

if [ -z "$PREFIXES" ]; then
    echo "Error: Failed to fetch any top-level prefixes."
    exit 1
fi

echo "Found $(echo "$PREFIXES" | wc -w) directories to scan. Starting the main scan..."

# --- Stage 2: Loop through each prefix and list its contents ---
for prefix in $PREFIXES; do
    echo "Processing prefix: $prefix" >&2
    
    aws s3api list-objects-v2 \
        --bucket "$BUCKET_NAME" \
        --prefix "$prefix" \
        --output json \
        --endpoint-url "$R2_ENDPOINT_URL" | \
    # Use jq to extract only the 'Key' from the JSON
    jq -r '.Contents[].Key' | \
    # Now that we have clean keys, we can reliably filter and clean them
    grep '\.zarr/0/\.zarray$' | \
    sed 's|/0/\.zarray$||' | \
    awk -v bucket="$BUCKET_NAME" '{print "s3://" bucket "/" $0}'

done > "$MANIFEST_FILE"

echo "----------------------------------------------------"
echo "Manifest generation complete!"
echo "Total Zarr stores found: $(wc -l < $MANIFEST_FILE)"
echo "File is ready at: $MANIFEST_FILE"