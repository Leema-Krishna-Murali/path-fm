#!/bin/bash
# --- Corrected Configuration ---
BUCKET_NAME="sophont"
# The path within the bucket you want to search. Note the trailing slash.
BASE_PREFIX="paul/data/omezarr-test/"
MANIFEST_FILE="manifest.txt"

# --- Validation ---
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to continue."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    exit 1
fi

if [ -z "$AWS_ENDPOINT_URL" ]; then
    echo "Error: AWS_ENDPOINT_URL is not set. Please export it first."
    exit 1
fi

# --- Stage 1: Get all top-level prefixes within the BASE_PREFIX ---
echo "Fetching the list of all directories within s3://${BUCKET_NAME}/${BASE_PREFIX}..."
PREFIXES=$(aws s3api list-objects-v2 \
    --bucket "$BUCKET_NAME" \
    --prefix "$BASE_PREFIX" \
    --delimiter "/" \
    --query "CommonPrefixes[].Prefix" \
    --output text \
    --endpoint-url "$AWS_ENDPOINT_URL")

if [ -z "$PREFIXES" ]; then
    echo "Error: Failed to fetch any prefixes within s3://${BUCKET_NAME}/${BASE_PREFIX}"
    exit 1
fi

echo "Found $(echo "$PREFIXES" | wc -w) directories to scan. Starting the main scan..."

# --- Stage 2: Loop through each prefix and list its contents ---
# Clear or create the manifest file before the loop
> "$MANIFEST_FILE"

for prefix in $PREFIXES; do
    echo "Processing prefix: $prefix" >&2
    
    aws s3api list-objects-v2 \
        --bucket "$BUCKET_NAME" \
        --prefix "$prefix" \
        --output json \
        --endpoint-url "$AWS_ENDPOINT_URL" | \
    # Use jq to extract only the 'Key' from the JSON
    jq -r '.Contents[].Key' | \
    # Now that we have clean keys, we can reliably filter and clean them
    grep '\.zarr/0/\.zarray$' | \
    sed 's|/0/\.zarray$||' | \
    # Append to the manifest file inside the loop
    awk -v bucket="$BUCKET_NAME" '{print "s3://" bucket "/" $0}' >> "$MANIFEST_FILE"

done

echo "----------------------------------------------------"
echo "Manifest generation complete!"
echo "Total Zarr stores found: $(wc -l < $MANIFEST_FILE)"
echo "File is ready at: $MANIFEST_FILE"