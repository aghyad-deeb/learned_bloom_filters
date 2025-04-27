#!/bin/bash
# Script to run the interactive Learned Bloom Filter testing with multiple batch sizes
#
# Usage:
#   ./many_interactive.sh                 # Run with default batch sizes

cd "$(dirname "$0")"  # Navigate to the script directory

# Default parameters
SAMPLE_SIZE=1024
BATCH_SIZES=(16 32 64 128 256)  # Default batch sizes if none provided

echo "Running tests with sample size: $SAMPLE_SIZE"
echo "Testing batch sizes: ${BATCH_SIZES[*]}"

# Loop through each batch size
for batch_size in "${BATCH_SIZES[@]}"; do
  echo
  echo "========================================"
  echo "Testing with batch size: $batch_size"
  echo "========================================"
  
  python interactive.py --batch --batch-size "$batch_size" --sample-size "$SAMPLE_SIZE" 
  
done