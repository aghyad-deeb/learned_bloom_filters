#!/bin/bash
# Script to run the interactive Learned Bloom Filter testing with multiple batch sizes
#
# Usage:
#   ./many_interactive.sh                 # Run with default batch sizes

cd "$(dirname "$0")"  # Navigate to the script directory

# Default parameters
SAMPLE_SIZE=1024
BATCH_SIZES=(16 32 64 128 256)  # Default batch sizes if none provided
RUNS=(run_20250427_115026_emb32_hid16 run_20250429_104423_emb4_hid2 run_20250429_120849_emb128_hid64 run_20250429_114727_emb64_hid32 run_20250429_123027_emb256_hid128 run_20250429_112526_emb16_hid8 run_20250429_110502_emb8_hid4)

echo "Running tests with sample size: $SAMPLE_SIZE"
echo "Testing batch sizes: ${BATCH_SIZES[*]}"

# Loop through each batch size
for run in "${RUNS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    echo
    echo "========================================"
    echo "Testing with run: $run and batch size: $batch_size"
    echo "========================================"

    python interactive.py --run "$run" --batch --batch-size "$batch_size" --sample-size "$SAMPLE_SIZE" 
    
  done
done