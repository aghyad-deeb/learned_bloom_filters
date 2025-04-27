#!/bin/bash
# Script to run the interactive Learned Bloom Filter testing environment
#
# Usage:
#   ./interactive.sh                      # Interactive mode with latest run
#   ./interactive.sh --run run_20240501_123456_emb8_hid4  # Specific run
#
#   # Sequential testing (one URL at a time):
#   ./interactive.sh --sequential         # Sequential testing with 100 samples
#   ./interactive.sh --sequential --sample-size 500 # Test 500 URLs sequentially
#   ./interactive.sh --sequential --verbose # Show details for each URL
#
#   # Batched inference (true parallelized model execution):
#   ./interactive.sh --batch              # Batched inference with default settings
#   ./interactive.sh --batch --batch-size 128  # Use larger batches of 128
#   ./interactive.sh --batch --sample-size 1000  # Test 1000 samples
#   ./interactive.sh --batch --batch-size 32 --sample-size 500  # Custom batch and sample size

cd "$(dirname "$0")"  # Navigate to the script directory
python interactive.py "$@"  # Pass all command line arguments to the script 