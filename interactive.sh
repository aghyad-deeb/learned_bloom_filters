#!/bin/bash
# Script to run the interactive Learned Bloom Filter testing environment
# Usage: ./interactive.sh [--run <run_directory>]

cd "$(dirname "$0")"  # Navigate to the script directory
python interactive.py "$@"  # Pass all command line arguments to the script 