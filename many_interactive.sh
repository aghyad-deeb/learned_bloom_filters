#!/bin/bash
# Script to run the interactive Learned Bloom Filter testing with multiple batch sizes
#
# Usage:
#   ./many_interactive.sh                 # Run with default batch sizes

cd "$(dirname "$0")"  # Navigate to the script directory

# Default parameters
SAMPLE_SIZE=4096
BATCH_SIZES=(16 32 64 128 256 512 1024 2048 4096)  # Default batch sizes if none provided
RUNS=(
run_20250508_120450_data12500_emb8_hid4
run_20250508_120541_data25000_emb8_hid4
run_20250508_120659_data50000_emb8_hid4
run_20250508_120926_data100000_emb8_hid4
run_20250508_121409_data200000_emb8_hid4
run_20250508_122324_data400000_emb8_hid4
run_20250508_124223_data800000_emb8_hid4
run_20250508_131809_data1600000_emb8_hid4
run_20250508_142636_data3200000_emb8_hid4
run_20250508_163931_data12500_emb16_hid8
run_20250508_164014_data25000_emb16_hid8
run_20250508_164202_data12500_emb4_hid2
run_20250508_164245_data50000_emb4_hid2
run_20250508_164458_data200000_emb4_hid2
run_20250508_165328_data800000_emb4_hid2
run_20250508_172939_data3200000_emb4_hid2
run_20250508_194526_data12500_emb16_hid8
run_20250508_194607_data50000_emb16_hid8
run_20250508_194818_data200000_emb16_hid8
run_20250508_195626_data800000_emb16_hid8
run_20250508_202851_data3200000_emb16_hid8
run_20250508_205728_data12500_emb32_hid16
run_20250508_205815_data50000_emb32_hid16
run_20250508_210034_data200000_emb32_hid16
run_20250508_210855_data800000_emb32_hid16
run_20250508_214359_data12500_emb64_hid32
run_20250508_214446_data50000_emb64_hid32
run_20250508_214728_data200000_emb64_hid32
run_20250508_215724_data800000_emb64_hid32
run_20250508_225519_data3200000_emb16_hid8
run_20250509_012951_data3200000_emb32_hid16
run_20250509_033949_data3200000_emb64_hid32
)
# RUN_ON_CPU=false
RUN_ON_CPU=true

echo "Running tests with sample size: $SAMPLE_SIZE"
echo "Testing batch sizes: ${BATCH_SIZES[*]}"

# Loop through each batch size
for run in "${RUNS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    echo
    echo "========================================"
    echo "Testing with run: $run and batch size: $batch_size"
    echo "========================================"

    if [ "$RUN_ON_CPU" = true ]; then
      echo "======= RUNNING ON CPU ======="
      python interactive.py --run "$run" --batch --batch-size "$batch_size" --sample-size "$SAMPLE_SIZE" --cpu
    else
      echo "======= Mapping Device Automatically ======="
      python interactive.py --run "$run" --batch --batch-size "$batch_size" --sample-size "$SAMPLE_SIZE"
    fi
    
  done
done