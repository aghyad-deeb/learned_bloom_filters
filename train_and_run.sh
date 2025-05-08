#!/bin/bash

# Define dimension pairs using an associative array
declare -A dim_pairs=(
    # [4]=2
    [8]=4
    # [16]=8
    # [64]=32
    # [128]=64
    # [256]=128
)

# Define data size pairs using an associative array
declare  data_sizes=(
    12500
    25000
    50000
    100000
    200000
    400000
    800000
)



# Run for each dimension pair
for embedding_dim in "${!dim_pairs[@]}"; do
    for data_size in "${data_sizes[@]}"; do
        hidden_dim="${dim_pairs[$embedding_dim]}"
        python train_and_run.py --embedding_dim "$embedding_dim" --hidden_dim "$hidden_dim" --data_cap "$data_size"
    done
done
