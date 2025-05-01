#!/bin/bash

# Define dimension pairs using an associative array
declare -A dim_pairs=(
    [4]=2
    [8]=4
    [16]=8
    [64]=32
    [128]=64
    [256]=128
)

# Run for each dimension pair
for embedding_dim in "${!dim_pairs[@]}"; do
    hidden_dim="${dim_pairs[$embedding_dim]}"
    python train_and_run.py --embedding_dim "$embedding_dim" --hidden_dim "$hidden_dim"
done
