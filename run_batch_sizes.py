#!/usr/bin/env python
"""
Script to run train_and_run.py with multiple batch sizes for comparison.
This allows evaluating how different batch sizes affect training performance.
"""
import os
import subprocess
import time
import json
from datetime import datetime

# List of batch sizes to test
BATCH_SIZES = [16, 32, 64, 128, 256]

# Create a results directory
results_dir = os.path.join("results", f"batch_size_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(results_dir, exist_ok=True)

# Log file to track all results
log_path = os.path.join(results_dir, "comparison_results.json")
results = []

print(f"Running training with {len(BATCH_SIZES)} different batch sizes: {BATCH_SIZES}")
print(f"Results will be saved to: {results_dir}")

for batch_size in BATCH_SIZES:
    print(f"\n{'='*60}")
    print(f"Training with batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Start timer
    start_time = time.time()
    
    # Run train_and_run.py with the current batch size
    # Use environment variable to pass the batch size
    env = os.environ.copy()
    env["BATCH_SIZE"] = str(batch_size)
    
    try:
        subprocess.run(
            ["python", "train_and_run.py", "--batch-size", str(batch_size)], 
            check=True
        )
        status = "success"
    except subprocess.CalledProcessError as e:
        print(f"Error running with batch size {batch_size}: {e}")
        status = "error"
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Collect results
    result = {
        "batch_size": batch_size,
        "status": status,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Try to get the latest run directory to extract model performance metrics
    try:
        latest_run_path = os.path.join("models", "latest_run")
        if os.path.exists(latest_run_path) and os.path.islink(latest_run_path):
            actual_path = os.readlink(latest_run_path)
            result["run_dir"] = actual_path
            
            # Try to read hyperparams
            hyperparams_path = os.path.join(actual_path, "hyperparams.json")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'r') as f:
                    hyperparams = json.load(f)
                    result["hyperparams"] = hyperparams
    except Exception as e:
        print(f"Error collecting run information: {e}")
    
    results.append(result)
    
    # Save updated results after each run
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed batch size {batch_size} in {elapsed_time:.2f} seconds")

print("\nAll training runs completed!")
print(f"Results saved to: {log_path}")

# Print summary
print("\nSummary:")
print("-" * 50)
print(f"{'Batch Size':<10} | {'Status':<10} | {'Time (s)':<10}")
print("-" * 50)
for result in results:
    print(f"{result['batch_size']:<10} | {result['status']:<10} | {result['elapsed_time']:<10.2f}") 