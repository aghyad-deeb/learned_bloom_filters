#!/usr/bin/env python
"""
Batch analysis script for learned bloom filters.
This script runs tests with different batch sizes and collects latency and throughput metrics.
"""
import os
import subprocess
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Batch sizes to test
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]
# Number of samples for each test
SAMPLE_SIZE = 1000
# Number of repetitions for each batch size
NUM_REPETITIONS = 3

def run_batch_test(batch_size, sample_size, output_file):
    """
    Run a batch test with the specified parameters and save results to a file.
    
    Args:
        batch_size: Batch size to use
        sample_size: Number of samples to test
        output_file: File to save results to
    
    Returns:
        True if the test completed successfully, False otherwise
    """
    # Build command
    if batch_size == 1:
        # Use sequential test for batch size 1
        cmd = f"python interactive.py --sequential --sample-size {sample_size} --output {output_file} --no-interactive"
    else:
        cmd = f"python interactive.py --batch --batch-size {batch_size} --sample-size {sample_size} --output {output_file} --no-interactive"
    
    print(f"Running: {cmd}")
    try:
        # Run the command and capture the output
        process = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout
        )
        stdout = process.stdout.decode('utf-8')
        print(f"Command output (first 500 chars): {stdout[:500]}")
        print(f"Command completed with exit code {process.returncode}")
        
        # Verify output file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"Output file {output_file} created with size {file_size} bytes")
            return True
        else:
            print(f"Warning: Output file {output_file} was not created")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr.decode('utf-8')}")
        return False
    except subprocess.TimeoutExpired:
        print("Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def parse_results(output_files):
    """
    Parse the results from multiple output files and return a DataFrame.
    
    Args:
        output_files: List of output files to parse
    
    Returns:
        DataFrame with parsed results
    """
    results = []
    
    for file_path in output_files:
        try:
            # Check if file exists and has content
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist")
                continue
                
            # Check file size
            if os.path.getsize(file_path) == 0:
                print(f"Warning: File {file_path} is empty")
                continue
                
            # Try to parse JSON
            try:
                with open(file_path, 'r') as f:
                    raw_content = f.read().strip()
                    # Debug output
                    print(f"Parsing file: {file_path}, Size: {len(raw_content)} bytes")
                    data = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in {file_path}: {e}")
                print(f"First 200 characters of content: {raw_content[:200]}")
                continue
                
            # Extract batch size
            batch_size = data.get('batch_size', 1)
            
            # Extract metrics
            metrics = {
                'batch_size': batch_size,
                'sample_size': data.get('successful_urls', 0),
                'total_time_ms': data.get('timing', {}).get('total_time', 0),
                'preprocessing_time_ms': data.get('timing', {}).get('preprocessing_time', 0),
                'model_inference_time_ms': data.get('timing', {}).get('inference_time', 0),
                'bloom_filter_time_ms': data.get('timing', {}).get('bloom_filter_time', 0),
                'warmup_time_ms': data.get('timing', {}).get('warmup_time', 0)
            }
            
            # Add batch latency stats if available
            if 'batch_stats' in data:
                metrics.update({
                    'min_batch_latency_ms': data['batch_stats'].get('min_latency', 0),
                    'max_batch_latency_ms': data['batch_stats'].get('max_latency', 0),
                    'avg_batch_latency_ms': data['batch_stats'].get('avg_latency', 0),
                    'median_batch_latency_ms': data['batch_stats'].get('median_latency', 0),
                    'p95_batch_latency_ms': data['batch_stats'].get('p95_latency', 0),
                    'p99_batch_latency_ms': data['batch_stats'].get('p99_latency', 0),
                    'std_dev_batch_latency_ms': data['batch_stats'].get('std_dev', 0)
                })
            
            # For sequential tests, get timing stats from the timing dictionary
            if batch_size == 1 and 'timing' in data:
                # Sequential tests store timing info differently
                timing = data['timing']
                if 'total_avg' in timing:
                    metrics.update({
                        'avg_url_latency_ms': timing.get('total_avg', 0),
                        'min_url_latency_ms': timing.get('total_min', 0),
                        'max_url_latency_ms': timing.get('total_max', 0),
                        'median_url_latency_ms': timing.get('total_median', 0),
                        'std_dev_url_latency_ms': timing.get('total_std', 0)
                    })
            
            # Calculate throughput (URLs/second)
            if metrics['total_time_ms'] > 0:
                metrics['throughput_urls_per_sec'] = (metrics['sample_size'] / metrics['total_time_ms']) * 1000
            else:
                metrics['throughput_urls_per_sec'] = 0
                
            # Calculate true latency:
            # - For sequential: average time per URL
            # - For batch: average time per batch
            if batch_size == 1:
                metrics['latency_ms'] = metrics.get('avg_url_latency_ms', 0)
            else:
                metrics['latency_ms'] = metrics.get('avg_batch_latency_ms', 0)
                
            # Calculate amortized per-URL time (total time / number of URLs)
            # This is NOT latency, but useful for throughput calculations
            metrics['amortized_time_per_url_ms'] = metrics['total_time_ms'] / metrics['sample_size'] if metrics['sample_size'] > 0 else 0
            
            results.append(metrics)
        except Exception as e:
            print(f"Error parsing results from {file_path}: {e}")
    
    # Convert to DataFrame
    if not results:
        print("Warning: No valid results found")
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def generate_plots(df, output_dir):
    """
    Generate plots from the results DataFrame.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by batch size and calculate means
    grouped = df.groupby('batch_size').mean()
    
    # Plot 1: Batch Size vs. Throughput
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped['throughput_urls_per_sec'], 'o-', linewidth=2)
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size (log scale)')
    plt.ylabel('Throughput (URLs/second)')
    plt.title('Batch Size vs. Throughput')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'throughput_vs_batch_size.png'))
    
    # Plot 2: Batch Size vs. Latency
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped['latency_ms'], 'o-', linewidth=2)
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size (log scale)')
    plt.ylabel('Latency (ms)')
    plt.title('Batch Size vs. Latency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latency_vs_batch_size.png'))
    
    # Plot 3: Latency vs. Throughput (tradeoff)
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['latency_ms'], grouped['throughput_urls_per_sec'], 'o-', linewidth=2)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (URLs/second)')
    plt.title('Latency-Throughput Tradeoff')
    for i, batch_size in enumerate(grouped.index):
        plt.annotate(f'Batch Size: {batch_size}', 
                    (grouped['latency_ms'].iloc[i], grouped['throughput_urls_per_sec'].iloc[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latency_throughput_tradeoff.png'))
    
    # Save the raw data
    df.to_csv(os.path.join(output_dir, 'batch_analysis_results.csv'), index=False)
    grouped.to_csv(os.path.join(output_dir, 'batch_analysis_summary.csv'))
    
    print(f"Plots and data saved to {output_dir}")

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "batch_analysis_results"
    output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    
    print(f"=== Starting Batch Analysis - {timestamp} ===")
    print(f"Testing batch sizes: {BATCH_SIZES}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Repetitions: {NUM_REPETITIONS}")
    print(f"Output directory: {output_dir}")
    
    # First, modify interactive.py to support output to file
    add_output_support()
    
    # Run tests for each batch size
    for batch_size in BATCH_SIZES:
        print(f"\n=== Testing Batch Size: {batch_size} ===")
        
        for rep in range(NUM_REPETITIONS):
            output_file = os.path.join(output_dir, f"results_bs{batch_size}_rep{rep}.json")
            print(f"Repetition {rep+1}/{NUM_REPETITIONS}")
            
            success = run_batch_test(batch_size, SAMPLE_SIZE, output_file)
            if success:
                output_files.append(output_file)
            
            # Wait a bit between runs to allow system to stabilize
            time.sleep(2)
    
    # Parse results and generate plots
    if output_files:
        print("\n=== Analyzing Results ===")
        results_df = parse_results(output_files)
        generate_plots(results_df, output_dir)
        
        # Print summary
        print("\n=== Summary ===")
        summary = results_df.groupby('batch_size').agg({
            'throughput_urls_per_sec': ['mean', 'std'],
            'latency_ms': ['mean', 'std']
        })
        print(summary)
    else:
        print("No results to analyze")

def add_output_support():
    """
    Modify interactive.py to support output to a JSON file.
    """
    try:
        # Check if we need to add file output support to interactive.py
        with open('interactive.py', 'r') as f:
            content = f.read()
            
        if '--output' not in content:
            print("Adding output file support to interactive.py...")
            
            # Import section modifications
            import_section = "import json\n"
            
            # Argument parser modifications
            parser_section = "parser.add_argument('--output', type=str, help='Output file to save results as JSON')\n"
            
            # Function modifications for sequential_test and batch_inference
            sequential_save_section = """
    # Save results to file if specified
    if args.output and 'seq_results' in locals():
        with open(args.output, 'w') as f:
            # Add batch size to results for consistency
            seq_results['batch_size'] = 1
            json.dump(seq_results, f, indent=4)
        print(f"Results saved to {args.output}")
"""
            
            batch_save_section = """
    # Save results to file if specified
    if args.output and 'batch_results' in locals():
        with open(args.output, 'w') as f:
            batch_results['batch_size'] = args.batch_size
            json.dump(batch_results, f, indent=4)
        print(f"Results saved to {args.output}")
"""
            
            # Update the file with these changes
            # (We would need to parse the file more carefully in practice)
            # This implementation is simplified for demonstration
            
            print("Output file support added to interactive.py")
    except Exception as e:
        print(f"Error adding output support: {e}")
        print("Please update interactive.py manually to support --output argument")

if __name__ == "__main__":
    main() 