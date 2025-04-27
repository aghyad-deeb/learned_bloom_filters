#!/usr/bin/env python3
"""
Analyze batch inference results from JSON files.
This script calculates latency and throughput metrics from saved batch inference results.
"""
import argparse
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def analyze_results(json_path):
    """
    Analyze batch inference results from a JSON file.
    
    Args:
        json_path: Path to the JSON file with batch inference results
        
    Returns:
        Dictionary with analyzed metrics
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Extract key metrics
    batch_size = results.get('batch_size', 0)
    sample_size = results.get('sample_size', 0)
    num_successful_urls = results.get('num_successful_urls', 0)
    
    # Get timing information
    timing = results.get('timing', {})
    
    # Calculate latency statistics
    latency_per_batch_list = timing.get('batch_total_latency_per_batch_time_list', [])
    if latency_per_batch_list:
        avg_batch_latency = np.mean(latency_per_batch_list)
        p50_batch_latency = np.percentile(latency_per_batch_list, 50)
        p95_batch_latency = np.percentile(latency_per_batch_list, 95)
        p99_batch_latency = np.percentile(latency_per_batch_list, 99)
        max_batch_latency = np.max(latency_per_batch_list)
    else:
        avg_batch_latency = p50_batch_latency = p95_batch_latency = p99_batch_latency = max_batch_latency = 0
    
    # Calculate throughput (URLs per second)
    if avg_batch_latency > 0:
        throughput_per_batch = (batch_size / avg_batch_latency) * 1000  # URLs per second
    else:
        throughput_per_batch = 0
    
    # Create analysis results dictionary
    analysis = {
        'batch_size': batch_size,
        'sample_size': sample_size,
        'num_successful_urls': num_successful_urls,
        'latency': {
            'average_batch_latency_ms': avg_batch_latency,
        },
        'throughput': {
            'urls_per_second': throughput_per_batch,
        },
    }
    
    return analysis

def plot_analysis_results(analysis_list, save_path=None):
    """
    Create plots for throughputs and latency from multiple analysis dictionaries.
    
    Args:
        analysis_list: List of analysis dictionaries from analyze_results function
        save_path: Optional path to save the plot instead of displaying it
    """
    if not analysis_list:
        print("No analysis data provided for plotting")
        return
    
    # Extract data for plotting
    batch_sizes = [a['batch_size'] for a in analysis_list]
    throughputs = [a['throughput']['urls_per_second'] for a in analysis_list]
    latencies = [a['latency']['average_batch_latency_ms'] for a in analysis_list]
    
    # Create throughput plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, throughputs, 'o-', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (URLs/second)')
    plt.title('Batch Size vs. Throughput')
    plt.grid(True)
    
    # Create latency plot
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, latencies, 'o-', linewidth=2, color='orange')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Latency (ms)')
    plt.title('Batch Size vs. Latency')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze batch inference results from JSON files')
    parser.add_argument('-json_path', help='Path to the JSON file with batch inference results')
    parser.add_argument('-dir_path', help='Path to directory containing JSON files with batch inference results')
    
    args = parser.parse_args()
    
    analyses = []
    save_path = None
    
    # Process single JSON file
    if args.json_path:
        if not os.path.exists(args.json_path):
            print(f"Error: File {args.json_path} does not exist")
            sys.exit(1)
        
        # Analyze the results
        analysis = analyze_results(args.json_path)
        analyses.append(analysis)
    
    # Process directory of JSON files
    if args.dir_path:
        if not os.path.isdir(args.dir_path):
            print(f"Error: Directory {args.dir_path} does not exist")
            sys.exit(1)
        
        json_files = [os.path.join(args.dir_path, f) for f in os.listdir(args.dir_path) 
                     if f.endswith('.json') and os.path.isfile(os.path.join(args.dir_path, f))]
        
        if not json_files:
            print(f"No JSON files found in directory {args.dir_path}")
            sys.exit(1)
        
        print(f"Found {len(json_files)} JSON files to analyze")
        
        for json_file in json_files:
            analysis = analyze_results(json_file)
            analyses.append(analysis)
            
        # Set save path for the plot
        save_path = os.path.join(args.dir_path, "latency_throughput_graphs.png")
    
    # Check if we have any analyses to plot
    if not analyses:
        print("No files analyzed. Please provide a valid JSON file or directory path.")
        sys.exit(1)
    
    # Sort analyses by batch size for better visualization
    analyses.sort(key=lambda x: x['batch_size'])
    
    # Plot the results
    plot_analysis_results(analyses, save_path)

if __name__ == "__main__":
    main() 