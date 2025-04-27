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
    inference_per_batch_list = timing.get('inference_per_batch_time_list', [])
    tensor_to_device_per_batch_list = timing.get('moving_tensor_to_device_per_batch_time_list', [])
    bloom_filter_per_batch_list = timing.get('overflow_bloom_filter_per_batch_time_list', [])
    
    # Calculate means for each metric
    if latency_per_batch_list:
        avg_batch_latency = np.mean(latency_per_batch_list)
        p50_batch_latency = np.percentile(latency_per_batch_list, 50)
        p95_batch_latency = np.percentile(latency_per_batch_list, 95)
        p99_batch_latency = np.percentile(latency_per_batch_list, 99)
        max_batch_latency = np.max(latency_per_batch_list)
    else:
        avg_batch_latency = p50_batch_latency = p95_batch_latency = p99_batch_latency = max_batch_latency = 0
    
    # Calculate means for additional metrics
    avg_inference_latency = np.mean(inference_per_batch_list) if inference_per_batch_list else 0
    avg_tensor_to_device_latency = np.mean(tensor_to_device_per_batch_list) if tensor_to_device_per_batch_list else 0
    avg_bloom_filter_latency = np.mean(bloom_filter_per_batch_list) if bloom_filter_per_batch_list else 0
    
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
            'average_inference_latency_ms': avg_inference_latency,
            'average_tensor_to_device_latency_ms': avg_tensor_to_device_latency,
            'average_bloom_filter_latency_ms': avg_bloom_filter_latency,
            'p50_batch_latency_ms': p50_batch_latency,
            'p95_batch_latency_ms': p95_batch_latency,
            'p99_batch_latency_ms': p99_batch_latency,
            'max_batch_latency_ms': max_batch_latency,
        },
        'throughput': {
            'urls_per_second': throughput_per_batch,
        },
        'raw_metrics': {
            'total_latency': latency_per_batch_list,
            'inference_latency': inference_per_batch_list,
            'tensor_to_device_latency': tensor_to_device_per_batch_list,
            'bloom_filter_latency': bloom_filter_per_batch_list,
        }
    }
    
    return analysis

def plot_analysis_results(analysis_list, save_path=None, metrics_to_plot=None):
    """
    Create plots for throughputs and latency from multiple analysis dictionaries.
    
    Args:
        analysis_list: List of analysis dictionaries from analyze_results function
        save_path: Optional path to save the plot instead of displaying it
        metrics_to_plot: List of metrics to plot (e.g. ['total', 'inference', 'tensor', 'bloom'])
    """
    if not analysis_list:
        print("No analysis data provided for plotting")
        return
    
    if not metrics_to_plot:
        metrics_to_plot = ['total']
    
    # Extract data for plotting
    batch_sizes = [a['batch_size'] for a in analysis_list]
    throughputs = [a['throughput']['urls_per_second'] for a in analysis_list]
    
    # Always include throughput, so total number of plots is len(metrics_to_plot) + 1
    total_plots = len(metrics_to_plot) + 1
    
    # Two plots per row
    plots_per_row = 2
    rows = (total_plots + plots_per_row - 1) // plots_per_row  # Ceiling division
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 5 * rows))
    
    # Colors for different metrics
    colors = ['orange', 'green', 'blue', 'red', 'purple', 'brown', 'pink']
    
    # Plot each metric in its own subplot
    for i in range(total_plots):
        plt.subplot(rows, plots_per_row, i + 1)
        
        if i == 0:
            # First plot is always throughput
            plt.plot(batch_sizes, throughputs, 'o-', linewidth=2)
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (URLs/second)')
            plt.title('Batch Size vs. Throughput')
            plt.grid(True)
        else:
            # Other plots are latency metrics
            metric = metrics_to_plot[i - 1]
            
            if metric == 'total':
                latencies = [a['latency']['average_batch_latency_ms'] for a in analysis_list]
                title = 'Total Latency'
            elif metric == 'inference':
                latencies = [a['latency']['average_inference_latency_ms'] for a in analysis_list]
                title = 'Inference Latency'
            elif metric == 'tensor':
                latencies = [a['latency']['average_tensor_to_device_latency_ms'] for a in analysis_list]
                title = 'Tensor to Device Latency'
            elif metric == 'bloom':
                latencies = [a['latency']['average_bloom_filter_latency_ms'] for a in analysis_list]
                title = 'Bloom Filter Latency'
            else:
                continue
            
            plt.plot(batch_sizes, latencies, 'o-', linewidth=2, color=colors[(i - 1) % len(colors)])
            plt.xlabel('Batch Size')
            plt.ylabel('Average Latency (ms)')
            plt.title(f'Batch Size vs. {title}')
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
    parser.add_argument('-metrics', nargs='+', choices=['total', 'inference', 'tensor', 'bloom'], default=['total'],
                        help='Metrics to plot: total (batch total latency), inference (model inference), '
                             'tensor (tensor movement to device), bloom (bloom filter lookup)')
    
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
        metrics_suffix = '_'.join(args.metrics)
        save_path = os.path.join(args.dir_path, f"latency_throughput_graphs_{metrics_suffix}.png")
    
    # Check if we have any analyses to plot
    if not analyses:
        print("No files analyzed. Please provide a valid JSON file or directory path.")
        sys.exit(1)
    
    # Sort analyses by batch size for better visualization
    analyses.sort(key=lambda x: x['batch_size'])
    
    # Plot the results
    plot_analysis_results(analyses, save_path, args.metrics)

if __name__ == "__main__":
    main() 