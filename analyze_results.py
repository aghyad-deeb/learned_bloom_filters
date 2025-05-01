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
import re
from collections import defaultdict

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
    overflow_bloom_filter_per_batch_list = timing.get('overflow_bloom_filter_per_batch_time_list', [])
    
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
    avg_overflow_bloom_filter_latency = np.mean(overflow_bloom_filter_per_batch_list) if overflow_bloom_filter_per_batch_list else 0
    
    # Calculate throughput (URLs per second)
    if avg_batch_latency > 0:
        throughput_per_batch = (batch_size / avg_batch_latency) * 1000  # URLs per second
    else:
        throughput_per_batch = 0
    
    # Get size metrics if available
    size_metrics = results.get('size_metrics', {})
    model_size = size_metrics.get('model_size_bytes', 0)
    overflow_bloom_filter_size = size_metrics.get('overflow_bloom_filter_size_bytes', 0)
    traditional_bloom_filter_size = size_metrics.get('traditional_bloom_filter_size_bytes', 0)
    
    # Get traditional Bloom filter metrics if available
    traditional_bloom = results.get('traditional_bloom', {})
    traditional_timing = traditional_bloom.get('timing', {})
    avg_traditional_latency = traditional_timing.get('time_per_url', 0)
    traditional_throughput = traditional_timing.get('urls_per_second', 0)
    
    # Create analysis results dictionary
    analysis = {
        'batch_size': batch_size,
        'sample_size': sample_size,
        'num_successful_urls': num_successful_urls,
        'latency': {
            'average_batch_latency_ms': avg_batch_latency,
            'average_inference_latency_ms': avg_inference_latency,
            'average_tensor_to_device_latency_ms': avg_tensor_to_device_latency,
            'average_overflow_bloom_filter_latency_ms': avg_overflow_bloom_filter_latency,
            'average_traditional_bloom_latency_ms': avg_traditional_latency,
            'p50_batch_latency_ms': p50_batch_latency,
            'p95_batch_latency_ms': p95_batch_latency,
            'p99_batch_latency_ms': p99_batch_latency,
            'max_batch_latency_ms': max_batch_latency,
        },
        'throughput': {
            'urls_per_second': throughput_per_batch,
            'traditional_urls_per_second': traditional_throughput,
        },
        'size_metrics': {
            'model_size_bytes': model_size,
            'overflow_bloom_filter_size_bytes': overflow_bloom_filter_size,
            'traditional_bloom_filter_size_bytes': traditional_bloom_filter_size,
        },
        'raw_metrics': {
            'total_latency': latency_per_batch_list,
            'inference_latency': inference_per_batch_list,
            'tensor_to_device_latency': tensor_to_device_per_batch_list,
            'overflow_bloom_filter_latency': overflow_bloom_filter_per_batch_list,
        }
    }
    
    return analysis

def load_model_hyperparams(model_path):
    """
    Load hyperparameters for a model from its hyperparams.json file.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary with model hyperparameters
    """
    hyperparams_path = os.path.join(model_path, "hyperparams.json")
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            return json.load(f)
    return None

def process_model_results(model_path, metrics_to_plot=None):
    """
    Process results for a single model and generate its plots.
    
    Args:
        model_path: Path to the model directory
        metrics_to_plot: List of metrics to plot
        
    Returns:
        Dictionary with model analysis results
    """
    # Find all JSON result files in the model's inference_outputs directory
    outputs_dir = os.path.join(model_path, "inference_outputs")
    if not os.path.exists(outputs_dir):
        print(f"No inference outputs found for model at {model_path}")
        return None
    
    json_files = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(outputs_dir, f))]
    
    if not json_files:
        print(f"No JSON files found in {outputs_dir}")
        return None
    
    # Load hyperparameters
    hyperparams = load_model_hyperparams(model_path)
    if not hyperparams:
        print(f"No hyperparameters found for model at {model_path}")
        return None
    
    # Process each batch size result
    analyses = []
    for json_file in json_files:
        analysis = analyze_results(json_file)
        analyses.append(analysis)
    
    # Sort analyses by batch size
    analyses.sort(key=lambda x: x['batch_size'])
    
    # Generate individual model plots
    metrics_suffix = '_'.join(metrics_to_plot) if metrics_to_plot else 'total'
    save_path = os.path.join(outputs_dir, f"latency_throughput_graphs_{metrics_suffix}.png")
    plot_analysis_results(analyses, save_path, metrics_to_plot)
    
    # Return model analysis data for comparison
    return {
        'hyperparams': hyperparams,
        'analyses': analyses,
        'model_path': model_path
    }

def prepare_model_comparison_data(model_analyses):
    """
    Prepare data for model comparison plots.
    
    Args:
        model_analyses: List of model analysis results
        
    Returns:
        Dictionary with prepared comparison data
    """
    comparison_data = {
        'model_sizes': [],
        'overflow_bloom_sizes': [],
        'model_keys': [],
        'batch_sizes': set(),
        'latencies': defaultdict(dict),  # batch_size -> {model_key -> latency}
        'throughputs': defaultdict(dict),  # batch_size -> {model_key -> throughput}
        'avg_traditional_latency': 0,  # single average across all models and batch sizes
        'avg_traditional_throughput': 0,  # single average across all models and batch sizes
        'traditional_bloom_size': 0,  # size of traditional Bloom filter
        'hyperparams': {}  # model_key -> hyperparams
    }
    
    # Collect all traditional metrics to calculate overall average
    all_traditional_latencies = []
    all_traditional_throughputs = []
    
    # Get traditional Bloom filter size from first analysis (should be same for all)
    if model_analyses and model_analyses[0]['analyses']:
        first_analysis = model_analyses[0]['analyses'][0]
        comparison_data['traditional_bloom_size'] = first_analysis['size_metrics']['traditional_bloom_filter_size_bytes']
    
    for model_data in model_analyses:
        analyses = model_data['analyses']
        for analysis in analyses:
            all_traditional_latencies.append(analysis['latency']['average_traditional_bloom_latency_ms'])
            all_traditional_throughputs.append(analysis['throughput']['traditional_urls_per_second'])
    
    # Calculate overall averages
    comparison_data['avg_traditional_latency'] = np.mean(all_traditional_latencies)
    comparison_data['avg_traditional_throughput'] = np.mean(all_traditional_throughputs)
    
    # Collect model-specific data
    for model_data in model_analyses:
        hyperparams = model_data['hyperparams']
        analyses = model_data['analyses']
        model_path = model_data['model_path']
        
        # Create model key from hyperparameters
        embedding_dim = hyperparams.get('embedding_dim', 0)
        hidden_dim = hyperparams.get('hidden_dim', 0)
        model_key = f"emb{embedding_dim}_hid{hidden_dim}"
        
        # Store hyperparameters
        comparison_data['hyperparams'][model_key] = hyperparams
        
        # Process each batch size analysis
        for analysis in analyses:
            batch_size = analysis['batch_size']
            model_size = analysis['size_metrics']['model_size_bytes']
            overflow_bloom_size = analysis['size_metrics']['overflow_bloom_filter_size_bytes']
            
            # Store sizes if not already stored
            if model_key not in comparison_data['model_keys']:
                comparison_data['model_sizes'].append(model_size)
                comparison_data['overflow_bloom_sizes'].append(overflow_bloom_size)
                comparison_data['model_keys'].append(model_key)
            
            # Store metrics
            comparison_data['batch_sizes'].add(batch_size)
            comparison_data['latencies'][batch_size][model_key] = analysis['latency']['average_batch_latency_ms']
            comparison_data['throughputs'][batch_size][model_key] = analysis['throughput']['urls_per_second']
    
    return comparison_data

def plot_model_comparisons(comparison_data, save_path=None, show_traditional=True, log_scale=False):
    """
    Create plots comparing different models.
    
    Args:
        comparison_data: Dictionary with prepared comparison data
        save_path: Optional path to save the plots
        show_traditional: Whether to show traditional Bloom filter results
        log_scale: Whether to use logarithmic scale for model size axis
    """
    if not comparison_data['model_keys']:
        print("No model comparison data provided for plotting")
        return
    
    # Convert model sizes to kilobytes
    model_sizes_kb = [size / 1024 for size in comparison_data['model_sizes']]
    overflow_bloom_filter_sizes_kb = [size / 1024 for size in comparison_data['overflow_bloom_sizes']]
    traditional_bloom_size_kb = comparison_data['traditional_bloom_size'] / 1024
    
    # Create figure with subplots
    if len(comparison_data['model_keys']) > 2:  # Only check for outliers if we have enough models
        fig = plt.figure(figsize=(15, 15))  # Make figure taller to accommodate extra plot
        n_rows = 3  # Add an extra row for the zoomed plot
    else:
        fig = plt.figure(figsize=(15, 10))
        n_rows = 2
    
    # Plot 1: Model Size vs Overflow Bloom Filter Size
    plt.subplot(n_rows, 2, 1)
    plt.scatter(model_sizes_kb, overflow_bloom_filter_sizes_kb, c='blue', alpha=0.6)
    for i, model_key in enumerate(comparison_data['model_keys']):
        plt.annotate(model_key, (model_sizes_kb[i], overflow_bloom_filter_sizes_kb[i]))
    log_scale_str = ''
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        log_scale_str = '(log scale)'
    plt.xlabel(f'Model Size (KB) {log_scale_str}')
    plt.ylabel(f'Overflow Bloom Filter Size (KB) {log_scale_str}')
    plt.title('Model Size vs Overflow Bloom Filter Size')
    plt.grid(True)
    
    # Plot 2: Model Size vs Latency for different batch sizes
    plt.subplot(n_rows, 2, 2)
    for batch_size in sorted(comparison_data['batch_sizes']):
        batch_latencies = [comparison_data['latencies'][batch_size].get(key, 0) 
                          for key in comparison_data['model_keys']]
        plt.scatter(model_sizes_kb, batch_latencies, label=f'Learned (Batch {batch_size})')
    
    if show_traditional:
        # Add point for traditional Bloom filter with its size
        plt.scatter([traditional_bloom_size_kb], [comparison_data['avg_traditional_latency']],
                    c='red', marker='x', s=100, label='Traditional (Avg)')
    if log_scale:
        plt.xscale('log')
    plt.xlabel(f'Predictor Size (KB) {log_scale_str}')
    plt.ylabel('Average Latency (ms)')
    plt.title('Predictor Size vs Latency')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Model Size vs Throughput for different batch sizes
    plt.subplot(n_rows, 2, 3)
    for batch_size in sorted(comparison_data['batch_sizes']):
        batch_throughputs = [comparison_data['throughputs'][batch_size].get(key, 0) 
                           for key in comparison_data['model_keys']]
        plt.scatter(model_sizes_kb, batch_throughputs, label=f'Learned (Batch {batch_size})')
    
    if show_traditional:
        # Add point for traditional Bloom filter with its size
        plt.scatter([traditional_bloom_size_kb], [comparison_data['avg_traditional_throughput']],
                    c='red', marker='x', s=100, label='Traditional (Avg)')
    if log_scale:
        plt.xscale('log')
    plt.xlabel(f'Predictor Size (KB) {log_scale_str}')
    plt.ylabel('Throughput (URLs/second)')
    plt.title('Size vs Throughput')
    plt.legend()
    plt.grid(True)
    
    # Collect all latency and throughput values for each model
    model_avg_latencies = {}
    model_avg_throughputs = {}
    for model_key in comparison_data['model_keys']:
        latencies = [comparison_data['latencies'][bs].get(model_key, 0) 
                    for bs in sorted(comparison_data['batch_sizes'])]
        throughputs = [comparison_data['throughputs'][bs].get(model_key, 0) 
                      for bs in sorted(comparison_data['batch_sizes'])]
        model_avg_latencies[model_key] = np.mean(latencies)
        model_avg_throughputs[model_key] = np.mean(throughputs)
    
    # Plot 4: Full Latency vs Throughput
    plt.subplot(n_rows, 2, 4)
    for model_key in comparison_data['model_keys']:
        plt.scatter([model_avg_latencies[model_key]], [model_avg_throughputs[model_key]], 
                   label=f'{model_key} (Learned)')
    
    if show_traditional:
        plt.scatter([comparison_data['avg_traditional_latency']], 
                   [comparison_data['avg_traditional_throughput']],
                   c='red', marker='x', s=100, label='Traditional (Avg)')
    
    plt.xlabel('Average Latency (ms)')
    plt.ylabel('Throughput (URLs/second)')
    plt.title('Latency vs Throughput (All Points)')
    plt.legend()
    plt.grid(True)
    
    # If we have enough models, create a zoomed version excluding outliers
    if len(comparison_data['model_keys']) > 2:
        plt.subplot(n_rows, 2, (5, 6))  # Make the zoomed plot wider
        
        # Calculate quartiles for latency to identify outliers
        latencies = list(model_avg_latencies.values())
        q1_lat = np.percentile(latencies, 25)
        q3_lat = np.percentile(latencies, 75)
        iqr_lat = q3_lat - q1_lat
        # factor = 1.5
        factor = 0.8
        upper_bound_lat = q3_lat + factor * iqr_lat
        
        # Plot non-outlier points
        for model_key in comparison_data['model_keys']:
            lat = model_avg_latencies[model_key]
            if lat <= upper_bound_lat:  # Only plot if not an outlier
                plt.scatter([lat], [model_avg_throughputs[model_key]], 
                          label=f'{model_key} (Learned)')
        
        if show_traditional:
            trad_lat = comparison_data['avg_traditional_latency']
            if trad_lat <= upper_bound_lat:  # Only plot if not an outlier
                plt.scatter([trad_lat], [comparison_data['avg_traditional_throughput']],
                          c='red', marker='x', s=100, label='Traditional (Avg)')
        
        plt.xlabel('Average Latency (ms)')
        plt.ylabel('Throughput (URLs/second)')
        plt.title('Latency vs Throughput (Zoomed, Excluding Outliers)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        # Modify save path to indicate if it includes outliers
        base_path = save_path.rsplit('.', 1)[0]
        plt.savefig(f"{base_path}_with_zoom.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {base_path}_with_zoom.png")
    else:
        plt.show()

def plot_analysis_results(analysis_list, save_path=None, metrics_to_plot=None, show_traditional=True, log_scale=False):
    """
    Create plots for throughputs and latency from multiple analysis dictionaries.
    
    Args:
        analysis_list: List of analysis dictionaries from analyze_results function
        save_path: Optional path to save the plot instead of displaying it
        metrics_to_plot: List of metrics to plot (e.g. ['total', 'inference', 'tensor', 'bloom'])
        show_traditional: Whether to show traditional Bloom filter results
        log_scale: Whether to use logarithmic scale for model size axis
    """
    if not analysis_list:
        print("No analysis data provided for plotting")
        return
    
    if not metrics_to_plot:
        metrics_to_plot = ['total']
    
    # Extract data for plotting
    batch_sizes = [a['batch_size'] for a in analysis_list]
    throughputs = [a['throughput']['urls_per_second'] for a in analysis_list]
    traditional_throughputs = [a['throughput']['traditional_urls_per_second'] for a in analysis_list]
    
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
            plt.scatter(batch_sizes, throughputs, c='blue', alpha=0.6, label='Learned Model')
            if show_traditional:
                plt.scatter(batch_sizes, traditional_throughputs, c='red', alpha=0.6, label='Traditional Bloom')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (URLs/second)')
            plt.title('Batch Size vs. Throughput')
            plt.legend()
            plt.grid(True)
        else:
            # Other plots are latency metrics
            metric = metrics_to_plot[i - 1]
            
            if metric == 'total':
                latencies = [a['latency']['average_batch_latency_ms'] for a in analysis_list]
                traditional_latencies = [a['latency']['average_traditional_bloom_latency_ms'] for a in analysis_list] if show_traditional else None
                title = 'Total Latency'
            elif metric == 'inference':
                latencies = [a['latency']['average_inference_latency_ms'] for a in analysis_list]
                traditional_latencies = None
                title = 'Inference Latency'
            elif metric == 'tensor':
                latencies = [a['latency']['average_tensor_to_device_latency_ms'] for a in analysis_list]
                traditional_latencies = None
                title = 'Tensor to Device Latency'
            elif metric == 'bloom':
                latencies = [a['latency']['average_overflow_bloom_filter_latency_ms'] for a in analysis_list]
                traditional_latencies = None
                title = 'Overflow Bloom Filter Latency'
            else:
                continue
            
            plt.scatter(batch_sizes, latencies, c=colors[(i - 1) % len(colors)], alpha=0.6, label='Learned Model')
            if show_traditional and traditional_latencies:
                plt.scatter(batch_sizes, traditional_latencies, c='red', alpha=0.6, label='Traditional Bloom')
            plt.xlabel('Batch Size')
            plt.ylabel('Average Latency (ms)')
            plt.title(f'Batch Size vs. {title}')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze batch inference results from JSON files')
    parser.add_argument('-json_path', help='Path to the JSON file with batch inference results')
    parser.add_argument('-dir_path', help='Path to directory containing JSON files with batch inference results')
    parser.add_argument('-models_path', help='Path to directory containing multiple model directories')
    parser.add_argument('-metrics', nargs='+', choices=['total', 'inference', 'tensor', 'bloom'], default=['total'],
                        help='Metrics to plot: total (batch total latency), inference (model inference), '
                             'tensor (tensor movement to device), bloom (bloom filter lookup)')
    parser.add_argument('--traditional', action='store_true',
                        help='Show traditional Bloom filter results in plots')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use logarithmic scale for model size axis')
    
    args = parser.parse_args()
    
    if args.models_path:
        # Process multiple models
        if not os.path.isdir(args.models_path):
            print(f"Error: Models directory {args.models_path} does not exist")
            sys.exit(1)
        
        # Find all model directories
        model_dirs = [os.path.join(args.models_path, d) for d in os.listdir(args.models_path)
                     if os.path.isdir(os.path.join(args.models_path, d)) and d.startswith('run_')]
        
        if not model_dirs:
            print(f"No model directories found in {args.models_path}")
            sys.exit(1)
        
        print(f"Found {len(model_dirs)} model directories to analyze")
        
        # Process each model
        model_analyses = []
        for model_dir in model_dirs:
            print(f"\nProcessing model: {os.path.basename(model_dir)}")
            model_analysis = process_model_results(model_dir, args.metrics)
            if model_analysis:
                model_analyses.append(model_analysis)
        
        if model_analyses:
            # Prepare and plot model comparisons
            comparison_data = prepare_model_comparison_data(model_analyses)
            metrics_suffix = '_'.join(args.metrics)
            save_path = os.path.join(args.models_path, f"model_comparison_{metrics_suffix}.png")
            print(f"{args.log_scale=}")
            plot_model_comparisons(comparison_data, save_path, args.traditional, args.log_scale)
    
    elif args.json_path or args.dir_path:
        # Process single model results
        json_files = []
        
        if args.json_path:
            if not os.path.exists(args.json_path):
                print(f"Error: File {args.json_path} does not exist")
                sys.exit(1)
            json_files.append(args.json_path)
        
        if args.dir_path:
            if not os.path.isdir(args.dir_path):
                print(f"Error: Directory {args.dir_path} does not exist")
                sys.exit(1)
            
            dir_json_files = [os.path.join(args.dir_path, f) for f in os.listdir(args.dir_path) 
                            if f.endswith('.json') and os.path.isfile(os.path.join(args.dir_path, f))]
            
            if not dir_json_files:
                print(f"No JSON files found in directory {args.dir_path}")
                sys.exit(1)
            
            print(f"Found {len(dir_json_files)} JSON files to analyze")
            json_files.extend(dir_json_files)
        
        if not json_files:
            print("No files to analyze. Please provide a valid JSON file or directory path.")
            sys.exit(1)
        
        # Process each file
        analyses = []
        for json_file in json_files:
            analysis = analyze_results(json_file)
            analyses.append(analysis)
        
        # Sort analyses by batch size
        analyses.sort(key=lambda x: x['batch_size'])
        
        # Set save path for the plot
        metrics_suffix = '_'.join(args.metrics)
        save_path = os.path.join(os.path.dirname(json_files[0]), f"latency_throughput_graphs_{metrics_suffix}.png")
        
        # Plot the results
        plot_analysis_results(analyses, save_path, args.metrics, args.traditional, args.log_scale)
    
    else:
        print("Please provide either -models_path, -json_path, or -dir_path")
        sys.exit(1)

if __name__ == "__main__":
    main() 