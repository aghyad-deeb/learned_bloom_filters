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

def process_model_results(model_path, metrics_to_plot=None, results_dir="inference_outputs", cpu=False, show_traditional=True, log_scale=False, show_throughput=True):
    """
    Process results for a single model and generate its plots.
    
    Args:
        model_path: Path to the model directory
        metrics_to_plot: List of metrics to plot
        results_dir: Directory containing inference results
        cpu: Whether to use CPU results
        show_traditional: Whether to show traditional Bloom filter results
        log_scale: Whether to use logarithmic scale for model size axis
        show_throughput: Whether to show throughput plot
        
    Returns:
        Dictionary with model analysis results
    """
    # Find all JSON result files in the model's inference_outputs directory
    outputs_dir = os.path.join(model_path, results_dir)
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
    save_name = f"latency_throughput_graphs_{metrics_suffix}_cpu.png" if cpu else f"latency_throughput_graphs_{metrics_suffix}.png"
    save_path = os.path.join(outputs_dir, save_name)
    plot_single_model_results(analyses, save_path, metrics_to_plot, model_path=model_path, show_traditional=show_traditional, show_throughput=show_throughput, cpu=cpu)
    
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
        'data_caps': set(),  # New: track dataset caps
        'batch_sizes': set(),
        'latencies': defaultdict(dict),  # batch_size -> {model_key -> latency}
        'throughputs': defaultdict(dict),  # batch_size -> {model_key -> throughput}
        'traditional_bloom_metrics': {},  # data_cap -> {latency, throughput, size}
        'hyperparams': {},  # model_key -> hyperparams
        'model_data_keys': [],  # New: model+data keys for combined identification
        'model_dims': [],  # Store (embedding_dim, hidden_dim) tuples instead of parameter counts
    }
    
    # Collect traditional metrics per data cap
    traditional_metrics_by_cap = defaultdict(lambda: {'latencies': [], 'throughputs': [], 'sizes': []})
    
    # Process each model to collect data cap specific traditional Bloom filter metrics
    for model_data in model_analyses:
        hyperparams = model_data['hyperparams']
        analyses = model_data['analyses']
        data_cap = hyperparams.get('data_cap', 0)
        
        # Add data cap to set of data caps
        comparison_data['data_caps'].add(data_cap)
        
        # Collect traditional Bloom filter metrics for this data cap
        for analysis in analyses:
            traditional_metrics_by_cap[data_cap]['latencies'].append(
                analysis['latency']['average_traditional_bloom_latency_ms'])
            traditional_metrics_by_cap[data_cap]['throughputs'].append(
                analysis['throughput']['traditional_urls_per_second'])
            traditional_metrics_by_cap[data_cap]['sizes'].append(
                analysis['size_metrics']['traditional_bloom_filter_size_bytes'])
    
    # Calculate averages per data cap
    for data_cap, metrics in traditional_metrics_by_cap.items():
        comparison_data['traditional_bloom_metrics'][data_cap] = {
            'avg_latency': np.mean(metrics['latencies']),
            'avg_throughput': np.mean(metrics['throughputs']),
            'avg_size': np.mean(metrics['sizes'])
        }
    
    # Collect model-specific data
    for model_data in model_analyses:
        hyperparams = model_data['hyperparams']
        analyses = model_data['analyses']
        model_path = model_data['model_path']
        
        # Create model key from hyperparameters
        embedding_dim = hyperparams.get('embedding_dim', 0)
        hidden_dim = hyperparams.get('hidden_dim', 0)
        data_cap = hyperparams.get('data_cap', 0)  # New: get data cap
        
        # Store dimensions as a tuple instead of calculating parameters
        model_dims = (embedding_dim, hidden_dim)
        
        model_key = f"emb{embedding_dim}_hid{hidden_dim}"
        model_data_key = f"{model_key}_data{data_cap}"  # Combined key with data cap
        
        # Store hyperparameters
        comparison_data['hyperparams'][model_data_key] = hyperparams
        
        # Process each batch size analysis
        for analysis in analyses:
            batch_size = analysis['batch_size']
            model_size = analysis['size_metrics']['model_size_bytes']
            overflow_bloom_size = analysis['size_metrics']['overflow_bloom_filter_size_bytes']
            
            # Store sizes if not already stored for this model+data combination
            if model_data_key not in comparison_data['model_data_keys']:
                comparison_data['model_sizes'].append(model_size)
                comparison_data['overflow_bloom_sizes'].append(overflow_bloom_size)
                comparison_data['model_keys'].append(model_key)
                comparison_data['model_data_keys'].append(model_data_key)
                comparison_data['model_dims'].append(model_dims)  # Store dimensions tuple
            
            # Store metrics - now indexed by model_data_key to track both model and data cap
            comparison_data['batch_sizes'].add(batch_size)
            comparison_data['latencies'][batch_size][model_data_key] = analysis['latency']['average_batch_latency_ms']
            comparison_data['throughputs'][batch_size][model_data_key] = analysis['throughput']['urls_per_second']
    
    return comparison_data

def plot_model_comparisons(comparison_data, save_path=None, show_traditional=True, log_scale=False, plot_types=None, cpu=False):
    """
    Create plots comparing different models.
    
    Args:
        comparison_data: Dictionary with prepared comparison data
        save_path: Optional path to save the plots
        show_traditional: Whether to show traditional Bloom filter results
        log_scale: Whether to use logarithmic scale for model size axis
        plot_types: List of plot types to generate (None means all)
    """
    if not comparison_data['model_data_keys']:
        print("No model comparison data provided for plotting")
        return
    
    # If plot_types is None, include all plots
    if plot_types is None:
        plot_types = ["predictor_size", "model_vs_bloom", "datacap", "efficiency", "heatmap", 
                      "min_bloom", "best_latency", "best_model_latency_throughput", "size_vs_datacap",
                      "model_vs_predictor", "size_vs_throughput"]
    
    # Convert sizes to kilobytes
    model_sizes_kb = [size / 1024 for size in comparison_data['model_sizes']]
    overflow_bloom_filter_sizes_kb = [size / 1024 for size in comparison_data['overflow_bloom_sizes']]
    
    # Find the largest data cap to use for overall comparisons
    largest_data_cap = max(comparison_data['data_caps'])
    
    # Traditional bloom filter metrics for the largest data cap 
    # (for plots that don't specifically handle different data caps)
    if show_traditional and largest_data_cap in comparison_data['traditional_bloom_metrics']:
        traditional_metrics = comparison_data['traditional_bloom_metrics'][largest_data_cap]
        traditional_bloom_size_kb = traditional_metrics['avg_size'] / 1024
        traditional_bloom_latency = traditional_metrics['avg_latency']
        traditional_bloom_throughput = traditional_metrics['avg_throughput']
    else:
        # Default values if not available
        traditional_bloom_size_kb = 0
        traditional_bloom_latency = 0
        traditional_bloom_throughput = 0
    
    # Calculate total predictor sizes (model + overflow Bloom filter)
    predictor_sizes_kb = [model + bloom for model, bloom in zip(model_sizes_kb, overflow_bloom_filter_sizes_kb)]
    
    # Basic figure setup - determine number of plots
    num_plots = sum([pt in plot_types for pt in ["predictor_size", "datacap", "efficiency", "heatmap", 
                                              "min_bloom", "best_latency", "best_model_latency_throughput",
                                              "size_vs_datacap", "model_vs_predictor", "size_vs_throughput"]])
    
    if num_plots == 0:
        print("No valid plot types specified")
        return
    
    # Organize model keys by unique architectures (ignoring data caps)
    unique_model_keys = list(set(comparison_data['model_keys']))
    all_model_data_keys = comparison_data['model_data_keys']
    
    # Set up Matplotlib style for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a color palette that's more visually distinct
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    figsize = (4.5, 3.5)
    
    # Create figure with appropriate number of subplots
    if "heatmap" in plot_types:
        # Heatmap needs special layout
        # fig = plt.figure(figsize=(15, 15))
        n_rows = (num_plots // 2) + (1 if num_plots % 2 != 0 else 0)
        current_plot = 1
    else:
        # fig = plt.figure(figsize=(15, 15))
        n_rows = (num_plots // 2) + (1 if num_plots % 2 != 0 else 0)
        current_plot = 1
    
    log_scale_str = ' (log scale)' if log_scale else ''
    
    # Generating plots based on requested types
    
    # NEW PLOT: Predictor Size vs Data Cap
    if "size_vs_datacap" in plot_types:
        # plt.subplot(n_rows, 2, current_plot)
        # current_plot += 1
        plt.figure(figsize=figsize)
        
        # Find the smallest model for each data cap
        smallest_models_by_datacap = {}  # data_cap -> (model_data_key, predictor_size, latency)
        
        # Find the model with smallest predictor size for each data cap
        for i, model_data_key in enumerate(all_model_data_keys):
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            predictor_size = predictor_sizes_kb[i]
            
            # Find average latency across all batch sizes (for reference)
            all_latencies = []
            for batch_size in comparison_data['batch_sizes']:
                if model_data_key in comparison_data['latencies'][batch_size]:
                    all_latencies.append(comparison_data['latencies'][batch_size][model_data_key])
            
            avg_latency = np.mean(all_latencies) if all_latencies else float('inf')
            
            # Check if this is the smallest model for this data cap
            if data_cap not in smallest_models_by_datacap or predictor_size < smallest_models_by_datacap[data_cap][1]:
                smallest_models_by_datacap[data_cap] = (model_data_key, predictor_size, avg_latency)
        
        # Extract data for plotting learned models
        data_caps_learned = []
        predictor_sizes_learned = []
        labels_learned = []
        
        for data_cap, (model_key, size, _) in sorted(smallest_models_by_datacap.items()):
            data_caps_learned.append(data_cap)
            predictor_sizes_learned.append(size)
            labels_learned.append(model_key)
        
        # Get traditional Bloom filter sizes for each data cap
        trad_data_caps = []
        trad_sizes = []
        
        if show_traditional:
            for data_cap, metrics in sorted(comparison_data['traditional_bloom_metrics'].items()):
                trad_data_caps.append(data_cap)
                trad_sizes.append(metrics['avg_size'] / 1024)  # Convert to KB
        
        # Create more elegant plot with better styling
        # plt.figure(fig.number)  # Make sure we're using the current figure
        
        # Plot learned models with nicer aesthetics
        plt.scatter(data_caps_learned, predictor_sizes_learned, s=70, color=colors[0], 
                   edgecolor='white', linewidth=1, label='Learned (Smallest Size)', zorder=3)
        plt.plot(data_caps_learned, predictor_sizes_learned, '-', color=colors[0], 
                alpha=0.8, linewidth=2, zorder=2)
        
        # Add annotations for learned model sizes
        for i, (x, y) in enumerate(zip(data_caps_learned, predictor_sizes_learned)):
            if i % 2 == 1:
                continue
            plt.annotate(f"{y:.1f} KB", 
                        xy=(x, y),
                        xytext=(5, 5),  # Small offset
                        textcoords='offset points',
                        fontsize=8,
                        color=colors[0],
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
                
                
        # Add subtle grid
        plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        
        # Show traditional Bloom filter with a different style
        if trad_data_caps:
            plt.scatter(trad_data_caps, trad_sizes, s=70, color='red', marker='x', 
                      linewidth=2, label='Traditional', zorder=3)
            plt.plot(trad_data_caps, trad_sizes, '-', color='red', 
                   alpha=0.8, linewidth=2, zorder=2)
            
            # Add annotations for first and last traditional points
            # First point
            plt.annotate(f"{trad_sizes[0]:.1f} KB", 
                        xy=(trad_data_caps[0], trad_sizes[0]),
                        xytext=(5, -15),  # Offset below point
                        textcoords='offset points',
                        fontsize=8,
                        color='red',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            
            # Last point
            plt.annotate(f"{trad_sizes[-1]:.1f} KB", 
                        xy=(trad_data_caps[-1], trad_sizes[-1]),
                        xytext=(5, -15),  # Offset below point
                        textcoords='offset points',
                        fontsize=8,
                        color='red',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            
            # Add a combined line+marker to legend instead of just marker
            plt.plot([], [], 'rx-', linewidth=2, label='_Traditional')  # Hidden duplicate for legend style
        
        # # Add annotations for key points
        # for i, (cap, size, label) in enumerate(zip(data_caps_learned, predictor_sizes_learned, labels_learned)):
        #     # Only annotate selected points to avoid clutter (first, last, and middle points)
        #     if i == 0 or i == len(data_caps_learned)-1 or i == len(data_caps_learned)//2:
        #         plt.annotate(label.split('_')[0], (cap, size), 
        #                    fontsize=9, ha='center', va='bottom', 
        #                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Set log scales if requested
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            # Configure grid to match log scale ticks
            plt.grid(True, which='major', alpha=0.5, linestyle='-', zorder=1)
            plt.grid(True, which='minor', alpha=0.2, linestyle='--', zorder=1)
        else:
            # Standard grid for linear scale
            plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        
        # Add better labels and title
        plt.xlabel(f'Dataset Size{log_scale_str}', fontsize=10, labelpad=10)
        plt.ylabel(f'Predictor Size (KB){log_scale_str}', fontsize=10, labelpad=10)
        title = 'data_size_vs_memory'
        # plt.title(title, fontsize=12, pad=10)
        
        # Add a nice legend with better positioning
        legend = plt.legend(loc='best', frameon=True, fontsize=9)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('grey')
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Predictor Size vs Latency for different batch sizes (simplified version)
    if "predictor_size" in plot_types:
        # plt.subplot(n_rows, 2, current_plot)
        # current_plot += 1
        plt.figure(figsize=figsize)
        
        # Find the largest data cap in the dataset
        # largest_data_cap = max(comparison_data['data_caps'])
        largest_data_cap = 3200000
        print(f"Using largest data cap: {largest_data_cap} for predictor_size plot")
        
        # Find the best configuration (lowest latency) for each model architecture with largest data cap
        best_configs = {}  # (embedding_dim, hidden_dim) -> (size, latency, batch_size, model_key)
        
        # Group all data by model architecture, filtering for largest data cap
        for i, model_data_key in enumerate(all_model_data_keys):
            model_dims = comparison_data['model_dims'][i]
            model_key = comparison_data['model_keys'][i]
            predictor_size = predictor_sizes_kb[i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            # Skip if not using the largest data cap
            if data_cap != largest_data_cap:
                continue
                
            # Find the lowest latency across all batch sizes for this model
            min_latency = float('inf')
            best_batch = None
            
            for batch_size in comparison_data['batch_sizes']:
                if model_data_key in comparison_data['latencies'][batch_size]:
                    latency = comparison_data['latencies'][batch_size][model_data_key]
                    if latency < min_latency:
                        min_latency = latency
                        best_batch = batch_size
            
            if best_batch is not None:
                # Store this model's info (no need to check for duplicates as we're filtering by data cap)
                best_configs[model_dims] = (predictor_size, min_latency, best_batch, model_key)
        
        # Extract data for plotting
        sizes = []
        latencies = []
        
        # Sort model dimensions for consistent ordering
        model_dims_sorted = sorted(best_configs.keys(), key=lambda dims: (dims[0], dims[1]))
        
        for dims in model_dims_sorted:
            predictor_size, latency, batch, model_key = best_configs[dims]
            sizes.append(predictor_size)
            latencies.append(latency)
            
        # Sort data points by size for the line plot
        sorted_indices = np.argsort(sizes)
        sorted_sizes = [sizes[i] for i in sorted_indices]
        sorted_latencies = [latencies[i] for i in sorted_indices]
        
        # Draw line first (so points appear on top) and then scatter plot
        plt.plot(sorted_sizes, sorted_latencies, alpha=0.7, color=colors[0], linewidth=2, label="Learned")
        plt.scatter(sizes, latencies, s=70, color=colors[0], edgecolor='white', linewidth=1, zorder=3)

        plt.annotate(f"{sorted_sizes[-1]:.0f} KB", 
                    xy=(sorted_sizes[-1], sorted_latencies[-1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color=colors[0],
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        
        # Add subtle grid
        plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        
        if show_traditional and largest_data_cap in comparison_data['traditional_bloom_metrics']:
            # Get traditional bloom metrics for the largest data cap
            trad_metrics = comparison_data['traditional_bloom_metrics'][largest_data_cap]
            trad_size_kb = trad_metrics['avg_size'] / 1024
            trad_latency = trad_metrics['avg_latency']
            
            # Add point for traditional Bloom filter with its size
            plt.scatter([trad_size_kb], [trad_latency],
                       s=70, color='red', marker='x', linewidth=2, zorder=3)
            
            # Add grid lines at the traditional point
            plt.axhline(y=trad_latency, color='red', linestyle='--', alpha=0.3, zorder=1)
            plt.axvline(x=trad_size_kb, color='red', linestyle='--', alpha=0.3, zorder=1)
            
            # Add line+marker to legend for consistent appearance
            plt.plot([], [], 'rx-', linewidth=2, label='Traditional')
        
        if log_scale:
            plt.xscale('log')
        
        plt.xlabel(f'Predictor Size (KB){log_scale_str}', fontsize=10, labelpad=10)
        plt.ylabel('Best Latency (ms)', fontsize=10, labelpad=10)
        title = 'predictor_size_vs_latency'
        
        # Better legend
        legend = plt.legend(loc='best', frameon=True, fontsize=9)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('grey')
        
        # Now that the plot is fully configured, get axis limits and add annotations for traditional values
        if show_traditional and largest_data_cap in comparison_data['traditional_bloom_metrics']:
            trad_metrics = comparison_data['traditional_bloom_metrics'][largest_data_cap]
            trad_size_kb = trad_metrics['avg_size'] / 1024
            trad_latency = trad_metrics['avg_latency']
            
            # Get current axis limits
            xlim = plt.xlim()
            ylim = plt.ylim()
            
            # Y-axis annotation (latency)
            plt.text(xlim[0]*1.05, trad_latency, f"{trad_latency:.3f} ms", 
                    color='red', fontsize=8, va='center', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            
            # X-axis annotation (predictor size)
            plt.text(trad_size_kb, ylim[-1] * 0.1, f"{trad_size_kb:.1f} KB", 
                    color='red', fontsize=8, va='bottom', ha='center', rotation=90,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # New Plot: Best Latency per Model Size (across all batch sizes and data caps)
    if "best_latency" in plot_types:
        # plt.subplot(n_rows, 2, current_plot)
        # current_plot += 1
        plt.figure(figsize=figsize)
        
        # Group by model dimensions
        model_dim_groups = defaultdict(list)
        for i, model_data_key in enumerate(all_model_data_keys):
            model_dims = comparison_data['model_dims'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            # Find the best (minimum) latency for this model across all batch sizes
            min_latency = float('inf')
            best_batch = None
            
            for batch_size in comparison_data['batch_sizes']:
                if model_data_key in comparison_data['latencies'][batch_size]:
                    latency = comparison_data['latencies'][batch_size][model_data_key]
                    if latency < min_latency:
                        min_latency = latency
                        best_batch = batch_size
            
            if best_batch is not None:
                model_dim_groups[model_dims].append((
                    model_dims,
                    model_data_key,
                    min_latency,
                    best_batch,
                    model_sizes_kb[i],                    # Add model size
                    overflow_bloom_filter_sizes_kb[i],    # Add overflow bloom filter size
                    predictor_sizes_kb[i],                # Add total predictor size (model + bloom)
                    data_cap                              # Add data cap
                ))
        
        # Find the best (minimum latency) model for each dimension
        best_latency_per_model = []
        for dims, group in model_dim_groups.items():
            # Sort by latency (ascending)
            group.sort(key=lambda x: x[2])
            # Take the one with lowest latency
            best_latency_per_model.append(group[0])
        
        # Sort by model dimensions (first by embedding_dim, then by hidden_dim)
        best_latency_per_model.sort(key=lambda x: (x[0][0], x[0][1]))
        
        # Extract data for plotting - now using predictor size for x-axis
        x_predictor_sizes = [entry[6] for entry in best_latency_per_model]  # Use predictor size in KB for x-axis
        y_latencies = [entry[2] for entry in best_latency_per_model]
        labels = [entry[1] for entry in best_latency_per_model]
        batches = [entry[3] for entry in best_latency_per_model]
        dim_labels = [f"({dims[0]},{dims[1]})" for dims, _, _, _, _, _, _, _ in best_latency_per_model]
        
        # Plot with improved aesthetics
        plt.plot(x_predictor_sizes, y_latencies, alpha=0.7, color=colors[0], linewidth=2)
        plt.scatter(x_predictor_sizes, y_latencies, s=70, color=colors[0], 
                   edgecolor='white', linewidth=1, zorder=3)
        
        # Add subtle grid
        plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        
        # Add annotations that are better formatted
        for i, (x, y, label, batch, dim) in enumerate(zip(x_predictor_sizes, y_latencies, labels, batches, dim_labels)):
            short_label = label.split('_')[0]  # Shorter label for annotation
            plt.annotate(f"{short_label}\n{dim}\nBatch {batch}", (x, y), 
                        fontsize=8, ha='center', va='bottom', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        if log_scale:
            plt.xscale('log')
        
        plt.xlabel(f'Total Predictor Size (KB){log_scale_str}', fontsize=10, labelpad=10)
        plt.ylabel('Best Latency (ms)', fontsize=10, labelpad=10)
        title = 'predictor_size_vs_latency_all_batch'
        # plt.title(title, fontsize=12, pad=10)
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Modified Plot: Overflow Bloom Filter Size per Model Size for a specific data cap
    if "min_bloom" in plot_types:
        plt.figure(figsize=figsize)
        
        # Set specific data cap to filter by
        target_data_cap = 800000  # Local variable specifying the target data cap
        
        # Filter model data to only include models with the target data cap
        filtered_model_data = []
        for i, model_data_key in enumerate(all_model_data_keys):
            model_dims = comparison_data['model_dims'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            # Only include models with the target data cap
            if data_cap == target_data_cap:
                filtered_model_data.append((
                    model_dims,
                    model_data_key,
                    overflow_bloom_filter_sizes_kb[i],
                    model_sizes_kb[i]
                ))
        
        # Sort by model dimensions (first by embedding_dim, then by hidden_dim)
        filtered_model_data.sort(key=lambda x: (x[0][0], x[0][1]))
        
        # Extract data for plotting
        x_model_sizes = [entry[3] for entry in filtered_model_data]  # Use model size in KB for x-axis
        y_bloom_sizes = [entry[2] for entry in filtered_model_data]  # Overflow bloom filter size
        labels = [entry[1] for entry in filtered_model_data]
        dim_labels = [f"({dims[0]},{dims[1]})" for dims, _, _, _ in filtered_model_data]
        
        # Plot as scatter with connecting line - sorting points by model size for the connecting line
        sort_idx = np.argsort(x_model_sizes)
        sorted_x = [x_model_sizes[i] for i in sort_idx]
        sorted_y = [y_bloom_sizes[i] for i in sort_idx]
        
        # Plot points using original order to maintain grouping by dimensions
        plt.scatter(x_model_sizes, y_bloom_sizes, color='C0')
        
        # Add connecting line in order of increasing model size
        plt.plot(sorted_x, sorted_y, alpha=1, color='C0', linewidth=2, label='Learned')
        
        plt.xlabel(f'Model Size (KB)')
        plt.ylabel(f'Overflow Bloom Filter Size (KB)')
        plt.legend()
        title = f'model_size_vs_overflow_size'
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # NEW PLOT: Efficiency Metrics (Throughput/Memory vs Data Cap)
    if "efficiency" in plot_types:
        # plt.subplot(n_rows, 2, current_plot)
        # current_plot += 1
        plt.figure(figsize=figsize)
        
        # Prepare data for efficiency plot - now tracking best efficiency per data cap
        best_efficiency_per_datacap = {}  # data_cap -> (efficiency, model_key, batch_size)
        
        # First pass: Find the best efficiency for each data cap across all models and batch sizes
        for i, model_data_key in enumerate(all_model_data_keys):
            model_key = comparison_data['model_keys'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            predictor_size = predictor_sizes_kb[i]
            
            # Only calculate if predictor has size
            if predictor_size > 0:
                # Check each batch size for this model and data cap
                for batch_size in comparison_data['batch_sizes']:
                    if model_data_key in comparison_data['throughputs'][batch_size]:
                        throughput = comparison_data['throughputs'][batch_size][model_data_key]
                        efficiency = throughput / predictor_size
                        
                        # Update if this is the best efficiency for this data cap
                        if data_cap not in best_efficiency_per_datacap or efficiency > best_efficiency_per_datacap[data_cap][0]:
                            best_efficiency_per_datacap[data_cap] = (efficiency, model_key, batch_size)
        
        # Get traditional Bloom filter efficiency data
        trad_data = {'data_caps': [], 'efficiency_scores': []}
        if show_traditional:
            for data_cap, metrics in comparison_data['traditional_bloom_metrics'].items():
                trad_throughput = metrics['avg_throughput']
                trad_size_kb = metrics['avg_size'] / 1024
                
                # Calculate efficiency for traditional Bloom filter
                if trad_size_kb > 0:
                    efficiency = trad_throughput / trad_size_kb
                    trad_data['data_caps'].append(data_cap)
                    trad_data['efficiency_scores'].append(efficiency)
        
        # Extract data for plotting best learned efficiencies
        data_caps = []
        efficiencies = []
        model_labels = []
        batch_labels = []
        
        # Sort data caps for consistent plot
        for data_cap in sorted(best_efficiency_per_datacap.keys()):
            efficiency, model_key, batch_size = best_efficiency_per_datacap[data_cap]
            data_caps.append(data_cap)
            efficiencies.append(efficiency)
            model_labels.append(model_key)
            batch_labels.append(batch_size)
        
        # Find max learned efficiency for scaling
        max_learned = max(efficiencies) if efficiencies else 0
        
        # Plot best model efficiencies
        plt.plot(data_caps, efficiencies, 'o-', color=colors[0], linewidth=2, label="Best Learned Model")
        
        # # Add annotations forRrodel architecture and batch size
        # for i, (x, y, model, batch) in enumerate(zip(data_caps, efficiencies, model_labels, batch_labels)):
        #     if i % 2 == 0:  # Annotate every other point to avoid clutter
        #         plt.annotate(f"{model}\nBatch {batch}", 
        #                     xy=(x, y),
        #                     xytext=(5, 5),  # Small offset
        #                     textcoords='offset points',
        #                     fontsize=8,
        #                     color=colors[0],
        #                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Set y-limit to focus on learned model values
        plt.ylim(0, max_learned * 1.2)
        
        # Plot traditional with 'x' markers and add annotations only for outliers
        if trad_data['data_caps']:
            sorted_indices = np.argsort(trad_data['data_caps'])
            trad_caps_sorted = [trad_data['data_caps'][i] for i in sorted_indices]
            trad_efficiency_sorted = [trad_data['efficiency_scores'][i] for i in sorted_indices]
            
            # We split the traditional data into regular and outlier points
            outlier_x = []
            outlier_y = []
            outlier_values = []
            regular_x = []
            regular_y = []
            
            for x, y in zip(trad_caps_sorted, trad_efficiency_sorted):
                if y > max_learned * 1.2:  # This is an outlier
                    outlier_x.append(x)
                    outlier_y.append(max_learned * 1.15)  # Place at top of visible area
                    outlier_values.append(y)
                else:
                    regular_x.append(x)
                    regular_y.append(y)
            
            # Plot the regular points first
            if regular_x:
                plt.plot(regular_x, regular_y, marker='x', color='red', linewidth=2, markersize=8)
            
            # Now plot connecting lines to all points (outliers and non-outliers)
            if len(trad_caps_sorted) > 1:
                plt.plot(trad_caps_sorted, [min(y, max_learned * 1.15) for y in trad_efficiency_sorted], 
                       'r-', alpha=0.2, linewidth=2)  # Increased linewidth to 2
            
            # Finally plot the outlier markers at the top with a label
            if outlier_x:
                plt.scatter(outlier_x, outlier_y, c='red', marker='x')
                
                # Add annotations for outliers
                for i, (x, y, val) in enumerate(zip(outlier_x, outlier_y, outlier_values)):
                    plt.annotate(f"{int(val)}", 
                               xy=(x, y), 
                               xytext=(x, y - max_learned * 0.05),  # Position slightly below the marker
                               fontsize=8, 
                               ha='center', 
                               va='top',
                               color='red')
                # Add Traditional to the legend only once - with consistent styling
                plt.plot([], [], 'rx-', alpha=0.2, linewidth=2, label='Traditional Outliers')
            
            # Add Traditional to the legend only once - with consistent styling
            plt.plot([], [], 'rx-', linewidth=2, label='Traditional')
        
        if log_scale:
            plt.xscale('log')
        
        plt.xlabel(f'Dataset Size (number of samples){log_scale_str}')
        plt.ylabel('Efficiency (Throughput per KB)')
        title = 'data_size_vs_best_efficiency'
        # plt.title(title, fontsize=12, pad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # NEW PLOT: Heatmap of Data Cap vs Model Size vs Performance
    if "heatmap" in plot_types and len(comparison_data['data_caps']) > 1 and len(unique_model_keys) > 1:
        plt.figure(figsize=figsize)
        
        # Find data caps that exist for all model architectures
        model_keys = sorted(unique_model_keys)
        data_caps = sorted(comparison_data['data_caps'])
        
        # Track which data caps have entries for each model architecture
        data_cap_coverage = {data_cap: set() for data_cap in data_caps}
        
        # Collect model architectures available for each data cap
        for i, model_data_key in enumerate(all_model_data_keys):
            model_key = comparison_data['model_keys'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            if data_cap in data_cap_coverage:
                data_cap_coverage[data_cap].add(model_key)
        
        # Filter to only include data caps that have all model architectures
        complete_data_caps = [data_cap for data_cap, models in data_cap_coverage.items() 
                             if len(models) == len(model_keys)]
        
        if not complete_data_caps:
            print("No data caps have complete data for all model architectures")
            # If no complete data caps, use all data caps
            complete_data_caps = data_caps
        else:
            print(f"Using {len(complete_data_caps)} data caps with complete data for all model architectures")
        
        # Sort filtered data caps
        complete_data_caps.sort()
        
        # NEW: Collect model information with model sizes for each data cap
        # We'll collect a list of entries for each data cap with (model_key, model_size)
        model_data_by_cap = {data_cap: [] for data_cap in complete_data_caps}
        
        # Collect model data for each data cap
        for i, model_data_key in enumerate(all_model_data_keys):
            model_key = comparison_data['model_keys'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            if data_cap in complete_data_caps:
                model_size = model_sizes_kb[i]  # Just the model size in KB (without overflow bloom filter)
                model_data_by_cap[data_cap].append((model_key, model_size, model_data_key))
        
        # Make sure all data caps have the same set of models, sorted by model size
        # For each data cap, sort models by model size
        for data_cap in complete_data_caps:
            model_data_by_cap[data_cap].sort(key=lambda x: x[1])  # Sort by model size
        
        # Get a reference ordering from the first data cap
        if complete_data_caps:
            first_cap = complete_data_caps[0]
            sorted_model_keys = [entry[0] for entry in model_data_by_cap[first_cap]]
            sorted_model_sizes = [entry[1] for entry in model_data_by_cap[first_cap]]
            
            # Create matrix for heatmap (rows=filtered_data_caps, cols=models sorted by model size)
            latency_matrix = np.zeros((len(complete_data_caps), len(sorted_model_keys)))
            
            # Fill matrix with latency values
            for row_idx, data_cap in enumerate(complete_data_caps):
                for col_idx, model_key in enumerate(sorted_model_keys):
                    # Find the model_data_key for this model and data cap
                    matching_entries = [entry[2] for entry in model_data_by_cap[data_cap] if entry[0] == model_key]
                    if matching_entries:
                        model_data_key = matching_entries[0]
                        
                        # Calculate average metrics across batch sizes
                        avg_latency = np.mean([
                            comparison_data['latencies'][bs].get(model_data_key, 0) 
                            for bs in sorted(comparison_data['batch_sizes'])
                        ])
                        
                        latency_matrix[row_idx, col_idx] = avg_latency
            
            # Use a better colormap with more contrast
            cmap = plt.cm.viridis
            
            # Plot latency heatmap with improved aesthetics
            im = plt.imshow(latency_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
            
            # Add grid lines to the heatmap to separate cells
            plt.grid(False)  # Turn off default grid
            
            # Add colorbar with more descriptive label
            cbar = plt.colorbar(im, label='Average Latency (ms)', pad=0.01)
            cbar.ax.tick_params(labelsize=8)  # smaller tick size
            
            # Create x-axis labels showing model sizes
            x_labels = [f"{size:.1f} KB" for size in sorted_model_sizes]
            
            # Set ticks and labels with better positioning and rotation
            plt.xticks(range(len(sorted_model_keys)), x_labels, rotation=45, ha='right', fontsize=9)
            plt.yticks(range(len(complete_data_caps)), [f"Data: {cap:,}" for cap in complete_data_caps], fontsize=9)
            
            # Add values in the cells for smaller matrices
            if len(complete_data_caps) * len(sorted_model_keys) <= 50:  # Threshold for adding text
                for i in range(len(complete_data_caps)):
                    for j in range(len(sorted_model_keys)):
                        if not np.isnan(latency_matrix[i, j]) and latency_matrix[i, j] > 0:
                            text_color = 'white' if latency_matrix[i, j] > np.max(latency_matrix) * 0.7 else 'black'
                            plt.text(j, i, f"{latency_matrix[i, j]:.1f}", 
                                    ha="center", va="center", color=text_color, fontsize=8)
            
            plt.xlabel('Model Size (KB)', fontsize=10, labelpad=10)
            plt.ylabel('Dataset Cap', fontsize=10, labelpad=10)
            title = 'heatmap_datacap_vs_modelsize_vs_latency'
            path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # New Plot: Latency vs Throughput for the best model
    if "best_model_latency_throughput" in plot_types:
        # plt.subplot(n_rows, 2, current_plot)
        # current_plot += 1
        plt.figure(figsize=figsize)
        
        # Find the best model with lowest latency across all batch sizes and data caps
        best_model_key = None
        best_latency = float('inf')
        best_data_cap = None
        
        for i, model_data_key in enumerate(all_model_data_keys):
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            # Check all batch sizes
            for batch_size in comparison_data['batch_sizes']:
                if model_data_key in comparison_data['latencies'][batch_size]:
                    latency = comparison_data['latencies'][batch_size][model_data_key]
                    if latency < best_latency:
                        best_latency = latency
                        best_model_key = model_data_key
                        best_data_cap = data_cap
        
        if best_model_key:
            # Get data for the best model across all batch sizes
            batch_sizes_list = sorted(list(comparison_data['batch_sizes']))
            latencies = []
            throughputs = []
            batch_labels = []
            
            for batch_size in batch_sizes_list:
                if best_model_key in comparison_data['latencies'][batch_size]:
                    latencies.append(comparison_data['latencies'][batch_size][best_model_key])
                    throughputs.append(comparison_data['throughputs'][batch_size][best_model_key])
                    batch_labels.append(f"Batch {batch_size}")
            
            # Get traditional bloom filter point if available for this data cap
            trad_latency = None
            trad_throughput = None
            if show_traditional and best_data_cap in comparison_data['traditional_bloom_metrics']:
                trad_metrics = comparison_data['traditional_bloom_metrics'][best_data_cap]
                trad_latency = trad_metrics['avg_latency']
                trad_throughput = trad_metrics['avg_throughput']

            # Plot learned model points and connecting line
            plt.scatter(latencies, throughputs, color='C0', s=60)
            plt.plot(latencies, throughputs, alpha=0.8, color='C0', label='Learned')
            
            # # Annotate points with batch sizes
            # for i, (x, y, label) in enumerate(zip(latencies, throughputs, batch_labels)):
                # plt.annotate(label, (x, y), fontsize=8, ha='center', va='bottom')
            
            # Set y-limit to focus on learned model values
            max_learned_throughput = max(throughputs) if throughputs else 100
            plt.ylim(0, max_learned_throughput * 1.2)
            
            # Add traditional bloom filter point with 'x' marker if available
            if trad_latency is not None and trad_throughput is not None:
                if trad_throughput > max_learned_throughput * 1.2:  # This is an outlier
                    # Remove the connecting line and just show the X marker at the top
                    plt.scatter([trad_latency], [max_learned_throughput * 1.15], 
                              c='red', marker='x', s=100)
                    
                    # Add annotation with actual value
                    plt.annotate(f"{int(trad_throughput)}", 
                               xy=(trad_latency, max_learned_throughput * 1.15), 
                               xytext=(trad_latency, max_learned_throughput * 1.05),
                               fontsize=8, 
                               ha='center', 
                               va='top',
                               color='red')
                else:
                    # If not an outlier, just show the point normally
                    plt.scatter([trad_latency], [trad_throughput], c='red', marker='x', s=100)
                
                # Add Traditional to the legend only once
                plt.plot([], [], 'rx-', linewidth=2, label='Traditional', markersize=8)
            
            plt.xlabel('Latency (ms)')
            plt.ylabel('Throughput (URLs/second)')
            title = f'latency_vs_throughput_best_model'
            # plt.title(title, fontsize=12, pad=10)
            path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.text(0.5, 0.5, "No data available for best model latency vs throughput plot",
                    horizontalalignment='center', verticalalignment='center')
    # NEW PLOT: Dataset Cap vs Performance
    if "datacap" in plot_types:
        # plt.subplot(n_rows, 2, current_plot)
        # current_plot += 1
        plt.figure(figsize=figsize)
        
        # Prepare data grouped by model architecture
        model_data = {}
        for unique_model in unique_model_keys:
            model_data[unique_model] = {'data_caps': [], 'avg_latencies': [], 'avg_throughputs': []}
        
        # For each model architecture, collect performance across data caps
        for i, model_data_key in enumerate(all_model_data_keys):
            model_key = comparison_data['model_keys'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            # Calculate average performance across batch sizes
            avg_latency = np.mean([
                comparison_data['latencies'][bs].get(model_data_key, 0) 
                for bs in sorted(comparison_data['batch_sizes'])
            ])
            
            avg_throughput = np.mean([
                comparison_data['throughputs'][bs].get(model_data_key, 0) 
                for bs in sorted(comparison_data['batch_sizes'])
            ])
            
            # Store in the model's data
            model_data[model_key]['data_caps'].append(data_cap)
            model_data[model_key]['avg_latencies'].append(avg_latency)
            model_data[model_key]['avg_throughputs'].append(avg_throughput)
        
        # Find max learned latency for scaling
        max_learned_latency = 0
        for m_data in model_data.values():
            if m_data['avg_latencies']:
                max_learned_latency = max(max_learned_latency, max(m_data['avg_latencies']))
        
        # Plot each model architecture as a line
        for model_key, data in model_data.items():
            # Sort by data cap
            sorted_indices = np.argsort(data['data_caps'])
            sorted_caps = [data['data_caps'][i] for i in sorted_indices]
            sorted_latencies = [data['avg_latencies'][i] for i in sorted_indices]
            
            if sorted_caps:  # Only plot if we have data
                plt.plot(sorted_caps, sorted_latencies, 'o-', label=model_key)
        
        # Set y-limit to focus on learned model values
        plt.ylim(0, max_learned_latency * 1.2)
        
        # Add traditional Bloom filter performance per data cap with 'x' markers and annotations for outliers
        if show_traditional:
            trad_caps = []
            trad_latencies = []
            
            for data_cap, metrics in comparison_data['traditional_bloom_metrics'].items():
                trad_caps.append(data_cap)
                trad_latencies.append(metrics['avg_latency'])
            
            # Sort by data cap
            sorted_indices = np.argsort(trad_caps)
            trad_caps_sorted = [trad_caps[i] for i in sorted_indices]
            trad_latencies_sorted = [trad_latencies[i] for i in sorted_indices]
            
            if trad_caps:  # Only plot if we have data
                # Split into regular and outlier points
                outlier_x = []
                outlier_y = []
                outlier_values = []
                regular_x = []
                regular_y = []
                
                for x, y in zip(trad_caps_sorted, trad_latencies_sorted):
                    if y > max_learned_latency * 1.2:  # This is an outlier
                        outlier_x.append(x)
                        outlier_y.append(max_learned_latency * 1.15)  # Place at top of visible area
                        outlier_values.append(y)
                    else:
                        regular_x.append(x)
                        regular_y.append(y)
                
                # Plot the regular points with connecting lines
                if regular_x:
                    plt.plot(regular_x, regular_y, 'rx-', linewidth=2, markersize=8)
                
                # Now plot connecting lines to all points (outliers and non-outliers)
                if len(trad_caps_sorted) > 1:
                    plt.plot(trad_caps_sorted, [min(y, max_learned_latency * 1.15) for y in trad_latencies_sorted], 
                           'r-', alpha=0.3, linewidth=2)  # Increased linewidth to 2
                
                # Plot the outlier markers at the top with labels
                if outlier_x:
                    plt.scatter(outlier_x, outlier_y, c='red', marker='x', s=100)
                    
                    # Add annotations for outliers
                    for i, (x, y, val) in enumerate(zip(outlier_x, outlier_y, outlier_values)):
                        plt.annotate(f"{int(val)}", 
                                   xy=(x, y), 
                                   xytext=(x, y - max_learned_latency * 0.05),  # Position slightly below the marker
                                   fontsize=8, 
                                   ha='center', 
                                   va='top',
                                   color='red')
                
                # Add Traditional to the legend only once - with consistent styling
                plt.plot([], [], 'rx-', linewidth=2, label='Traditional')
        
        if log_scale:
            plt.xscale('log')
        
        plt.xlabel(f'Dataset Size (number of samples){log_scale_str}')
        plt.ylabel('Average Latency (ms)')
        title = 'data_size_vs_latency'
        # plt.title(title, fontsize=12, pad=10)
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # NEW PLOT: Model Size vs Predictor Size for a specific data cap
    if "model_vs_predictor" in plot_types:
        plt.figure(figsize=figsize)
        
        # Set specific data cap to filter by
        target_data_cap = 800000  # Use the same data cap as other plots
        
        # Filter model data to only include models with the target data cap
        filtered_model_data = []
        for i, model_data_key in enumerate(all_model_data_keys):
            model_dims = comparison_data['model_dims'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            
            # Only include models with the target data cap
            if data_cap == target_data_cap:
                filtered_model_data.append((
                    model_dims,
                    model_data_key,
                    model_sizes_kb[i],  # Model size in KB
                    predictor_sizes_kb[i]  # Predictor size (model + bloom) in KB
                ))
        
        # Sort by model dimensions for consistent presentation
        filtered_model_data.sort(key=lambda x: (x[0][0], x[0][1]))
        
        # Extract data for plotting
        x_model_sizes = [entry[2] for entry in filtered_model_data]  # Model size in KB for x-axis
        y_predictor_sizes = [entry[3] for entry in filtered_model_data]  # Predictor size in KB for y-axis
        model_dims = [f"({dims[0]},{dims[1]})" for dims, _, _, _ in filtered_model_data]  # Dimensions for annotations
        
        # Plot the data points
        plt.scatter(x_model_sizes, y_predictor_sizes, color='C0',  zorder=3)
        
        # Add connecting line to show trend
        # Sort points by model size for the connecting line
        sorted_indices = np.argsort(x_model_sizes)
        sorted_x = [x_model_sizes[i] for i in sorted_indices]
        sorted_y = [y_predictor_sizes[i] for i in sorted_indices]
        plt.plot(sorted_x, sorted_y, '-', alpha=1, color='C0', linewidth=2, zorder=2)
        plt.plot([], [], '-o', alpha=1, color='C0', linewidth=2, zorder=2, label='Learned')
        
        # # Add annotations for each point showing the model dimensions
        # for i, (x, y, dims) in enumerate(zip(x_model_sizes, y_predictor_sizes, model_dims)):
        #     plt.annotate(dims, 
        #                 xy=(x, y),
        #                 xytext=(5, 5),  # Small offset
        #                 textcoords='offset points',
        #                 fontsize=8,
        #                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add the y=x line (where predictor size = model size)
        min_size = min(x_model_sizes)
        max_size = max(x_model_sizes)
        
        # Add grid and labels
        plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        plt.xlabel(f'Model Size (KB)', fontsize=10, labelpad=10)
        plt.ylabel(f'Predictor Size (KB)', fontsize=10, labelpad=10)
        title = f'model_size_vs_predictor_size'
        
        # Add legend
        # plt.legend(loc='best')
        plt.legend()
        
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # NEW PLOT: Predictor Size vs Throughput (best performer per size)
    if "size_vs_throughput" in plot_types:
        plt.figure(figsize=figsize)
        
        # Find the largest data cap in the dataset for consistency with other plots
        largest_data_cap = 3200000
        print(f"Using largest data cap: {largest_data_cap} for size_vs_throughput plot")
        
        # Group models by predictor size and find the best throughput for each size
        size_throughput_map = {}  # predictor_size -> (throughput, model_key, batch_size)
        
        # Collect data for each model and batch size
        for i, model_data_key in enumerate(all_model_data_keys):
            model_key = comparison_data['model_keys'][i]
            data_cap = comparison_data['hyperparams'][model_data_key].get('data_cap', 0)
            predictor_size = predictor_sizes_kb[i]
            
            # Only consider models with the target data cap
            if data_cap == largest_data_cap:
                # Find the best throughput across all batch sizes
                best_throughput = 0
                best_batch = None
                
                for batch_size in sorted(comparison_data['batch_sizes']):
                    if model_data_key in comparison_data['throughputs'][batch_size]:
                        throughput = comparison_data['throughputs'][batch_size][model_data_key]
                        if throughput > best_throughput:
                            best_throughput = throughput
                            best_batch = batch_size
                
                # Round predictor size to nearest KB to group similar sizes
                rounded_size = round(predictor_size)
                
                # Update the map if this is the best throughput for this size
                if rounded_size not in size_throughput_map or best_throughput > size_throughput_map[rounded_size][0]:
                    size_throughput_map[rounded_size] = (best_throughput, model_key, best_batch)
        
        # Extract the data for plotting
        predictor_sizes = []
        throughputs = []
        model_keys = []
        batch_sizes = []
        
        for size in sorted(size_throughput_map.keys()):
            throughput, model_key, batch_size = size_throughput_map[size]
            predictor_sizes.append(size)
            throughputs.append(throughput)
            model_keys.append(model_key)
            batch_sizes.append(batch_size)
        
        # Plot the learned model data
        plt.plot(predictor_sizes, throughputs, '-o', color=colors[0], linewidth=2, label='Learned Models')
        
        # Get traditional bloom filter data
        trad_data = {'sizes': [], 'throughputs': []}
        if show_traditional:
            # Get traditional bloom filter throughput for the largest data cap
            if largest_data_cap in comparison_data['traditional_bloom_metrics']:
                trad_metrics = comparison_data['traditional_bloom_metrics'][largest_data_cap]
                trad_size_kb = trad_metrics['avg_size'] / 1024  # Convert to KB
                trad_throughput = trad_metrics['avg_throughput']
                
                trad_data['sizes'].append(trad_size_kb)
                trad_data['throughputs'].append(trad_throughput)
        
        # Find max learned throughput for scaling outliers
        max_learned_throughput = max(throughputs) if throughputs else 100
        
        # Plot traditional with handling for outliers, if available
        if show_traditional and trad_data['throughputs']:
            # Sort traditional data points by size
            sorted_indices = np.argsort(trad_data['sizes'])
            trad_sizes = [trad_data['sizes'][i] for i in sorted_indices]
            trad_throughputs = [trad_data['throughputs'][i] for i in sorted_indices]
            
            # Split into regular and outlier points
            outlier_x = []
            outlier_y = []
            outlier_values = []
            regular_x = []
            regular_y = []
            
            for x, y in zip(trad_sizes, trad_throughputs):
                if y > max_learned_throughput * 1.2:  # This is an outlier
                    outlier_x.append(x)
                    outlier_y.append(max_learned_throughput * 1.15)  # Place at top of visible area
                    outlier_values.append(y)
                else:
                    regular_x.append(x)
                    regular_y.append(y)
            
            # Plot the regular points
            if regular_x:
                plt.plot(regular_x, regular_y, marker='x', color='red', linewidth=2, markersize=8)
            
            # Plot the outlier markers at the top with a label
            if outlier_x:
                plt.scatter(outlier_x, outlier_y, c='red', marker='x')
                
                # Add annotations for outliers
                for i, (x, y, val) in enumerate(zip(outlier_x, outlier_y, outlier_values)):
                    plt.annotate(f"{int(val)}", 
                               xy=(x, y), 
                               xytext=(x, y - max_learned_throughput * 0.05),  # Position slightly below the marker
                               fontsize=8, 
                               ha='center', 
                               va='top',
                               color='red')
                
                # Add legend item for traditional outliers
                # plt.plot([], [], 'rx-', alpha=0.2, linewidth=2, label='Traditional Outliers')
            
            # Add Traditional to the legend
            plt.plot([], [], 'rx-', linewidth=2, label='Traditional')
        
        # Set axis labels and limits
        plt.xlabel('Predictor Size (KB)', fontsize=10, labelpad=10)
        plt.ylabel('Throughput (URLs/second)', fontsize=10, labelpad=10)
        plt.ylim(0, max_learned_throughput * 1.2)  # Set y-limit to focus on learned models
        
        # if log_scale:
        #     plt.xscale('log')
        
        # Add grid and legend
        plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        plt.legend()
        
        # Save the plot
        title = 'predictor_size_vs_throughput'
        path = f"plots/{title}.pdf" if not cpu else f"plots/cpu/cpu_{title}.pdf"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_single_model_results(analysis_list, save_path=None, metrics_to_plot=None, show_traditional=True, log_scale=False, show_throughput=True, model_path=None, cpu=False):
    """
    Create plots for throughputs and latency from multiple analysis dictionaries.
    
    Args:
        analysis_list: List of analysis dictionaries from analyze_results function
        save_path: Optional path to save the plot instead of displaying it
        metrics_to_plot: List of metrics to plot (e.g. ['total', 'inference', 'tensor', 'bloom'])
        show_traditional: Whether to show traditional Bloom filter results
        log_scale: Whether to use logarithmic scale for model size axis
        show_throughput: Whether to show throughput plot (if False, only shows latency breakdown)
        model_path: Path to the model for display in the title
        cpu: Whether to save to cpu directory
    """
    if not analysis_list:
        print("No analysis data provided for plotting")
        return
    
    if not metrics_to_plot:
        metrics_to_plot = ['total']
    
    # Extract data for plotting
    batch_sizes = [a['batch_size'] for a in analysis_list]
    
    # Get base path for saving files
    if save_path:
        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
    else:
        base_dir = "plots"
        base_name = "analysis"
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Add cpu subdirectory if needed
    if cpu and not os.path.exists(os.path.join(base_dir, "cpu")):
        os.makedirs(os.path.join(base_dir, "cpu"))
    
    # Create Throughput Plot if requested
    if show_throughput:
        plt.figure(figsize=(4.5, 3.5))
        
        # Get throughput data
        throughputs = [a['throughput']['urls_per_second'] for a in analysis_list]
        traditional_throughputs = [a['throughput']['traditional_urls_per_second'] for a in analysis_list]
        
        # Plot throughput
        plt.plot(batch_sizes, throughputs, color='blue', marker='o', label='Learned Model')
        if show_traditional:
            plt.plot(batch_sizes, traditional_throughputs, color='red', marker='x', label='Traditional Bloom')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (URLs/second)')
        # plt.title('Batch Size vs. Throughput')
        plt.legend()
        if log_scale:
            plt.xscale('log')
            plt.xlabel('Batch Size (log scale)')
        plt.grid(True)
        
        # Save throughput plot
        throughput_filename = f"{base_name}_throughput.pdf"
        throughput_path = os.path.join(base_dir, "cpu", throughput_filename) if cpu else os.path.join(base_dir, throughput_filename)
        plt.savefig(throughput_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create Latency Breakdown Plot
    plt.figure(figsize=(4.5, 3.5))
    
    # Extract each component of latency
    tensor_latencies = [a['latency']['average_tensor_to_device_latency_ms'] for a in analysis_list]
    inference_latencies = [a['latency']['average_inference_latency_ms'] for a in analysis_list]
    bloom_latencies = [a['latency']['average_overflow_bloom_filter_latency_ms'] for a in analysis_list]
    total_latencies = [a['latency']['average_batch_latency_ms'] for a in analysis_list]
    
    # Calculate "other" latency as the difference between total and the sum of the components
    other_latencies = []
    for i in range(len(total_latencies)):
        components_sum = tensor_latencies[i] + inference_latencies[i] + bloom_latencies[i]
        other = max(0, total_latencies[i] - components_sum)  # Ensure non-negative
        other_latencies.append(other)
    
    # Determine the number of data points to include (skip last 3 as in user's modification)
    data_indices = slice(0, -3)
    
    # Plot: Stacked Area Chart for Latency Components
    plt.stackplot(batch_sizes[data_indices], 
                  tensor_latencies[data_indices],       # Bottom layer
                  inference_latencies[data_indices],    # Second layer
                  bloom_latencies[data_indices],        # Third layer
                  other_latencies[data_indices],        # Top layer
                  labels=['Predictor Data Movement', 
                          'Predictor Inference', 
                          'Backup BF Query', 
                          'Other/Overhead'],
                  colors=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'])
    
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)' )
    # plt.title('Latency Breakdown by Component')
    plt.grid(True)
    
    if log_scale:   
        plt.xscale('log')
        plt.xlabel('Batch Size (log scale)')
        plt.yscale('log')
        plt.ylabel('Time (ms) (log scale)')
    
    plt.legend(loc='upper left')
    
    # Add model info if provided
    # if model_path:
        # plt.suptitle(f"Model: {model_path}", y=0.98)
    
    # Save latency breakdown plot
    latency_filename = f"LBF_cpu_latency_breakdown.pdf" if cpu else f"LBF_latency_breakdown.pdf"
    latency_path = os.path.join(base_dir, "cpu", latency_filename) if cpu else os.path.join(base_dir, latency_filename)
    plt.savefig(latency_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency breakdown plot saved to: {latency_path}")

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
    parser.add_argument('--traditional-individual', action='store_true',
                        help='Show traditional Bloom filter results in plots')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use logarithmic scale for model size axis')
    parser.add_argument('--cpu', action='store_true',
                        help='Use the cpu output directory json files')
    parser.add_argument('--no-throughput', action='store_true',
                        help='Do not show throughput plot, only show latency breakdown')
    # Add new argument for specifying plot types
    parser.add_argument('--plot-types', nargs='+', 
                        choices=['predictor_size', 'model_vs_bloom', 'datacap', 'efficiency', 'heatmap', 
                                'min_bloom', 'best_latency', 'best_model_latency_throughput', 'all'],
                        default=['all'],
                        help='Types of comparison plots to generate')
    
    args = parser.parse_args()
    
    # Process plot types
    plot_types = None  # Default is all plots
    if 'all' not in args.plot_types:
        plot_types = args.plot_types
    
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
            results_dir = "inference_outputs" if not args.cpu else "inference_outputs_cpu"
            model_analysis = process_model_results(model_dir, args.metrics, results_dir, args.cpu, show_traditional=args.traditional_individual, log_scale=args.log_scale, show_throughput=not args.no_throughput)
            if model_analysis:
                model_analyses.append(model_analysis)
        
        if model_analyses:
            # Prepare and plot model comparisons
            comparison_data = prepare_model_comparison_data(model_analyses)
            metrics_suffix = '_'.join(args.metrics)
            output_name = f"model_comparison_{metrics_suffix}_cpu.png" if args.cpu else f"model_comparison_{metrics_suffix}.png"
            save_path = os.path.join(args.models_path, output_name)
            print(f"{args.log_scale=}")
            plot_model_comparisons(comparison_data, save_path, args.traditional, args.log_scale, plot_types, args.cpu)
    
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
        plot_single_model_results(analyses, save_path, args.metrics, show_traditional=args.traditional, 
                                log_scale=args.log_scale, show_throughput=not args.no_throughput, cpu=args.cpu)
    
    else:
        print("Please provide either -models_path, -json_path, or -dir_path")
        sys.exit(1)

if __name__ == "__main__":
    main() 