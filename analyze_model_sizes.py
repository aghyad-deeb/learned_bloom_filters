#!/usr/bin/env python3
"""
Analyze results across different model architectures (different embedding and hidden layer sizes).
This script compares performance metrics and model sizes across different model configurations.
"""
import argparse
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re

def extract_model_config(run_name):
    """
    Extract embedding and hidden dimensions from run name.
    Example: run_20250429_104423_emb4_hid2 -> (4, 2)
    """
    match = re.search(r'emb(\d+)_hid(\d+)', run_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def analyze_model_results(results_dir):
    """
    Analyze results for different model architectures.
    
    Args:
        results_dir: Directory containing results for different model architectures
        
    Returns:
        Dictionary with analyzed metrics for each model configuration
    """
    model_analyses = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(results_dir):
        # Look for JSON files in inference_outputs directories
        if os.path.basename(root) == 'inference_outputs':
            # Get the run name from the parent directory
            run_name = os.path.basename(os.path.dirname(root))
            emb_dim, hid_dim = extract_model_config(run_name)
            
            if emb_dim is None or hid_dim is None:
                continue
                
            # Find the latest JSON file for this run (assuming timestamp in filename)
            json_files = [f for f in files if f.endswith('.json')]
            if not json_files:
                continue
                
            latest_json = max(json_files, key=lambda x: os.path.getctime(os.path.join(root, x)))
            json_path = os.path.join(root, latest_json)
            
            # Load and analyze the results
            with open(json_path, 'r') as f:
                results = json.load(f)
            
            # Extract metrics
            size_metrics = results.get('size_metrics', {})
            timing = results.get('timing', {})
            
            # Calculate average latencies
            latency_list = timing.get('batch_total_latency_per_batch_time_list', [])
            inference_list = timing.get('inference_per_batch_time_list', [])
            tensor_list = timing.get('moving_tensor_to_device_per_batch_time_list', [])
            bloom_list = timing.get('overflow_bloom_filter_per_batch_time_list', [])
            
            avg_latency = np.mean(latency_list) if latency_list else 0
            avg_inference = np.mean(inference_list) if inference_list else 0
            avg_tensor = np.mean(tensor_list) if tensor_list else 0
            avg_bloom = np.mean(bloom_list) if bloom_list else 0
            
            # Store analysis for this model configuration
            model_analyses[(emb_dim, hid_dim)] = {
                'model_size_bytes': size_metrics.get('model_size_bytes', 0),
                'overflow_bloom_size_bytes': size_metrics.get('overflow_bloom_filter_size_bytes', 0),
                'traditional_bloom_size_bytes': size_metrics.get('traditional_bloom_filter_size_bytes', 0),
                'latency': {
                    'total_ms': avg_latency,
                    'inference_ms': avg_inference,
                    'tensor_ms': avg_tensor,
                    'bloom_ms': avg_bloom
                }
            }
    
    return model_analyses

def plot_model_analysis(model_analyses, save_path=None):
    """
    Create plots comparing different model architectures.
    
    Args:
        model_analyses: Dictionary of analysis results for each model configuration
        save_path: Optional path to save the plots
    """
    if not model_analyses:
        print("No model analysis data to plot")
        return
    
    # Prepare data for plotting
    configs = sorted(model_analyses.keys())  # Sort by (emb_dim, hid_dim)
    model_sizes = [model_analyses[cfg]['model_size_bytes'] for cfg in configs]
    bloom_sizes = [model_analyses[cfg]['overflow_bloom_size_bytes'] for cfg in configs]
    trad_sizes = [model_analyses[cfg]['traditional_bloom_size_bytes'] for cfg in configs]
    latencies = [model_analyses[cfg]['latency']['total_ms'] for cfg in configs]
    
    # Create x-axis labels
    x_labels = [f'emb{emb}_hid{hid}' for emb, hid in configs]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot sizes
    x = np.arange(len(configs))
    width = 0.25
    
    ax1.bar(x - width, model_sizes, width, label='Model Size')
    ax1.bar(x, bloom_sizes, width, label='Overflow Bloom Size')
    ax1.bar(x + width, trad_sizes, width, label='Traditional Bloom Size')
    
    ax1.set_ylabel('Size (bytes)')
    ax1.set_title('Model and Bloom Filter Sizes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend()
    
    # Plot latencies
    ax2.plot(x, latencies, 'o-', label='Average Latency')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Model Inference Latency')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze results across different model architectures')
    parser.add_argument('--results_dir', default='models',
                        help='Directory containing results for different model architectures')
    parser.add_argument('--save_path',
                        help='Path to save the analysis plots')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist")
        sys.exit(1)
    
    # Analyze results
    model_analyses = analyze_model_results(args.results_dir)
    
    if not model_analyses:
        print("No model analysis data found")
        sys.exit(1)
    
    # Print summary
    print("\nModel Analysis Summary:")
    print("======================")
    for (emb_dim, hid_dim), analysis in sorted(model_analyses.items()):
        print(f"\nModel: emb{emb_dim}_hid{hid_dim}")
        print(f"Model Size: {analysis['model_size_bytes']/1024:.2f} KB")
        print(f"Overflow Bloom Size: {analysis['overflow_bloom_size_bytes']/1024:.2f} KB")
        print(f"Traditional Bloom Size: {analysis['traditional_bloom_size_bytes']/1024:.2f} KB")
        print(f"Average Latency: {analysis['latency']['total_ms']:.2f} ms")
    
    # Create plots
    plot_model_analysis(model_analyses, args.save_path)

if __name__ == "__main__":
    main() 