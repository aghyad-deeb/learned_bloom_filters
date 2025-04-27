"""
Interactive testing script for the learned bloom filter model.
This script loads the trained model and provides an interactive IPython session
for testing URLs against the model and overflow bloom filter.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from IPython import embed
import pickle
import argparse
import json
import time

# Import necessary classes and functions from train_and_run.py
from train_and_run import URLClassifier, BloomFilter, url_to_indices, build_vocab, load_data

def load_model(model, load_path, device):
    """
    Load a trained model from disk.
    
    Args:
        model: An initialized model of the correct architecture
        load_path: Path to the saved model file
        device: The device to load the model to
        
    Returns:
        The loaded model
    """
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Model loaded from {load_path}")
    else:
        print(f"No saved model found at {load_path}")
    return model

def load_bloom_filter(load_path):
    """
    Load a saved Bloom filter from disk.
    
    Args:
        load_path: Path to the saved Bloom filter file
        
    Returns:
        The loaded Bloom filter
    """
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            bloom_filter = pickle.load(f)
        print(f"Bloom filter loaded from {load_path}")
        return bloom_filter
    else:
        print(f"No saved Bloom filter found at {load_path}")
        return None

def load_hyperparams(load_path):
    """
    Load hyperparameters from a JSON file.
    
    Args:
        load_path: Path to the hyperparameters JSON file
        
    Returns:
        Dictionary of hyperparameters
    """
    if os.path.exists(load_path):
        with open(load_path, 'r') as f:
            hyperparams = json.load(f)
        print(f"Hyperparameters loaded from {load_path}")
        return hyperparams
    else:
        print(f"No hyperparameters file found at {load_path}")
        return None

def test_url(url, model, threshold, char2idx, device, overflow_bloom=None):
    """
    Test a URL against the model and overflow Bloom filter.
    
    Args:
        url: The URL to test
        model: The trained model
        threshold: The threshold for the model to classify a URL as malicious
        char2idx: Character to index mapping
        device: The device to run the model on
        overflow_bloom: The overflow Bloom filter
        
    Returns:
        A tuple of (model_prediction, model_probability, is_in_overflow, final_result, timing_info)
    """
    timing_info = {}
    
    # Convert URL to indices
    try:
        indices = url_to_indices(url, char2idx)
    except KeyError as e:
        print(f"Error: URL contains characters not in the vocabulary: {e}")
        return None, None, None, None, None
    
    # Measure data transfer time
    transfer_start = time.time()
    idx_seq = torch.tensor([indices], dtype=torch.long)
    transfer_cpu_ready = time.time()
    idx_seq = idx_seq.to(device)
    transfer_end = time.time()
    
    timing_info['time_to_create_tensor_from_indices'] = (transfer_cpu_ready - transfer_start) * 1000  # ms
    timing_info['time_to_transfer_tensor_to_device'] = (transfer_end - transfer_cpu_ready) * 1000  # ms
    
    # Measure model inference time
    model.eval()
    inference_start = time.time()
    with torch.no_grad():
        prob = model(idx_seq).cpu().item()
    inference_end = time.time()
    
    timing_info['time_to_run_model'] = (inference_end - inference_start) * 1000  # ms
    
    # Check if above threshold
    model_prediction = prob >= threshold
    
    # Check if in overflow Bloom filter (if applicable)
    is_in_overflow = False
    timing_info['time_to_check_overflow_bloom_filter'] = 0
    
    if overflow_bloom is not None:
        bloom_start = time.time()
        is_in_overflow = url in overflow_bloom
        bloom_end = time.time()
        timing_info['time_to_check_overflow_bloom_filter'] = (bloom_end - bloom_start) * 1000  # ms
    
    # Final result is positive if either model predicts positive or URL is in overflow Bloom filter
    final_result = model_prediction or is_in_overflow
    
    # Total time
    timing_info['total'] = timing_info['time_to_create_tensor_from_indices'] + timing_info['time_to_transfer_tensor_to_device'] + timing_info['time_to_run_model'] + timing_info['time_to_check_overflow_bloom_filter']
    
    return model_prediction, prob, is_in_overflow, final_result, timing_info

def get_run_dirs(model_dir):
    """Get a list of all run directories, sorted by creation time (newest first)"""
    runs = []
    for entry in os.listdir(model_dir):
        if entry.startswith("run_"):
            path = os.path.join(model_dir, entry)
            if os.path.isdir(path):
                runs.append(path)
    # Sort by folder name (which includes timestamp) in reverse order
    return sorted(runs, reverse=True)

def batch_inference(urls, model, threshold, char2idx, device, overflow_bloom=None, batch_size=64, return_detailed_results=False):
    """
    Process multiple URLs using batched model inference for better parallelization.
    
    Args:
        urls: List of URLs to test
        model: The trained model
        threshold: Classification threshold
        char2idx: Character to index mapping
        device: The device to run the model on
        overflow_bloom: The overflow Bloom filter
        batch_size: Size of batches for model inference
        return_detailed_results: Whether to return detailed results for each URL
        
    Returns:
        Dictionary with results and timing information
    """
    import time
    import numpy as np
    from tqdm import tqdm
    
    total_urls = len(urls)
    num_batches = (total_urls + batch_size - 1) // batch_size
    
    # Prepare to collect results
    results = {
        'model_positives': 0,
        'bloom_positives': 0,
        'total_positives': 0,
        'failed_urls': [],
        'successful_urls': 0,
        'timing': {
            'preprocessing_time': 0,
            'inference_time': 0,
            'bloom_filter_time': 0,
            'total_time': 0,
            'warmup_time': 0  # Track warmup time
        },
        'batch_latencies': []  # Track individual batch latencies
    }
    
    # Add storage for detailed URL results if requested
    if return_detailed_results:
        results['url_results'] = {}
    
    print(f"Processing {total_urls} URLs in {num_batches} batches (batch size: {batch_size})...")
    start_time = time.time()
    
    # Pre-processing: Convert all URLs to indices first
    preprocessing_start = time.time()
    indices_list = []
    valid_urls = []
    
    for url in urls:
        try:
            indices = url_to_indices(url, char2idx)
            indices_list.append(indices)
            valid_urls.append(url)
        except KeyError:
            results['failed_urls'].append(url)
    
    preprocessing_end = time.time()
    results['timing']['preprocessing_time'] = (preprocessing_end - preprocessing_start) * 1000
    
    # Now we have only valid URLs and their indices
    results['successful_urls'] = len(valid_urls)
    
    # Process in batches
    model.eval()
    inference_time = 0
    bloom_filter_time = 0
    
    # Do a warm-up run with a small batch to eliminate first-batch overhead
    if len(valid_urls) > 0:
        print("Performing warmup run...")
        warmup_start = time.time()
        
        warmup_indices = [indices_list[0]]
        warmup_tensor = torch.tensor(warmup_indices, dtype=torch.long).to(device)
        
        with torch.no_grad():
            _ = model(warmup_tensor)
            
        warmup_end = time.time()
        results['timing']['warmup_time'] = (warmup_end - warmup_start) * 1000
        print(f"Warmup completed in {results['timing']['warmup_time']:.2f} ms")
    
    for i in tqdm(range(0, len(valid_urls), batch_size)):
        batch_end = min(i + batch_size, len(valid_urls))
        batch_indices = indices_list[i:batch_end]
        batch_urls = valid_urls[i:batch_end]
        current_batch_size = len(batch_indices)
        
        # Track complete batch processing time (true latency for this batch)
        batch_start = time.time()
        
        # Create tensor and move to device
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
        
        # Inference on batch
        inference_start = time.time()
        with torch.no_grad():
            batch_probs = model(batch_tensor).cpu().numpy()
        inference_end = time.time()
        batch_inference_time = (inference_end - inference_start) * 1000
        inference_time += batch_inference_time
        
        # Check model predictions (vectorized)
        batch_model_predictions = (batch_probs >= threshold)
        results['model_positives'] += np.sum(batch_model_predictions)
        
        # Check bloom filter (can't be vectorized)
        bloom_start = time.time()
        batch_bloom_positives = 0
        
        # Track which URLs match the bloom filter
        bloom_matches = np.zeros(current_batch_size, dtype=bool)
        
        if overflow_bloom is not None:
            for j, url in enumerate(batch_urls):
                in_bloom = url in overflow_bloom
                if in_bloom:
                    batch_bloom_positives += 1
                    bloom_matches[j] = True
        
        bloom_end = time.time()
        batch_bloom_time = (bloom_end - bloom_start) * 1000
        bloom_filter_time += batch_bloom_time
        
        # Record the end of batch processing
        batch_end_time = time.time()
        batch_latency = (batch_end_time - batch_start) * 1000  # in milliseconds
        
        # Store this batch's complete latency
        results['batch_latencies'].append(batch_latency)
        
        # If detailed results are requested, store per-URL information
        if return_detailed_results:
            for j, url in enumerate(batch_urls):
                model_pred = bool(batch_model_predictions[j])
                in_bloom = bool(bloom_matches[j])
                is_positive = model_pred or in_bloom
                
                results['url_results'][url] = {
                    'model_prediction': model_pred,
                    'probability': float(batch_probs[j]),
                    'in_bloom': in_bloom,
                    'is_positive': is_positive,
                    'timing': {
                        'inference': batch_inference_time / current_batch_size,  # Approximate per-URL timing
                        'bloom_filter': batch_bloom_time / current_batch_size if overflow_bloom is not None else 0
                    }
                }
        
        # Calculate total positives as the number of URLs that are either model-positive or bloom-positive
        total_positives_in_batch = sum(1 for j in range(current_batch_size) if batch_model_predictions[j] or bloom_matches[j])
        results['bloom_positives'] += batch_bloom_positives
        results['total_positives'] += total_positives_in_batch
    
    end_time = time.time()
    
    # Record timing information
    results['timing']['inference_time'] = inference_time
    results['timing']['bloom_filter_time'] = bloom_filter_time
    results['timing']['total_time'] = (end_time - start_time) * 1000
    
    # Calculate batch latency statistics
    if results['batch_latencies']:
        results['batch_stats'] = {
            'min_latency': np.min(results['batch_latencies']),
            'max_latency': np.max(results['batch_latencies']),
            'avg_latency': np.mean(results['batch_latencies']),
            'median_latency': np.median(results['batch_latencies']),
            'p95_latency': np.percentile(results['batch_latencies'], 95),
            'p99_latency': np.percentile(results['batch_latencies'], 99),
            'std_dev': np.std(results['batch_latencies'])
        }
    
    # Calculate per-URL metrics (for throughput estimation, not true latency)
    if results['successful_urls'] > 0:
        results['timing']['preprocessing_per_url'] = results['timing']['preprocessing_time'] / results['successful_urls']
        results['timing']['inference_per_url'] = results['timing']['inference_time'] / results['successful_urls']
        results['timing']['bloom_filter_per_url'] = results['timing']['bloom_filter_time'] / results['successful_urls']
        results['timing']['total_per_url'] = results['timing']['total_time'] / results['successful_urls']
    
    # Print summary
    print("\nBatch Inference Summary:")
    print(f"Total URLs: {total_urls}")
    print(f"Successfully processed: {results['successful_urls']}")
    print(f"Failed: {len(results['failed_urls'])}")
    print(f"Model positives: {results['model_positives']} ({results['model_positives']/results['successful_urls']*100:.2f}%)")
    
    if overflow_bloom is not None:
        print(f"Bloom filter positives: {results['bloom_positives']} ({results['bloom_positives']/results['successful_urls']*100:.2f}%)")
    
    print(f"Total positives: {results['total_positives']} ({results['total_positives']/results['successful_urls']*100:.2f}%)")
    print(f"Total execution time: {results['timing']['total_time']/1000:.2f} seconds")
    
    print("\nTiming Information (Total):")
    print(f"- Warmup time: {results['timing']['warmup_time']:.2f} ms")
    print(f"- Preprocessing: {results['timing']['preprocessing_time']:.2f} ms")
    print(f"- Model inference: {results['timing']['inference_time']:.2f} ms")
    print(f"- Bloom filter: {results['timing']['bloom_filter_time']:.2f} ms")
    print(f"- Total: {results['timing']['total_time']:.2f} ms")
    
    print("\nBatch Latency (true response time per batch):")
    if 'batch_stats' in results:
        print(f"- Minimum: {results['batch_stats']['min_latency']:.2f} ms")
        print(f"- Maximum: {results['batch_stats']['max_latency']:.2f} ms")
        print(f"- Average: {results['batch_stats']['avg_latency']:.2f} ms")
        print(f"- Median: {results['batch_stats']['median_latency']:.2f} ms")
        print(f"- 95th percentile: {results['batch_stats']['p95_latency']:.2f} ms")
        print(f"- 99th percentile: {results['batch_stats']['p99_latency']:.2f} ms")
        print(f"- Standard Deviation: {results['batch_stats']['std_dev']:.2f} ms")
    
    print("\nThroughput Information (Per URL):")
    print(f"- Preprocessing: {results['timing']['preprocessing_per_url']:.2f} ms/URL")
    print(f"- Model inference: {results['timing']['inference_per_url']:.2f} ms/URL")
    print(f"- Bloom filter: {results['timing']['bloom_filter_per_url']:.2f} ms/URL")
    print(f"- Total: {results['timing']['total_per_url']:.2f} ms/URL")
    print(f"- Estimated throughput: {1000 / results['timing']['total_per_url']:.2f} URLs/second")
    
    return results

# Rename existing batch_test to sequential_test
def sequential_test(urls, model, threshold, char2idx, device, overflow_bloom=None, verbose=False):
    """
    Test URLs sequentially (one by one) and provide detailed timing statistics.
    This is not optimized for throughput but provides detailed per-URL statistics.
    
    Args:
        urls: List of URLs to test
        model: The trained model
        threshold: Classification threshold
        char2idx: Character to index mapping
        device: The device to run the model on
        overflow_bloom: The overflow Bloom filter
        verbose: Whether to print results for each URL
        
    Returns:
        Dictionary with results and timing statistics
    """
    import time
    import numpy as np
    
    total_urls = len(urls)
    successful_tests = 0
    results = {
        'model_positives': 0,
        'bloom_positives': 0,
        'total_positives': 0,
        'failed_tests': 0,
        'timing': {
            'time_to_create_tensor_from_indices': [],
            'time_to_transfer_tensor_to_device': [],
            'time_to_run_model': [],
            'time_to_check_overflow_bloom_filter': [],
            'total': []
        }
    }
    
    print(f"Testing {total_urls} URLs sequentially...")
    start_time = time.time()
    
    for i, url in enumerate(urls):
        if verbose:
            print(f"\nTesting URL {i+1}/{total_urls}: {url}")
        
        model_prediction, prob, is_in_overflow, final_result, timing_info = test_url(
            url, model, threshold, char2idx, device, overflow_bloom
        )
        
        if model_prediction is None:
            results['failed_tests'] += 1
            if verbose:
                print("  Failed to process URL (contains unknown characters)")
            continue
        
        successful_tests += 1
        
        # Record results
        if model_prediction:
            results['model_positives'] += 1
        if is_in_overflow:
            results['bloom_positives'] += 1
        if final_result:
            results['total_positives'] += 1
            
        # Record timing
        for key, value in timing_info.items():
            if key in results['timing']:
                results['timing'][key].append(value)
            
        if verbose:
            print(f"  Model: {'Malicious' if model_prediction else 'Benign'} ({prob:.4f})")
            if overflow_bloom is not None:
                print(f"  Bloom: {'Yes' if is_in_overflow else 'No'}")
            print(f"  Final: {'Malicious' if final_result else 'Benign'}")
            print(f"  Time: {timing_info['total']:.2f} ms")
    
    # Calculate statistics
    total_time = time.time() - start_time
    
    if successful_tests > 0:
        # Calculate timing averages
        timing_keys = list(results['timing'].keys())
        for key in timing_keys:
            if results['timing'][key]:
                results['timing'][key + '_avg'] = np.mean(results['timing'][key])
                results['timing'][key + '_min'] = np.min(results['timing'][key])
                results['timing'][key + '_max'] = np.max(results['timing'][key])
                results['timing'][key + '_median'] = np.median(results['timing'][key])
                results['timing'][key + '_std'] = np.std(results['timing'][key])
    
    # Print summary
    print("\nSequential Test Summary:")
    print(f"Total URLs tested: {total_urls}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {results['failed_tests']}")
    print(f"Model positives: {results['model_positives']} ({results['model_positives']/successful_tests*100:.2f}%)")
    
    if overflow_bloom is not None:
        print(f"Bloom filter positives: {results['bloom_positives']} ({results['bloom_positives']/successful_tests*100:.2f}%)")
    
    print(f"Total positives: {results['total_positives']} ({results['total_positives']/successful_tests*100:.2f}%)")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("\nAverage Timing (ms):")
    timing_display = [
        ('time_to_create_tensor_from_indices', 'CPU Processing'),
        ('time_to_transfer_tensor_to_device', 'Device Transfer'),
        ('time_to_run_model', 'Model Inference'),
        ('time_to_check_overflow_bloom_filter', 'Bloom Filter Lookup'),
        ('total', 'Total')
    ]
    
    for key, display_name in timing_display:
        if key + '_avg' in results['timing']:
            print(f"- {display_name}: {results['timing'][key + '_avg']:.2f} ms (±{results['timing'][key + '_std']:.2f})")
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Learned Bloom Filter Testing')
    parser.add_argument('--run', type=str, help='Specific run directory to use (default: latest)')
    parser.add_argument('--sequential', action='store_true', help='Run sequential testing using test data')
    parser.add_argument('--batch', action='store_true', help='Run batched inference using test data')
    parser.add_argument('--batch-size', type=int, default=64, help='Size of batches for model inference')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--verbose', action='store_true', help='Show detailed results for sequential testing')
    parser.add_argument('--output', type=str, help='Output file to save results as JSON')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive mode (useful for batch testing)')
    args = parser.parse_args()
    
    data_dir = "data"
    model_dir = "models"
    training_dataset_name = "training_dataset.csv"
    negative_dataset_name = "negative_dataset.csv"
    training_dataset_path = os.path.join(data_dir, training_dataset_name)
    negative_dataset_path = os.path.join(data_dir, negative_dataset_name)
    
    # Determine which run directory to use
    if args.run:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(model_dir, args.run)
    else:
        # Try to use the latest_run symlink
        latest_run_link = os.path.join(model_dir, "latest_run")
        if os.path.exists(latest_run_link) and os.path.islink(latest_run_link):
            run_dir = os.path.realpath(latest_run_link)
        else:
            # If no symlink, find the latest run by timestamp
            run_dirs = get_run_dirs(model_dir)
            if run_dirs:
                run_dir = run_dirs[0]  # First one is the newest
            else:
                print("No run directories found. Please train a model first.")
                return
    
    print(f"Using model artifacts from: {run_dir}")
    
    # Set up paths for model artifacts
    model_save_path = os.path.join(run_dir, "url_classifier.pt")
    overflow_bloom_filter_path = os.path.join(run_dir, "overflow_bloom.pkl")
    threshold_path = os.path.join(run_dir, "threshold.txt")
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")
    
    # Check for CUDA, MPS, or CPU availability
    device = torch.device(
        'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load hyperparameters
    hyperparams = load_hyperparams(hyperparams_path)
    
    # Load data to build vocabulary
    print("Loading datasets to build vocabulary...")
    train_df, val_df, test_df, negative_df = load_data(training_dataset_path, negative_dataset_path)
    
    # Build vocabulary
    print("Building vocabulary...")
    char2idx, idx2char = build_vocab([train_df, val_df, test_df, negative_df])
    vocab_size = len(char2idx)
    
    # Initialize the model with hyperparameters from file if available
    print("Initializing model...")
    if hyperparams:
        embedding_dim = hyperparams.get('embedding_dim', 32)
        hidden_dim = hyperparams.get('hidden_dim', 16)
    else:
        embedding_dim = 32
        hidden_dim = 16
    
    model = URLClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
    
    # Load the trained model
    if os.path.exists(model_save_path):
        model = load_model(model, model_save_path, device)
    else:
        print(f"Error: No model found at {model_save_path}")
        return
    
    # Load threshold
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Threshold loaded: {threshold}")
    else:
        # Default threshold if not saved
        threshold = 0.5
        print(f"No threshold file found. Using default: {threshold}")
    
    # Load the overflow Bloom filter
    overflow_bloom = load_bloom_filter(overflow_bloom_filter_path)
    
    # Run sequential testing if requested
    if args.sequential:
        print("\nRunning sequential test...")
        # Sample URLs from test set
        sample_size = min(args.sample_size, len(test_df))
        test_sample = test_df.sample(sample_size)
        sample_urls = test_sample['url'].tolist()
        
        # Run sequential test
        seq_results = sequential_test(sample_urls, model, threshold, char2idx, device, overflow_bloom, verbose=args.verbose)
        
        # Save results to file if specified
        if args.output and 'seq_results' in locals():
            try:
                output_data = seq_results.copy()  # Create a copy to avoid modifying the original
                # Add batch size and other metadata
                output_data['batch_size'] = 1
                output_data['sample_size'] = args.sample_size
                
                # Ensure all values are JSON serializable
                for key in output_data:
                    if isinstance(output_data[key], np.ndarray):
                        output_data[key] = output_data[key].tolist()
                    elif isinstance(output_data[key], np.integer):
                        output_data[key] = int(output_data[key])
                    elif isinstance(output_data[key], np.floating):
                        output_data[key] = float(output_data[key])
                    elif key == 'batch_latencies' and isinstance(output_data[key], list):
                        # Convert any numpy values in the list to Python types
                        output_data[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in output_data[key]]
                    elif isinstance(output_data[key], dict):
                        # Handle nested dictionaries
                        for nested_key in output_data[key]:
                            if isinstance(output_data[key][nested_key], np.ndarray):
                                output_data[key][nested_key] = output_data[key][nested_key].tolist()
                            elif isinstance(output_data[key][nested_key], np.integer):
                                output_data[key][nested_key] = int(output_data[key][nested_key])
                            elif isinstance(output_data[key][nested_key], np.floating):
                                output_data[key][nested_key] = float(output_data[key][nested_key])
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=4)
                print(f"Results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results to {args.output}: {e}")
        
        # # Also test some negative samples
        # print("\nTesting negative samples...")
        # neg_sample_size = min(args.sample_size, len(negative_df))
        # neg_sample = negative_df.sample(neg_sample_size)
        # neg_sample_urls = neg_sample['url'].tolist()
        
        # neg_seq_results = sequential_test(neg_sample_urls, model, threshold, char2idx, device, overflow_bloom, verbose=args.verbose)
        
        # print("\nSequential testing complete. Entering interactive mode...")
    
    # Run batched inference if requested
    if args.batch:
        print("\nRunning batched inference...")
        # Sample URLs from test set
        sample_size = min(args.sample_size, len(test_df))
        test_sample = test_df.sample(sample_size)
        sample_urls = test_sample['url'].tolist()
        
        # Run batched inference with specified batch size
        batch_results = batch_inference(sample_urls, model, threshold, char2idx, device, overflow_bloom, batch_size=args.batch_size)
        
        # Save results to file if specified
        if args.output and 'batch_results' in locals():
            try:
                output_data = batch_results.copy()  # Create a copy to avoid modifying the original
                # Add batch size and other metadata
                output_data['batch_size'] = args.batch_size
                output_data['sample_size'] = args.sample_size
                
                # Ensure all values are JSON serializable
                for key in output_data:
                    if isinstance(output_data[key], np.ndarray):
                        output_data[key] = output_data[key].tolist()
                    elif isinstance(output_data[key], np.integer):
                        output_data[key] = int(output_data[key])
                    elif isinstance(output_data[key], np.floating):
                        output_data[key] = float(output_data[key])
                    elif key == 'batch_latencies' and isinstance(output_data[key], list):
                        # Convert any numpy values in the list to Python types
                        output_data[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in output_data[key]]
                    elif isinstance(output_data[key], dict):
                        # Handle nested dictionaries
                        for nested_key in output_data[key]:
                            if isinstance(output_data[key][nested_key], np.ndarray):
                                output_data[key][nested_key] = output_data[key][nested_key].tolist()
                            elif isinstance(output_data[key][nested_key], np.integer):
                                output_data[key][nested_key] = int(output_data[key][nested_key])
                            elif isinstance(output_data[key][nested_key], np.floating):
                                output_data[key][nested_key] = float(output_data[key][nested_key])
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=4)
                print(f"Results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results to {args.output}: {e}")
        
        # # Also test some negative samples
        # print("\nTesting negative samples...")
        # neg_sample_size = min(args.sample_size, len(negative_df))
        # neg_sample = negative_df.sample(neg_sample_size)
        # neg_sample_urls = neg_sample['url'].tolist()
        
        # neg_batch_results = batch_inference(neg_sample_urls, model, threshold, char2idx, device, overflow_bloom, batch_size=args.batch_size)
        
        print("\nBatched inference complete. Entering interactive mode...")
    
    # Skip interactive mode if --no-interactive is specified
    if args.no_interactive:
        print("Interactive mode disabled. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("Interactive Learned Bloom Filter Testing")
    print("=" * 50)
    print(f"Using run: {os.path.basename(run_dir)}")
    print("Available functions:")
    print("- test(url): Test a URL against the model and bloom filter (short version)")
    print("- test_url(url, model, threshold, char2idx, device, overflow_bloom): Full test function with timing")
    print("- sequential_test(urls, model, threshold, char2idx, device, overflow_bloom, verbose): Test URLs sequentially")
    print("- batch_inference(urls, model, threshold, char2idx, device, overflow_bloom, batch_size): Run batched inference")
    print("\nVariables available:")
    print("- model: The trained URL classifier")
    print("- threshold: Classification threshold")
    print("- overflow_bloom: The overflow Bloom filter")
    print("- char2idx: Character to index mapping")
    print("- test_df, train_df, val_df, negative_df: Original datasets")
    if hyperparams:
        print("\nModel hyperparameters:")
        for key, value in hyperparams.items():
            print(f"- {key}: {value}")
    print("=" * 50 + "\n")
    
    # Define a helper function for the interactive session
    def test(url):
        """Helper function to test a URL in the interactive session."""
        model_prediction, prob, is_in_overflow, final_result, timing_info = test_url(
            url, model, threshold, char2idx, device, overflow_bloom
        )
        
        if model_prediction is None:
            return "Error processing URL"
        
        print(f"URL: {url}")
        print(f"Model probability: {prob:.4f} (threshold: {threshold:.4f})")
        print(f"Model prediction: {'Malicious' if model_prediction else 'Benign'}")
        
        if overflow_bloom is not None:
            print(f"In overflow Bloom filter: {'Yes' if is_in_overflow else 'No'}")
        
        print(f"Final result: {'Malicious' if final_result else 'Benign'}")
        
        if timing_info:
            print("\nTiming information:")
            for key, value in timing_info.items():
                print(f"- {key}: {value:.2f} ms")
        
        return final_result
    
    # Start interactive session
    embed()

if __name__ == "__main__":
    main() 