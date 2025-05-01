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
import datetime
from tqdm import tqdm
import math
import bitarray

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
        return hyperparams
    else:
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
    
    # Check if in overflow Bloom filter ONLY if model prediction is negative
    is_in_overflow = False
    timing_info['time_to_check_overflow_bloom_filter'] = 0
    
    if overflow_bloom is not None and not model_prediction:
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

def calculate_size_metrics(model, overflow_bloom, run_dir):
    """
    Calculate size metrics for the model, overflow bloom filter, and traditional bloom filter.
    
    Args:
        model: The trained model
        overflow_bloom: The overflow Bloom filter
        model_save_path: Path to the saved model file (used to find hyperparameters)
        
    Returns:
        Dictionary containing size metrics in bytes
    """
    size_metrics = {
        'model_size_bytes': 0,
        'overflow_bloom_filter_size_bytes': 0,
        'traditional_bloom_filter_size_bytes': 0
    }
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    size_metrics['model_size_bytes'] = model_size
    
    # Calculate overflow bloom filter size if available
    if overflow_bloom is not None:
        size_metrics['overflow_bloom_filter_size_bytes'] = overflow_bloom.__sizeof__()
    
    # Calculate traditional bloom filter size using test dataset
    data_dir = "data"
    training_dataset_name = "training_dataset.csv"
    negative_dataset_name = "negative_dataset.csv"
    training_dataset_path = os.path.join(data_dir, training_dataset_name)
    negative_dataset_path = os.path.join(data_dir, negative_dataset_name)
    
    # Get desired FPR and random seed from hyperparameters
    hyperparams_path = os.path.abspath(os.path.join(run_dir, "hyperparams.json"))
    print(f"Hyperparams path: {hyperparams_path}")
    hyperparams = load_hyperparams(hyperparams_path)
    desired_fpr = hyperparams['desired_fpr']  # Default to 1% if not found
    random_seed = hyperparams['random_seed']  # Default to 42 if not found
    
    # Load test dataset with the same random seed used during training
    _, _, test_df, _ = load_data(training_dataset_path, negative_dataset_path, random_seed=random_seed)
    
    # Get number of positive items in test set
    n_positive = np.sum(test_df['label'] == 1)
    
    # Calculate traditional bloom filter size using the same formula as in train_and_run.py
    m_per_element = -math.log(desired_fpr) / (math.log(2)**2)
    traditional_size = n_positive * m_per_element / 8  # Convert bits to bytes
    size_metrics['traditional_bloom_filter_size_bytes'] = traditional_size
    
    return size_metrics

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
    
    total_urls = len(urls)
    num_batches = (total_urls + batch_size - 1) // batch_size
    
    # Prepare to collect results
    results = {
        'model_positives': 0,
        'bloom_positives': 0,
        'total_positives': 0,
        'num_successful_urls': 0,
        'failed_urls': [],
        'timing': {
            'total_preprocessing_time': 0,
            'preprocessing_time_per_url': 0,
            'warmup_batch_time': 0,
            'moving_tensor_to_device_per_batch_time_list': [],
            'inference_per_batch_time_list': [],
            'overflow_bloom_filter_per_batch_time_list': [],
            'batch_total_latency_per_batch_time_list': [],
        }
    }
    
    # Add storage for detailed URL results if requested
    if return_detailed_results:
        results['per_url_results'] = {}
    
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
    results['timing']['total_preprocessing_time'] = (preprocessing_end - preprocessing_start) * 1000
    results['timing']['preprocessing_time_per_url'] = results['timing']['total_preprocessing_time'] / len(valid_urls)
    
    # Now we have only valid URLs and their indices
    results['num_successful_urls'] = len(valid_urls)
    
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
        results['timing']['warmup_batch_time'] = (warmup_end - warmup_start) * 1000
    
    results['timing']['moving_tensor_to_device_per_batch_time_list'] = []
    results['timing']['inference_per_batch_time_list'] = []
    results['timing']['overflow_bloom_filter_per_batch_time_list'] = []
    results['timing']['batch_total_latency_per_batch_time_list'] = []

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
        results['timing']['moving_tensor_to_device_per_batch_time_list'].append((inference_start - batch_start) * 1000)
        with torch.no_grad():
            batch_probs = model(batch_tensor).cpu().numpy()
        inference_end = time.time()
        
        batch_inference_time = (inference_end - inference_start) * 1000
        results['timing']['inference_per_batch_time_list'].append(batch_inference_time)
        
        batch_model_predictions = (batch_probs >= threshold)
        results['model_positives'] += np.sum(batch_model_predictions)
        
        # Check bloom filter (can't be vectorized)
        bloom_start = time.time()
        batch_bloom_positives = 0
        
        # Track which URLs match the bloom filter
        bloom_matches = np.zeros(current_batch_size, dtype=bool)
        
        if overflow_bloom is not None:
            for j, url in enumerate(batch_urls):
                # Only check the bloom filter if the model prediction is negative
                if not batch_model_predictions[j]:
                    in_bloom = url in overflow_bloom
                    if in_bloom:
                        batch_bloom_positives += 1
                        bloom_matches[j] = True
        
        bloom_end = time.time()
        batch_bloom_time = (bloom_end - bloom_start) * 1000
        results['timing']['overflow_bloom_filter_per_batch_time_list'].append(batch_bloom_time)
        
        # Record the end of batch processing
        batch_end_time = time.time()
        batch_latency = (batch_end_time - batch_start) * 1000  # in milliseconds
        
        # Store this batch's complete latency
        results['timing']['batch_total_latency_per_batch_time_list'].append(batch_latency)
        
        # If detailed results are requested, store per-URL information
        if return_detailed_results:
            for j, url in enumerate(batch_urls):
                model_pred = bool(batch_model_predictions[j])
                in_bloom = bool(bloom_matches[j])
                is_positive = model_pred or in_bloom
                
                results['per_url_results'][url] = {
                    'model_prediction': model_pred,
                    'probability': float(batch_probs[j]),
                    'in_bloom': in_bloom,
                    'is_positive': is_positive,
                }
        
        # Calculate total positives as the number of URLs that are either model-positive or bloom-positive
        total_positives_in_batch = sum(1 for j in range(current_batch_size) if batch_model_predictions[j] or bloom_matches[j])
        results['bloom_positives'] += batch_bloom_positives
        results['total_positives'] += total_positives_in_batch
    
    return results

def measure_traditional_bloom_performance(urls, desired_fpr, test_df):
    """
    Measure the performance of a traditional Bloom filter on a set of URLs.
    
    Args:
        urls: List of URLs to test
        desired_fpr: Desired false positive rate
        test_df: DataFrame containing the test data
        
    Returns:
        Dictionary with results and timing information
    """
    # Get number of positive items in test set
    n_positive = np.sum(test_df['label'] == 1)
    
    # Create and populate traditional Bloom filter
    print("\nBuilding traditional Bloom filter...")
    traditional_bloom = BloomFilter(n_positive, desired_fpr)
    
    # Add all positive URLs from test set
    positive_urls = test_df[test_df['label'] == 1]['url'].tolist()
    for url in tqdm(positive_urls, desc="Adding URLs to traditional Bloom filter"):
        traditional_bloom.add(url)
    
    # Prepare results dictionary
    results = {
        'traditional_bloom_positives': 0,
        'timing': {
            'total_time': 0,
            'time_per_url': 0,
            'urls_per_second': 0
        }
    }
    
    # Test URLs against traditional Bloom filter
    print("\nTesting URLs against traditional Bloom filter...")
    start_time = time.time()
    
    for url in tqdm(urls, desc="Testing URLs"):
        if url in traditional_bloom:
            results['traditional_bloom_positives'] += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate timing metrics
    results['timing']['total_time'] = total_time * 1000  # Convert to ms
    results['timing']['time_per_url'] = (total_time * 1000) / len(urls)  # ms per URL
    results['timing']['urls_per_second'] = len(urls) / total_time
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Learned Bloom Filter Testing')
    parser.add_argument('--run', type=str, help='Specific run directory to use (default: latest)')
    parser.add_argument('--batch', action='store_true', help='Run batched inference using test data')
    parser.add_argument('--batch-size', type=int, default=64, help='Size of batches for model inference')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--verbose', action='store_true', help='Show detailed results for sequential testing')
    parser.add_argument('--interactive', action='store_true', default=False, help='Enable interactive mode (useful for batch testing)')
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
        run_dirs = get_run_dirs(model_dir)
        if run_dirs:
            run_dir = os.path.abspath(run_dirs[0])  # First one is the newest
        if os.path.exists(latest_run_link) and os.path.islink(latest_run_link):
            assert run_dir == os.path.realpath(latest_run_link), \
                f"Latest run link {latest_run_link} points to " \
                f"{os.path.realpath(latest_run_link)}, but run_dir is {run_dir}"
    
    # Set up paths for model artifacts
    model_save_path = os.path.join(run_dir, "url_classifier.pt")
    overflow_bloom_filter_path = os.path.join(run_dir, "overflow_bloom.pkl")
    threshold_path = os.path.join(run_dir, "threshold.txt")
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")
    outputs_path = os.path.join(run_dir, "inference_outputs")
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    # Check for CUDA, MPS, or CPU availability
    device = torch.device(
        'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
    )
    
    # Load hyperparameters
    hyperparams = load_hyperparams(hyperparams_path)
    
    # Load data to build vocabulary
    train_df, val_df, test_df, negative_df = load_data(training_dataset_path, negative_dataset_path)
    
    # Build vocabulary
    char2idx, idx2char = build_vocab([train_df, val_df, test_df, negative_df])
    vocab_size = len(char2idx)
    
    # Initialize the model with hyperparameters from file if available
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
    else:
        raise ValueError(f"No threshold file found at {threshold_path}")
    
    
    # Load the overflow Bloom filter
    overflow_bloom = load_bloom_filter(overflow_bloom_filter_path)
    
    # Run batched inference if requested
    if args.batch:
        print("\nRunning batched inference...")
        # Sample URLs from test set
        sample_size = min(args.sample_size, len(test_df))
        test_sample = test_df.sample(sample_size)
        sample_urls = test_sample['url'].tolist()
        
        # Run batched inference with specified batch size
        batch_results = batch_inference(sample_urls, model, threshold, char2idx, device, overflow_bloom, batch_size=args.batch_size)
        
        # Calculate size metrics
        batch_results['size_metrics'] = calculate_size_metrics(model, overflow_bloom, run_dir)
        
        # Measure traditional Bloom filter performance
        print("\nMeasuring traditional Bloom filter performance...")
        traditional_results = measure_traditional_bloom_performance(sample_urls, hyperparams['desired_fpr'], test_df)
        
        # Add traditional Bloom filter results to batch results
        batch_results['traditional_bloom'] = traditional_results
        
        # Save results to json file if specified
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"batch_results_{args.sample_size}_{args.batch_size}.json"
        output_path = os.path.join(outputs_path, output_name)
        try:
            output_data = batch_results.copy()  # Create a copy to avoid modifying the original
            # Add batch size and other metadata
            output_data['batch_size'] = args.batch_size
            output_data['sample_size'] = args.sample_size
            
            # Ensure all values are JSON serializable
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            # Apply the conversion recursively to the entire data structure
            output_data = convert_to_serializable(output_data)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")
        
    # Skip interactive mode if --no-interactive is specified
    if not args.interactive:
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
            if model_prediction:
                print(f"Overflow Bloom filter: Not checked (model already predicted Malicious)")
            else:
                print(f"Overflow Bloom filter: {'Malicious' if is_in_overflow else 'Benign'}")
        
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