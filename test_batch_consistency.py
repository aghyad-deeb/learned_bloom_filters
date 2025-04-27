#!/usr/bin/env python
"""
Test script to verify that batch inference produces the same results as sequential testing.
This script compares the classification outputs of batch and sequential methods for the same inputs.
"""
import os
import argparse
import json
import random
import numpy as np
import torch
import pickle
from tqdm import tqdm

# Import required classes from train_and_run first (needed for unpickling bloom filter)
from train_and_run import URLClassifier, BloomFilter

# Import required functions from interactive.py
from interactive import (
    load_model, load_hyperparams, 
    test_url, batch_inference, sequential_test,
    get_run_dirs, build_vocab, load_data
)

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

def setup_test_environment(run_dir=None, sample_size=100, random_seed=42):
    """Set up the test environment by loading model, threshold, and bloom filter."""
    print("Setting up test environment...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Determine which run directory to use
    data_dir = "data"
    model_dir = "models"
    
    if run_dir is None:
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
                raise ValueError("No run directories found. Please train a model first.")
    
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
    training_dataset_name = "training_dataset.csv"
    negative_dataset_name = "negative_dataset.csv"
    training_dataset_path = os.path.join(data_dir, training_dataset_name)
    negative_dataset_path = os.path.join(data_dir, negative_dataset_name)
    
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
    
    # Import URLClassifier here to avoid circular import
    from train_and_run import URLClassifier
    model = URLClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
    
    # Load the trained model
    if os.path.exists(model_save_path):
        model = load_model(model, model_save_path, device)
    else:
        raise FileNotFoundError(f"No model found at {model_save_path}")
    
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
    
    # Sample URLs from test set
    print(f"Sampling {sample_size} URLs for testing...")
    test_sample = test_df.sample(sample_size, random_state=random_seed)
    sample_urls = test_sample['url'].tolist()
    
    return {
        'model': model,
        'threshold': threshold,
        'char2idx': char2idx,
        'device': device,
        'overflow_bloom': overflow_bloom,
        'sample_urls': sample_urls,
        'test_df': test_df,
        'hyperparams': hyperparams
    }

def run_sequential_test(env, verbose=False, return_url_results=False):
    """Run sequential test on sample URLs."""
    print("\nRunning sequential test...")
    
    # If we need detailed per-URL results, we'll collect them
    url_results = {}
    total_positives = 0
    model_positives = 0
    bloom_positives = 0
    total_time = 0
    
    if return_url_results:
        # Test each URL individually and store results
        for url in tqdm(env['sample_urls']):
            model_pred, prob, is_in_bloom, result, timing = test_url(
                url, 
                env['model'], 
                env['threshold'], 
                env['char2idx'], 
                env['device'], 
                env['overflow_bloom']
            )
            
            # Store result for this URL
            url_results[url] = {
                'model_prediction': bool(model_pred),
                'probability': float(prob),
                'in_bloom': bool(is_in_bloom),
                'is_positive': bool(model_pred or is_in_bloom),
                'timing': timing
            }
            
            # Update counters
            if model_pred or is_in_bloom:
                total_positives += 1
            if model_pred:
                model_positives += 1
            if is_in_bloom:
                bloom_positives += 1
            
            # Add timing
            total_time += sum(timing.values()) if timing else 0
        
        # Create results object similar to sequential_test
        results = {
            'total_urls': len(env['sample_urls']),
            'total_positives': total_positives,
            'model_positives': model_positives,
            'bloom_positives': bloom_positives,
            'total_time': total_time,
            'avg_time_per_url': total_time / len(env['sample_urls']) if env['sample_urls'] else 0,
            'url_results': url_results  # Add detailed results
        }
    else:
        # Use the original sequential_test function
        results = sequential_test(
            env['sample_urls'], 
            env['model'], 
            env['threshold'], 
            env['char2idx'], 
            env['device'], 
            env['overflow_bloom'],
            verbose=verbose
        )
    
    return results

def run_batch_inference(env, batch_size=64, return_url_results=False):
    """Run batch inference on sample URLs."""
    print(f"\nRunning batch inference with batch_size={batch_size}...")
    
    if return_url_results:
        # We need to modify our approach since batch_inference doesn't return per-URL results
        # We'll run batch_inference but also track which URLs are positive
        
        # First get the standard batch results
        batch_results = batch_inference(
            env['sample_urls'], 
            env['model'], 
            env['threshold'], 
            env['char2idx'], 
            env['device'], 
            env['overflow_bloom'],
            batch_size=batch_size,
            return_detailed_results=True  # This is a new parameter we'll add to batch_inference
        )
        
        return batch_results
    else:
        # Use the original batch_inference function
        batch_results = batch_inference(
            env['sample_urls'], 
            env['model'], 
            env['threshold'], 
            env['char2idx'], 
            env['device'], 
            env['overflow_bloom'],
            batch_size=batch_size
        )
        
        return batch_results

def compare_results(seq_results, batch_results):
    """Compare the results of sequential and batch processing."""
    print("\nComparing sequential and batch results...")
    
    # Check if key result metrics match
    seq_positives = seq_results.get('total_positives', 0)
    batch_positives = batch_results.get('total_positives', 0)
    
    print(f"Sequential positives: {seq_positives}")
    print(f"Batch positives: {batch_positives}")
    
    if seq_positives == batch_positives:
        print("✅ Total positives match!")
    else:
        print("❌ Total positives do not match!")
    
    seq_model_positives = seq_results.get('model_positives', 0)
    batch_model_positives = batch_results.get('model_positives', 0)
    
    print(f"Sequential model positives: {seq_model_positives}")
    print(f"Batch model positives: {batch_model_positives}")
    
    if seq_model_positives == batch_model_positives:
        print("✅ Model positives match!")
    else:
        print("❌ Model positives do not match!")
    
    seq_bloom_positives = seq_results.get('bloom_positives', 0)
    batch_bloom_positives = batch_results.get('bloom_positives', 0)
    
    print(f"Sequential bloom positives: {seq_bloom_positives}")
    print(f"Batch bloom positives: {batch_bloom_positives}")
    
    if seq_bloom_positives == batch_bloom_positives:
        print("✅ Bloom filter positives match!")
    else:
        print("❌ Bloom filter positives do not match!")
    
    # Check if we have detailed URL results to compare
    if 'url_results' in seq_results and 'url_results' in batch_results:
        print("\nComparing per-URL results...")
        seq_url_results = seq_results['url_results']
        batch_url_results = batch_results['url_results']
        
        # Count mismatches
        mismatches = 0
        urls_with_mismatches = []
        
        for url, seq_result in seq_url_results.items():
            if url in batch_url_results:
                batch_result = batch_url_results[url]
                
                # Compare the is_positive results
                if seq_result['is_positive'] != batch_result['is_positive']:
                    mismatches += 1
                    urls_with_mismatches.append(url)
        
        if mismatches == 0:
            print(f"✅ All {len(seq_url_results)} URLs have matching results between sequential and batch processing!")
        else:
            print(f"❌ Found {mismatches} URLs with different results out of {len(seq_url_results)} total URLs.")
            print("First few mismatches:")
            for i, url in enumerate(urls_with_mismatches[:5]):
                print(f"  URL {i+1}: {url}")
                print(f"    Sequential: {seq_url_results[url]['is_positive']}")
                print(f"    Batch: {batch_url_results[url]['is_positive']}")
    
    # Create a summary
    matches = (seq_positives == batch_positives and 
               seq_model_positives == batch_model_positives and 
               seq_bloom_positives == batch_bloom_positives)
    
    # Add URL-level matching to summary if available
    if 'url_results' in seq_results and 'url_results' in batch_results:
        url_level_matches = (mismatches == 0)
        matches = matches and url_level_matches
    
    if matches:
        print("\n🎉 All results match! Batch inference produces the same classification results as sequential testing.")
    else:
        print("\n⚠️ Some results do not match! Batch inference may have issues.")
    
    return matches

def run_detailed_comparison(env, batch_size=64):
    """Run a detailed comparison by testing each URL individually in both modes."""
    print("\nRunning detailed URL-by-URL comparison...")
    
    # Get per-URL results from both methods
    seq_results = run_sequential_test(env, return_url_results=True)
    batch_results = run_batch_inference(env, batch_size, return_url_results=True)
    
    # Compare the results
    return compare_results(seq_results, batch_results)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test batch inference consistency')
    parser.add_argument('--run', type=str, help='Specific run directory to use (default: latest)')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--detailed', action='store_true', default=True, help='Run URL-by-URL comparison')
    args = parser.parse_args()
    
    # Set up test environment
    env = setup_test_environment(args.run, args.sample_size, args.seed)
    
    # Run tests
    if args.detailed:
        # If detailed comparison requested, use that function which compares URL by URL
        test_passed = run_detailed_comparison(env, args.batch_size)
    else:
        # Otherwise just compare the aggregate counts
        seq_results = run_sequential_test(env, args.verbose)
        batch_results = run_batch_inference(env, args.batch_size)
        test_passed = compare_results(seq_results, batch_results)
    
    # Final verdict
    if test_passed:
        print("\n✅ PASS: Batch inference produces the same results as sequential testing.")
        return 0
    else:
        print("\n❌ FAIL: Batch inference results differ from sequential testing.")
        return 1

if __name__ == "__main__":
    main() 