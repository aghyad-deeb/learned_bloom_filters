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
        A tuple of (model_prediction, model_probability, is_in_overflow, final_result)
    """
    # Convert URL to tensor
    try:
        idx_seq = torch.tensor([url_to_indices(url, char2idx)], dtype=torch.long).to(device)
    except KeyError as e:
        print(f"Error: URL contains characters not in the vocabulary: {e}")
        return None, None, None, None
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        prob = model(idx_seq).cpu().item()
    
    # Check if above threshold
    model_prediction = prob >= threshold
    
    # Check if in overflow Bloom filter
    is_in_overflow = url in overflow_bloom if overflow_bloom is not None else False
    
    # Final result is positive if either model predicts positive or URL is in overflow Bloom filter
    final_result = model_prediction or is_in_overflow
    
    return model_prediction, prob, is_in_overflow, final_result

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Learned Bloom Filter Testing')
    parser.add_argument('--run', type=str, help='Specific run directory to use (default: latest)')
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
    
    print("\n" + "=" * 50)
    print("Interactive Learned Bloom Filter Testing")
    print("=" * 50)
    print(f"Using run: {os.path.basename(run_dir)}")
    print("Available functions:")
    print("- test(url): Test a URL against the model and bloom filter (short version)")
    print("- test_url(url, model, threshold, char2idx, device, overflow_bloom): Full test function")
    print("\nVariables available:")
    print("- model: The trained URL classifier")
    print("- threshold: Classification threshold")
    print("- overflow_bloom: The overflow Bloom filter")
    print("- char2idx: Character to index mapping")
    if hyperparams:
        print("\nModel hyperparameters:")
        for key, value in hyperparams.items():
            print(f"- {key}: {value}")
    print("=" * 50 + "\n")
    
    # Define a helper function for the interactive session
    def test(url):
        """Helper function to test a URL in the interactive session."""
        model_prediction, prob, is_in_overflow, final_result = test_url(
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
        
        return final_result
    
    # Start interactive session
    embed()

if __name__ == "__main__":
    main() 