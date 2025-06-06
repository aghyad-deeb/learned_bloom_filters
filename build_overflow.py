import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import json
from train_and_run import (
    load_data, build_vocab, prepare_dataset, URLClassifier,
    build_overflow_filter,
    BloomFilter
)

def determine_threshold_batched(model, X_val, y_val, model_desired_fpr, batch_size=1024):
    """
    Determine the threshold for the model where a url is classified as 
    postive or negative, using batched processing to avoid memory issues.
    """
    print("\nEvaluating on validation set to determine threshold...")
    model.eval()
    
    # Process validation set in batches
    val_probs = []
    val_labels = []
    
    num_samples = len(X_val)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing validation batches"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_x = X_val[start_idx:end_idx]
            batch_y = y_val[start_idx:end_idx]
            
            batch_probs = model(batch_x).cpu().numpy()
            val_probs.extend(batch_probs)
            val_labels.extend(batch_y.cpu().numpy())
    
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # To compute FPR: among the negatives (label == 0), FPR = fraction predicted positive.
    negatives_val = val_probs[val_labels == 0]
    # We search for a threshold tau that gives approximately target_fpr among negatives.
    tau = np.quantile(negatives_val, 1 - model_desired_fpr)
    print(f"Chosen threshold tau: {tau:.4f}")
    
    return tau

def evaluate_model_batched(model, X_test, y_test, tau, batch_size=1024):
    """
    Evaluate the model on the test set using batched processing to avoid memory issues.
    """
    print("\nEvaluating on test set...")
    model.eval()
    
    # Process test set in batches
    test_probs = []
    test_labels = []
    
    num_samples = len(X_test)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing test batches"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_x = X_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]
            
            batch_probs = model(batch_x).cpu().numpy()
            test_probs.extend(batch_probs)
            test_labels.extend(batch_y.cpu().numpy())
    
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    # Predictions based on threshold tau.
    test_predictions = (test_probs >= tau).astype(int)

    # Calculate false positive rate (FPR) and false negative rate (FNR) on test set.
    negatives_test = test_labels == 0
    positives_test = test_labels == 1
    fpr = np.sum((test_predictions == 1) & negatives_test) / np.sum(negatives_test)
    fnr = np.sum((test_predictions == 0) & positives_test) / np.sum(positives_test)
    print(f"Model Only: Test FPR: {fpr*100:.2f}%, Test FNR: {fnr*100:.2f}%")
    
    return test_probs, test_labels, test_predictions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build overflow Bloom filter for learned Bloom filters')
    parser.add_argument('--model_dir', type=str, default="models/run_20250509_033949_data3200000_emb64_hid32",
                        help='Directory containing the model and hyperparameters')
    args = parser.parse_args()
    
    # Paths
    data_dir = "data"
    model_dir = args.model_dir
    training_dataset_path = os.path.join(data_dir, "training_dataset_extended_balanced.csv")
    negative_dataset_path = os.path.join(data_dir, "negative_dataset.csv")
    model_path = os.path.join(model_dir, "url_classifier.pt")
    threshold_path = os.path.join(model_dir, "threshold.txt")
    overflow_bloom_filter_save_path = os.path.join(model_dir, "overflow_bloom.pkl")
    hyperparams_path = os.path.join(model_dir, "hyperparams.json")
    
    # Load hyperparameters from JSON file
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        print(f"Loaded hyperparameters from {hyperparams_path}")
    else:
        print(f"Warning: Hyperparameters file not found at {hyperparams_path}. Using default values.")
        hyperparams = {}
    
    # Set hyperparameters, using defaults if not in the JSON file
    embedding_dim = hyperparams["embedding_dim"]
    hidden_dim = hyperparams["hidden_dim"]
    batch_size = hyperparams["batch_size"]  # Batch size for validation set processing
    batch_size_for_getting_false_negatives = 2**10  # Match original script
    
    # Use the half of the desired_fpr for model and half for bloom filter
    desired_fpr = hyperparams["desired_fpr"]
    model_desired_fpr = desired_fpr / 2
    overflow_bloom_filter_desired_fpr = desired_fpr / 2
    
    # Use random seed from hyperparameters or default
    random_seed = hyperparams["random_seed"]
    
    # Command line argument overrides the hyperparameter
    data_cap = hyperparams["data_cap"]
    
    print("Using hyperparameters:")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  batch_size: {batch_size}")
    print(f"  model_desired_fpr: {model_desired_fpr}")
    print(f"  overflow_bloom_filter_desired_fpr: {overflow_bloom_filter_desired_fpr}")
    print(f"  random_seed: {random_seed}")
    print(f"  data_cap: {data_cap}")
    
    # Check for CUDA, MPS, or CPU availability
    device = torch.device(
        'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load data with same random seed and data_cap
    train_df, val_df, test_df, negative_df = load_data(
        training_dataset_path, 
        negative_dataset_path,
        data_cap=data_cap,
        random_seed=random_seed
    )
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Build vocabulary
    char2idx, idx2char = build_vocab([train_df, val_df, test_df, negative_df])
    vocab_size = len(char2idx)
    
    # Prepare datasets
    x_val, y_val = prepare_dataset(val_df, char2idx, device)
    x_test, y_test = prepare_dataset(test_df, char2idx, device)
    
    # Initialize and load model
    model = URLClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # Determine threshold using validation set
    print("\nDetermining threshold using validation set...")
    threshold = determine_threshold_batched(model, x_val, y_val, model_desired_fpr, batch_size)
    print(f"Determined threshold: {threshold}")
    
    # Save threshold
    with open(threshold_path, 'w') as f:
        f.write(str(threshold))
    print(f"Threshold saved to {threshold_path}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_probs, test_labels, test_predictions = evaluate_model_batched(model, x_test, y_test, threshold, batch_size)
    
    # Build overflow bloom filter
    print("\nBuilding overflow Bloom filter...")
    overflow_bloom = build_overflow_filter(
        test_df, model, threshold, char2idx, device,
        overflow_bloom_filter_desired_fpr,
        batch_size_for_getting_false_negatives
    )
    
    # Save overflow bloom filter
    with open(overflow_bloom_filter_save_path, 'wb') as f:
        pickle.dump(overflow_bloom, f)
    print(f"Overflow bloom filter saved to {overflow_bloom_filter_save_path}")

if __name__ == "__main__":
    main() 