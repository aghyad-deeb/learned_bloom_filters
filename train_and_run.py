"""
What does it mean to test if a b
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import math
import pandas as pd
from tqdm import tqdm
import time
import datetime
import json

"""
TODO
    - Check if the overflow bloom filter is working correctly (trained on false
    negatives of the model).
    - Check if the FPR calculation for the model is correct (page 16 FPR_\tau)
    - Check if the bloom filter training works properly.
    - Checks how the model is chosen for some FPR.
    - Need to add model size to results.
!   - Paper reports bloom filter size to be > 1.31 MB, why am I getting 64 bytes
!   - When they test the bloom filter, they test that a set of items not in the
!   the set does not return true.

"""

def build_vocab(df_list):
    # Extract all URLs from the DataFrames
    all_urls = []
    for df in df_list:
        if 'url' in df.columns:
            all_urls.extend(df['url'].tolist())
    
    # Build a set of unique characters from all URLs
    vocab = set("".join(all_urls))
    vocab = sorted(list(vocab))
    # Create mapping from char to index (start from 1; reserve 0 for padding)
    char2idx = {ch: i+1 for i, ch in enumerate(vocab)}
    idx2char = {i+1: ch for i, ch in enumerate(vocab)}
    return char2idx, idx2char

class URLClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=16):
        super(URLClassifier, self).__init__()
        #! Need to compare to their network, seems fine on quick check
        # Embedding layer converts characters to vectors.
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim, padding_idx=0)
        # GRU layer to process the sequence of embeddings.
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        # Fully connected layer to produce a single logit.
        self.fc = nn.Linear(hidden_dim, 1)
        # Sigmoid activation to produce a probability.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: batch of sequences (batch_size x seq_length)
        embed = self.embedding(x)  # (batch_size x seq_length x embedding_dim)
        _, hidden = self.gru(embed)  # hidden: (1 x batch_size x hidden_dim)
        hidden = hidden.squeeze(0)   # (batch_size x hidden_dim)
        logits = self.fc(hidden)     # (batch_size x 1)
        prob = self.sigmoid(logits)  # (batch_size x 1)
        return prob.squeeze(1)

class BloomFilter:
    def __init__(self, m, k):
        self.m = m                    # size of bit array
        self.k = k                    # number of hash functions
        self.bit_array = [0] * m      # initialize bit array to zeros

    def _hashes(self, item):
        # Generate k different hash values for the item.
        hashes = []
        # Use Python's built-in hash function with different salts.
        for i in range(self.k):
            # Create a combined hash from the item and salt i.
            combined = hash(str(item) + str(i))
            # Map the hash value to an index in [0, m).
            idx = combined % self.m
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        # Check if all bits for item are set to 1.
        return all(self.bit_array[idx] for idx in self._hashes(item))

    def memory_usage_bytes(self):
        # Rough estimate: number of bits / 8
        return self.m / 8

def load_data(training_dataset_path, negative_dataset_path, data_cap=None, random_seed=42):
    # Load the dataset from the CSV file
    df = pd.read_csv(training_dataset_path)

    # We assume df has columns: 'url' and 'type' (e.g., "benign", "defacement", "phishing").
    # We'll create a binary label: 0 for benign, 1 for anything else.
    df['label'] = (df['type'] == 'phishing').astype(int)

    #! Limit the size of data for testing
    if data_cap is not None:
        df = df.iloc[:data_cap]

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split the dataset into training, validation, and test sets (60/20/20 split)
    train_split = int(0.6 * len(df))
    val_split = int(0.8 * len(df))
    
    train_df = df.iloc[:train_split]
    val_df = df.iloc[train_split:val_split]
    test_df = df.iloc[val_split:]
    
    # Load negative dataset
    negative_df = pd.read_csv(negative_dataset_path)
    negative_df['label'] = (negative_df['type'] == 'phishing').astype(int)
    # Shuffle the negative dataset
    negative_df = negative_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # Assert that all labels in the negative dataset are 0
    assert (negative_df['label'] == 0).all(), "Expected all labels in negative dataset to be 0"

    print(f"Dataset sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df, negative_df

#* Confirmed to be faithful to paper (truncates too long urls and pads to make
#* all inputs of the same length)
def url_to_indices(url, char2idx, max_length=50):
    # Truncate or pad the URL to a fixed max_length.
    #! Replaced with dict lookup so error if not in dataset
    indices = [char2idx[ch] for ch in url]
    # If the url is shorter than max length, pad
    if len(indices) < max_length:
        indices += [0]*(max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

def prepare_dataset(df, char2idx, device, max_length=50):
    """
    Convert the dataframe into a tensor of indices, rather than strings, and a
    tensor of labels.
    """
    urls = df['url'].tolist()
    labels = df['label'].tolist()
    xs = [url_to_indices(url, char2idx, max_length) for url in urls]
    return torch.tensor(xs, dtype=torch.long).to(device), torch.tensor(labels, dtype=torch.float).to(device)

def train_model(model, X, y, num_epochs, batch_size, criterion, optimizer, device):
    model.train()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # Update statistics
            epoch_loss += loss.item() * batch_x.size(0)
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Average loss: {epoch_loss / len(dataset):.4f}")
        print(f"Training accuracy: {100 * correct / total:.2f}%")
    
    return model

def determine_threshold(model, X_val, y_val, model_desired_fpr):
    """
    Determine the threshold for the model where a url is classified as 
    postive or negative.
    """
    print("\nEvaluating on validation set to determine threshold...")
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val).cpu().numpy()
        val_labels = y_val.cpu().numpy()

    # To compute FPR: among the negatives (label == 0), FPR = fraction predicted positive.
    # This picks the probabilities for the values who's y value is 0 (negative) #!make sure y=0 is negative
    negatives_val = val_probs[val_labels == 0]
    # We search for a threshold tau that gives approximately target_fpr among negatives.
    # One simple method: sort the negative probabilities and pick the (1 - target_fpr) quantile.
    #! Check if this makes sense
    tau = np.quantile(negatives_val, 1 - model_desired_fpr)
    print(f"Chosen threshold tau: {tau:.4f}")
    
    return tau

def evaluate_model(model, X_test, y_test, tau):
    print("\nEvaluating on test set...")
    with torch.no_grad():
        test_probs = model(X_test).cpu().numpy()
        test_labels = y_test.cpu().numpy()

    # Predictions based on threshold tau.
    test_predictions = (test_probs >= tau).astype(int)

    # Calculate false positive rate (FPR) and false negative rate (FNR) on test set.
    negatives_test = test_labels == 0
    positives_test = test_labels == 1
    fpr = np.sum((test_predictions == 1) & negatives_test) / np.sum(negatives_test)
    fnr = np.sum((test_predictions == 0) & positives_test) / np.sum(positives_test)
    print(f"Model Only: Test FPR: {fpr*100:.2f}%, Test FNR: {fnr*100:.2f}%")
    
    return test_probs, test_labels, test_predictions

def build_overflow_filter(df, model, tau, char2idx, device, overflow_bloom_filter_desired_fpr):
    print("\nBuilding overflow Bloom filter...")
    # The overflow Bloom filter will store those URLs that are actual positives but were predicted negative.
    # In a real system, we would store the keys that were "missed" by the model.
    overflow_urls = []
    urls = df['url'].tolist()
    labels = df['label'].tolist()
    
    print("Processing test URLs to identify false negatives...")
    for url, label in tqdm(zip(urls, labels)):
        # Convert URL to tensor and get model probability.
        idx_seq = torch.tensor([url_to_indices(url, char2idx)], dtype=torch.long).to(device)
        with torch.no_grad():
            prob = model(idx_seq).cpu().item()
        # If the model predicted negative (prob < tau) but the URL is positive (label 1), it is a false negative.
        if label == 1 and prob < tau:
            #! Should these also include the true negatives? probably not.
            overflow_urls.append(url)

    # Calculate optimal Bloom filter parameters
    n = len(overflow_urls)  # number of elements to store
    p = overflow_bloom_filter_desired_fpr  # desired false positive rate
    m = int(-n * math.log(p) / (math.log(2)**2))  # optimal number of bits
    k = int(m/n * math.log(2))  # optimal number of hash functions
    k = max(k, 1)  # ensure at least one hash function

    print(f"Adding {len(overflow_urls)} false negatives to overflow Bloom filter...")
    overflow_bloom = BloomFilter(m, k)
    for url in tqdm(overflow_urls):
        overflow_bloom.add(url)

    print(f"Number of false negatives stored in the overflow Bloom filter: {len(overflow_urls)}")
    # print(f"{m_per_element=}")
    total_bits = m/8
    print(f"Estimated memory usage of overflow Bloom filter: {total_bits:.2f} bytes")
    
    return overflow_bloom

#! The calculation is correct, but currently my model is larger than theirs
#! relative to the dataset size because their dataset is 1.7 true positives and
#! random URLs
def compare_with_traditional_filter(test_labels, desired_fpr):
    # For comparison, suppose we want a traditional Bloom filter for all positive URLs in the test set.
    # For a given FPR, standard Bloom filter sizing formulas can be used.
    n_positive = np.sum(test_labels == 1)
    # For example, for a desired FPR of 1%, the optimal number of bits per element is:
    # m_per_element = -log(FPR) / (log(2)**2)

    m_per_element = -math.log(desired_fpr) / (math.log(2)**2)
    print(f"{m_per_element=}")
    total_bits = n_positive * m_per_element
    print(f"Traditional Bloom filter size for {n_positive} positives at {desired_fpr*100:.1f}% FPR: {total_bits/8:.2f} bytes")

def save_model(model, save_path):
    """
    Save a trained model to disk.
    
    Args:
        model: The trained PyTorch model to save
        save_path: Path where the model will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

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

def main():
    import datetime
    import json
    
    # model hyperparameters
    # embedding_dim = 32
    # hidden_dim = 16
    embedding_dim = 8
    hidden_dim = 4

    # training hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include model architecture in the run directory name
    model_config = f"emb{embedding_dim}_hid{hidden_dim}"
    
    data_dir = "data"
    model_dir = "models"
    model_run_dir = os.path.join(model_dir, f"run_{timestamp}_{model_config}")
    
    training_dataset_name = "training_dataset.csv"
    negative_dataset_name = "negative_dataset.csv"
    model_save_path = os.path.join(model_run_dir, "url_classifier.pt")
    overflow_bloom_filter_save_path = os.path.join(model_run_dir, "overflow_bloom.pkl")
    threshold_save_path = os.path.join(model_run_dir, "threshold.txt")
    hyperparams_save_path = os.path.join(model_run_dir, "hyperparams.json")
    
    training_dataset_path = os.path.join(data_dir, training_dataset_name)
    negative_dataset_path = os.path.join(data_dir, negative_dataset_name)
    
    # Create model run directory
    os.makedirs(model_run_dir, exist_ok=True)
    # print(f"Saving model artifacts to: {model_run_dir}")    
    # Set target FPR
    desired_fpr = 1/100
    random_seed = 42
    
    # Check for CUDA, MPS, or CPU availability
    device = torch.device(
        'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load data
    # data_cap = 10**4 * 2
    data_cap = None
    train_df, val_df, test_df, negative_df = load_data(training_dataset_path, negative_dataset_path, data_cap, random_seed)
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Negative set size: {len(negative_df)}")
    
    # Build vocabulary
    char2idx, idx2char = build_vocab([train_df, val_df, test_df, negative_df])
    vocab_size = len(char2idx)
    
    # Prepare PyTorch datasets
    x_train, y_train = prepare_dataset(train_df, char2idx, device)
    x_val, y_val = prepare_dataset(val_df, char2idx, device)
    x_test, y_test = prepare_dataset(test_df, char2idx, device)
    x_negative, y_negative = prepare_dataset(negative_df, char2idx, device)
    
    # Save hyperparameters as JSON
    hyperparams = {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "desired_fpr": desired_fpr,
        "random_seed": random_seed,
        "vocab_size": vocab_size,
        "timestamp": timestamp,
        "device": str(device)
    }
    
    with open(hyperparams_save_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_save_path}")
    
    # Initialize the model, loss function, and optimizer
    model = URLClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
    
    # Train the model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = train_model(model, x_train, y_train, num_epochs, batch_size, criterion, optimizer, device)
    
    # Save the trained model
    save_model(model, model_save_path)
    
    # Create a link to the latest run
    latest_run_link = os.path.join(model_dir, "latest_run")
    if os.path.exists(latest_run_link) and os.path.islink(latest_run_link):
        os.unlink(latest_run_link)
    if os.path.exists(latest_run_link):
        os.remove(latest_run_link)
    os.symlink(model_run_dir, latest_run_link)
    print(f"Created link to latest run: {latest_run_link}")
    
    # Determine threshold
    model_desired_fpr = desired_fpr / 2
    threshold = determine_threshold(model, x_val, y_val, model_desired_fpr)
    
    # Save threshold
    with open(threshold_save_path, 'w') as f:
        f.write(str(threshold))
    print(f"Threshold saved to {threshold_save_path}")
    
    # Evaluate model
    test_probs, test_labels, test_predictions = evaluate_model(model, x_test, y_test, threshold)
    
    # of the total fpr.
    overflow_bloom_filter_desired_fpr = desired_fpr - model_desired_fpr
    # Build overflow bloom filter
    overflow_bloom = build_overflow_filter(test_df, model, threshold, char2idx, device, overflow_bloom_filter_desired_fpr)
    
    # Save overflow bloom filter
    import pickle
    with open(overflow_bloom_filter_save_path, 'wb') as f:
        pickle.dump(overflow_bloom, f)
    print(f"Overflow bloom filter saved to {overflow_bloom_filter_save_path}")
    
    # Compare with traditional bloom filter
    compare_with_traditional_filter(test_labels, desired_fpr)

if __name__ == "__main__":
    main()
