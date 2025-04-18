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

"""
TODO
    - Check if the overflow bloom filter is working correctly (trained on false
    negatives of the model).
    - Check if the FPR calculation for the model is correct (page 16 FPR_\tau)
    - Check if the bloom filter training works properly. 
    - Checks how the model is chosen for some FPR.
    - Need to add model size to results
!   - Paper reports bloom filter size to be > 1.31 MB, why am I getting 64 bytes

"""

# Check for MPS availability
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# 1. Utility: Build a character vocabulary for URLs.
# -----------------------------
def build_vocab(urls):
    # Build a set of unique characters from all URLs.
    vocab = set("".join(urls))
    vocab = sorted(list(vocab))
    # Create mapping from char to index (start from 1; reserve 0 for padding)
    char2idx = {ch: i+1 for i, ch in enumerate(vocab)}
    idx2char = {i+1: ch for i, ch in enumerate(vocab)}
    return char2idx, idx2char

# -----------------------------
# 2. Define the GRU-based classifier model.
# -----------------------------
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

# -----------------------------
# 3. A simple Bloom filter implementation.
# -----------------------------
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

# -----------------------------
# 4. Data Loading using malicious_phish.csv
# -----------------------------
# Load the dataset from the CSV file. Adjust the file path as needed.
df = pd.read_csv("malicious_phish.csv")

# We assume df has columns: 'url' and 'type' (e.g., "benign", "defacement", "phishing").
# We'll create a binary label: 0 for benign, 1 for anything else.
df['label'] = (df['type'] != 'benign').astype(int)

# Build a list of (url, label) tuples.
data = list(zip(df['url'].tolist(), df['label'].tolist()))

#! Limit the size of data for testing
data_cap = 10 ** 4 * 2
data = data[:data_cap]

# Shuffle and split the dataset into training, validation, and test sets (60/20/20 split).
random.shuffle(data)
train_split = int(0.6 * len(data))
val_split = int(0.8 * len(data))
train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

# Build vocabulary from all URLs in the dataset
all_url_texts = [url for url, _ in data]
char2idx, idx2char = build_vocab(all_url_texts)
vocab_size = len(char2idx)

# Define a function to convert a URL into a list of character indices.
#* Confirmed to be faithful to paper (truncates too long urls and pads to make
#* all inputs of the same length)
def url_to_indices(url, max_length=50):
    # Truncate or pad the URL to a fixed max_length.
    #! Replaced with dict lookup so error if not in dataset
    indices = [char2idx[ch] for ch in url]
    # If the url is shorter than max length, pad
    if len(indices) < max_length:
        indices += [0]*(max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

# Prepare PyTorch datasets.
def prepare_dataset(dataset, max_length=50):
    xs = [url_to_indices(url, max_length) for url, _ in dataset]
    ys = [label for _, label in dataset]
    return torch.tensor(xs, dtype=torch.long).to(device), torch.tensor(ys, dtype=torch.float).to(device)

X_train, y_train = prepare_dataset(train_data)
X_val, y_val = prepare_dataset(val_data)
X_test, y_test = prepare_dataset(test_data)

# -----------------------------
# 5. Train the GRU classifier.
# -----------------------------
# Hyperparameters.
embedding_dim = 32
hidden_dim = 16
#! Should do a hyper parameter search
num_epochs = 5      # For demo; increase for real experiments.
batch_size = 64
learning_rate = 0.001

# Initialize the model, loss function, and optimizer.
model = URLClassifier(vocab_size, embedding_dim, hidden_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Simple training loop.
def train_model(model, X, y, num_epochs, batch_size):
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

train_model(model, X_train, y_train, num_epochs, batch_size)

# -----------------------------
# 6. Determine threshold τ on validation set to achieve desired FPR.
# -----------------------------
# We choose a target FPR (for the model alone) – for instance, 0.5%.
target_fpr = 0.005

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
tau = np.quantile(negatives_val, 1 - target_fpr)
print(f"Chosen threshold tau: {tau:.4f}")

# -----------------------------
# 7. Evaluate on the test set.
# -----------------------------
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
print(f"Test FPR: {fpr*100:.2f}%, Test FNR: {fnr*100:.2f}%")

# -----------------------------
# 8. Build the overflow Bloom filter.
# -----------------------------
print("\nBuilding overflow Bloom filter...")
# The overflow Bloom filter will store those URLs that are actual positives but were predicted negative.
# In a real system, we would store the keys that were "missed" by the model.
overflow_urls = []
print("Processing test URLs to identify false negatives...")
for url, label in tqdm(test_data):
    # Convert URL to tensor and get model probability.
    idx_seq = torch.tensor([url_to_indices(url)], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = model(idx_seq).cpu().item()
    # If the model predicted negative (prob < tau) but the URL is positive (label 1), it is a false negative.
    if label == 1 and prob < tau:
        overflow_urls.append(url)

# Define Bloom filter parameters.
# For demonstration, we set m (size) and k (number of hash functions) arbitrarily.
#! Tune these
m = 10000  # size of bit array (in bits)
k = 3      # number of hash functions

#! This should use the train set to train the bloom filter and test on test set
#! does it do this?

print(f"Adding {len(overflow_urls)} false negatives to overflow Bloom filter...")
overflow_bloom = BloomFilter(m, k)
for url in tqdm(overflow_urls):
    overflow_bloom.add(url)

print(f"Number of false negatives stored in the overflow Bloom filter: {len(overflow_urls)}")
print(f"Estimated memory usage of overflow Bloom filter: {overflow_bloom.memory_usage_bytes():.2f} bytes")


# -----------------------------
# 9. Compare with a traditional Bloom filter.
# -----------------------------
# For comparison, suppose we want a traditional Bloom filter for all positive URLs in the test set.
# For a given FPR, standard Bloom filter sizing formulas can be used.
#! It's sus that this doesn't define a new bloom filter
n_positive = np.sum(test_labels == 1)
# For example, for a desired FPR of 1%, the optimal number of bits per element is:
# m_per_element = -log(FPR) / (log(2)**2)
desired_fpr = 0.01
m_per_element = -math.log(desired_fpr) / (math.log(2)**2)
total_bits = n_positive * m_per_element
print(f"Traditional Bloom filter size for {n_positive} positives at {desired_fpr*100:.1f}% FPR: {total_bits/8:.2f} bytes")
    
# -----------------------------
# 10. Summary:
# -----------------------------
# Here, we trained a small GRU model to classify URLs.
# We set a threshold tau on the validation set to control the model's FPR.
# On the test set, false negatives (i.e. missed positives) are captured by an auxiliary (overflow) Bloom filter.
# This combination of a learned model plus a small Bloom filter can yield a lower overall memory footprint
# compared to a traditional Bloom filter while guaranteeing zero false negatives.
