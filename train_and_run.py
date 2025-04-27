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
from typing import Final, Optional, Callable, List, Tuple, Generic, TypeVar
import xxhash
import struct
from bitarray import bitarray

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

KeyType = TypeVar("KeyType")
Serializer = Callable[[KeyType], bytes]
XXH_SEED1: Final[int] = 0
XXH_SEED2: Final[int] = 6917

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

class BloomFilter(Generic[KeyType]):
    """
    A generic, high-performance Bloom filter optimized for speed.

    Requires the user to provide a `serializer` function during initialization
    to convert items of `KeyType` into bytes before hashing. The core filter
    logic operates exclusively on these bytes.

    Features:
    - Generic over KeyType.
    - Requires user-provided serialization function (KeyType -> bytes).
    - xxhash (xxh64) for fast hashing.
    - bitarray package for C-optimized bit manipulation.
    - Kirsch-Mitzenmacher optimization (double hashing).
    - No runtime type checks in hot paths.
    """

    __slots__ = (
        "capacity",
        "error_rate",
        "serializer",
        "size",
        "num_hashes",
        "bit_array",
        "num_items",
        "_hasher1_intdigest",
        "_hasher2_intdigest",
    )

    # Type alias for the internal hash function signature (bytes -> int)
    _BytesHasher = Callable[[bytes], int]

    def __init__(
        self,
        capacity: int,
        error_rate: float,
        serializer: Optional[Serializer[KeyType]] = None,
    ):
        """
        Initializes the generic Bloom filter.

        Args:
            capacity: The expected number of items to be stored (n).
            error_rate: The desired false positive probability (p), e.g., 0.001.
            serializer: An optional function that takes an item of KeyType and returns bytes.
                        If None, built-in support for int, float, str, bytes is used.

        Raises:
            ValueError: If capacity is non-positive or error_rate is not in (0, 1).
            TypeError: If serializer is provided but not callable, or if an
                       unsupported key type is encountered at insertion.
        """
        if not capacity > 0:
            raise ValueError("Capacity must be positive")
        if not 0 < error_rate < 1:
            raise ValueError("Error rate must be between 0 and 1")

        if serializer is None:
            serializer = self._default_serializer
        self.serializer: Final[Serializer[KeyType]] = serializer

        self.capacity: Final[int] = capacity
        self.error_rate: Final[float] = error_rate

        size, num_hashes = self._calculate_optimal_params(capacity, error_rate)
        self.size: Final[int] = size
        self.num_hashes: Final[int] = num_hashes

        # Initialize bit array using the C-backed bitarray
        self.bit_array: bitarray = bitarray(self.size)
        self.bit_array.setall(0)

        self.num_items: int = 0

        # Initialize hashers using xxh64_intdigest for direct integer output
        # These always operate on bytes internally.
        self._hasher1_intdigest: BloomFilter._BytesHasher = (
            lambda b: xxhash.xxh64_intdigest(b, seed=XXH_SEED1)
        )
        self._hasher2_intdigest: BloomFilter._BytesHasher = (
            lambda b: xxhash.xxh64_intdigest(b, seed=XXH_SEED2)
        )

    @staticmethod
    def _default_serializer(item: KeyType) -> bytes:
        """
        Default serialization for int, float, str, bytes.
        Raises TypeError on other types.
        """
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)  # no-op
        if isinstance(item, str):
            return item.encode("utf-8")
        if isinstance(item, float): # float: 8-byte IEEE-754 big-endian
            return struct.pack(">d", item)
        if isinstance(item, int): # int: two's-complement 64-bit little-endian
            return item.to_bytes(8, byteorder="little", signed=True)
        raise TypeError(
            f"No default serializer for type {type(item).__name__}; "
            "please provide a custom serializer"
        )

    @staticmethod
    def _calculate_optimal_params(capacity: int, error_rate: float) -> Tuple[int, int]:
        """Calculates optimal size (m) and hash count (k)."""
        # m = - (n * ln(p)) / (ln(2)^2)
        m_float: float = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        size: int = max(1, int(math.ceil(m_float)))  # Ensure size is at least 1

        # k = (m / n) * ln(2)
        # Handle potential division by zero if capacity is somehow <= 0 despite check
        k_float: float = (size / capacity) * math.log(2) if capacity > 0 else 1.0
        num_hashes: int = max(1, int(math.ceil(k_float)))  # Ensure at least 1 hash

        return size, num_hashes

    def _get_indices(self, item_bytes: bytes) -> List[int]:
        """Generates k indices using double hashing with xxhash on bytes."""
        h1: int = self._hasher1_intdigest(item_bytes)
        h2: int = self._hasher2_intdigest(item_bytes)
        m: int = self.size
        # Generate k indices using Kirsch-Mitzenmacher optimization
        return [(h1 + i * h2) % m for i in range(self.num_hashes)]

    def _add_indices(self, indices: List[int]) -> None:
        """Sets the bits at the given indices in the bit array."""
        bit_arr: bitarray = self.bit_array
        for index in indices:
            bit_arr[index] = 1

    def _check_indices(self, indices: List[int]) -> bool:
        """Checks if all bits at the given indices are set."""
        bit_arr: bitarray = self.bit_array
        for index in indices:
            if not bit_arr[index]:
                return False  # Definitely not present (early exit)
        return True  # Possibly present

    # --- Public Add/Contains Methods ---

    def add(self, item: KeyType) -> None:
        """
        Adds an item to the Bloom filter.

        The item is first converted to bytes using the serializer provided
        during initialization.

        Args:
            item: The item of KeyType to add.
        """
        try:
            item_bytes: bytes = self.serializer(item)
        except Exception as e:
            raise TypeError(
                f"Failed to serialize item of type {type(item).__name__} with provided serializer: {e}"
            ) from e

        indices: List[int] = self._get_indices(item_bytes)
        self._add_indices(indices)
        self.num_items += 1

    def __contains__(self, item: KeyType) -> bool:
        """
        Checks if an item might be in the Bloom filter.

        The item is first converted to bytes using the serializer provided
        during initialization.

        Args:
            item: The item of KeyType to check.

        Returns:
            True if the item is possibly in the set (may be a false positive).
            False if the item is definitely not in the set.
        """
        try:
            item_bytes: bytes = self.serializer(item)
        except Exception as e:
            # If serialization fails, the item cannot have been added
            raise TypeError(
                f"Warning: Failed to serialize item for checking. Returning False. Error: {e}"
            ) from e

        indices: List[int] = self._get_indices(item_bytes)
        return self._check_indices(indices)

    # --- Other Public Methods ---

    def __len__(self) -> int:
        """Returns the number of items added."""
        return self.num_items

    @property
    def bit_size(self) -> int:
        """Returns the size of the underlying bit array (m)."""
        return self.size

    def __sizeof__(self) -> int:
        """Returns the size of the underlying bit array in bytes"""
        return math.ceil(self.bit_size / 8)

    def get_current_false_positive_rate(self) -> float:
        """
        Estimates the current theoretical false positive rate based on the
        number of items added (`num_items`).

        Formula: (1 - exp(-k * n / m))^k
        Where: k = num_hashes, n = num_items, m = size

        Returns:
            The estimated false positive probability (float between 0.0 and 1.0).
        """
        k: int = self.num_hashes
        n: int = self.num_items
        m: int = self.size

        if m == 0 or n == 0:  # Avoid division by zero or calculation for empty filter
            return 0.0

        try:
            exponent: float = -k * n / float(m)
            rate: float = (1.0 - math.exp(exponent)) ** k
        except (OverflowError, ValueError):
            rate = 1.0  # Theoretical rate approaches 1 if calculations fail

        return max(0.0, min(1.0, rate))  # Clamp result

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the filter."""
        # Determine serializer name if possible, otherwise show type
        serializer_name = getattr(
            self.serializer, "__name__", str(type(self.serializer))
        )
        return (
            f"{self.__class__.__name__}("
            f"capacity={self.capacity}, "
            f"error_rate={self.error_rate:.2e}, "
            f"serializer={serializer_name}, "
            f"size={self.size}, "
            f"num_hashes={self.num_hashes}, "
            f"num_items={self.num_items})"
        )

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
    
    # Process URLs in batches for better efficiency
    batch_size = 2**10
    num_samples = len(urls)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Processing {num_samples} URLs in {num_batches} batches (batch size: {batch_size})...")
    model.eval()
    
    for batch_idx in tqdm(range(num_batches)):
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Extract batch data
        batch_urls = urls[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Convert batch to tensors
        batch_indices = [url_to_indices(url, char2idx) for url in batch_urls]
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
        
        # Get model predictions
        with torch.no_grad():
            batch_probs = model(batch_tensor).cpu().numpy()
            
        # Find false negatives (predicted negative but actually positive)
        for i, (url, prob, label) in enumerate(zip(batch_urls, batch_probs, batch_labels)):
            if label == 1 and prob < tau:
                overflow_urls.append(url)
    
    # Calculate optimal Bloom filter parameters
    n = len(overflow_urls)  # number of elements to store
    p = overflow_bloom_filter_desired_fpr  # desired false positive rate
    m = int(-n * math.log(p) / (math.log(2)**2))  # optimal number of bits
    k = int(m/n * math.log(2)) if n > 0 else 1  # optimal number of hash functions
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
    embedding_dim = 32
    hidden_dim = 16
    # embedding_dim = 8
    # hidden_dim = 4

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
    try:
        # Check if latest_run_link already exists
        if os.path.lexists(latest_run_link):  # Use lexists instead of exists to handle broken symlinks
            # Remove the existing link/file/directory
            if os.path.islink(latest_run_link):
                os.unlink(latest_run_link)
                print(f"Removed existing symlink: {latest_run_link}")
            elif os.path.isdir(latest_run_link):
                import shutil
                shutil.rmtree(latest_run_link)
                print(f"Removed existing directory: {latest_run_link}")
            else:
                os.remove(latest_run_link)
                print(f"Removed existing file: {latest_run_link}")
        
        # Create the new symlink (using absolute paths to avoid relative path issues)
        abs_model_run_dir = os.path.abspath(model_run_dir)
        os.symlink(abs_model_run_dir, latest_run_link)
        print(f"Created link to latest run: {latest_run_link} -> {abs_model_run_dir}")
    except Exception as e:
        print(f"Warning: Could not create latest_run symlink: {str(e)}")
        print(f"To fix this, manually remove {latest_run_link} and retry or just access this run directly at: {model_run_dir}")
    
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
