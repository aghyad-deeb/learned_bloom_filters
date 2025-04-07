import unittest
import torch
import numpy as np
from chat_attempt import URLClassifier, BloomFilter, build_vocab, url_to_indices

class TestURLClassifier(unittest.TestCase):
    def setUp(self):
        # Create a small vocabulary for testing
        self.test_urls = [
            "http://example.com",
            "https://test.com",
            "http://malicious.com"
        ]
        self.char2idx, self.idx2char = build_vocab(self.test_urls)
        self.vocab_size = len(self.char2idx)
        
        # Initialize model with small dimensions for testing
        self.model = URLClassifier(self.vocab_size, embedding_dim=8, hidden_dim=4)
        
        # Create test data
        self.test_sequence = url_to_indices("http://test.com", max_length=20)
        self.test_tensor = torch.tensor([self.test_sequence], dtype=torch.long)

    def test_model_initialization(self):
        """Test if model initializes correctly with given parameters"""
        self.assertEqual(self.model.embedding.num_embeddings, self.vocab_size + 1)
        self.assertEqual(self.model.embedding.embedding_dim, 8)
        self.assertEqual(self.model.gru.hidden_size, 4)

    def test_model_forward(self):
        """Test if model forward pass works and returns correct shape"""
        output = self.model(self.test_tensor)
        self.assertEqual(output.shape, torch.Size([1]))  # Should return one probability per input
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # Should be probabilities

    def test_vocabulary_creation(self):
        """Test if vocabulary is created correctly"""
        # Check if all characters are in vocabulary
        all_chars = set("".join(self.test_urls))
        for char in all_chars:
            self.assertIn(char, self.char2idx)
        
        # Check if padding index 0 is reserved
        self.assertNotIn(0, self.idx2char)

    def test_url_to_indices(self):
        """Test URL to indices conversion"""
        # Test padding
        indices = url_to_indices("http", max_length=10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices.count(0), 6)  # Should pad with zeros
        
        # Test truncation
        indices = url_to_indices("http://verylongurl.com", max_length=10)
        self.assertEqual(len(indices), 10)

class TestBloomFilter(unittest.TestCase):
    def setUp(self):
        self.m = 100  # Small size for testing
        self.k = 3
        self.bf = BloomFilter(self.m, self.k)
        
        # Test items
        self.test_items = ["item1", "item2", "item3"]

    def test_bloom_filter_initialization(self):
        """Test if Bloom filter initializes correctly"""
        self.assertEqual(len(self.bf.bit_array), self.m)
        self.assertEqual(self.bf.k, self.k)
        self.assertTrue(all(bit == 0 for bit in self.bf.bit_array))

    def test_add_and_contains(self):
        """Test adding items and checking membership"""
        # Add items
        for item in self.test_items:
            self.bf.add(item)
        
        # Check if added items are present
        for item in self.test_items:
            self.assertIn(item, self.bf)
        
        # Check if non-added items are not present
        self.assertNotIn("nonexistent", self.bf)

    def test_false_positives(self):
        """Test false positive rate with small filter"""
        # Add many items to a small filter to test false positives
        small_bf = BloomFilter(10, 2)  # Very small filter to force collisions
        items = [f"item{i}" for i in range(5)]
        for item in items:
            small_bf.add(item)
        
        # Check some random strings that weren't added
        false_positives = 0
        test_items = [f"test{i}" for i in range(100)]
        for item in test_items:
            if item in small_bf:
                false_positives += 1
        
        # We expect some false positives due to the small filter size
        self.assertGreater(false_positives, 0)

    def test_memory_usage(self):
        """Test memory usage calculation"""
        expected_bytes = self.m / 8
        self.assertEqual(self.bf.memory_usage_bytes(), expected_bytes)

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestURLClassifier))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBloomFilter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests() 