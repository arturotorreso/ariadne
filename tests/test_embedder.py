import unittest
import sys
import os
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from embedder import MockRoPEModel, SequenceEmbedder

class TestSequenceEmbedder(unittest.TestCase):
    def setUp(self):
        # Initialize the mock model and the embedder
        self.embedding_dim = 512
        mock_model = MockRoPEModel(embedding_dim=self.embedding_dim)
        
        # If your interactive node has a GPU, this will automatically use it
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SequenceEmbedder(mock_model, device=self.device)
        
        # Create our carefully crafted test sequences
        self.seq_len = 100
        self.seq1 = "A" * self.seq_len  # All Adenine
        self.seq2 = "C" * self.seq_len  # All Cytosine
        self.seq3 = "A" * self.seq_len  # Exact duplicate of Seq1
        
        self.batch = [self.seq1, self.seq2, self.seq3]

    def test_embedding_generation(self):
        # Run the forward pass
        embeddings = self.embedder.embed_batch(self.batch)
        
        # 1. Type Check: Must be a NumPy array (FAISS requirement)
        self.assertIsInstance(embeddings, np.ndarray, "Output must be a numpy array")
        
        # 2. Data Type Check: Must be strictly float32 (FAISS requirement)
        self.assertEqual(embeddings.dtype, np.float32, "Output must be strictly float32")
        
        # 3. Dimensionality Check
        expected_shape = (3, self.embedding_dim)
        self.assertEqual(embeddings.shape, expected_shape, "Shape mismatch")
        
        # 4. Biological Fidelity Check (Determinism)
        # Vector 0 (Seq1) and Vector 2 (Seq3) must be perfectly identical
        np.testing.assert_array_equal(
            embeddings[0], 
            embeddings[2], 
            err_msg="Identical DNA sequences produced different embeddings"
        )
        
        # 5. Differentiation Check
        # Vector 0 (Seq1) and Vector 1 (Seq2) must be different
        is_identical = np.array_equal(embeddings[0], embeddings[1])
        self.assertFalse(is_identical, "Different DNA sequences produced identical embeddings")
        
        print(f"\n[INFO] Test ran successfully on device: {self.device.upper()}")

if __name__ == '__main__':
    unittest.main(verbosity=2)

# python -m unittest tests/test_embedder.py