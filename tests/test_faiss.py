import unittest
import sys
import os
import torch
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from embedder import MockRoPEModel, SequenceEmbedder
from faiss_indexer import FaissIndexer

class TestFaissIndexer(unittest.TestCase):
    def setUp(self):
        # 1. Setup the GPU Embedder
        self.embedding_dim = 512
        mock_model = MockRoPEModel(embedding_dim=self.embedding_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SequenceEmbedder(mock_model, device=self.device)
        
        # 2. Generate 300 mock sequences to satisfy FAISS PQ codebook training
        # (2^8 bits = 256 centroids, requiring at least 256 training points)
        random.seed(42)
        self.batch = []
        for i in range(300):
            if i == 0 or i == 2:
                self.batch.append("A" * 100) # Seq 0 and 2 are identical
            elif i == 1:
                self.batch.append("C" * 100) # Seq 1 is distinct
            else:
                # Generate random DNA for the rest to provide K-means entropy
                seq = "".join(random.choices(['A', 'C', 'G', 'T'], k=100))
                self.batch.append(seq)
        
        # Generate the dense vectors on the GPU
        self.vectors = self.embedder.embed_batch(self.batch)
        
        # 3. Setup the FAISS Indexer
        self.indexer = FaissIndexer(embedding_dim=self.embedding_dim, nlist=1, m=8)
        self.index_path = os.path.join(current_dir, "test.index")

    def tearDown(self):
        # Clean up the index artifact
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except PermissionError:
                pass

    def test_faiss_lifecycle(self):
        # 1. Train the Index
        self.indexer.train(self.vectors)
        self.assertTrue(self.indexer.index.is_trained, "FAISS index failed to train")
        
        # 2. Add the Vectors with mock SQLite IDs (1000 to 1299)
        mock_ids = list(range(1000, 1300))
        self.indexer.add_batch(self.vectors, mock_ids)
        self.assertEqual(self.indexer.index.ntotal, 300, "Index should contain exactly 300 vectors")
        
        # 3. The Query (Mapping a Read)
        # We will query the index using Vector 0
        query_vector = self.vectors[0:1] # Keep it as a 2D array
        distances, indices = self.indexer.search(query_vector, k=1)
        
        # Assertion 1: ID Retrieval
        # Because we queried with Vector 0, the closest match must be ID 1000
        self.assertEqual(indices[0][0], 1000, "FAISS failed to retrieve the correct ID")
        
        # Assertion 2: Proxy Accuracy (Cosine Similarity)
        self.assertGreater(distances[0][0], 0.95, "Cosine similarity score is fatally degraded")
        
        # 4. Serialization
        self.indexer.save(self.index_path)
        self.assertTrue(os.path.exists(self.index_path), "Failed to save the .index file to disk")

if __name__ == '__main__':
    unittest.main(verbosity=2)

# python -m unittest tests/test_faiss.py