import unittest
import sys
import os
import sqlite3
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from mapper import MetagenomicMapper
from embedder import MockRoPEModel, SequenceEmbedder
from faiss_indexer import FaissIndexer

class TestMapper(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(current_dir, "test_mapper.db")
        self.index_path = os.path.join(current_dir, "test_mapper.index")
        
        # Clean up old artifacts
        for path in [self.db_path, self.index_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # 1. Create Mock SQLite Database
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE metadata (id INTEGER PRIMARY KEY, header TEXT, start_pos INTEGER)''')
        # Insert our target sequence at ID 1000
        conn.execute("INSERT INTO metadata (id, header, start_pos) VALUES (?, ?, ?)", (1000, "TEST_SEQ_A", 0))
        conn.commit()
        conn.close()
        
        # 2. Create Mock FAISS Index
        random.seed(42)
        batch = ["A" * 100] 
        for _ in range(299):
            batch.append("".join(random.choices(['A', 'C', 'G', 'T'], k=100)))
            
        # SAVE THE MODEL INSTANCE SO THE MAPPER CAN USE IT
        self.shared_model = MockRoPEModel(embedding_dim=512)
        embedder = SequenceEmbedder(self.shared_model, device='cpu')
        
        vectors = embedder.embed_batch(batch)
        indexer = FaissIndexer(embedding_dim=512, nlist=1, m=8, use_gpu=False)
        indexer.train(vectors)
        indexer.add_batch(vectors, list(range(1000, 1300)))
        indexer.save(self.index_path)

    def tearDown(self):
        for path in [self.db_path, self.index_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_mapping_logic(self):
        # Initialize Mapper strictly on CPU for the test
        mapper = MetagenomicMapper(self.db_path, self.index_path, model=self.shared_model, use_gpu=False)
        
        # THE TEST READ: 100 bases of A, followed by 20 bases of garbage (G)
        read_seq = ("A" * 100) + ("G" * 20)
        mock_reads = [{'id': 'read_001', 'seq': read_seq}]
        
        # Execute mapping with a stride of 100 (forces non-overlapping, should discard tail)
        results = mapper.map_reads(mock_reads, query_stride=100, k=1)
        
        # Assertion 1: Chunking Logic
        self.assertEqual(len(results), 1, "Should return exactly 1 chunk for a 120bp read with stride 100")
        
        res = results[0]
        self.assertEqual(res['chunk_seq'], "A" * 100, "Chunker failed to accurately slice the first 100bp")
        
        hit = res['hits'][0]
        
        # Assertion 2: Index Retrieval
        self.assertEqual(hit['faiss_id'], 1000, "Failed to retrieve the correct FAISS ID")
        
        # Assertion 3: SQLite Batch Metadata Connection
        self.assertEqual(hit['header'], "TEST_SEQ_A", "Failed to retrieve the correct SQLite header")
        
        # Assertion 4: Mismatch Proxy Math
        self.assertEqual(hit['mismatches'], 0, "Perfect math should translate to exactly 0 mismatches")
        
        mapper.close()
        print("\n[INFO] Mapper Test Validated Successfully!")

if __name__ == '__main__':
    unittest.main(verbosity=2)

# python -m unittest tests/test_mapper.py