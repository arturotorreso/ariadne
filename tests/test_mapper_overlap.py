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

class TestMapperOverlap(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(current_dir, "test_overlap.db")
        self.index_path = os.path.join(current_dir, "test_overlap.index")
        
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

    def test_overlapping_chunks(self):
        mapper = MetagenomicMapper(self.db_path, self.index_path, model=self.shared_model, use_gpu=False)
        
        # THE TEST READ: 100 bases of A, followed by 20 bases of garbage (G)
        read_seq = ("A" * 100) + ("G" * 20)
        mock_reads = [{'id': 'read_001', 'seq': read_seq}]
        
        # Execute mapping with a stride of 20!
        results = mapper.map_reads(mock_reads, query_stride=20, k=1)
        
        self.assertEqual(len(results), 2, "Should return exactly 2 chunks for a 120bp read with stride 20")
        
        chunk1 = results[0]
        chunk2 = results[1]
        
        print(f"\n\n--- Overlapping Window Results ---")
        print(f"Chunk 1 Sequence (0-100)  : {chunk1['chunk_seq'][:20]}...{chunk1['chunk_seq'][-20:]}")
        print(f"Chunk 1 Best Match        : {chunk1['hits'][0]['header']} (ID {chunk1['hits'][0]['faiss_id']})")
        print(f"Chunk 1 Mismatches        : {chunk1['hits'][0]['mismatches']}")
        print(f"Chunk 1 Cosine Similarity : {chunk1['hits'][0]['cosine_sim']:.4f}\n")
        
        print(f"Chunk 2 Sequence (20-120) : {chunk2['chunk_seq'][:20]}...{chunk2['chunk_seq'][-20:]}")
        print(f"Chunk 2 Best Match        : {chunk2['hits'][0]['header']} (ID {chunk2['hits'][0]['faiss_id']})")
        print(f"Chunk 2 Mismatches        : {chunk2['hits'][0]['mismatches']}")
        print(f"Chunk 2 Cosine Similarity : {chunk2['hits'][0]['cosine_sim']:.4f}")
        print(f"----------------------------------\n")
        
        # Assertions
        self.assertEqual(chunk1['hits'][0]['mismatches'], 0)
        
        # Because Chunk 2 has 20 G's, it should NOT be a perfect match
        self.assertGreater(chunk2['hits'][0]['mismatches'], 0)
        
        mapper.close()

if __name__ == '__main__':
    unittest.main(verbosity=2)

# python tests/test_mapper_overlap.py