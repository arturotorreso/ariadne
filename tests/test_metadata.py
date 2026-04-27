import unittest
import sys
import os
import sqlite3

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from ingestion import sliding_window_fasta
from metadata_store import MetadataStore

class TestMetadataStore(unittest.TestCase):
    def setUp(self):
        self.fasta_path = os.path.join(current_dir, "test_db.fasta")
        # Define a temporary test database file
        self.db_path = os.path.join(current_dir, "test_metadata.db")
        
        # Clean up any leftover database from previous failed tests
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.window_size = 100
        self.stride = 50
        self.batch_size = 2

    def tearDown(self):
        # Clean up the test database after the test finishes
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass

    def test_pipeline_synchronization(self):
        # 1. Initialize the Store
        store = MetadataStore(self.db_path)
        
        # 2. Initialize the Generator
        generator = sliding_window_fasta(
            self.fasta_path, 
            self.window_size, 
            self.stride, 
            self.batch_size
        )
        
        # 3. The Synchronization Loop (Mimicking the real pipeline)
        global_faiss_id = 0
        for sequences_batch, metadata_batch in generator:
            # Insert the metadata batch and update the global ID counter
            global_faiss_id = store.insert_batch(global_faiss_id, metadata_batch)
            
        # Assertion 1: Total IDs generated
        # Based on Step 1, we expect exactly 6 valid windows across the entire FASTA.
        self.assertEqual(global_faiss_id, 6, "Should have generated exactly 6 sequential IDs")

        # Assertion 2: Direct Retrieval Accuracy
        # Let's query ID 3. 
        # IDs 0, 1, 2, 3 belong to Sequence A. 
        # ID 3 should be the 4th window of Sequence A (bases 150 to 250).
        retrieved_data = store.fetch_metadata(3)
        self.assertIsNotNone(retrieved_data, "Metadata for ID 3 should exist")
        self.assertEqual(retrieved_data[0], 'seqA_length_250', "Header mismatch")
        self.assertEqual(retrieved_data[1], 150, "Start position mismatch")
        self.assertEqual(retrieved_data[2], 250, "End position mismatch")
        
        # Assertion 3: Sequence B Boundary Check
        # ID 4 should be the 1st window of Sequence B (bases 0 to 100).
        retrieved_data_b = store.fetch_metadata(4)
        self.assertEqual(retrieved_data_b[0], 'seqB_length_160', "Sequence transition failed")
        self.assertEqual(retrieved_data_b[1], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)


# python -m unittest tests/test_metadata.py