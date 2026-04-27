import unittest
import sys
import os

# 1. Dynamically add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)

# Now we can safely import from the src folder
from ingestion import sliding_window_fasta

class TestFastaIngestion(unittest.TestCase):
    def setUp(self):
        # 2. Use absolute paths so the test finds the file 
        # no matter what directory you run the test from
        self.file_path = os.path.join(current_dir, "test_db.fasta")
        self.window_size = 100
        self.stride = 50
        self.batch_size = 2

    def test_sliding_window(self):
        # Initialize the generator
        generator = sliding_window_fasta(
            self.file_path, 
            self.window_size, 
            self.stride, 
            self.batch_size
        )
        
        # Exhaust the generator into a list to inspect the outputs
        batches = list(generator)
        
        # 1. Yield Count: The generator must yield exactly 3 batches
        self.assertEqual(len(batches), 3, "Generator should yield exactly 3 batches")
        
        # 2. Batch Size Limit: Ensure memory boundaries are respected
        for seqs, metas in batches:
            self.assertEqual(len(seqs), 2, "Each batch must contain exactly 2 windows")
            self.assertEqual(len(metas), 2, "Each batch must contain exactly 2 metadata entries")
            # Verify the physical string length matches the window requirement
            self.assertEqual(len(seqs[0]), 100, "Sequence length must exactly match window_size")
            
        # 3. Coordinate Accuracy (First Batch - Sequence A)
        first_batch_metas = batches[0][1]
        self.assertEqual(first_batch_metas[0], ('seqA_length_250', 0, 100))
        self.assertEqual(first_batch_metas[1], ('seqA_length_250', 50, 150))
        
        # 4. Tail Discarding (Final Batch - Sequence B)
        # batches[2] is the third batch. It should contain window 5 and 6.
        # Window 7 (100-200) must not exist because sequence B ends at 160.
        third_batch_metas = batches[2][1]
        self.assertEqual(third_batch_metas[0], ('seqB_length_160', 0, 100))
        self.assertEqual(third_batch_metas[1], ('seqB_length_160', 50, 150))

if __name__ == '__main__':
    unittest.main(verbosity=2)

# python -m unittest tests/test_ingestion.py