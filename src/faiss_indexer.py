import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, embedding_dim, nlist=1, m=32, use_gpu=False):
        self.embedding_dim = embedding_dim
        
        # We explicitly force use_gpu to False for the Indexer. 
        # The PyTorch Embedder will use the GPU, but the massive uncompressed 
        # Flat index MUST stay in system RAM to prevent VRAM OOM.
        self.use_gpu = False 
        
        # EXACT SEARCH: No compression, pure mathematical fidelity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # GPU push is completely disabled for the Flat Index
        # if self.use_gpu:
        #     self.res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

    def _normalize(self, vectors):
        norm_vectors = vectors.copy()
        faiss.normalize_L2(norm_vectors)
        return norm_vectors

    def train(self, vectors):
        # IndexFlatIP does not require Voronoi training
        pass

    def add_batch(self, vectors, ids):
        norm_vectors = self._normalize(vectors)
        # FAISS IndexFlat does not support add_with_ids.
        # Since our ingestion pipeline guarantees that 'ids' are strictly 
        # sequential starting from 0, we can safely use the standard .add() 
        # which automatically assigns sequential IDs matching our SQLite database.
        self.index.add(norm_vectors)

    def search(self, query_vectors, k=1):
        norm_queries = self._normalize(query_vectors)
        distances, indices = self.index.search(norm_queries, k)
        return distances, indices

    def save(self, filepath):
        # Since it's already on the CPU, we just write it directly to disk
        faiss.write_index(self.index, filepath)