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

# class FaissIndexer:
#     def __init__(self, embedding_dim, nlist=1, m=8, use_gpu=False):
#         """
#         Initializes the IVF_PQ index and optionally pushes it to the GPU.
#         """
#         self.embedding_dim = embedding_dim
#         self.nlist = nlist
#         self.use_gpu = use_gpu
        
#         # 1. Build the Base CPU Index
#         self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
#         self.index = faiss.IndexIVFPQ(
#             self.quantizer, 
#             self.embedding_dim, 
#             self.nlist, 
#             m, 
#             8, 
#             faiss.METRIC_INNER_PRODUCT
#         )
        
#         # 2. Push to GPU if requested
#         if self.use_gpu:
#             self.res = faiss.StandardGpuResources()
#             # The '0' assigns it to your first GPU. 
#             # We no longer need cloner_options because m=32 fits natively!
#             self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

#     def _normalize(self, vectors):
#         """L2 normalizes the vectors for Cosine Similarity."""
#         norm_vectors = vectors.copy()
#         faiss.normalize_L2(norm_vectors)
#         return norm_vectors

#     def train(self, vectors):
#         """Trains the Voronoi clusters and PQ codebooks."""
#         norm_vectors = self._normalize(vectors)
#         self.index.train(norm_vectors)

#     def add_batch(self, vectors, ids):
#         """Adds a batch of vectors with their IDs."""
#         norm_vectors = self._normalize(vectors)
#         ids_array = np.array(ids, dtype=np.int64)
#         self.index.add_with_ids(norm_vectors, ids_array)

#     def search(self, query_vectors, k=1):
#         """Searches the index and returns distances and IDs."""
#         norm_queries = self._normalize(query_vectors)
#         distances, indices = self.index.search(norm_queries, k)
#         return distances, indices

#     def save(self, filepath):
#         """Serializes the index to disk. Must pull to CPU first if on GPU."""
#         if self.use_gpu:
#             print("  -> Pulling FAISS index from GPU VRAM back to CPU RAM for saving...")
#             cpu_index = faiss.index_gpu_to_cpu(self.index)
#             faiss.write_index(cpu_index, filepath)
#         else:
#             faiss.write_index(self.index, filepath)