import faiss
import numpy as np

class FaissIndexer:
    # Removed 'm' because we are storing raw uncompressed vectors
    def __init__(self, embedding_dim, nlist=1, use_gpu=False):
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.use_gpu = use_gpu
        
        # 1. Build the Base CPU Index 
        # IVFFlat = Clustered lookup + Uncompressed raw vectors!
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, 
            self.embedding_dim, 
            self.nlist, 
            faiss.METRIC_INNER_PRODUCT
        )
        
        # 2. Push to GPU if requested (Great for building/training) - Remove this for memory handling
        # if self.use_gpu:
        #     self.res = faiss.StandardGpuResources()
        #     cloner_options = faiss.GpuClonerOptions()
        #     cloner_options.useFloat16 = True 
            
        #     try:
        #         # Pushing IVFFlat to GPU is much easier than IVFPQ because 
        #         # there are no sub-quantizer memory blocks to overflow!
        #         self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index, cloner_options)
        #     except Exception as e:
        #         print(f"  -> [Hardware Warning] GPU push failed: {e}")
        #         print(f"  -> Falling back to CPU for FAISS. (Embeddings will still run on GPU)")
        #         self.use_gpu = False

    def _normalize(self, vectors):
        norm_vectors = vectors.copy()
        faiss.normalize_L2(norm_vectors)
        return norm_vectors

    def train(self, vectors):
        """Trains the Voronoi clusters. Leverages GPU for speed, then returns to CPU."""
        norm_vectors = self._normalize(vectors)
        
        if self.use_gpu:
            print("  -> [FAISS] Pushing to GPU for lightning-fast K-Means training...")
            res = faiss.StandardGpuResources()
            
            # Push the empty index to GPU just for training
            gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            gpu_index.train(norm_vectors)
            
            # Pull the trained centroids back to the CPU index
            print("  -> [FAISS] Training complete. Pulling centroids back to CPU...")
            self.index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            self.index.train(norm_vectors)

    def add_batch(self, vectors, ids):
        """Adds vectors to the CPU RAM index, preventing GPU VRAM OOM."""
        norm_vectors = self._normalize(vectors)
        ids_array = np.array(ids, dtype=np.int64)
        
        # This now safely appends to system RAM
        self.index.add_with_ids(norm_vectors, ids_array)

    def search(self, query_vectors, k=1):
        """Searches the CPU index."""
        norm_queries = self._normalize(query_vectors)
        distances, indices = self.index.search(norm_queries, k)
        return distances, indices

    def save(self, filepath):
        """Serializes the CPU index to disk."""
        faiss.write_index(self.index, filepath)