import faiss
import numpy as np
import torch

class FaissIndexer:
    # Removed 'm' because we are storing raw uncompressed vectors
    def __init__(self, embedding_dim, nlist=1, use_gpu=False, train_mode="auto", quantizer="SQ8", m=128):
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.train_mode = train_mode
        
        # 1. Build the Base CPU Index 
        # IVFSQ8 = Clustered lookup + 8-bit scalar quantized vectors (4x compression)!
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        if quantizer == "PQ":
            self.index = faiss.IndexIVFPQ(
                self.quantizer,
                self.embedding_dim,
                self.nlist,
                m,
                8, # 8 bits per subquantizer
                faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexIVFScalarQuantizer(
                self.quantizer, 
                self.embedding_dim, 
                self.nlist, 
                faiss.ScalarQuantizer.QT_8bit,
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
        #     except RuntimeError as e:
        #         print(f"Warning: Could not push index to GPU. Falling back to CPU. Error: {e}")
        #         self.use_gpu = False 

    def _normalize(self, vectors):
        """FAISS Inner Product becomes Cosine Similarity if vectors are L2 Normalized."""
        # Convert to contiguous float32 array
        norm_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # In-place L2 normalization
        faiss.normalize_L2(norm_vectors)
        return norm_vectors

    def train(self, vectors):
        """Trains the Voronoi clusters. Leverages GPU for speed, then returns to CPU."""
        norm_vectors = self._normalize(vectors)
        
        use_gpu_for_training = self.use_gpu
        
        if self.train_mode == "cpu":
            use_gpu_for_training = False
        elif self.train_mode == "auto" and self.use_gpu:
            try:
                n_vectors, d = norm_vectors.shape
                required_vram = n_vectors * d * 4 * 1.5
                free_vram, _ = torch.cuda.mem_get_info()
                
                if free_vram < required_vram:
                    print(f"  -> [Warning] Insufficient VRAM for GPU training (Need {required_vram/1e9:.2f}GB, Free {free_vram/1e9:.2f}GB). Falling back to CPU training.")
                    use_gpu_for_training = False
            except Exception as e:
                print(f"  -> [Warning] Could not detect GPU memory: {e}. Falling back to CPU training.")
                use_gpu_for_training = False
        elif self.train_mode == "gpu":
            use_gpu_for_training = True
            
        if use_gpu_for_training:
            print("  -> [FAISS] Pushing to GPU for lightning-fast K-Means training...")
            res = faiss.StandardGpuResources()
            
            try:
                # Push the empty index to GPU just for training
                gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                gpu_index.train(norm_vectors)
                
                # Pull the trained centroids back to the CPU index
                print("  -> [FAISS] Training complete. Pulling centroids back to CPU...")
                self.index = faiss.index_gpu_to_cpu(gpu_index)
            except RuntimeError as e:
                print(f"  -> [Warning] GPU training failed (unsupported settings): {e}")
                print("  -> [FAISS] Falling back to CPU training...")
                self.index.train(norm_vectors)
        else:
            print("  -> [FAISS] Executing K-Means on CPU...")
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
        """Serializes the index to disk."""
        faiss.write_index(self.index, filepath)

    @classmethod
    def load(cls, filepath, use_gpu=False):
        """Loads a pre-built index from disk."""
        # Note: We bypass the init because we are loading an existing C++ object
        instance = cls.__new__(cls)
        instance.use_gpu = use_gpu
        instance.index = faiss.read_index(filepath)
        instance.embedding_dim = instance.index.d
        
        # If GPU was requested for searching
        if use_gpu:
            res = faiss.StandardGpuResources()
            cloner_options = faiss.GpuClonerOptions()
            cloner_options.useFloat16 = True 
            try:
                instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index, cloner_options)
            except RuntimeError as e:
                print(f"Warning: Could not push index to GPU for search. Using CPU. Error: {e}")
                instance.use_gpu = False
                
        return instance