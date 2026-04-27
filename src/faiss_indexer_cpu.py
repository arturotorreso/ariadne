import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, embedding_dim, nlist=1, m=8):
        """
        Initializes the Inverted File with Product Quantization (IVF_PQ) index.
        """
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        
        # 1. The Quantizer: We use Inner Product (IP) to enable Cosine Similarity
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        # 2. The Index: 
        # The '8' specifies that each sub-vector is encoded with 8 bits.
        self.index = faiss.IndexIVFPQ(
            self.quantizer, 
            self.embedding_dim, 
            self.nlist, 
            m, 
            8, 
            faiss.METRIC_INNER_PRODUCT
        )

    def _normalize(self, vectors):
        """
        L2 normalizes the vectors. This is mathematically MANDATORY 
        to ensure Inner Product equals Cosine Similarity.
        """
        # Create a copy so we don't accidentally mutate the arrays used elsewhere
        norm_vectors = vectors.copy()
        faiss.normalize_L2(norm_vectors)
        return norm_vectors

    def train(self, vectors):
        """Trains the Voronoi clusters and PQ codebooks."""
        norm_vectors = self._normalize(vectors)
        self.index.train(norm_vectors)

    def add_batch(self, vectors, ids):
        """Adds a batch of vectors with their corresponding metadata IDs."""
        norm_vectors = self._normalize(vectors)
        # FAISS will crash if IDs are not strictly 64-bit integers
        ids_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(norm_vectors, ids_array)

    def search(self, query_vectors, k=1):
        """Searches the index and returns distances (cosine similarities) and IDs."""
        norm_queries = self._normalize(query_vectors)
        distances, indices = self.index.search(norm_queries, k)
        return distances, indices

    def save(self, filepath):
        """Serializes the index to disk."""
        faiss.write_index(self.index, filepath)