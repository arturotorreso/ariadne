import os
import faiss
import torch
import numpy as np
import sqlite3

from embedder import SequenceEmbedder

class MetagenomicMapper:
    # Removed model and embedding_dim arguments
    def __init__(self, db_path, index_path, window_size=100, use_gpu=None):
        self.window_size = window_size
        
        # 1. Hardware Toggle
        if use_gpu is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            
        print(f"[Mapper] Initializing pipeline on device: {self.device.upper()}")
        
        # 2. Embedder (Pure RotorMap Math)
        self.embedder = SequenceEmbedder(device=self.device)
        
        # 3. FAISS Index & Memory Mapping
        print(f"[Mapper] Connecting to FAISS index...")
        self.index = faiss.read_index(index_path)
        
        if self.device == 'cuda':
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        
        self.set_nprobe(8)
        
        # 4. SQLite Metadata (Batched Read-Only Connection)
        print(f"[Mapper] Connecting to SQLite metadata...")
        db_uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        self.conn = sqlite3.connect(db_uri, uri=True)
        self.conn.row_factory = sqlite3.Row

    def set_nprobe(self, nprobe):
        """Adjusts the Voronoi cluster search radius."""
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        else:
            faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", nprobe)

    def _chunk_sequence(self, sequence, stride):
        """
        Dynamically slices a sequence into windows. 
        If stride = window_size, it is non-overlapping.
        """
        chunks = []
        seq_len = len(sequence)
        
        # Pad sequence if it is mathematically too short for the model
        if seq_len < self.window_size:
            chunks.append(sequence.ljust(self.window_size, 'N'))
        else:
            for i in range(0, seq_len - self.window_size + 1, stride):
                chunks.append(sequence[i : i + self.window_size])
        return chunks

    def map_reads(self, reads, query_stride=50, k=1):
        """
        The main ingestion engine. Takes a list of read dictionaries, chunks them,
        embeds them, queries FAISS, translates the math, and fetches the metadata.
        """
        all_chunks = []
        chunk_to_read_map = [] 
        
        # 1. Read Chunking
        for read in reads:
            chunks = self._chunk_sequence(read['seq'], stride=query_stride)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_read_map.append(read['id'])
                
        if not all_chunks:
            return []
            
        # 2. Embedding Generation
        vectors = self.embedder.embed_batch(all_chunks)
        faiss.normalize_L2(vectors)
        
        # 3. FAISS AVX/SIMD Search
        distances, indices = self.index.search(vectors, k)
        
        # 4. Fast SQLite Batch Retrieval
        unique_ids = list(set(indices.flatten().tolist()))
        # Remove FAISS's '-1' marker for missing clusters
        unique_ids = [idx for idx in unique_ids if idx != -1]
        metadata_map = self._fetch_metadata_batch(unique_ids)
        
        # 5. Math Translation & Formatting
        results = []
        for i in range(len(all_chunks)):
            read_id = chunk_to_read_map[i]
            chunk_seq = all_chunks[i]
            chunk_hits = []
            
            for j in range(k):
                hit_id = int(indices[i][j])
                if hit_id == -1: continue
                    
                cosine_sim = float(distances[i][j])
                
                # Proxy Formula: Mismatches = 100 * (1 - Cosine Similarity)
                mismatches = max(0, int(round(self.window_size * (1.0 - cosine_sim))))
                
                meta = metadata_map.get(hit_id, {'header': 'UNKNOWN', 'start_pos': 0})
                
                chunk_hits.append({
                    'faiss_id': hit_id,
                    'cosine_sim': cosine_sim,
                    'mismatches': mismatches,
                    'header': meta['header'],
                    'start_pos': meta['start_pos'] # Plugs directly into map_reads.py!
                })
                
            results.append({
                'read_id': read_id,
                'chunk_seq': chunk_seq,
                'hits': chunk_hits
            })
            
        return results

    def _fetch_metadata_batch(self, ids):
        """Executes a single ultra-fast query using the IN clause."""
        if not ids: return {}
            
        placeholders = ','.join('?' * len(ids))
        query = f"SELECT id, header, start_pos FROM metadata WHERE id IN ({placeholders})"
        
        cursor = self.conn.cursor()
        cursor.execute(query, ids)
        rows = cursor.fetchall()
        
        return {row['id']: {'header': row['header'], 'start_pos': row['start_pos']} for row in rows}

    def close(self):
        self.conn.close()