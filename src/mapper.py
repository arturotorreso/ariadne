import os
import faiss
import torch
import numpy as np
import sqlite3

from embedder import SequenceEmbedder
from chainer import SpatialChainer

class MetagenomicMapper:
    def __init__(self, db_path, index_path, window_size=100, use_gpu=None, use_mmap=False, nprobe=128, search_reverse_complement=True, include_terminal_window=True):
        self.window_size = window_size
        self.search_reverse_complement = search_reverse_complement
        self.include_terminal_window = include_terminal_window
        self._rc_table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
        
        # 1. Hardware Toggle
        if use_gpu is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            
        print(f"[Mapper] Initializing pipeline on device: {self.device.upper()}")
        
        # 2. Embedder (Pure RotorMap Math)
        # Use the GPU for rapid sequence embedding
        self.embedder = SequenceEmbedder(device=self.device, window_size=self.window_size)
        
        # 3. FAISS Index & Memory Mapping
        print(f"[Mapper] Connecting to FAISS index (MMAP={use_mmap})")
        
        #### Loading index ####
        
        if use_mmap:
            # Memory mapping
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        else:
            # Without memory mapping
            self.index = faiss.read_index(index_path)

        # I have commented this here because it keep memory crashing the test dataset
        # if self.device == 'cuda':
        #     self.res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        
        self.set_nprobe(nprobe)
        
        # 4. SQLite Metadata (Batched Read-Only Connection)
        print(f"[Mapper] Connecting to SQLite metadata...")
        db_uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        self.conn = sqlite3.connect(db_uri, uri=True)
        self.conn.row_factory = sqlite3.Row

    def set_nprobe(self, nprobe):
        """Adjusts the Voronoi cluster search radius safely based on DB size."""
        try:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
            else:
                faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", nprobe)
        except RuntimeError:
            print(f"  -> [Warning] Database is too small for nprobe={nprobe}. Falling back to exhaustive search.")

    def _reverse_complement(self, sequence):
        return sequence.translate(self._rc_table)[::-1].upper()

    def _chunk_sequence(self, sequence, stride):
        """Dynamically slices a sequence into windows and tracks their offset.

        Query windowing mirrors the database policy for full-length sequences:
        emit regular stride-spaced chunks, then add one terminal chunk if the
        stride grid misses the read end. Short reads are still padded with N;
        the embedder masks N so padded positions do not contribute signal.
        """
        chunks = []
        offsets = []
        seq_len = len(sequence)

        # Short reads are represented by one N-padded query window.
        if seq_len < self.window_size:
            chunks.append(sequence.ljust(self.window_size, 'N'))
            offsets.append(0)
        else:
            last_start = None
            for i in range(0, seq_len - self.window_size + 1, stride):
                chunks.append(sequence[i : i + self.window_size])
                offsets.append(i)
                last_start = i

            # Add a final full-length terminal chunk if the stride did not
            # already place a chunk ending exactly at the read end. This avoids
            # missing the 3' tail while preventing duplicate terminal chunks.
            if self.include_terminal_window:
                terminal_start = seq_len - self.window_size
                if last_start != terminal_start:
                    chunks.append(sequence[terminal_start:seq_len])
                    offsets.append(terminal_start)

        return chunks, offsets

    def map_reads(self, reads, query_stride=1, k=3, chain_alignments=False):
        """
        Sweeps across the read to find perfect phase-locks, then aggregates 
        and returns the absolute best hits. Can chain hits spatially.
        Searches both query strands when search_reverse_complement=True.
        """
        all_chunks = []
        chunk_offsets = []
        chunk_to_read_map = [] 
        chunk_strands = []
        read_lengths = {}
        
        # 1. Read Chunking Sweep
        for read in reads:
            read_lengths[read['id']] = len(read['seq'])
            chunks, offsets = self._chunk_sequence(read['seq'], stride=query_stride)
            for chunk, offset in zip(chunks, offsets):
                all_chunks.append(chunk)
                chunk_offsets.append(offset)
                chunk_to_read_map.append(read['id'])
                chunk_strands.append('+')

                if self.search_reverse_complement:
                    all_chunks.append(self._reverse_complement(chunk))
                    chunk_offsets.append(offset)
                    chunk_to_read_map.append(read['id'])
                    chunk_strands.append('-')
                
        if not all_chunks:
            return []
            
        # 2. Embedding Generation
        vectors = self.embedder.embed_batch(all_chunks)
        faiss.normalize_L2(vectors)
        
        # 3. FAISS AVX/SIMD Search 
        # If chaining, fetch a wider net (20) so the bins have enough data to form chains.
        search_k = 20 if chain_alignments else k
        distances, indices = self.index.search(vectors, search_k)
        
        # 4. Fast SQLite Batch Retrieval
        unique_ids = list(set(indices.flatten().tolist()))
        unique_ids = [idx for idx in unique_ids if idx != -1]
        metadata_map = self._fetch_metadata_batch(unique_ids)
        
        # 5. Aggregate all chunk hits by their parent read
        read_results = {read['id']: [] for read in reads}
        for i in range(len(all_chunks)):
            read_id = chunk_to_read_map[i]
            
            for j in range(search_k):
                hit_id = int(indices[i][j])
                if hit_id == -1: continue
                    
                cosine_sim = float(distances[i][j])
                meta = metadata_map.get(hit_id, {'header': 'UNKNOWN', 'start_pos': 0})
                
                read_results[read_id].append({
                    'faiss_id': hit_id,
                    'cosine_sim': cosine_sim,
                    'strand': chunk_strands[i],
                    'header': meta['header'],
                    'start_pos': meta['start_pos'],
                    'query_offset': chunk_offsets[i] # Required for Chainer
                })
        
        final_results = []
        
        # 6a. NEW CHAINING LOGIC
        if chain_alignments:
            chainer = SpatialChainer(window_size=self.window_size)
            for read_id, hits in read_results.items():
                chained_hits = chainer.chain(read_lengths[read_id], hits)
                final_results.append({
                    'read_id': read_id,
                    'hits': chained_hits[:k] # Return the top K chained alignments
                })
                
        # 6b. ORIGINAL UNCHAINED LOGIC
        else:
            for read_id, hits in read_results.items():
                sorted_hits = sorted(hits, key=lambda x: x['cosine_sim'], reverse=True)
                unique_targets = {}
                filtered_hits = []
                for hit in sorted_hits:
                    target_key = (hit['header'], hit['start_pos'], hit['strand'])
                    if target_key not in unique_targets:
                        unique_targets[target_key] = True
                        filtered_hits.append(hit)
                    if len(filtered_hits) >= k:
                        break
                final_results.append({
                    'read_id': read_id,
                    'hits': filtered_hits
                })
            
        return final_results

    def _fetch_metadata_batch(self, ids):
        if not ids: return {}
        # Chunk the SQL query to prevent SQLite from crashing if 'search_k' pushes 
        # the unique_ids past SQLite's variable limits (usually 999).
        ids_list = list(ids)
        result = {}
        chunk_size = 999
        cursor = self.conn.cursor()
        
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i:i+chunk_size]
            placeholders = ','.join('?' * len(chunk))
            query = f"SELECT id, header, start_pos FROM metadata WHERE id IN ({placeholders})"
            cursor.execute(query, chunk)
            for row in cursor.fetchall():
                result[row['id']] = {'header': row['header'], 'start_pos': row['start_pos']}
        return result

    def close(self):
        self.conn.close()
