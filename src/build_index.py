import os
import time
import math
import torch
import numpy as np

# Import our custom pipeline modules
from ingestion import sliding_window_fasta
from metadata_store import MetadataStore
from embedder import SequenceEmbedder
from faiss_indexer import FaissIndexer

# def estimate_database_parameters(file_path, window_size, stride, max_ram_gb=64):
#     file_bytes = os.path.getsize(file_path)
#     estimated_bases = file_bytes * 3.5 if file_path.endswith('.gz') else file_bytes
#     estimated_n = int(estimated_bases / stride)
    
#     raw_nlist = 4 * math.sqrt(estimated_n)
#     nlist = 1 << int(math.log2(raw_nlist))
    
#     # Calculate absolute bounds for FAISS K-Means
#     min_t = 39 * nlist
#     ideal_t = 64 * nlist
#     max_t = 256 * nlist
    
#     # 1 vector = 2048 dims * 4 bytes = 8192 bytes. 
#     # We want the matrix to take up no more than 30% of total system RAM
#     max_safe_vectors = int((max_ram_gb * 1024**3 * 0.30) / 8192)

#     # We target 64x nlist for good cluster resolution, but it MUST be at least min_t
#     target_t = max(min_t, min(ideal_t, max_safe_vectors))
    
#     # Matrix Memory calculation: target_t * 2048 dimensions * 4 bytes (float32)
#     matrix_gb = (target_t * 2048 * 4) / (1024**3)
    
#     print(f"  -> [Memory] Required Training Matrix: ~{matrix_gb:.2f} GB")
#     if matrix_gb > 25.0:
#         print(f"  -> [WARNING] High memory required. Ensure your Slurm node has at least {matrix_gb * 2.5:.0f}GB RAM to prevent OOM!")
        
#     sampling_fraction = max(1, int(estimated_n / target_t))
    
#     return estimated_n, nlist, target_t, sampling_fraction


def estimate_database_parameters(file_path, window_size, stride, max_ram_gb=64):
    file_bytes = os.path.getsize(file_path)
    estimated_bases = file_bytes * 3.5 if file_path.endswith('.gz') else file_bytes
    estimated_n = max(100, int(estimated_bases / stride)) # Prevent division by zero
    
    raw_nlist = 4 * math.sqrt(estimated_n)
    nlist = max(1, 1 << int(math.log2(raw_nlist))) # Ensure at least 1 cluster
    
    # Calculate bounds
    # m=64 requires at least ~10,000 vectors for the PQ subquantizers to train
    min_t = max(39 * nlist, 10000) 
    ideal_t = 64 * nlist
    max_safe_vectors = int((max_ram_gb * 1024**3 * 0.30) / 8192)

    # Target the sweet spot, but respect the floor and hardware ceiling
    target_t = max(min_t, min(ideal_t, max_safe_vectors))
    
    # CRITICAL FOR TOY DATABASES: 
    # Do not demand more training vectors than actual windows in the database!
    target_t = min(target_t, estimated_n)
    
    matrix_gb = (target_t * 2048 * 4) / (1024**3)
    
    print(f"  -> [Memory] Required Training Matrix: ~{matrix_gb:.4f} GB")
        
    sampling_fraction = max(1, int(estimated_n / target_t))
    
    return estimated_n, nlist, target_t, sampling_fraction

def build_pipeline(fasta_path, db_path, index_path, window_size=100, stride=50, batch_size=100000):
    start_time = time.time()
    
    print(f"\n[1/5] Analyzing Database: {fasta_path}")
    est_n, nlist, target_t, fraction = estimate_database_parameters(fasta_path, window_size, stride)
    print(f"  -> Estimated Windows (N): ~{est_n:,}")
    print(f"  -> FAISS Clusters (nlist): {nlist:,}")
    print(f"  -> Required Training Size (T): {target_t:,}")
    print(f"  -> Sampling Strategy: 1 out of every {fraction} windows\n")

    # Initialize Hardware and Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Hardware] PyTorch initialized on: {device.upper()}")
    
    embedder = SequenceEmbedder(device=device)
    dim = embedder.embedding_dim # Dynamically grabs 768
    
    # Initialize FAISS with the GPU flag
    use_faiss_gpu = (device == 'cuda')
    print(f"[Hardware] FAISS initialized on: {'GPU' if use_faiss_gpu else 'CPU'}")
    indexer = FaissIndexer(embedding_dim=dim, nlist=nlist, m=32, use_gpu=use_faiss_gpu)
    store = MetadataStore(db_path)

    # ==========================================
    # PASS 1: SYSTEMATIC FRACTIONAL SAMPLING
    # ==========================================
    print(f"\n[2/5] Pass 1: Extracting Representative Sample for FAISS Training...")
    training_strings = []
    generator_pass_1 = sliding_window_fasta(fasta_path, window_size, stride, batch_size)
    
    windows_scanned = 0
    for seq_batch, _ in generator_pass_1:
        for i in range(0, len(seq_batch), fraction):
            if len(training_strings) < target_t:
                training_strings.append(seq_batch[i])
        
        windows_scanned += len(seq_batch)
        if len(training_strings) >= target_t:
            break 

    print(f"  -> Gathered {len(training_strings):,} strings. Converting to dense vectors...")
    
    # MEMORY OPTIMIZATION: Pre-allocate the final 10GB matrix to avoid np.vstack duplication
    full_training_matrix = np.zeros((len(training_strings), dim), dtype=np.float32)
    
    chunk_size = batch_size
    for i in range(0, len(training_strings), chunk_size):
        chunk = training_strings[i : i + chunk_size]
        vecs = embedder.embed_batch(chunk)
        # Drop the vectors directly into the pre-allocated matrix
        full_training_matrix[i : i + len(chunk)] = vecs
        
    del training_strings # Free up the 1GB string list
    
    print(f"\n[3/5] Training FAISS.")
    indexer.train(full_training_matrix)
    del full_training_matrix # Free up the 10GB array immediately
    print("  -> FAISS Index successfully trained.")

    # ==========================================
    # PASS 2: THE MAIN INGESTION LOOP
    # ==========================================
    print(f"\n[4/5] Pass 2: Main Ingestion Loop (Streaming, Embedding, Indexing)...")
    generator_pass_2 = sliding_window_fasta(fasta_path, window_size, stride, batch_size)
    
    global_faiss_id = 0
    batches_processed = 0
    
    for seq_batch, meta_batch in generator_pass_2:
        # 1. Embed on GPU
        vectors = embedder.embed_batch(seq_batch)
        
        # 2. Insert to SQLite and get IDs
        # store.insert_batch returns the NEXT available ID, so we subtract batch length 
        # to get the exact ID list for the current batch
        next_global_id = store.insert_batch(global_faiss_id, meta_batch)
        id_list = list(range(global_faiss_id, next_global_id))
        
        # 3. Add to FAISS
        indexer.add_batch(vectors, id_list)
        
        global_faiss_id = next_global_id
        batches_processed += 1
        
        if batches_processed % 10 == 0:
            print(f"  -> Processed {global_faiss_id:,} windows...")

    # ==========================================
    # SERIALIZATION
    # ==========================================
    print(f"\n[5/5] Pipeline Complete. Serializing Assets to Disk...")
    indexer.save(index_path)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Indexed {global_faiss_id:,} genomic windows in {total_time:.2f} minutes.")
    print(f"  -> Metadata Database: {db_path}")
    print(f"  -> Vector Index: {index_path}\n")

if __name__ == '__main__':
    # Define paths based on your architecture
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Pointing to your specific fda_argos folder
    FASTA_INPUT = os.path.join(project_root, "fda_argos", "fda_argos_subsampled.fa")
    
    # We will create an 'output' folder in your root directory for the database artifacts
    SQLITE_OUTPUT = os.path.join(project_root, "output", "fda_argos.db")
    FAISS_OUTPUT = os.path.join(project_root, "output", "fda_argos.index")
    
    os.makedirs(os.path.dirname(SQLITE_OUTPUT), exist_ok=True)
    
    # Execute the pipeline
    build_pipeline(
        fasta_path=FASTA_INPUT,
        db_path=SQLITE_OUTPUT,
        index_path=FAISS_OUTPUT,
        window_size=100,
        stride=50,
        batch_size=100000 
    )