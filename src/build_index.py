import os
import time
import math
import torch
import numpy as np

# Import our custom pipeline modules
from ingestion import sliding_window_fasta
from manifest import build_manifest, update_manifest_config
from embedder import SequenceEmbedder
from faiss_indexer import FaissIndexer


def estimate_index_parameters(total_windows, embedding_dim=2048, max_ram_gb=64):
    """
    Estimate FAISS training parameters from the exact manifest window count.

    The old implementation estimated N from FASTA file size. The manifest now
    owns exact window counts, including terminal windows and padded short contigs.
    """
    estimated_n = max(100, int(total_windows))
    raw_nlist = 4 * math.sqrt(estimated_n)
    nlist = max(1, 1 << int(math.log2(raw_nlist))) # Ensure at least 1 cluster
    
    # Calculate bounds
    # pq_m=64 usually needs at least ~10,000 vectors for stable PQ training.
    min_t = max(39 * nlist, 10000) 
    ideal_t = 64 * nlist
    bytes_per_vector = embedding_dim * 4
    max_safe_vectors = int((max_ram_gb * 1024**3 * 0.30) / bytes_per_vector)

    # Target the sweet spot, but respect the floor and hardware ceiling.
    target_t = max(min_t, min(ideal_t, max_safe_vectors))
    target_t = min(target_t, estimated_n)
    
    matrix_gb = (target_t * embedding_dim * 4) / (1024**3)
    print(f"  -> [Memory] Required Training Matrix: ~{matrix_gb:.4f} GB")
        
    sampling_fraction = max(1, int(estimated_n / target_t))
    return estimated_n, nlist, target_t, sampling_fraction


# pq_m is the FAISS PQ subquantizer count; RotorMap uses rotor_m internally.
def build_pipeline(
    fasta_path,
    db_path,
    index_path,
    window_size=100,
    stride=50,
    batch_size=100000,
    train_mode="auto",
    quantizer="PQ",
    pq_m=64,
    include_terminal_window=True,
    index_short_contigs=True,
    min_short_contig_len=50,
    pad_short_contigs=True,
):
    start_time = time.time()
    
    print(f"\n[1/5] Building Exact Manifest: {fasta_path}")
    manifest_summary = build_manifest(
        db_path=db_path,
        fasta_paths=fasta_path,
        window_size=window_size,
        stride=stride,
        include_terminal_window=include_terminal_window,
        index_short_contigs=index_short_contigs,
        min_short_contig_len=min_short_contig_len,
        pad_short_contigs=pad_short_contigs,
        quantizer=quantizer,
        pq_m=pq_m,
        overwrite=True,
    )
    print(f"  -> Total Contigs: {manifest_summary['total_contigs']:,}")
    print(f"  -> Indexed Contigs: {manifest_summary['total_indexed_contigs']:,}")
    print(f"  -> Exact Windows (N): {manifest_summary['total_windows']:,}")
    print(f"  -> Metadata Store: compact contig manifest at {db_path}")

    # Initialize Hardware and Models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Hardware] PyTorch initialized on: {device.upper()}")
    
    embedder = SequenceEmbedder(device=device, window_size=window_size)
    dim = embedder.embedding_dim

    est_n, nlist, target_t, fraction = estimate_index_parameters(
        manifest_summary['total_windows'],
        embedding_dim=dim,
    )
    update_manifest_config(
        db_path,
        embedding_dim=dim,
        nlist=nlist,
        training_size=target_t,
        training_sampling_fraction=fraction,
    )
    print(f"  -> FAISS Clusters (nlist): {nlist:,}")
    print(f"  -> Required Training Size (T): {target_t:,}")
    print(f"  -> Sampling Strategy: 1 out of every {fraction} windows\n")
    
    # Initialize FAISS with the GPU flag
    use_faiss_gpu = (device == 'cuda')
    print(f"[Hardware] FAISS initialized on: {'GPU' if use_faiss_gpu else 'CPU'}")
    
    # Passing the FAISS PQ parameter down to the indexer
    indexer = FaissIndexer(embedding_dim=dim, nlist=nlist, use_gpu=use_faiss_gpu, train_mode=train_mode, quantizer=quantizer, pq_m=pq_m)

    # ==========================================
    # PASS 1: SYSTEMATIC FRACTIONAL SAMPLING
    # ==========================================
    print(f"\n[2/5] Pass 1: Extracting Representative Sample for FAISS Training...")
    training_strings = []
    generator_pass_1 = sliding_window_fasta(
        fasta_path,
        window_size,
        stride,
        batch_size,
        include_terminal_window=include_terminal_window,
        index_short_contigs=index_short_contigs,
        min_short_contig_len=min_short_contig_len,
        pad_short_contigs=pad_short_contigs,
    )
    
    windows_scanned = 0
    for seq_batch, _ in generator_pass_1:
        for i in range(0, len(seq_batch), fraction):
            if len(training_strings) < target_t:
                training_strings.append(seq_batch[i])
        
        windows_scanned += len(seq_batch)
        if len(training_strings) >= target_t:
            break 

    print(f"  -> Gathered {len(training_strings):,} strings. Converting to dense vectors...")
    
    # MEMORY OPTIMIZATION: Pre-allocate the final training matrix to avoid np.vstack duplication.
    full_training_matrix = np.zeros((len(training_strings), dim), dtype=np.float32)
    
    chunk_size = batch_size
    for i in range(0, len(training_strings), chunk_size):
        chunk = training_strings[i : i + chunk_size]
        vecs = embedder.embed_batch(chunk)
        # Drop the vectors directly into the pre-allocated matrix.
        full_training_matrix[i : i + len(chunk)] = vecs
        
    del training_strings
    
    print(f"\n[3/5] Training FAISS.")
    indexer.train(full_training_matrix)
    del full_training_matrix
    print("  -> FAISS Index successfully trained.")

    # ==========================================
    # PASS 2: THE MAIN INGESTION LOOP
    # ==========================================
    print(f"\n[4/5] Pass 2: Main Ingestion Loop (Streaming, Embedding, Indexing)...")
    print("  -> Using compact manifest metadata; no per-window SQLite inserts.")
    generator_pass_2 = sliding_window_fasta(
        fasta_path,
        window_size,
        stride,
        batch_size,
        include_terminal_window=include_terminal_window,
        index_short_contigs=index_short_contigs,
        min_short_contig_len=min_short_contig_len,
        pad_short_contigs=pad_short_contigs,
    )
    
    global_faiss_id = 0
    batches_processed = 0
    
    for seq_batch, _ in generator_pass_2:
        # 1. Embed on GPU/CPU through the configured embedder.
        vectors = embedder.embed_batch(seq_batch)
        
        # 2. IDs are assigned deterministically by the manifest/window order.
        next_global_id = global_faiss_id + len(seq_batch)
        id_list = list(range(global_faiss_id, next_global_id))
        
        # 3. Add to FAISS.
        indexer.add_batch(vectors, id_list)
        
        global_faiss_id = next_global_id
        batches_processed += 1
        
        if batches_processed % 10 == 0:
            print(f"  -> Processed {global_faiss_id:,} windows...")

    if global_faiss_id != manifest_summary['total_windows']:
        raise RuntimeError(
            "Window generation mismatch: "
            f"manifest has {manifest_summary['total_windows']:,} windows but ingestion emitted {global_faiss_id:,}."
        )

    update_manifest_config(db_path, indexed_windows=global_faiss_id)

    # ==========================================
    # SERIALIZATION
    # ==========================================
    print(f"\n[5/5] Pipeline Complete. Serializing Assets to Disk...")
    indexer.save(index_path)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Indexed {global_faiss_id:,} genomic windows in {total_time:.2f} minutes.")
    print(f"  -> Compact Manifest Database: {db_path}")
    print(f"  -> Vector Index: {index_path}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mode", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--quantizer", type=str, choices=["SQ8", "PQ"], default="PQ")
    parser.add_argument("--pq-m", dest="pq_m", type=int, default=64)
    parser.add_argument("--m", dest="pq_m", type=int, help=argparse.SUPPRESS) # Backward-compatible alias
    args, _ = parser.parse_known_args()

    # Define paths based on your architecture
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    FASTA_INPUT = os.path.join(project_root, "fda_argos", "fda_argos_subsampled_100.fa")

    # We will create an 'output' folder in your root directory for the database artifacts
    SQLITE_OUTPUT = os.path.join(project_root, "new_output", "fda_argos.db")
    FAISS_OUTPUT = os.path.join(project_root, "new_output", "fda_argos.index")
    
    os.makedirs(os.path.dirname(SQLITE_OUTPUT), exist_ok=True)
    
    # Execute the pipeline
    build_pipeline(
        fasta_path=FASTA_INPUT,
        db_path=SQLITE_OUTPUT,
        index_path=FAISS_OUTPUT,
        window_size=100,
        stride=50,
        batch_size=100000,
        train_mode=args.train_mode,
        quantizer=args.quantizer,
        pq_m=args.pq_m
    )
