import math
import os
import time
import traceback

import numpy as np
import torch

from embedder import SequenceEmbedder
from faiss_indexer import FaissIndexer
from manifest import (
    assign_shards,
    build_manifest,
    file_sha256,
    get_manifest_config,
    get_manifest_summary,
    get_shard,
    get_shards,
    stream_windows_from_manifest,
    update_manifest_config,
    update_shard_status,
    utc_now_iso,
    validate_shard_index,
)


def estimate_index_parameters(total_windows, embedding_dim=2048, max_ram_gb=64):
    """
    Estimate FAISS training parameters from the exact manifest window count.

    The old implementation estimated N from FASTA file size. The manifest now
    owns exact window counts, including terminal windows and padded short contigs.
    """
    estimated_n = max(100, int(total_windows))
    raw_nlist = 4 * math.sqrt(estimated_n)
    nlist = max(1, 1 << int(math.log2(raw_nlist)))

    # pq_m=64 usually needs at least ~10,000 vectors for stable PQ training.
    min_t = max(39 * nlist, 10000)
    ideal_t = 64 * nlist
    bytes_per_vector = embedding_dim * 4
    max_safe_vectors = int((max_ram_gb * 1024**3 * 0.30) / bytes_per_vector)

    target_t = max(min_t, min(ideal_t, max_safe_vectors))
    target_t = min(target_t, estimated_n)

    matrix_gb = (target_t * embedding_dim * 4) / (1024**3)
    print(f"  -> [Memory] Required Training Matrix: ~{matrix_gb:.4f} GB")

    sampling_fraction = max(1, int(estimated_n / target_t))
    return estimated_n, nlist, target_t, sampling_fraction


def _normalize_for_faiss(vectors):
    """Normalize vectors for cosine-similarity search with FAISS inner product."""
    import faiss

    norm_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(norm_vectors)
    return norm_vectors


def _default_template_path(index_path):
    base, _ = os.path.splitext(index_path)
    return base + ".template.index"


def _default_shards_dir(index_path):
    base, _ = os.path.splitext(index_path)
    return base + "_shards"


def _prepare_manifest(
    fasta_path,
    db_path,
    window_size,
    stride,
    quantizer,
    pq_m,
    include_terminal_window,
    index_short_contigs,
    min_short_contig_len,
    pad_short_contigs,
    resume,
):
    """Create or reuse the exact compact manifest."""
    if resume and os.path.exists(db_path):
        print(f"\n[1/6] Reusing Existing Manifest: {db_path}")
        return get_manifest_summary(db_path)

    print(f"\n[1/6] Building Exact Manifest: {fasta_path}")
    return build_manifest(
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


def train_faiss_template(
    db_path,
    template_path,
    embedder,
    embedding_dim,
    nlist,
    target_training_size,
    sampling_fraction,
    batch_size,
    train_mode,
    quantizer,
    pq_m,
    use_faiss_gpu,
    resume=False,
):
    """
    Train a global empty FAISS template and save it to disk.

    Shards must all share the same trained IVF/PQ geometry. The saved template
    contains trained centroids/codebooks but no database vectors.
    """
    if resume and os.path.exists(template_path):
        print(f"\n[3/6] Reusing Trained FAISS Template: {template_path}")
        update_manifest_config(db_path, trained_template_path=template_path)
        return template_path

    print("\n[3/6] Training Global FAISS Template...")
    training_strings = []

    for seq_batch, id_batch in stream_windows_from_manifest(db_path, batch_size=batch_size):
        for seq, faiss_id in zip(seq_batch, id_batch):
            # Sample by exact global FAISS ID so sampling is reproducible and
            # independent of batch boundaries.
            if faiss_id % sampling_fraction == 0:
                training_strings.append(seq)
                if len(training_strings) >= target_training_size:
                    break
        if len(training_strings) >= target_training_size:
            break

    if not training_strings:
        raise RuntimeError("No training windows were collected from the manifest.")

    print(f"  -> Gathered {len(training_strings):,} windows. Converting to dense vectors...")
    full_training_matrix = np.zeros((len(training_strings), embedding_dim), dtype=np.float32)

    for i in range(0, len(training_strings), batch_size):
        chunk = training_strings[i : i + batch_size]
        full_training_matrix[i : i + len(chunk)] = embedder.embed_batch(chunk)

    del training_strings

    indexer = FaissIndexer(
        embedding_dim=embedding_dim,
        nlist=nlist,
        use_gpu=use_faiss_gpu,
        train_mode=train_mode,
        quantizer=quantizer,
        pq_m=pq_m,
    )
    indexer.train(full_training_matrix)
    del full_training_matrix

    os.makedirs(os.path.dirname(os.path.abspath(template_path)), exist_ok=True)
    indexer.save(template_path)
    checksum = file_sha256(template_path)

    update_manifest_config(
        db_path,
        trained_template_path=template_path,
        trained_template_checksum=checksum,
        training_size_observed=target_training_size,
    )
    print(f"  -> Saved trained empty template: {template_path}")
    return template_path


def build_monolithic_index_from_template(
    db_path,
    template_path,
    index_path,
    embedder,
    batch_size,
):
    """
    Compatibility path: build one full FAISS index from the trained template.

    This preserves the current one-index mapper workflow while using the new
    compact manifest and separated training-template architecture.
    """
    import faiss

    print("\n[4/6] Building Monolithic Index From Template...")
    index = faiss.read_index(template_path)
    total_added = 0

    for seq_batch, id_batch in stream_windows_from_manifest(db_path, batch_size=batch_size):
        vectors = embedder.embed_batch(seq_batch)
        index.add_with_ids(_normalize_for_faiss(vectors), np.asarray(id_batch, dtype=np.int64))
        total_added += len(id_batch)
        if total_added % 1_000_000 < len(id_batch):
            print(f"  -> Added {total_added:,} windows...")

    expected = int(get_manifest_config(db_path).get("total_windows", total_added))
    if total_added != expected:
        raise RuntimeError(
            f"Window generation mismatch: manifest has {expected:,} windows but builder emitted {total_added:,}."
        )
    if int(index.ntotal) != expected:
        raise RuntimeError(f"FAISS ntotal mismatch: observed {index.ntotal:,}, expected {expected:,}.")

    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    faiss.write_index(index, index_path)
    update_manifest_config(db_path, monolithic_index_path=index_path, indexed_windows=total_added)
    print(f"  -> Saved monolithic index: {index_path}")
    return index_path


def build_shard(
    db_path,
    shard_id,
    template_path,
    embedder,
    batch_size,
    resume=True,
):
    """Build exactly one FAISS shard from the global trained template."""
    import faiss

    shard = get_shard(db_path, shard_id)
    final_path = shard["index_path"]
    tmp_path = shard["tmp_index_path"] or (final_path + ".tmp")
    expected = int(shard["ntotal_expected"] or shard["n_windows"])

    if resume:
        valid, observed, error = validate_shard_index(final_path, expected)
        if valid:
            print(f"  -> Shard {shard_id:06d} already complete ({observed:,} vectors). Skipping.")
            if shard.get("status") != "complete":
                update_shard_status(
                    db_path,
                    shard_id,
                    "complete",
                    ntotal_observed=observed,
                    completed_at=utc_now_iso(),
                    error_message=None,
                )
            return final_path
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"  -> Building shard {shard_id:06d}: expected {expected:,} windows")
    update_shard_status(
        db_path,
        shard_id,
        "running",
        started_at=utc_now_iso(),
        completed_at=None,
        error_message=None,
    )

    try:
        index = faiss.read_index(template_path)
        total_added = 0

        for seq_batch, id_batch in stream_windows_from_manifest(
            db_path,
            batch_size=batch_size,
            shard_id=shard_id,
        ):
            vectors = embedder.embed_batch(seq_batch)
            index.add_with_ids(_normalize_for_faiss(vectors), np.asarray(id_batch, dtype=np.int64))
            total_added += len(id_batch)
            if total_added % 1_000_000 < len(id_batch):
                print(f"     shard {shard_id:06d}: added {total_added:,} windows...")

        if total_added != expected:
            raise RuntimeError(
                f"Shard {shard_id} emitted {total_added:,} windows, expected {expected:,}."
            )
        if int(index.ntotal) != expected:
            raise RuntimeError(
                f"Shard {shard_id} FAISS ntotal {index.ntotal:,}, expected {expected:,}."
            )

        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
        faiss.write_index(index, tmp_path)

        valid, observed, error = validate_shard_index(tmp_path, expected)
        if not valid:
            raise RuntimeError(f"Temporary shard validation failed: {error}")

        os.replace(tmp_path, final_path)
        checksum = file_sha256(final_path)
        update_shard_status(
            db_path,
            shard_id,
            "complete",
            checksum=checksum,
            ntotal_observed=observed,
            completed_at=utc_now_iso(),
            error_message=None,
        )
        print(f"  -> Shard {shard_id:06d} complete: {final_path}")
        return final_path

    except Exception as exc:
        update_shard_status(
            db_path,
            shard_id,
            "failed",
            error_message="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        raise


def build_all_shards(db_path, template_path, embedder, batch_size, resume=True):
    """Build all assigned shards sequentially, skipping valid completed shards."""
    shards = get_shards(db_path)
    if not shards:
        raise RuntimeError("No shards are assigned. Run assign_shards(...) before building shards.")

    print("\n[5/6] Building Independent FAISS Shards...")
    built_or_reused = []
    for shard in shards:
        built_or_reused.append(
            build_shard(
                db_path=db_path,
                shard_id=int(shard["shard_id"]),
                template_path=template_path,
                embedder=embedder,
                batch_size=batch_size,
                resume=resume,
            )
        )
    return built_or_reused


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
    sharded=False,
    target_shard_windows=10_000_000,
    template_path=None,
    shards_dir=None,
    resume=False,
):
    """
    Build a database index.

    sharded=False preserves the current one-index output, but uses the new
    manifest + trained-template internals. sharded=True builds independent shard
    indexes with global FAISS IDs and resumable shard status.
    """
    start_time = time.time()
    template_path = template_path or _default_template_path(index_path)
    shards_dir = shards_dir or _default_shards_dir(index_path)

    manifest_summary = _prepare_manifest(
        fasta_path=fasta_path,
        db_path=db_path,
        window_size=window_size,
        stride=stride,
        quantizer=quantizer,
        pq_m=pq_m,
        include_terminal_window=include_terminal_window,
        index_short_contigs=index_short_contigs,
        min_short_contig_len=min_short_contig_len,
        pad_short_contigs=pad_short_contigs,
        resume=resume,
    )
    print(f"  -> Total Contigs: {manifest_summary['total_contigs']:,}")
    print(f"  -> Indexed Contigs: {manifest_summary['total_indexed_contigs']:,}")
    print(f"  -> Exact Windows (N): {manifest_summary['total_windows']:,}")
    print(f"  -> Compact Manifest Database: {db_path}")

    if manifest_summary["total_windows"] <= 0:
        raise RuntimeError("Manifest has zero indexed windows; nothing to build.")

    print("\n[2/6] Initializing Hardware and Embedder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] PyTorch initialized on: {device.upper()}")
    embedder = SequenceEmbedder(device=device, window_size=window_size)
    embedding_dim = embedder.embedding_dim

    _, nlist, target_t, sampling_fraction = estimate_index_parameters(
        manifest_summary["total_windows"],
        embedding_dim=embedding_dim,
    )
    update_manifest_config(
        db_path,
        embedding_dim=embedding_dim,
        nlist=nlist,
        training_size=target_t,
        training_sampling_fraction=sampling_fraction,
        quantizer=quantizer,
        pq_m=pq_m,
    )
    print(f"  -> FAISS Clusters (nlist): {nlist:,}")
    print(f"  -> Required Training Size (T): {target_t:,}")
    print(f"  -> Sampling Strategy: global FAISS ID % {sampling_fraction} == 0\n")

    use_faiss_gpu = device == "cuda"
    print(f"[Hardware] FAISS initialized on: {'GPU' if use_faiss_gpu else 'CPU'}")

    train_faiss_template(
        db_path=db_path,
        template_path=template_path,
        embedder=embedder,
        embedding_dim=embedding_dim,
        nlist=nlist,
        target_training_size=target_t,
        sampling_fraction=sampling_fraction,
        batch_size=batch_size,
        train_mode=train_mode,
        quantizer=quantizer,
        pq_m=pq_m,
        use_faiss_gpu=use_faiss_gpu,
        resume=resume,
    )

    if sharded:
        if not (resume and get_shards(db_path)):
            print("\n[4/6] Assigning Shards...")
            shard_summary = assign_shards(
                db_path=db_path,
                shards_dir=shards_dir,
                target_shard_windows=target_shard_windows,
                overwrite=True,
            )
            print(f"  -> Shards: {shard_summary['num_shards']:,}")
            print(f"  -> Target shard windows: {target_shard_windows:,}")
            print(f"  -> Shards directory: {shards_dir}")
        else:
            print("\n[4/6] Reusing Existing Shard Assignment...")
            print(f"  -> Shards: {len(get_shards(db_path)):,}")

        build_all_shards(
            db_path=db_path,
            template_path=template_path,
            embedder=embedder,
            batch_size=batch_size,
            resume=resume,
        )
        result_path = shards_dir
    else:
        result_path = build_monolithic_index_from_template(
            db_path=db_path,
            template_path=template_path,
            index_path=index_path,
            embedder=embedder,
            batch_size=batch_size,
        )

    total_time = (time.time() - start_time) / 60
    print("\n[6/6] Build Complete.")
    print(f"[SUCCESS] Finished in {total_time:.2f} minutes.")
    print(f"  -> Compact Manifest Database: {db_path}")
    print(f"  -> Trained Template: {template_path}")
    if sharded:
        print(f"  -> Shard Index Directory: {result_path}\n")
    else:
        print(f"  -> Vector Index: {result_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mode", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--quantizer", type=str, choices=["SQ8", "PQ"], default="PQ")
    parser.add_argument("--pq-m", dest="pq_m", type=int, default=64)
    parser.add_argument("--m", dest="pq_m", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sharded", action="store_true", help="Build independent shard indexes instead of one monolithic index")
    parser.add_argument("--target-shard-windows", type=int, default=10_000_000)
    parser.add_argument("--resume", action="store_true", help="Reuse existing manifest/template and skip valid completed shards")
    parser.add_argument("--build-shard", type=int, default=None, help="Build one assigned shard from an existing manifest/template")
    parser.add_argument("--batch-size", type=int, default=100000)
    args, _ = parser.parse_known_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    FASTA_INPUT = os.path.join(project_root, "fda_argos", "fda_argos_subsampled_100.fa")
    SQLITE_OUTPUT = os.path.join(project_root, "new_output", "fda_argos.db")
    FAISS_OUTPUT = os.path.join(project_root, "new_output", "fda_argos.index")
    TEMPLATE_OUTPUT = _default_template_path(FAISS_OUTPUT)

    os.makedirs(os.path.dirname(SQLITE_OUTPUT), exist_ok=True)

    if args.build_shard is not None:
        # Manual/scheduler-friendly entry point. Assumes manifest, shard assignment,
        # and trained template already exist.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = SequenceEmbedder(device=device, window_size=100)
        build_shard(
            db_path=SQLITE_OUTPUT,
            shard_id=args.build_shard,
            template_path=TEMPLATE_OUTPUT,
            embedder=embedder,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    else:
        build_pipeline(
            fasta_path=FASTA_INPUT,
            db_path=SQLITE_OUTPUT,
            index_path=FAISS_OUTPUT,
            window_size=100,
            stride=50,
            batch_size=args.batch_size,
            train_mode=args.train_mode,
            quantizer=args.quantizer,
            pq_m=args.pq_m,
            sharded=args.sharded,
            target_shard_windows=args.target_shard_windows,
            resume=args.resume,
        )
