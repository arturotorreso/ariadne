import pyfastx

from manifest import compute_window_layout


def sliding_window_fasta(
    file_path,
    window_size,
    stride,
    batch_size,
    include_terminal_window=True,
    index_short_contigs=True,
    min_short_contig_len=50,
    pad_short_contigs=True,
):
    """
    Generator that streams a FASTA file and yields batches of sequence windows.

    Window emission order is intentionally matched to manifest.compute_window_layout:
      1. regular stride-grid windows,
      2. one optional terminal full-length window if the stride grid misses the end,
      3. one optional N-padded short-contig window for informative short contigs.

    Metadata stores true reference coordinates. For padded short contigs, end_pos
    is the true contig length, not the padded window length.
    """
    sequences_batch = []
    metadata_batch = []

    def append_window(header, window_seq, start_pos, end_pos):
        sequences_batch.append(window_seq)
        metadata_batch.append((header, start_pos, end_pos))

    def maybe_yield_batch():
        if len(sequences_batch) == batch_size:
            out_seq = list(sequences_batch)
            out_meta = list(metadata_batch)
            sequences_batch.clear()
            metadata_batch.clear()
            return out_seq, out_meta
        return None
    
    # build_index=False is CRITICAL here. It prevents pyfastx from trying 
    # to build a massive SQLite index of RefSeq on your hard drive. 
    # It forces pyfastx to act as a pure, lightweight stream.
    for name, seq in pyfastx.Fasta(file_path, build_index=False):
        seq_len = len(seq)
        layout = compute_window_layout(
            seq_len,
            window_size,
            stride,
            include_terminal_window=include_terminal_window,
            index_short_contigs=index_short_contigs,
            min_short_contig_len=min_short_contig_len,
            pad_short_contigs=pad_short_contigs,
        )

        if layout["n_windows"] == 0:
            continue

        if layout["is_short_contig"]:
            # Short references are padded to the embedding window length, but
            # the metadata keeps the true coordinate span [0, seq_len]. The
            # embedder masks N, so padded positions do not add artificial signal.
            window_seq = seq[0:seq_len].ljust(window_size, 'N')
            append_window(name, window_seq, 0, seq_len)
            batch = maybe_yield_batch()
            if batch is not None:
                yield batch
            continue

        # Regular stride-grid windows. These always have exactly window_size bp.
        for ordinal in range(layout["regular_window_count"]):
            start_pos = ordinal * stride
            end_pos = start_pos + window_size
            append_window(name, seq[start_pos:end_pos], start_pos, end_pos)
            batch = maybe_yield_batch()
            if batch is not None:
                yield batch

        # Add one final full-length terminal window if the stride grid did not
        # already end exactly at the contig boundary. This avoids losing the tail
        # while also avoiding duplicate terminal windows.
        if layout["has_terminal_window"]:
            start_pos = seq_len - window_size
            end_pos = seq_len
            append_window(name, seq[start_pos:end_pos], start_pos, end_pos)
            batch = maybe_yield_batch()
            if batch is not None:
                yield batch

    # Yield any remaining sequences that didn't perfectly fill the final batch.
    if sequences_batch:
        yield sequences_batch, metadata_batch
