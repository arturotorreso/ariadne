import pyfastx

def sliding_window_fasta(
    file_path,
    window_size,
    stride,
    batch_size,
    include_terminal_window=True,
    index_short_contigs=True,
    min_short_contig_len=50,
    pad_short_contigs=True
):
    """
    Generator that streams a FASTA file and yields batches of sequence windows.

    Windowing policy:
      1. Emit regular stride-spaced full-length windows.
      2. If the stride grid misses the contig end, emit one extra terminal
         full-length window ending exactly at seq_len.
      3. For short contigs, optionally emit one N-padded window when the
         contig is long enough to be informative. The metadata keeps the true
         biological end position rather than the padded length.
    """
    sequences_batch = []
    metadata_batch = []

    def append_window(header, window_seq, start_pos, end_pos):
        """Append one sequence window and its biological coordinates."""
        sequences_batch.append(window_seq)
        metadata_batch.append((header, start_pos, end_pos))

    # build_index=False is CRITICAL here. It prevents pyfastx from trying
    # to build a massive SQLite index of RefSeq on your hard drive.
    # It forces pyfastx to act as a pure, lightweight stream.
    for name, seq in pyfastx.Fasta(file_path, build_index=False):
        seq_len = len(seq)

        # Short-contig policy: retain sufficiently informative short contigs.
        # Padded N bases preserve the fixed embedding length and are masked by
        # the embedder, so they should not contribute artificial A-like signal.
        if seq_len < window_size:
            if index_short_contigs and seq_len >= min_short_contig_len:
                window_seq = seq.ljust(window_size, 'N') if pad_short_contigs else seq
                append_window(name, window_seq, 0, seq_len)  # end_pos is the true contig length.

                if len(sequences_batch) == batch_size:
                    yield sequences_batch, metadata_batch
                    sequences_batch = []
                    metadata_batch = []
            continue

        last_start = None

        # Sliding window over the current chromosome/contig. This bounded range
        # emits full-length windows only; terminal-tail coverage is handled below.
        for start_pos in range(0, seq_len - window_size + 1, stride):
            end_pos = start_pos + window_size
            window_seq = seq[start_pos:end_pos]
            append_window(name, window_seq, start_pos, end_pos)
            last_start = start_pos

            if len(sequences_batch) == batch_size:
                yield sequences_batch, metadata_batch
                sequences_batch = []
                metadata_batch = []

        # Terminal window policy: add one final full-length window ending at
        # seq_len if the stride grid did not already land exactly there.
        if include_terminal_window:
            terminal_start = seq_len - window_size
            if last_start != terminal_start:
                append_window(name, seq[terminal_start:seq_len], terminal_start, seq_len)

                if len(sequences_batch) == batch_size:
                    yield sequences_batch, metadata_batch
                    sequences_batch = []
                    metadata_batch = []

    # Yield any remaining sequences that didn't perfectly fill the final batch.
    if sequences_batch:
        yield sequences_batch, metadata_batch
