import pyfastx

def sliding_window_fasta(file_path, window_size, stride, batch_size):
    """
    Generator that streams a FASTA file and yields batches of sequence windows.
    """
    sequences_batch = []
    metadata_batch = []
    
    # build_index=False is CRITICAL here. It prevents pyfastx from trying 
    # to build a massive SQLite index of RefSeq on your hard drive. 
    # It forces pyfastx to act as a pure, lightweight stream.
    for name, seq in pyfastx.Fasta(file_path, build_index=False):
        seq_len = len(seq)
        
        # Sliding window over the current chromosome/contig
        for start_pos in range(0, seq_len, stride):
            end_pos = start_pos + window_size
            
            # Discard any tail that is shorter than the required window_size
            if end_pos > seq_len:
                break
            
            # Extract the string slice and append
            window_seq = seq[start_pos:end_pos]
            sequences_batch.append(window_seq)
            
            # start_pos is the physical base-pair coordinate on the reference
            metadata_batch.append((name, start_pos, end_pos))
            
            # If the batch is full, yield it and clear RAM
            if len(sequences_batch) == batch_size:
                yield sequences_batch, metadata_batch
                # Re-initialize empty lists to trigger Python garbage collection
                sequences_batch = []
                metadata_batch = []

    # Yield any remaining sequences that didn't perfectly fill the final batch            
    if sequences_batch:
        yield sequences_batch, metadata_batch
