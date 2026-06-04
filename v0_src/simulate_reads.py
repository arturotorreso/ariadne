import os
import random
import argparse
import pyfastx

def apply_mutations(seq, snps, dels, ins):
    """
    Applies SNPs, Deletions, and Insertions to a sequence list.
    Applies them in reverse order (right-to-left) to prevent coordinate shifting.
    Returns the mutated sequence and a formatted log of the mutations.
    """
    seq = list(seq)
    bases = ['A', 'C', 'G', 'T']
    
    # 1. Build a list of all requested operations
    operations = []
    for _ in range(snps): operations.append(('SNP', 0))
    for d in dels: operations.append(('DEL', d))
    for i in ins: operations.append(('INS', i))
    
    if not operations:
        return "".join(seq), ""
        
    # 2. Determine safe bounds for picking mutation indices
    max_del = max(dels) if dels else 0
    safe_end = len(seq) - max_del - 1
    
    if safe_end < 5:
        safe_end = len(seq) - 1
        
    # 3. Sample unique positions and sort DESCENDING
    positions = random.sample(range(1, safe_end), len(operations))
    positions.sort(reverse=True)
    
    mut_log = []
    
    # 4. Apply the mutations and record the exact offset
    for pos, op in zip(positions, operations):
        op_type = op[0]
        if op_type == 'SNP':
            current_base = seq[pos]
            new_base = random.choice([b for b in bases if b != current_base.upper() and b != 'N'])
            seq[pos] = new_base
            mut_log.append(f"SNP_{pos}_{current_base}>{new_base}")
            
        elif op_type == 'DEL':
            size = op[1]
            del seq[pos : pos+size]
            mut_log.append(f"DEL_{pos}_{size}bp")
            
        elif op_type == 'INS':
            size = op[1]
            insert_seq = [random.choice(bases) for _ in range(size)]
            seq[pos:pos] = insert_seq 
            mut_log.append(f"INS_{pos}_{size}bp")
            
    # Reverse the log so it reads naturally from left-to-right (5' to 3') in the header
    mut_log.reverse()
    return "".join(seq), ",".join(mut_log)

def main():
    parser = argparse.ArgumentParser(description="Simulate FASTQ reads from a FASTA database.")
    parser.add_argument("-f", "--fasta", required=True, help="Path to reference FASTA")
    parser.add_argument("-o", "--output", required=True, help="Path to output FASTQ")
    parser.add_argument("-n", "--num_reads", type=int, default=10, help="Number of reads to simulate")
    parser.add_argument("-l", "--length", type=int, default=200, help="Length of simulated reads")
    
    # New Mutation Flags
    parser.add_argument("--snps", type=int, default=0, help="Number of SNPs to introduce")
    parser.add_argument("--dels", type=int, nargs="*", default=[], help="List of deletion sizes, e.g., --dels 2 5")
    parser.add_argument("--ins", type=int, nargs="*", default=[], help="List of insertion sizes, e.g., --ins 3 1")
    
    args = parser.parse_args()

    print(f"[Simulator] Indexing and loading FASTA: {args.fasta}...")
    fa = pyfastx.Fasta(args.fasta)
    keys = list(fa.keys())
    
    # Calculate the exact physical extraction length required from the reference
    # to guarantee the final mutated read is exactly args.length
    extract_length = args.length + sum(args.dels) - sum(args.ins)
    
    print(f"[Simulator] Generating {args.num_reads} reads of final length {args.length}bp...")
    print(f"  -> Mutations per read: {args.snps} SNPs, {len(args.dels)} Deletions, {len(args.ins)} Insertions")
    print(f"  -> Pre-mutation extraction length: {extract_length}bp")
    
    if extract_length <= 0:
        print("[Error] Insertions are larger than the requested read length!")
        return

    with open(args.output, "w") as out_fq:
        valid_reads = 0
        while valid_reads < args.num_reads:
            # 1. Pick a random genome/contig
            seq_id = random.choice(keys)
            seq_obj = fa[seq_id]
            seq_len = len(seq_obj)
            
            if seq_len <= extract_length:
                continue
                
            # 2. Pick a random start position
            start_pos = random.randint(0, seq_len - extract_length)
            
            # 3. Extract the sequence
            raw_seq = seq_obj[start_pos : start_pos + extract_length].seq.upper()
            
            if raw_seq.count('N') > (extract_length * 0.1):
                continue
                
            # 4. Apply Mutations & Get Log
            mutated_seq, mut_log_str = apply_mutations(raw_seq, args.snps, args.dels, args.ins)
            
            # 5. Generate fake Phred scores (Q40 = 'I')
            qual = "I" * len(mutated_seq)
            
            # 6. Format Read ID
            mut_field = f"|MUTS:{mut_log_str}" if mut_log_str else ""
            read_name = f"read_{valid_reads}|TRUTH:{seq_id}|POS:{start_pos}{mut_field}"
            
            out_fq.write(f"@{read_name}\n{mutated_seq}\n+\n{qual}\n")
            valid_reads += 1

    print(f"[SUCCESS] Saved simulated reads to {args.output}")

if __name__ == "__main__":
    main()


# python src/simulate_reads.py \
#     -f fda_argos/fda_argos.fa \
#     -o output/simulated_mutations.fastq \
#     -n 5 \
#     -l 200 \
#     --snps 3 \
#     --dels 10 2 \
#     --ins 5