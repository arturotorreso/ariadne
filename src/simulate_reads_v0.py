import os
import random
import argparse
import pyfastx

def main():
    parser = argparse.ArgumentParser(description="Simulate FASTQ reads from a FASTA database.")
    parser.add_argument("-f", "--fasta", required=True, help="Path to reference FASTA")
    parser.add_argument("-o", "--output", required=True, help="Path to output FASTQ")
    parser.add_argument("-n", "--num_reads", type=int, default=10, help="Number of reads to simulate")
    parser.add_argument("-l", "--length", type=int, default=200, help="Length of simulated reads")
    args = parser.parse_args()

    print(f"[Simulator] Indexing and loading FASTA: {args.fasta}...")
    # pyfastx automatically builds a fast SQLite index (.fxi) for massive files
    fa = pyfastx.Fasta(args.fasta)
    keys = list(fa.keys())
    
    print(f"[Simulator] Generating {args.num_reads} reads of length {args.length}bp...")
    
    with open(args.output, "w") as out_fq:
        valid_reads = 0
        while valid_reads < args.num_reads:
            # 1. Pick a random genome/contig
            seq_id = random.choice(keys)
            seq_obj = fa[seq_id]
            seq_len = len(seq_obj)
            
            if seq_len <= args.length:
                continue
                
            # 2. Pick a random start position
            start_pos = random.randint(0, seq_len - args.length)
            
            # 3. Extract the sequence
            read_seq = seq_obj[start_pos : start_pos + args.length].seq.upper()
            
            # Skip sequences with lots of Ns
            if read_seq.count('N') > (args.length * 0.1):
                continue
                
            # 4. Generate fake Phred scores (Q40 = 'I')
            qual = "I" * args.length
            
            # 5. Format Read ID to contain the GROUND TRUTH
            # Example: read_0|TRUTH:NC_002516.2|POS:1050
            read_name = f"read_{valid_reads}|TRUTH:{seq_id}|POS:{start_pos}"
            
            out_fq.write(f"@{read_name}\n{read_seq}\n+\n{qual}\n")
            valid_reads += 1

    print(f"[SUCCESS] Saved simulated reads to {args.output}")

if __name__ == "__main__":
    main()

# python src/simulate_reads.py \
#     -f fda_argos/fda_argos.fa \
#     -o output/simulated_test.fastq \
#     -n 5 \
#     -l 200