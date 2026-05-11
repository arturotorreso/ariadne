import os
import argparse
import time
import pyfastx
from mapper import MetagenomicMapper

def main():
    parser = argparse.ArgumentParser(description="Map FASTQ reads against a Deep Learning RoPE Index.")
    parser.add_argument("-i", "--fastq", required=True, help="Path to input FASTQ file")
    parser.add_argument("-d", "--db", required=True, help="Path to SQLite metadata (.db)")
    parser.add_argument("-x", "--index", required=True, help="Path to FAISS index (.index)")
    parser.add_argument("-o", "--output", required=True, help="Path to output TSV file")
    
    # Tuning Parameters (Notice default stride is now 1)
    parser.add_argument("--stride", type=int, default=1, help="Window stride for chunking long reads (default: 1)")
    parser.add_argument("-k", "--top_k", type=int, default=3, help="Number of best matches to return per read (default: 3)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Reads to process per batch (default: 10000)")
    parser.add_argument("--cpu-only", action="store_true", help="Force execution on CPU even if GPU is available")
    parser.add_argument("--chain", action="store_true", help="Enable spatial chaining of chunks into read-level alignments")
    parser.add_argument("--mmap", action="store_true", help="Enable memory mapping for the FAISS index (faiss.IO_FLAG_MMAP)")
    parser.add_argument("--nprobe", type=int, default=128, help="Number of Voronoi clusters to search in FAISS (default: 128)")

    args = parser.parse_args()
    
    start_time = time.time()
    
    use_gpu = False if args.cpu_only else None
    mapper = MetagenomicMapper(
        db_path=args.db, 
        index_path=args.index, 
        use_gpu=use_gpu,
        use_mmap=args.mmap,
        nprobe=args.nprobe
    )
    
    print(f"\n[Mapper] Beginning sequence mapping...")
    print(f"  -> Input: {args.fastq}")
    print(f"  -> Output: {args.output}")
    print(f"  -> Stride: {args.stride}bp")
    
    # build_index=False forces pyfastx to stream the text file directly,
    # completely ignoring any stale .fxi caching files.
    fq = pyfastx.Fastq(args.fastq, build_index=False)
    
    total_reads = 0
    total_hits_written = 0
    batch = []
    
    with open(args.output, 'w') as out_f:
        if args.chain:
            out_f.write("Read_ID\tHit_Rank\tTarget_Header\tPosition\tTarget_ID\tMismatches\tCosine_Sim\tMatched_Len\tAlign_Score\n")
        else:
            out_f.write("Read_ID\tHit_Rank\tTarget_Header\tPosition\tTarget_ID\tMismatches\tCosine_Sim\n")
            
        for name, seq, qual in fq:
            batch.append({'id': name, 'seq': seq})
            
            if len(batch) >= args.batch_size:
                results = mapper.map_reads(batch, query_stride=args.stride, k=args.top_k, chain_alignments=args.chain)
                
                for res in results:
                    read_id = res['read_id']
                    for hit_rank, hit in enumerate(res['hits']):
                        pos = hit.get('start_pos', 'N/A')
                        if args.chain:
                            out_f.write(f"{read_id}\t{hit_rank}\t{hit['header']}\t{pos}\t{hit['faiss_id']}\t{hit['mismatches']}\t{hit['cosine_sim']:.4f}\t{hit['matched_length']}\t{hit['alignment_score']:.1f}\n")
                        else:
                            out_f.write(f"{read_id}\t{hit_rank}\t{hit['header']}\t{pos}\t{hit['faiss_id']}\t{hit['mismatches']}\t{hit['cosine_sim']:.4f}\n")
                        total_hits_written += 1
                        
                total_reads += len(batch)
                print(f"  -> Processed {total_reads:,} reads...", flush=True)
                batch = []
                
        if batch:
            results = mapper.map_reads(batch, query_stride=args.stride, k=args.top_k, chain_alignments=args.chain)
            for res in results:
                read_id = res['read_id']
                for hit_rank, hit in enumerate(res['hits']):
                    pos = hit.get('start_pos', 'N/A')
                    if args.chain:
                        out_f.write(f"{read_id}\t{hit_rank}\t{hit['header']}\t{pos}\t{hit['faiss_id']}\t{hit['mismatches']}\t{hit['cosine_sim']:.4f}\t{hit['matched_length']}\t{hit['alignment_score']:.1f}\n")
                    else:
                        out_f.write(f"{read_id}\t{hit_rank}\t{hit['header']}\t{pos}\t{hit['faiss_id']}\t{hit['mismatches']}\t{hit['cosine_sim']:.4f}\n")
                    total_hits_written += 1
            total_reads += len(batch)
            
    mapper.close()
    
    end_time = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Mapped {total_reads:,} reads ({total_hits_written:,} target hits) in {end_time:.2f} minutes.")

if __name__ == "__main__":
    main()

# Must run simulate_reads.py first

# python src/map_reads.py \
#     -i output/simulated_test.fastq \
#     -d output/fda_argos.db \
#     -x output/fda_argos.index \
#     -o output/simulated_results.tsv \
#     -k 3 \
#     --chain

# column -t -s $'\t' output/simulated_results.tsv