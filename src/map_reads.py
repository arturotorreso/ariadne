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
    
    # Tuning Parameters
    parser.add_argument("--stride", type=int, default=50, help="Window stride for chunking long reads (default: 50)")
    parser.add_argument("-k", "--top_k", type=int, default=1, help="Number of best matches to return per chunk (default: 1)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Reads to process per batch (default: 10000)")
    parser.add_argument("--cpu-only", action="store_true", help="Force execution on CPU even if GPU is available")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. Initialize Mapper
    use_gpu = False if args.cpu_only else None
    mapper = MetagenomicMapper(
        db_path=args.db, 
        index_path=args.index, 
        use_gpu=use_gpu
    )
    
    # 2. Open Outputs and Stream Fastq
    print(f"\n[Mapper] Beginning sequence mapping...")
    print(f"  -> Input: {args.fastq}")
    print(f"  -> Output: {args.output}")
    print(f"  -> Stride: {args.stride}bp")
    
    fq = pyfastx.Fastq(args.fastq)
    
    total_reads = 0
    total_chunks = 0
    batch = []
    
    with open(args.output, 'w') as out_f:
        # Write TSV Header
        out_f.write("Read_ID\tChunk_Num\tTarget_Header\tPosition\tTarget_ID\tMismatches\tCosine_Sim\n")
        
        for read in fq:
            batch.append({'id': read.name, 'seq': read.seq})
            
            # When batch is full, process and write
            if len(batch) >= args.batch_size:
                results = mapper.map_reads(batch, query_stride=args.stride, k=args.top_k)
                
                for res in results:
                    read_id = res['read_id']
                    # A single read might have multiple chunks, we enumerate them to track them
                    for chunk_num, hit in enumerate(res['hits']):
                        # Safely pulling 'start_pos' from the hit dictionary
                        pos = hit.get('start_pos', 'N/A')
                        out_f.write(f"{read_id}\t{chunk_num}\t{hit['header']}\t{pos}\t{hit['faiss_id']}\t{hit['mismatches']}\t{hit['cosine_sim']:.4f}\n")
                        total_chunks += 1
                        
                total_reads += len(batch)
                print(f"  -> Processed {total_reads:,} reads...", flush=True)
                batch = [] # Clear the batch

        # 3. Process any remaining reads in the final partial batch        
        if batch:
            results = mapper.map_reads(batch, query_stride=args.stride, k=args.top_k)
            for res in results:
                read_id = res['read_id']
                for chunk_num, hit in enumerate(res['hits']):
                    pos = hit.get('start_pos', 'N/A')
                    out_f.write(f"{read_id}\t{chunk_num}\t{hit['header']}\t{pos}\t{hit['faiss_id']}\t{hit['mismatches']}\t{hit['cosine_sim']:.4f}\n")
                    total_chunks += 1
            total_reads += len(batch)
            
    mapper.close()
    
    end_time = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Mapped {total_reads:,} reads ({total_chunks:,} chunks) in {end_time:.2f} minutes.")

if __name__ == "__main__":
    main()

# Must run simulate_reads.py first

# python src/map_reads.py \
#     -i output/simulated_test.fastq \
#     -d output/fda_argos.db \
#     -x output/fda_argos.index \
#     -o output/simulated_results.tsv \
#     -k 3 \
#     --stride 100

# python src/map_reads.py \
#     -i /path/to/your/reads.fastq \
#     -d output/fda_argos.db \
#     -x output/fda_argos.index \
#     -o output/mapping_results.tsv \
#     --stride 50

# column -t -s $'\t' output/simulated_results.tsv