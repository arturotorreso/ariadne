from collections import defaultdict

class SpatialChainer:
    def __init__(self, window_size=100, mismatch_penalty=3.0, cluster_tolerance=50):
        """
        Groups fragmented FAISS chunk hits into contiguous biological alignments.
        
        :param window_size: The k-mer window size used in the database.
        :param mismatch_penalty: Weight to subtract from the Matched Length per mismatch.
        :param cluster_tolerance: How many base pairs of projected variance to tolerate. 
                                  Setting this to ~50 allows the chain to absorb database 
                                  boundary artifacts and minor Indels automatically.
        """
        self.window_size = window_size
        self.mismatch_penalty = mismatch_penalty
        self.cluster_tolerance = cluster_tolerance

    def _intervals_overlap(self, int1, int2):
        """Checks if two [start, end] intervals overlap."""
        return max(0, min(int1[1], int2[1]) - max(int1[0], int2[0])) > 0

    def chain(self, read_length, raw_hits):
        """
        Evaluates a list of chunk hits for a single read and returns chained alignments.
        """
        # 1. Project Origins & Group by Chromosome/Contig
        by_header = defaultdict(list)
        for hit in raw_hits:
            projected_start = hit['start_pos'] - hit['query_offset']
            hit['projected_start'] = projected_start
            by_header[hit['header']].append(hit)
        
        chained_alignments = []
        
        # 2. 1D Clustering (Group hits that project to the same genomic origin)
        for header, hits in by_header.items():
            # Sort by projected start to sweep spatially
            hits.sort(key=lambda x: x['projected_start'])
            
            clusters = []
            current_cluster = []
            
            for hit in hits:
                if not current_cluster:
                    current_cluster.append(hit)
                else:
                    # If this hit projects an origin within our tolerance of the cluster's origin, merge it
                    if abs(hit['projected_start'] - current_cluster[0]['projected_start']) <= self.cluster_tolerance:
                        current_cluster.append(hit)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [hit]
            if current_cluster:
                clusters.append(current_cluster)
                
            # 3. Evaluate Each Cluster (The "Best-First" Anchor Phase)
            for cluster in clusters:
                # Matched Length (Based on the physical spread of all chunks in the cluster)
                min_offset = min(x['query_offset'] for x in cluster)
                max_offset = max(x['query_offset'] for x in cluster)
                matched_length = (max_offset + self.window_size) - min_offset
                matched_length = min(matched_length, read_length)
                
                # Sort the cluster by cosine_sim DESCENDING to pick the best anchors first!
                cluster.sort(key=lambda x: x['cosine_sim'], reverse=True)
                
                best_hit = cluster[0]
                consensus_start = best_hit['projected_start']
                
                # Mismatch Proxy Calculation (Strictly Non-Overlapping Best-First)
                total_mismatches = 0
                covered_intervals = []
                
                for hit in cluster:
                    interval = (hit['query_offset'], hit['query_offset'] + self.window_size)
                    
                    # Check if this hit overlaps with any already selected higher-quality anchors
                    overlaps = any(self._intervals_overlap(interval, cov) for cov in covered_intervals)
                    
                    if not overlaps:
                        total_mismatches += hit['mismatches']
                        covered_intervals.append(interval)
                        
                # 4. Final Scoring: Reward length, penalize errors
                alignment_score = matched_length - (self.mismatch_penalty * total_mismatches)
                
                # Only keep biologically plausible alignments (positive scores)
                if alignment_score > 0:
                    chained_alignments.append({
                        'header': header,
                        'start_pos': consensus_start,
                        'faiss_id': best_hit['faiss_id'], # The ID of the best anchor
                        'cosine_sim': best_hit['cosine_sim'], # Peak similarity
                        'mismatches': total_mismatches,
                        'matched_length': matched_length,
                        'alignment_score': alignment_score
                    })
        
        # 5. Sort all chained alignments globally by their Alignment Score
        chained_alignments.sort(key=lambda x: x['alignment_score'], reverse=True)
        return chained_alignments