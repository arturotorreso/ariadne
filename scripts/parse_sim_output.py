import sys
import argparse
import pandas as pd

def parse_fasta(fasta_path):
    """Creates a dictionary mapping IDs to full headers."""
    fasta_dict = {}
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                # Header is everything after '>'
                full_header = line.strip()[1:]
                # ID is the first whitespace-separated string
                header_id = full_header.split(' ', 1)[0]
                fasta_dict[header_id] = full_header
    return fasta_dict

def get_full_header(id_val, fasta_dict):
    """Looks up header, handling optional 'NZ_' prefix differences."""
    # Defensive check to ensure we are only operating on strings
    if not isinstance(id_val, str) or id_val == "N/A":
        return "N/A"
        
    if id_val in fasta_dict:
        return fasta_dict[id_val]
        
    # Check if adding or removing 'NZ_' finds a match
    if id_val.startswith("NZ_"):
        stripped = id_val[3:]
        if stripped in fasta_dict:
            return fasta_dict[stripped]
    else:
        with_nz = "NZ_" + id_val
        if with_nz in fasta_dict:
            return fasta_dict[with_nz]
    return "N/A"

def process_file(file_path, fasta_path=None):
    # keep_default_na=False prevents pandas from turning "N/A" into a NaN float
    df = pd.read_csv(file_path, sep='\t', keep_default_na=False)

    def parse_read_id(read_id):
        parts = read_id.split('|')
        return parts[1].replace('TRUTH:', ''), parts[2].replace('POS:', '')

    extracted = df['Read_ID'].apply(lambda x: pd.Series(parse_read_id(x)))
    df['Read_ID TRUTH'] = extracted[0]
    df['Read_ID POS'] = extracted[1]

    # Comparison logic
    norm_truth = df['Read_ID TRUTH'].apply(lambda x: x[3:] if str(x).startswith('NZ_') else x)
    norm_target = df['Target_Header'].apply(lambda x: x[3:] if str(x).startswith('NZ_') else x)
    df['Match'] = (norm_truth == norm_target)

    # If FASTA is provided, add full headers
    if fasta_path:
        fasta_map = parse_fasta(fasta_path)
        df['Truth_Full_Header'] = df['Read_ID TRUTH'].apply(lambda x: get_full_header(x, fasta_map))
        df['Target_Full_Header'] = df['Target_Header'].apply(lambda x: get_full_header(x, fasta_map))

    # Reorder columns
    cols = ['Read_ID', 'Read_ID TRUTH', 'Read_ID POS', 'Target_Header', 'Position', 'Match']
    if fasta_path:
        cols += ['Truth_Full_Header', 'Target_Full_Header']
    
    output = df[cols]
    output.columns = [c.replace('Target_Header', 'Target_ID') for c in output.columns]
    
    print(output.to_csv(sep='\t', index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse read TSV and optionally map to FASTA headers.")
    parser.add_argument("file", help="Path to the TSV file")
    parser.add_argument("--fasta", help="Optional path to reference genome FASTA")
    args = parser.parse_args()
    
    process_file(args.file, args.fasta)