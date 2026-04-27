#!/bin/bash

#SBATCH --account=msalomon_1385
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --time=6:00:00
#SBATCH --job-name=fda_argos_dl
#SBATCH --output=/scratch1/atortiz/temp/download_%j.out
#SBATCH --error=/scratch1/atortiz/temp/download_%j.err

# 1. Get the bashrc information for conda.
source ~/.bash_profile

# 2. Load required modules
module load gcc/13.3.0
module load aria2/1.37.0

# 3. Navigate to your working directory
cd /scratch1/atortiz/ml_mapping/fda_argos/

echo "Starting FDA-ARGOS prep..."

# 4. Fetch the NCBI Datasets CLI locally if it isn't already installed
if ! command -v datasets &> /dev/null; then
    curl -sLO "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets"
    chmod +x datasets
    DATASETS_CMD="./datasets"
else
    DATASETS_CMD="datasets"
fi

# 5. Download the "dehydrated" package (metadata only, very fast)
# BioProject PRJNA231221 is the FDA-ARGOS database.
$DATASETS_CMD download genome accession PRJNA231221 --dehydrated --filename fda_argos_dehydrated.zip

# 6. Unzip the dehydrated package
unzip -q -o fda_argos_dehydrated.zip -d fda_argos_db

# 7. Generate the aria2c input file from the NCBI fetch.txt map.
# This tells aria2c the URL ($1) and the exact sub-folder/filename ($3) to save it as.
awk '{print $1 "\n  dir=fda_argos_db\n  out=" $3}' fda_argos_db/ncbi_dataset/fetch.txt > aria2c_links.txt

echo "List generated. Starting aria2c parallel download..."

# 8. Execute aria2c
# Optimized for a large number of smaller files:
# -j 20 : Matches your --cpus-per-task, downloading 20 genomes simultaneously.
# -x 2 -s 2 : Uses 2 connections per file (keeps NCBI from blocking you).
aria2c -i aria2c_links.txt \
       -j 3 \
       -c \
       -x 1 \
       -s 1 \
       --max-connection-per-server=3 \
       --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
       --file-allocation=none \
       --timeout=120 \
       --max-tries=10 \
       --retry-wait=30

echo "Download complete! Files are located in fda_argos_db/ncbi_dataset/data/"
