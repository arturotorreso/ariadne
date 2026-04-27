#!/bin/bash

#SBATCH --account=msalomon_1385
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=feiss_train
#SBATCH --output=/scratch1/atortiz/temp/feiss_train_%j.out
#SBATCH --error=/scratch1/atortiz/temp/feiss_train_%j.err

echo "Job started at: $(date)"

# Get the bashrc information for conda.
source ~/.bash_profile

conda activate mapping_ml

cd /scratch1/atortiz/ml_mapping

python -u src/build_index.py

echo "Job ended at: $(date)"