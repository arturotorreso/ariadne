#!/bin/bash

#SBATCH --account=msalomon_1385
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=feiss_train
#SBATCH --output=/scratch1/atortiz/temp/feiss_train_%j.out
#SBATCH --error=/scratch1/atortiz/temp/feiss_train_%j.err

echo "Job started at: $(date)"

# Get the bashrc information for conda.
source ~/.bash_profile

conda activate mapping_ml

cd /scratch1/atortiz/ml_mapping

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u src/build_index.py --quantizer PQ --m 256

echo "Job ended at: $(date)"
