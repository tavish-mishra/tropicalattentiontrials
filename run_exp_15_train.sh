#!/bin/bash
#SBATCH -A m4790
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -J exp_15
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Load necessary modules
module load cudatoolkit
module load pytorch

# Activate your conda environment
source ~/.bashrc
conda activate tropical_attention

# Optional: print some GPU info
echo "CUDA available? $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# Run the experiment
python experiment.py --job_file jobs_to_do_train --job_id 22 --tag 15_exp