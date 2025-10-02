#!/bin/bash
#SBATCH --job-name=cace_water        
#SBATCH --account=st-ortner-1-gpu    
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --nodes=1                       # 1 node
#SBATCH --cpus-per-task=12             # Number of CPU cores per task
#SBATCH --mem=32G                       # Total memory per node
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --output=/scratch/st-ortner-1/torabit/cace/cace-water/paper/t1/cace_output.out
#SBATCH --error=/scratch/st-ortner-1/torabit/cace/cace-water/paper/t1/cace_error.err

# Load necessary modules
module load python
module load cuda/11.3

# Activate your environment
source /scratch/st-ortner-1/torabit/cace/cace-env/bin/activate

# Run your Python script
python /scratch/st-ortner-1/torabit/cace/cace-water/paper/t1/water.py

