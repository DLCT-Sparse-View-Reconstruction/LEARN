#!/bin/bash

#SBATCH --account=dyq@v100
#SBATCH --job-name=LEARN
#SBATCH --output=LEARN_Train%j.out 
#SBATCH --error=LEARN_Train%j.err

#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=01:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread

#SBATCH --cpus-per-task=3

module purge
module load pytorch-gpu/py3/1.12.1

set -x

export HYDRA_FULL_ERROR=1
srun python -u learn_train.py
