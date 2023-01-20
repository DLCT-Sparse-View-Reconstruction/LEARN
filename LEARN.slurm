#!/bin/bash

#SBATCH --account=dyq@v100
#SBATCH --job-name=LEARN
#SBATCH --output=LEARN_Test%j.out 
#SBATCH --error=LEARN_Test%j.err

#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

#SBATCH --cpus-per-task=1

module purge
module load pytorch-gpu/py3/1.12.1

set -x

export HYDRA_FULL_ERROR=1
srun python -u learn_train.py
