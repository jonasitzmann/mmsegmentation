#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --out=log/%j.out
#SBATCH --time=200:00:00

source activate mmseg
srun python -u tools/train.py $1
