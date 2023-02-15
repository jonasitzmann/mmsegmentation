#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --out=log/%j.out
#SBATCH --time=200:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100:1

source activate mmseg
srun python -u tools/train.py $1
