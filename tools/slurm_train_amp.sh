#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10G
#SBATCH --out=log/%j.out
#SBATCH --time=200:00:00
#SBATCH --exclude=gpu04

source activate mmseg
srun python -u tools/train.py --amp $1
