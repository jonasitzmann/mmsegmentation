#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --out=log/%j.out
#SBATCH --time=20:00
#SBATCH --mem=10G

source activate mmseg
srun python -u tools/train.py $1
