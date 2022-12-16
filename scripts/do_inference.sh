#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1080:1
#SBATCH --out=log/%j.out
#SBATCH --time=20:00

source activate mmseg
srun python -u scripts/do_inference.py --config $1
