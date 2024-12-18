#!/bin/bash
#SBATCH -J hello_slurm
#SBATCH --gres=gpu:1
#SBATCH -o /home/sunuk/Research/LightGCN-PyTorch/code/runs/slurm_log/amazon_book.out
hostname
#conda activate LightGCN

python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="amazon-book" --topks="[20]" --recdim=64