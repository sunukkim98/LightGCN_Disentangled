#!/bin/bash
#SBATCH -J amazon-book
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -o /home/sunuk/Research/LightGCN-PyTorch/code/runs/slurm_log/amazon_book.out
hostname
#conda activate LightGCN

python main.py -u --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="amazon-book" --topks="[20]" --recdim=64