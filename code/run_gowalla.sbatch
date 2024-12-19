#!/bin/bash
#SBATCH -J gowalla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -o /home/sunuk/Research/LightGCN-PyTorch/code/runs/slurm_log/gowalla_early_stop.out
hostname
#conda activate LightGCN

python -u main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20,40]" --recdim=64 --eval_metric="recall"
python -u main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20,40]" --recdim=64 --eval_metric="ndcg"