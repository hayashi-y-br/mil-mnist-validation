#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=12:00:00
#$ -N 012-0

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mil

python main.py -m seed=0 lr=0.001,0.0005,0.0001,0.00005,0.00001 reg=0.001,0.0005,0.0001,0.00005,0.00001