#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=4:00:00

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mil

python main.py -m
# python main.py -m settings.batch_size=8,16,32 settings.lr=0.001,0.0001,0.00001 settings.reg=0.001,0.0001,0.00001 seed='range(3)' 