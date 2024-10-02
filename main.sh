#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=12:00:00

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mil

python main.py -m model=additive,attention dataset=config-012,config-169,config-348,config-750,config-926