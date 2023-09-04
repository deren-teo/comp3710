#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=my_email_address
#SBATCH -o /path_for_out/test_out.txt
#SBATCH -e /path_for_error/test_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate conda-env
python ./part2b.py
