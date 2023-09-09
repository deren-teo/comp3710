#!/bin/bash
#SBATCH --job-name=s4528554-resnet-cifar
#SBATCH --partition=vgpu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1

python ./2b_cifar_resnet.py --epochs=16
