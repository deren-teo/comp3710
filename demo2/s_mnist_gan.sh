#!/bin/bash
#SBATCH --job-name=mnist-gan
#SBATCH --partition=vgpu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1

python ./3_mnist_gan.py --epochs=16
