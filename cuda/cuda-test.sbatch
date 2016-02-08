#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=cuda_integ
#SBATCH --output=cuda.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=m2090

module load cuda
./simpson_cuda 4
./simpson_cuda 8
./simpson_cuda 16
./simpson_cuda 20
./simpson_cuda 24

./mc_cuda 192000
./mc_cuda 1920000
./mc_cuda 19200000
./mc_cuda 192000000
##./mc_cuda 1920000000
