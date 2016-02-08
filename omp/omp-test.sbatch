#!/bin/bash

#SBATCH --job-name=omp_integ
#SBATCH --output=omp.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=ib

#./simpson_omp 4 16
#./simpson_omp 8 16
#./simpson_omp 16 16
#./simpson_omp 20 16
#./simpson_omp 24 16

./mc_omp 192000 16
./mc_omp 1920000 16
./mc_omp 19200000 16
./mc_omp 192000000 16
./mc_omp 1920000000 16

