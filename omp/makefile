CFLAGS = -Wall -g

all: mc_omp simpson_omp

mc_omp: mc_omp.c class-func.c class-func.h
	gcc -o mc_omp mc_omp.c class-func.c -fopenmp -lrt -lm

simpson_omp: simpson_omp.c class-func.c class-func.h
	gcc -o simpson_omp simpson_omp.c class-func.c -fopenmp -lrt -lm

clean:
	-@rm -f mc_omp simpson_omp omp.out
