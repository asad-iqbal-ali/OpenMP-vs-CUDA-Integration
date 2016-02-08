CFLAGS = -Wall -g

all: mc_cuda simpson_cuda 

mc_cuda: mc.cu class-func.cu class-func.h
	nvcc -o mc_cuda mc.cu class-func.cu -lrt -lm -arch sm_20 

simpson_cuda: simpson.cu class-func.cu class-func.h
	nvcc -o simpson_cuda simpson.cu class-func.cu -lrt -lm -arch sm_20

clean:
	-@rm -f mc_cuda simpson_cuda cuda.out
