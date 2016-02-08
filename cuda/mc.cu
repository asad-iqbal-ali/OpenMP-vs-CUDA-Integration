#include"class-func.h"
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include<ctime>
#define TPB 192 

//Initialize the random number generation state for the
//device. Uses "clock()" for the seed to help get new,
//random values on each run
__global__ void rand_kernel_init(curandState_t *state)
{
 curand_init((unsigned long long)clock(),0 , 0, state);

}

//taken from Nvidia's CUDA C Programming Guide
__device__ double atomicDoubleAdd(double *address, double val)
{
 unsigned long long int *address_as_ull = (unsigned long long int*) address;
 unsigned long long int old = *address_as_ull, assumed;
 do{
  assumed = old;
  old = atomicCAS(address_as_ull, assumed,
		__double_as_longlong(val +
		__longlong_as_double(assumed)));
 } while(assumed != old);

 return __longlong_as_double(old);
}


//Device operation
__global__ void dev_f(double *result, curandState_t *state)
{
 //temp is used to store values of f
 __shared__ double temp[TPB];

//jump ahead on the state by a unique value to help ensure a random
//sex of values to run the function
 skipahead((unsigned long long)(threadIdx.x + blockIdx.x * blockDim.x), state); 

 double x1 = curand_uniform_double(state);
 double x2 = curand_uniform_double(state);
 double x3 = curand_uniform_double(state);
 double x4 = curand_uniform_double(state);
 double x5 = curand_uniform_double(state);

 temp[threadIdx.x] = exp(
		-(
		(x1*x1)+
		(x2*x2)+
		(x3*x3)+
		(x4*x4)+
		(x5*x5)
		));

 __syncthreads();

 //synchronize and atomically sum
 if(0 == threadIdx.x)
 {
  double sum = 0.0;
  for(int i = 0; i < TPB; ++i)
   sum += temp[i];
  atomicDoubleAdd(result, sum); 
 } 

}
 
int main(int argc, char **argv)
{
 int N, blocks, requests, BPR, i; 

 double *total; 
 double *dev_total;
 curandState_t *dev_state;

 double integral;
 
 struct timespec start, stop;
 FILE *fp;
 double data[4];
 
 if(argc != 2)
 {
  fprintf(stderr, "Usage: %s (sample size)\n", argv[0]);
  exit(-1);
 }

 N = atoi(argv[1]);
 blocks = N/TPB;

//"Blocks Per Request" has to be lower than just N/TPB if more than the
//maximum number of blocks per function call are necessary. In this case,
//figure out how many times we need to request BPR blocks to be calculated
 requests = 1;
 BPR = blocks;
 while(BPR > 65535)
 {
  BPR /= 10;
  requests *= 10;
 }


 clock_gettime(CLOCK_MONOTONIC, &start);
 
 total = (double *)malloc(sizeof(double));
 cudaMalloc((void **)&dev_total, sizeof(double));
 cudaMemset(dev_total, 0.0, sizeof(double));
 cudaMalloc((void **)&dev_state, sizeof(curandState_t));

//initialize the random number generator state
 rand_kernel_init<<<1,1>>>(dev_state);
 
 for(i = 0; i < requests; ++i)
  dev_f<<<BPR, TPB>>>(dev_total, dev_state);

 //copy back the result
 cudaMemcpy(total, dev_total, sizeof(double), cudaMemcpyDeviceToHost);

 
 integral = (*total)/((double)N); 
 
 free(total);
 cudaFree(dev_total);

 clock_gettime(CLOCK_MONOTONIC, &stop);

 //store results as an array and use a function to print,
 //for simplicity/portability
 data[0] = integral;
 data[1] = (double) TPB;
 data[2] = get_time(start, stop);
 data[3] = (double) N;

 fp = fopen("mc_cuda.dat", "a");
 print_data(data, 4, fp);
 fclose(fp);

 printf("integral = %.20lf\n", integral);

 exit(0);
}
