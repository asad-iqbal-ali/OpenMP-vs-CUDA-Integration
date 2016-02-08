#include"class-func.h"
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#define TPB 192 

//taken from NVidia's CUDA C Programming Guide
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

__global__ void dev_f(double *result, int a, int N, double delta)
{
  //temp is used to store values of f
 __shared__ double temp[TPB];

 double w[3];
 w[0] = delta/6.0;
 w[1] = 4.0*w[0];
 w[2] = w[0];

//The actual index of the thread is offset by a
 int index = (threadIdx.x + blockIdx.x * blockDim.x)+a;
//Confusingly, "N" here refers to "n" from main(). That's how I was raised
//and I'm not changing now.
//Values for i, ii, j, jj, etc. as from the given simpson's rule code
//are calculated for the (3n)^5 threads as mods of the thread's index
//since there are 3n possible values for each of x1,x2,etc.
 int i = index % (3*N);
 int ii =  i % 3;
 i /= 3;

 int j = (index % (9*N*N))/(3*N);
 int jj = j % 3;
 j /= 3;

 int k = (index % (27*N*N*N))/(9*N*N);
 int kk = k % 3;
 k /= 3;
 
 int l = (index % (81*N*N*N*N))/(27*N*N*N);
 int ll = l % 3;
 l /= 3;
 
 int m = (index % (243*N*N*N*N*N))/(81*N*N*N*N);
 int mm = m % 3;
 m /= 3;

 double x5 = (i * delta) + (ii * (delta/2.0));
 double x4 = (j * delta) + (jj * (delta/2.0));
 double x3 = (k * delta) + (kk * (delta/2.0));
 double x2 = (l * delta) + (ll * (delta/2.0));
 double x1 = (m * delta) + (mm * (delta/2.0));

 temp[threadIdx.x] = w[ii]*w[jj]*w[kk]*w[ll]*w[mm]*
		exp(
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
  for(int count = 0; count < TPB; ++count)
   sum += temp[count];
  atomicDoubleAdd(result, sum); 
 } 

}
 
int main(int argc, char **argv)
{
 int i, n, blocks, BPR, requests; 
 double delta;
 unsigned long N;
 double *total; 
 double *dev_total;

 struct timespec start, stop;
 FILE *fp;
 double data[4];
 
 if(argc != 2)
 {
  fprintf(stderr, "Usage: %s (sample size)\n", argv[0]);
  exit(-1);
 }

 n = atoi(argv[1]);

//The actual number of calculations (threads) to be done
 N = 243*n*n*n*n*n;

 blocks = N/TPB;

//"Blocks Per Request" has to be lower than just N/TPB if more than the
//maximum number of blocks per function call are necessary. In this case,
//figure out how many times we need to request BPR blocks to be calculated
 requests = 1;
 BPR = blocks;
 while(BPR > 65535)
 {
  BPR /= n;
  requests *= n;
 }


 delta = 1.0/((double) n);
 total = (double *)malloc(sizeof(double));
 

 clock_gettime(CLOCK_MONOTONIC, &start);

 
 cudaMalloc((void **)&dev_total, sizeof(double));
 cudaMemset(dev_total, 0, sizeof(double));

//Run the device code as many times as necessary to compute all of the requested
//threads. The second argument is an offset, indicating how many threads have already
//been done so that this invocation knkows where to start.
 for(i = 0; i < requests; ++i)
 {
  dev_f<<<BPR, TPB>>>(dev_total, i*BPR*TPB, n, delta);
 }

 //copy back the result
 cudaMemcpy(total, dev_total, sizeof(double), cudaMemcpyDeviceToHost);

 
 //housekeeping
 cudaFree(dev_total);

 clock_gettime(CLOCK_MONOTONIC, &stop);

 //store results as an array and use a function to print,
 //for simplicity/portability
 data[0] = *total;
 data[1] = (double) TPB;
 data[2] = get_time(start, stop);
 data[3] = (double) n;

 fp = fopen("simpson_cuda.dat", "a");
 print_data(data, 4, fp);
 fclose(fp);

 printf("integral = %.20lf\n", *total);

 free(total);
 exit(0);
}
