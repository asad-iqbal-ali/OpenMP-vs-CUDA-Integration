#include"class-func.h"
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<omp.h>

int main(int argc, char **argv)
{
 int n, i, seed;
 double total, integral;
 struct timespec start, stop;
 FILE *fp;
 double data[4];
 int nprocs;
 
 if(argc != 3)
 {
  fprintf(stderr, "Usage: %s (sample size) (threads)\n", argv[0]);
  exit(-1);
 }

 n = atoi(argv[1]);
 nprocs = atoi(argv[2]);

 clock_gettime(CLOCK_MONOTONIC, &start);

 total = 0.0;

#pragma omp parallel for reduction (+:total) private(i) shared(n) num_threads(nprocs)
 for(i = 0; i < n; ++i)
 {
  total += f(
	get_double(0.0, 1.0, &seed),
	get_double(0.0, 1.0, &seed),
	get_double(0.0, 1.0, &seed),
	get_double(0.0, 1.0, &seed),
	get_double(0.0, 1.0, &seed)
	);
 }

 integral = total/((double) n);

 clock_gettime(CLOCK_MONOTONIC, &stop);

 data[0] = integral;
 data[1] = (double) nprocs;
 data[2] = get_time(start, stop);
 data[3] = (double) n;

 fp = fopen("mc_omp.dat", "a");
 print_data(data, 4, fp);
 fclose(fp);

 printf("integral = %lf\n", integral);

 exit(0);
}
