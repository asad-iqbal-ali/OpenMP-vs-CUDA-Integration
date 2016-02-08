#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<assert.h>
#include"class-func.h"
#include<omp.h>

int main(int argc, char **argv)
{
  // inputs
  int i,ii,j,jj,k,kk,l,ll,m,mm,n;
  double a, b;
  double delta, h, integral = 0.0;
  double x1,x2,x3,x4,x5;
  double w[3];
  struct timespec start, stop;
  FILE *fp;
  double data[4];
  int nprocs;

 if(argc != 3)
 {
  fprintf(stderr, "Usage: %s (sample size) (threads)\n", argv[0]);
  exit(1);
 }
  n = atoi(argv[1]); /* "resolution", ie number of points at which to sample f */
  nprocs = atoi(argv[2]);

  a = 0; b = 1; /* upper and lower bounds of the integral */

  delta =(b-a)/n;   /* "grid spacing" -- fixed interval between function sample points */

  h = delta / 2.0;  /* h is used for convenience to find half distance between adjacent samples */
  integral = 0.0;   /* the accumulated integral estimate */

  /* three point weights that define Simpson's rule */
  w[0] = h/3.; w[1] = 4.*w[0]; w[2] = w[0];

  clock_gettime(CLOCK_MONOTONIC, &start);
  omp_set_nested(1);
  #pragma omp parallel for reduction (+:integral) shared(w, delta, h, a, b) private(j, jj, i, ii, k, kk, l, ll, m, mm, x1, x2, x3, x4, x5) num_threads(nprocs)
  for (j = 0; j < n; j++){
    for (jj=0;jj<3;++jj){
      x1 = a + j * delta + jj * h;
      for( i = 0; i < n; i++ ){
	for (ii=0;ii<3;++ii){
	  x2 = a + i * delta + ii * h;
	  for(k = 0; k < n; ++k){
 	    for(kk=0;kk<3;++kk){
	      x3 = a + k * delta + kk * h;
              for(l = 0; l < n; ++l){
	        for(ll=0;ll<3;++ll){
		  x4 = a + l * delta + ll * h;
		  for(m = 0; m < n; m++){
		    for(mm=0;mm<3;++mm){
  	             x5 = a + m * delta + mm * h;
	  	     integral += w[ii]*w[jj]*w[kk]*w[ll]*w[mm]*f(x1,x2,x3,x4,x5);
		    }
		  }
                }
	     }
	   }
         }
	}
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &stop);
  data[0] = integral;
  data[1] = (double) nprocs;
  data[2] = get_time(start, stop);
  data[3] = (double) n;


  fp = fopen("simpson_omp.dat", "a");
  print_data(data, 4, fp);
  fclose(fp);
      
  printf("integral is: %lf\n", integral);

  return 0;
}
