#include<time.h>
#include<stdio.h>

double get_double(double min, double max, unsigned int *seed);
double get_time(struct timespec start, struct timespec stop);
double f(double x1, double x2, double x3, double x4, double x5);
void print_data(double *data, int size, FILE *fp);
