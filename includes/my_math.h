#pragma once
#define MYMATHH
#include <math.h>

#define ll long long

static inline __attribute__((always_inline)) void swapint(int *a, int *b)
{
    const int temp = *a;
    *a = *b;
    *b = temp;
}

static inline __attribute__((always_inline)) void swap(double *a, double *b)
{
    const int temp = *a;
    *a = *b;
    *b = temp;
}

typedef double (*math_function)(double);
typedef enum {
    inf,
    minus_inf,
    integer
} math_func_limit;

typedef struct {
    math_function f;
    math_func_limit left_type;
    double left;
    math_func_limit right_type;
    double right;
    size_t n_undef;
    double *undefined;
} my_func_t;

double my_power(double x, unsigned int n);
int my_randint(int minN, int maxN);
double my_randfloat(double minN, double maxN);
unsigned ll my_factorial(const unsigned int n);
double my_dot_product(double* a, double* b, unsigned int size);
double my_exp(double x, int n);
double my_ln(double x, int n);
double my_abs(double x);
int my_gcd(int a, int b);
double my_func_max(math_function func, double start, double end);
double my_func_min(math_function func, double start, double end);
double my_max(double *x, size_t size);
double my_max_between(double x1, double x2);
double my_min_between(double x1, double x2);
double my_min(double *x, size_t size);
double my_max_i(int *x, size_t size);
