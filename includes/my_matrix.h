#pragma once
#define MYMATRIXH

#include "./my_math.h"

#define FALSE 0
#define TRUE 1

#define MAT_PRINT(A) my_matrix_print(1, &A)
#define MAT_FREE(A) my_matrix_free(1, &A)

#define MAT_PRINT_DIM(A) printf("%s's dim: m = %u, n = %u\n", A.name, A.m, A.n)

#define MAT_DECLA(A) my_matrix_t A = {.n = 0, .m = 0, .name = #A}

static inline __attribute__((always_inline)) void check_alloc(void *A)
{
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed!");
        exit(1);
    }
}

typedef enum {
    my_false = FALSE,
    my_true = TRUE
} my_bool_t;

typedef struct my_matrix {
    char *name;
    uint32_t m;
    uint32_t n;
    double **arr;
} my_matrix_t;

typedef double (*UnaryFunction)(double);

void my_matrix_create(unsigned int m, unsigned int n, \
    const unsigned int count, ...);
void my_matrix_identity(const unsigned int count, ...);
void my_matrix_set(my_matrix_t *A, const unsigned int x, \
    const unsigned int y, const double n);
void my_matrix_print(const unsigned int count, ...);
void my_matrix_free(const unsigned int count, ...);
void my_matrix_multiplybyscalar(my_matrix_t *A, double scalar, \
    my_matrix_t *result);
void my_matrix_add(my_matrix_t *result, const unsigned int count, ...);
void my_matrix_transpose(my_matrix_t *A, my_matrix_t* T);
void my_matrix_getcolumn(my_matrix_t *A, unsigned int n, double result[]);
double* my_matrix_getrow(my_matrix_t *A, unsigned int i);
void my_matrix_product(my_matrix_t *result, const unsigned int count, ...);
void my_matrix_copy(my_matrix_t *A, my_matrix_t *copy);
void my_matrix_powerint(my_matrix_t *A, const unsigned int n, \
    my_matrix_t *result);
void my_matrix_randint(const int minN, const int maxN,\
    const unsigned int count, ...);
void my_matrix_randfloat(const double minN, const double maxN, \
    const unsigned int count, ...);
double my_matrix_det(my_matrix_t *A);
void my_matrix_getsubmatrix(my_matrix_t *A, \
    const unsigned int m, const unsigned int n, my_matrix_t *result);
void my_matrix_adjugate(my_matrix_t *A, my_matrix_t *result);
void my_matrix_inverse(my_matrix_t *A, my_matrix_t *result);
void my_matrix_addscalar(my_matrix_t *A, double scalar, my_matrix_t *result);
void my_matrix_applyfunc(my_matrix_t *A, UnaryFunction func, \
    my_matrix_t *result);
double my_matrix_sum(my_matrix_t *A);
void my_matrix_one(my_matrix_t *A, my_matrix_t *result);
void my_matrix_addcol(my_matrix_t *A, const unsigned int n, \
    my_matrix_t *result);
void my_matrix_addrow(my_matrix_t *A, const unsigned int m, \
    my_matrix_t *result);
void my_matrix_concatcol(my_matrix_t *result, my_matrix_t *A, my_matrix_t *B);
void my_matrix_concatrow(my_matrix_t *result, my_matrix_t *A, my_matrix_t *B);
int my_matrix_equals(my_matrix_t *A, my_matrix_t *B);
void my_matrix_swaprow(my_matrix_t *A, const unsigned int a, \
    const unsigned int b, my_matrix_t *result);
void my_matrix_swapcol(my_matrix_t *A, const unsigned int a,\
    const unsigned int b, my_matrix_t *result);
void my_matrix_sumcol(my_matrix_t *A, my_matrix_t *result);
void my_matrix_sumrow(my_matrix_t *A, my_matrix_t *result);
void my_matrix_setrow(my_matrix_t *A, const unsigned int m, const double x);
void my_matrix_setcol(my_matrix_t *A, const unsigned int n, const double x);
void my_matrix_broadcasting(my_matrix_t *A, const unsigned int m, \
    const unsigned int n, my_matrix_t *result);
double my_matrix_max(my_matrix_t *A);
double my_matrix_min(my_matrix_t *A);
void my_matrix_product_elementwise(my_matrix_t *result,\
    const unsigned int count, ...);
double my_matrix_maxcol(my_matrix_t *A, const unsigned int n);
double my_matrix_maxrow(my_matrix_t *A, const unsigned int m);
double my_matrix_mincol(my_matrix_t *A, const unsigned int n);
double my_matrix_minrow(my_matrix_t *A, const unsigned int m);
void my_matrix_free_array(my_matrix_t **arr, uint8_t size);
void my_matrix_create_array(my_matrix_t **arr, char *common_name, \
                                const uint32_t count, ...);
void my_matrix_print_array(my_matrix_t **arr, uint8_t size);
void my_matrix_fill_from_array(my_matrix_t *A, double *arr, uint32_t arr_size);

static inline __attribute__((always_inline)) char *init_str(char *str, int i)
{
    size_t ite = (size_t)log10(i == 0 ? 1 : i) + 1;
    char *m_str = malloc(strlen(str) + 1 + ite);
    if (m_str == NULL) {
        fprintf(stderr, "Memory Allocation Failed!");
        exit(1);
    }
    strcpy(m_str, str);
    m_str[strlen(str) + ite] = '\0';
    for (size_t j = 1; j <= ite; ++j) {
        m_str[strlen(str) + ite - j] = 48 + i % 10;
        i = (i - i % 10) / 10;
    }
    return m_str;
}
