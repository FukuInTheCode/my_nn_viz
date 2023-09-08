#pragma once
#define MYMATRIXH

#include "./my_math.h"

//  TODO add meshgrid

#define FALSE 0

#define TRUE 1

#define MAT_PRINT(A) my_matrix_print(1, &A)

#define MAT_FREE(A) my_matrix_free(1, &A)

#define MAT_PRINT_DIM(A) printf("%s's dim: m = %u, n = %u\n", A.name, A.m, A.n)

#define MAT_DECLA(A) my_matrix_t A = {.n = 0, .m = 0, .name = #A}

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

typedef double (*temp_func)(double);

void my_matrix_create(uint32_t m, uint32_t n, \
    uint32_t const count, ...);
void my_matrix_identity(uint32_t const count, ...);
void my_matrix_set(my_matrix_t *A, uint32_t const m,\
                        uint32_t const n, double const  x);
void my_matrix_print(uint32_t const count, ...);
void my_matrix_free(uint32_t const count, ...);
void my_matrix_multiplybyscalar(my_matrix_t *A, double scalar, \
    my_matrix_t *result);
void my_matrix_add(my_matrix_t *result, uint32_t const count, ...);
void my_matrix_transpose(my_matrix_t *A, my_matrix_t* T);
void my_matrix_getcolumn(my_matrix_t *A, uint32_t n, double result[]);
double *my_matrix_getrow(my_matrix_t *A, uint32_t i);
void my_matrix_product(my_matrix_t* result, uint32_t const count, ...);
void my_matrix_copy(my_matrix_t *A, my_matrix_t *copy);
void my_matrix_powerint(my_matrix_t *A, uint32_t const n, my_matrix_t *result);
void my_matrix_randint(int const minN, int const maxN, \
    uint32_t const count, ...);
void my_matrix_randfloat(double const minN, double const maxN,\
    uint32_t const count, ...);
void my_matrix_getsubmatrix(my_matrix_t *A, uint32_t const m, \
    uint32_t const n, my_matrix_t *result);
void my_matrix_adjugate(my_matrix_t *A, my_matrix_t *result);
void my_matrix_inverse(my_matrix_t *A, my_matrix_t *result);
void my_matrix_addscalar(my_matrix_t *A, double scalar, my_matrix_t *result);
void my_matrix_applyfunc(my_matrix_t *A, temp_func func, \
    my_matrix_t *result);
double my_matrix_sum(my_matrix_t *A);
void my_matrix_one(my_matrix_t *A, my_matrix_t *result);
void my_matrix_addcol(my_matrix_t *A, uint32_t const n, my_matrix_t *result);
void my_matrix_addrow(my_matrix_t *A, uint32_t const m, my_matrix_t *result);
void my_matrix_concatcol(my_matrix_t *result, my_matrix_t *A, my_matrix_t *B);
void my_matrix_concatrow(my_matrix_t *result, my_matrix_t *A, my_matrix_t *B);
my_bool_t my_matrix_equals(my_matrix_t *A, my_matrix_t *B);
void my_matrix_swaprow(my_matrix_t *A, uint32_t const a, \
    uint32_t const b, my_matrix_t *result);
void my_matrix_swapcol(my_matrix_t *A, uint32_t const a, \
    uint32_t const b, my_matrix_t *result);
void my_matrix_sumcol(my_matrix_t *A, my_matrix_t *result);
void my_matrix_sumrow(my_matrix_t *A, my_matrix_t *result);
void my_matrix_setrow(my_matrix_t *A, uint32_t const m, double const x);
void my_matrix_setcol(my_matrix_t *A, uint32_t const n, double const x);
void my_matrix_broadcasting(my_matrix_t *A, uint32_t m, \
    uint32_t n, my_matrix_t *result);
double my_matrix_max(my_matrix_t *A);
double my_matrix_min(my_matrix_t *A);
void my_matrix_product_elementwise(my_matrix_t *result, \
    uint32_t const count, ...);
double my_matrix_maxcol(my_matrix_t *A, uint32_t const n);
double my_matrix_maxrow(my_matrix_t *A, uint32_t const m);
double my_matrix_mincol(my_matrix_t *A, uint32_t const n);
double my_matrix_minrow(my_matrix_t *A, uint32_t const m);
void my_matrix_free_array(my_matrix_t **arr, uint32_t size);
void my_matrix_create_array(my_matrix_t **arr, char *common_name, \
                                const uint32_t count, ...);
void my_matrix_print_array(my_matrix_t **arr, uint32_t size);
void my_matrix_fill_from_array(my_matrix_t *A, double *arr, uint32_t arr_size);
void my_matrix_addcol_2(my_matrix_t *A, uint32_t const n);
void my_matrix_addrow_2(my_matrix_t *A, uint32_t const n);
void my_matrix_applyfunc_2(my_matrix_t *A, temp_func func);
void my_matrix_broadcasting_2(my_matrix_t *A, uint32_t m, \
    uint32_t n);
void my_matrix_inverse_2(my_matrix_t *A);
void my_matrix_one_2(my_matrix_t *A);
void my_matrix_powerint_2(my_matrix_t *A, uint32_t const n);
void my_matrix_addscalar_2(my_matrix_t *A, double scalar);
void my_matrix_sumcol_2(my_matrix_t *A);
void my_matrix_sumrow_2(my_matrix_t *A);
void my_matrix_swapcol_2(my_matrix_t *A, uint32_t const a, \
    uint32_t const b);
void my_matrix_swaprow_2(my_matrix_t *A, uint32_t const a, \
    uint32_t const b);
void my_matrix_multiplybyscalar_2(my_matrix_t *A, double scalar);
void my_matrix_transpose_2(my_matrix_t *A);
double my_matrix_det(my_matrix_t *A);
void my_matrix_copy_array(my_matrix_t **arr, my_matrix_t **copy,\
                                    uint32_t size, char *common_name);
uint32_t my_matrix_find_col_index(my_matrix_t *A, uint32_t m,  double x);
uint32_t my_matrix_find_row_index(my_matrix_t *A, uint32_t n,  double x);
void my_matrix_linspace(my_matrix_t *m, double start,\
                                double stop, uint32_t num);
void my_matrix_setall(my_matrix_t *A, double const  x);
void my_matrix_transform(my_matrix_t *A, uint32_t m,\
                            uint32_t n, my_matrix_t *result);
void my_matrix_ravel(my_matrix_t *A, my_matrix_t *result);
void my_matrix_transform_2(my_matrix_t *A, uint32_t m, uint32_t n);
void my_matrix_ravel_2(my_matrix_t *A);
void my_matrix_meshgrid(my_matrix_t *x, my_matrix_t *y,\
                            my_matrix_t *xx, my_matrix_t *yy);
void my_matrix_meshgrid_2(my_matrix_t *x, my_matrix_t *y);
void my_matrix_to_array(my_matrix_t *A, double **arr);

#ifdef MATRIX_INIT_STR

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

#endif

#ifdef MATRIX_CHECK_ALLOC
static inline __attribute__((always_inline)) void check_alloc(void *A)
{
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed!");
        exit(1);
    }
}
#endif