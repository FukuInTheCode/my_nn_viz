#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

#define NN_DECLA(n) my_nn_t n = {.name = #n, .size = 0}

typedef double (*activ_func_temp)(double);
typedef double (*activ_func_temp2)(double *, double);

typedef enum {
    base_type,
    param_type
} activ_f_type_t;

typedef struct {
    uint32_t size;
    uint32_t *dims;
    char *name;
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
    my_matrix_t *activations;
    my_matrix_t *gradients_theta;
    my_matrix_t *gradients_bias;
    activ_f_type_t acti_type;
    union {
        struct
        {
            activ_func_temp af;
            activ_func_temp grad_af;
        };
        struct
        {
            double *params;
            activ_func_temp2 af_p;
            activ_func_temp2 grad_af_p;
        };
    } funcs;
} my_nn_t;

typedef struct {
    double alpha;
    uint32_t epoch;
    double threshold;
    bool show_tqdm;
} my_params_t;

void my_nn_create(my_nn_t *nn);
void my_nn_forward(my_nn_t *nn, my_matrix_t *x);
double my_nn_sigmoid(double x);
double my_nn_relu(double x);
double my_nn_sigmoid_grad(double x);
double my_nn_relu_grad(double x);
void my_nn_backprogation(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y);
uint32_t my_nn_train(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, my_params_t *hp);
void my_nn_predict(my_nn_t *nn, my_matrix_t *x, my_matrix_t *res);
double my_nn_calc_error(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y);
double my_nn_linear(double x);
double my_nn_linear_grad(double x);
void my_nn_free(my_nn_t *nn);
double my_nn_tanh_grad(double x);
double my_nn_tanh(double x);
double my_nn_sin(double x);
double my_nn_sin_grad(double x);
double my_nn_sinc(double x);
double my_nn_sinc_grad(double x);
double my_nn_softplus_grad(double x);
double my_nn_softplus(double x);
double my_nn_idc_grad(double x);
double my_nn_idc(double x);
double my_nn_leaky_grad(double x);
double my_nn_leaky(double x);
double my_nn_silu_grad(double x);
double my_nn_silu(double x);
double my_nn_gelu_grad(double x);
double my_nn_gelu(double x);
double my_nn_gaussian(double x);
double my_nn_gaussian_grad(double x);
double my_nn_binarystep_grad(double x);
double my_nn_binarystep(double x);
double my_nn_elu_grad(double *params, double x);
double my_nn_elu(double *params, double x);
double my_nn_prelu_grad(double *params, double x);
double my_nn_prelu(double *params, double x);
double my_nn_selu(double *params, double x);
double my_nn_selu_grad(double *params, double x);
double my_nn_softexp_grad(double *params, double x);
double my_nn_softexp(double *params, double x);
double my_nn_softsign(double x);
double my_nn_softsign_grad(double x);
double my_nn_atan_grad(double x);
double my_nn_atan(double x);
void my_nn_print(my_nn_t *nn);
void my_nn_copy(my_nn_t *nn, my_nn_t *copy);
void my_nn_to_array(my_nn_t *nn, double **res);
uint32_t my_nn_get_n_params(my_nn_t *nn);
void my_nn_from_array(my_nn_t *nn, double *arr);
