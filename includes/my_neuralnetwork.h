#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef double (*double_func_double)(double);

typedef struct {
    uint32_t size;
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
    my_matrix_t *activations;
    my_matrix_t *gradients_theta;
    my_matrix_t *gradients_bias;
    struct {
        double_func_double af;
        double_func_double grad_af;
    } funcs;
} my_nn_t;

typedef struct {
    double alpha;
    uint32_t epoch;
    double threshold;
} my_params_t;

void my_nn_create(my_nn_t *nn, uint32_t *dimensions);
void my_nn_forward(my_nn_t *nn, my_matrix_t *x);
double my_nn_sigmoid(double x);
double my_nn_relu(double x);
double my_nn_sig_grad(double x);
double my_nn_relu_grad(double x);
void my_nn_backprogation(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y);
void my_nn_train(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, my_params_t *hp);
void my_nn_predict(my_nn_t *nn, my_matrix_t *x, my_matrix_t *res);
double my_nn_calc_error(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y);
double my_nn_linear(double x);
double my_nn_linear_grad(double x);

