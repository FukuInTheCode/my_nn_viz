#pragma once
#define MY_VIZ_H

#include "my_plot.h"
#include "my_neuralnetwork.h"

void my_nn_viz_error(my_nn_t *N, my_plot_t *plt, my_params_t *params, my_matrix_t *features, my_matrix_t *y);
