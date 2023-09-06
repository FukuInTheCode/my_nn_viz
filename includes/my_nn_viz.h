#ifndef MY_NN_VIZ_H
#define MY_NN_VIZ_H

#include "my_neuralnetwork.h"
#include "my_plot.h"

void my_nn_viz_get_error_graph(my_graph_t *g, my_nn_t *nn, my_matrix_t *x,  my_matrix_t *y, my_params_t *hp);
void my_nn_viz_arch(my_nn_t *nn, sfRenderWindow *window);

#endif