#ifndef MY_NN_VIZ_H
#define MY_NN_VIZ_H

#include "my_neuralnetwork.h"
#include "my_plot.h"

#define PAD_X 50.
#define PAD_Y 50.
#define COLOR1_W {0, 255, 255, 255}
#define COLOR2_W {255, 0, 0, 255}
#define COLOR1_B {0, 255, 0, 255}
#define COLOR2_B {255, 0, 255, 255}
#define COLOR_0 sfRed

void my_nn_viz_get_error_graph(my_graph_t *g, my_nn_t *nn, my_matrix_t *x,  my_matrix_t *y, my_params_t *hp);
void my_nn_viz_arch(my_nn_t *nn, sfRenderWindow *window);
void my_nn_viz_repr(my_nn_t *nn, sfVideoMode mode);

#endif