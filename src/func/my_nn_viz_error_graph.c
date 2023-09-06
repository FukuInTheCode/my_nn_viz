#include "../../includes/my.h"

void my_nn_viz_get_error_graph(my_graph_t *g, my_nn_t *nn, my_matrix_t *x,  my_matrix_t *y, my_params_t *hp)
{
    my_params_t hparams = {
        .alpha = hp->alpha,
        .epoch = 1,
        .threshold = hp->threshold
    };
    g->pts_n = hp->epoch;
    for (uint32_t i = 0; i < g->pts_n; ++i) {
        my_nn_train(nn, x, y, &hparams);
        g->xs[i] = i;
        g->ys[i] = my_nn_calc_error(nn, x, y);
        if (g->ys[i] <= hp->threshold) {
            n = i + 1;
            return;
        }
    }
}
