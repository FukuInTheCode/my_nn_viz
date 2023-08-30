#include "../../includes/my.h"

static inline __attribute__((always_inline)) void type_handler(my_plot_t *plt,\
                                                                my_graph_t *g)
{
    if (g->type == func)
        my_plot_func(plt, g);
    else if (g->type == dynamic_pts)
        calc_ratio(plt, g, 2);
}

static inline __attribute__((always_inline)) void mouse_handler(my_plot_t *plt)
{
    if (sfMouse_isButtonPressed(sfMouseLeft)) {
        sfVector2i mouse_vec = sfMouse_getPosition(plt->window);
        if (!plt->is_dragged) {
            plt->is_dragged = sfTrue;
            plt->last_shift = mouse_vec;
        } else {
            plt->shift.x += mouse_vec.x - plt->last_shift.x;
            plt->shift.y += mouse_vec.y - plt->last_shift.y;
            plt->last_shift = mouse_vec;
        }
    } else
        plt->is_dragged = sfFalse;
}

void my_nn_viz_error(my_nn_t *N, my_plot_t *plt, my_params_t *params, my_matrix_t *features, my_matrix_t *y)
{
    my_params_t hp = {
        .iterations = 10,
        .alpha = params->alpha,
        .threshold = params->threshold
    };
    while (sfRenderWindow_isOpen(plt->window)) {
        my_plot_handle_event(plt);
        mouse_handler(plt);

        if (plt->graph[0]->data_num < params->iterations / hp.iterations && my_nn_calcerror_mse(N, features, y) > params->threshold) {
            plt->graph[0]->points[plt->graph[0]->data_num].x = plt->graph[0]->data_num * hp.iterations;
            plt->graph[0]->points[plt->graph[0]->data_num].y = my_nn_calcerror_mse(N, features, y);
            printf("%llu | Error : %f\n", plt->graph[0]->data_num, my_nn_calcerror_mse(N, features, y));
            my_nn_train(N, features, y, &hp);
            plt->graph[0]->data_num += 1;
        }

        sfRenderWindow_clear(plt->window, plt->theme->plot.bg);
        my_plot_axis(plt);
        for (uint32_t i = 0; i < plt->graph_n; ++i) {
            type_handler(plt, plt->graph[i]);
            my_plot_points(plt, plt->graph[i]);
        }

        sfRenderWindow_display(plt->window);
    }
}
