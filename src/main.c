#include "../includes/my.h"

// AND - gates
double and_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double and_train_tar[] = {
    0,
    0,
    0,
    1
};

int main(void)
{
    my_params_t hp = {
        .alpha = 1e-2,
        .iterations = 1000,
        .threshold = 1e-3
    };

    sfVector2f error_pts[hp.iterations];
    sfVector2f c_pts[hp.iterations];

    my_theme_t th_plt = {
        .type = plot_th,
        .plot.bg = sfBlack,
        .plot.axis = sfWhite
    };
    my_theme_t th_g1 = {
        .type = graph_th,
        .graph.radius = 10,
        .graph.pt = sfRed
    };

    my_graph_t g = {
        .points = points,
        .computed_pts = c_pts,
        .data_num = num,
        .type = dynamic_pts,
        .theme = &th_g1
    };

    my_graph_t *g_arr[] = {
        &g,
        &g2
    };

    my_plot_t plt = {
        .graph = g_arr,
        .graph_n = 2,
        .theme = &th_plt
    };

    sfVideoMode mode = {500 * SCALE, 500 * SCALE, 32};
    char *title = "Hello World";
    sfEvent evt;

    my_plot_create(&plt, title, &mode, &evt);

    my_plot_show(&plt);

    sfRenderWindow_destroy(plt.window);

    return 0;
}
