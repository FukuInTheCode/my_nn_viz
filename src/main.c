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
    sfVector2f points[num];
    sfVector2f c_pts[num];

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
    my_theme_t th_g2 = {
        .type = graph_th,
        .graph.radius = 5,
        .graph.pt = sfYellow
    };

    my_graph_t g = {
        .points = points,
        .computed_pts = c_pts,
        .data_num = num,
        .type = static_pts,
        .theme = &th_g1
    };

    my_graph_t g2 = {
        .points = points2,
        .computed_pts = c_pts2,
        .data_num = num2,
        .type = static_func,
        .st_func = {
            .func = f,
            .max_pts = num2
        },
        .theme = &th_g2
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
