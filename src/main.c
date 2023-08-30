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

static double fun(double x)
{
    return x * 2;
}

int main(void)
{
    srand(time(0));
    my_matrix_t features = {.m = 0, .n = 0, .name = "features"};
    my_matrix_t targets = {.m = 0, .n = 0, .name = "targets"};
    my_matrix_create(100, 1, 1, &features);
    my_matrix_randint(-100, 100, 1, &features);
    // my_matrix_randfloat(1, 100, 1, &features);
    // my_matrix_fill_from_array(&features, and_train_fea, 8);
    // my_matrix_create(4, 1, 1, &targets);
    // my_matrix_fill_from_array(&targets, and_train_tar, 4);
    my_matrix_applyfunc(&features, fun, &targets);
    my_matrix_print(2, &features, &targets);

    uint32_t layers[] = {features.n, 32, targets.n};
    my_nn_t nn = {.layers = layers, .layers_size = sizeof(layers) / sizeof(uint32_t), .name = "Viz-Neuro", .apply_all = my_false};
    my_nn_create(&nn);
    my_matrix_t predictions = {.m = 0, .n = 0, .name = "predictions"};

    my_nn_print(&nn);

    my_nn_predict(&nn, &features, &predictions);

    my_matrix_print(1, &predictions);

    my_params_t hp = {
        .alpha = 1e-2,
        .iterations = 10*1000,
        .threshold = 1e-5
    };

    sfVector2f error_pts[hp.iterations];
    sfVector2f c_pts[hp.iterations];

    my_theme_t th_g1 = {
        .type = graph_th,
        .graph.radius = 10,
        .graph.pt = sfRed
    };

    my_graph_t g = {
        .points = error_pts,
        .computed_pts = c_pts,
        .data_num = 0,
        .max_pts = hp.iterations,
        .type = dynamic_pts,
        .theme = &th_g1
    };

    my_graph_t *g_arr[] = {
        &g
    };
    my_theme_t th_plt = {
        .type = plot_th,
        .plot.bg = sfBlack,
        .plot.axis = sfWhite
    };

    my_plot_t plt = {
        .graph = g_arr,
        .graph_n = 1,
        .theme = &th_plt
    };

    sfVideoMode mode = {500 * SCALE, 500 * SCALE, 32};
    char *title = "Hello World";
    sfEvent evt;

    my_plot_create(&plt, title, &mode, &evt);

    my_nn_viz_error(&nn, &plt, &hp, &features, &targets);
    printf("\n\n\n\nfinish\n\n\n\n");
    my_nn_print(&nn);
    my_nn_predict(&nn, &features, &predictions);
    my_matrix_print(1, &predictions);

    sfRenderWindow_destroy(plt.window);

    my_matrix_free(3, &features, &targets, &predictions);
    my_nn_free(&nn);
    return 0;
}
