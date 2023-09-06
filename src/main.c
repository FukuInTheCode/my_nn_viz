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
// OR - gates
double or_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double or_train_tar[] = {
    0,
    1,
    1,
    1
};
// XOR - gates
double xor_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double xor_train_tar[] = {
    0,
    1,
    1,
    0
};
// UNKNW - gates
double unknw_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double unknw_train_tar[] = {
    1,
    0,
    1,
    0
};

double fun(double x)
{
    return x * x;
}

int main(void)
{
    srand(time(0));

    MAT_DECLA(features_tr);
    MAT_DECLA(features);
    MAT_DECLA(targets_tr);
    MAT_DECLA(targets);

    // my_matrix_create(4, 2, 1, &features_tr);
    // my_matrix_fill_from_array(&features_tr, xor_train_fea, 8);
    // my_matrix_create(4, 1, 1, &targets_tr);
    // my_matrix_fill_from_array(&targets_tr, xor_train_tar, 4);

    my_matrix_create(25, 1, 1, &features_tr);
    // my_matrix_create(25, 1, 1, &targets_tr);
    my_matrix_randint(10, -10, 1, &features_tr);
    my_matrix_applyfunc(&features_tr, fun, &targets_tr);

    MAT_PRINT(features_tr);
    MAT_PRINT(targets_tr);

    double tmp_min = my_matrix_min(&features_tr);
    my_matrix_addscalar_2(&features_tr, -1. * tmp_min);
    double tmp_max = my_matrix_max(&features_tr);
    if (tmp_max != 0)
        my_matrix_multiplybyscalar_2(&features_tr, 1. / tmp_max);

    double tmp_min_tar = my_matrix_min(&targets_tr);
    my_matrix_addscalar_2(&targets_tr, -1. * tmp_min_tar);
    double tmp_max_tar = my_matrix_max(&targets_tr);
    if (tmp_max_tar != 0)
        my_matrix_multiplybyscalar_2(&targets_tr, 1. / tmp_max_tar);

    my_matrix_transpose(&features_tr, &features);
    my_matrix_transpose(&targets_tr, &targets);

    // MAT_PRINT(features);
    // MAT_PRINT(targets);

    my_nn_t neuro = {.name = "neuro"};

    neuro.size = 3;
    uint32_t dims[] = {features.m, 3, 3, targets.m};

    neuro.dims = dims;

    neuro.acti_type = base_type;
    neuro.funcs.af = my_nn_sin;
    neuro.funcs.grad_af = my_nn_sin_grad;

    my_nn_create(&neuro);

    my_params_t hparams = {
        .alpha = 1e-1,
        .epoch = 10*1000,
        .threshold = 1e-4
    };
    double xs[hparams.epoch];
    double ys[hparams.epoch];

    my_theme_t g_th = {
        .point = sfRed,
        .radius = 10
    };

    my_graph_t g = {
        .type = points,
        .xs = xs,
        .ys = ys,
        .th = &g_th
    };

    my_nn_viz_get_error_graph(&g, &neuro, &features, &targets, &hparams);

    neuro.acti_type = base_type;
    neuro.funcs.af = my_nn_gelu;
    neuro.funcs.grad_af = my_nn_gelu_grad;

    my_nn_create(&neuro);
    double xs2[hparams.epoch];
    double ys2[hparams.epoch];

    my_theme_t g2_th = {
        .point = sfBlue,
        .radius = 10
    };

    my_graph_t g2 = {
        .type = points,
        .xs = xs2,
        .ys = ys2,
        .th = &g2_th
    };

    my_nn_viz_get_error_graph(&g2, &neuro, &features, &targets, &hparams);
    neuro.acti_type = base_type;
    neuro.funcs.af = my_nn_sigmoid;
    neuro.funcs.grad_af = my_nn_sigmoid_grad;

    my_nn_create(&neuro);
    double xs3[hparams.epoch];
    double ys3[hparams.epoch];

    my_theme_t g3_th = {
        .point = sfGreen,
        .radius = 10
    };

    my_graph_t g3 = {
        .type = points,
        .xs = xs3,
        .ys = ys3,
        .th = &g3_th
    };

    my_nn_viz_get_error_graph(&g3, &neuro, &features, &targets, &hparams);
    neuro.acti_type = base_type;
    neuro.funcs.af = my_nn_tanh;
    neuro.funcs.grad_af = my_nn_tanh_grad;

    my_nn_create(&neuro);
    double xs4[hparams.epoch];
    double ys4[hparams.epoch];

    my_theme_t g4_th = {
        .point = sfYellow,
        .radius = 10
    };

    my_graph_t g4 = {
        .type = points,
        .xs = xs4,
        .ys = ys4,
        .th = &g4_th
    };

    my_nn_viz_get_error_graph(&g4, &neuro, &features, &targets, &hparams);

    my_graph_t *gs[] = {
        &g,
        &g2,
        &g3,
        &g4
    };

    my_plot_t plt = {
        .title = "neuro viz",
        .gs = gs,
        .gs_n = 4
    };

    my_theme_t plt_th = {
        .bg = sfBlack,
        .axis = sfWhite
    };

    my_plot_create(&plt, &plt_th);

    my_plot_show(&plt);

    my_matrix_free(4, &features, &targets, &targets_tr, &features_tr);
    my_nn_free(&neuro);

    sfRenderWindow_destroy(plt.window);

    return 0;
}
