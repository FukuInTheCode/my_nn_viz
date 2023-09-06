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
    return x * x * x;
}

typedef struct {
    my_nn_t *nn;
    double x_max;
    double y_max;
    double y_min;
    double x_min;

} my_p_t;

double f2(double x, my_p_t *p)
{
    MAT_DECLA(A);
    my_matrix_create(1, 1, 1, &A);
    my_matrix_set(&A, 0, 0, (x - p->x_min) / p->x_max);
    MAT_DECLA(pre);
    my_nn_predict(p->nn, &A, &pre);
    double res = pre.arr[0][0] * p->y_max + p->y_min;
    my_matrix_free(2, &A, &pre);
    // printf("%lf, %lf\n", x, res);
    return res;
}

sfColor interpolateColor(sfColor color1, sfColor color2, float value)
{
    sfColor result;

    result.r = (sfUint8)((1 - value) * color1.r + value * color2.r);
    result.g = (sfUint8)((1 - value) * color1.g + value * color2.g);
    result.b = (sfUint8)((1 - value) * color1.b + value * color2.b);
    result.a = (sfUint8)((1 - value) * color1.a + value * color2.a);

    return result;
}

int main(void)
{
    srand(time(0));

    MAT_DECLA(features_tr);
    MAT_DECLA(features);
    MAT_DECLA(targets_tr);
    MAT_DECLA(targets);

    my_matrix_create(4, 2, 1, &features_tr);
    my_matrix_fill_from_array(&features_tr, xor_train_fea, 8);
    my_matrix_create(4, 1, 1, &targets_tr);
    my_matrix_fill_from_array(&targets_tr, xor_train_tar, 4);

    // my_matrix_create(50, 1, 1, &features_tr);
    // // my_matrix_create(25, 1, 1, &targets_tr);
    // my_matrix_randfloat(10, -10, 1, &features_tr);
    // my_matrix_applyfunc(&features_tr, fun, &targets_tr);

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

    // // MAT_PRINT(features);
    // // MAT_PRINT(targets);

    // my_params_t hparams = {
    //     .alpha = 1e-1,
    //     .epoch = 100*1000,
    //     .threshold = 1e-6
    // };

    my_nn_t neuro = {.name = "neuro"};

    uint32_t dims[] = {features.m, 3, 3, targets.m};
    neuro.size = sizeof(dims) / sizeof(dims[0]);

    neuro.dims = dims;

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_gelu
    // neuro.funcs.grad_af = my_nn_gelu_grad;

    my_nn_create(&neuro);

    // my_nn_train(&neuro, &features, &targets, &hparams);

    // my_matrix_free(4, &features, &targets, &targets_tr, &features_tr);
    // my_nn_free(&neuro);

    my_nn_print(&neuro);

    sfColor start = sfBlue;
    sfColor end = sfRed;

    sfVideoMode mode = {1000, 1000, 32};
    sfRenderWindow *window = sfRenderWindow_create(mode, "test", sfDefaultStyle, NULL);

    sfEvent event;

    sfVector2u window_size = sfRenderWindow_getSize(window);

    sfVector2f padding = {
        .x = 50,
        .y = 50
    };

    double layer_hpad = (window_size.x - padding.x * 2) / (double)neuro.size;

    double radius = 1. / 4. * layer_hpad;

    while (sfRenderWindow_isOpen(window)) {
        while (sfRenderWindow_pollEvent(window, &event)) {
            if (event.type == sfEvtClosed)
                sfRenderWindow_close(window);
        }

        sfRenderWindow_clear(window, sfBlack);
        for (uint32_t i = 0; i < neuro.size; ++i) {
            double layer_vpad = (window_size.y - padding.y * 2) / (double)(neuro.dims[i]);
            for (uint32_t j = 0; j < neuro.dims[i]; ++j) {
                sfVector2f pos = {
                    .x = i * layer_hpad + padding.x + radius * 2,
                    .y = j * layer_vpad + padding.y + layer_vpad / 2
                };
                if (i != neuro.size - 1) {
                    double nlayer_vpad = (window_size.y - padding.y * 2) / (double)(neuro.dims[i + 1]);
                    for (uint32_t k = 0; k < neuro.dims[i + 1]; ++k) {
                        sfVector2f pos2 = {
                            .x = (i + 1) * layer_hpad + padding.x + radius * 2,
                            .y = k * nlayer_vpad + padding.y + nlayer_vpad / 2
                        };
                        sfVertex connection[] = {
                            {pos, sfWhite, {0, 0}},
                            {pos2, sfWhite, {0, 0}}
                        };
                        sfRenderWindow_drawPrimitives(window, connection, 2, sfLines, NULL);

                    }
                }
                pos.x -= radius;
                pos.y -= radius;
                sfCircleShape *pt = sfCircleShape_create();
                if (i == 0)
                    sfCircleShape_setFillColor(pt, sfGreen);
                else
                    sfCircleShape_setFillColor(pt, interpolateColor(start, end, neuro.bias_arr[i - 1].arr[j][0]));

                sfCircleShape_setRadius(pt, radius);
                sfCircleShape_setPosition(pt, pos);
                sfRenderWindow_drawCircleShape(window, pt, NULL);
                sfCircleShape_destroy(pt);
            }
        }

        sfRenderWindow_display(window);
    }

    sfRenderWindow_destroy(window);
    return 0;
}
