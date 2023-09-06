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

    my_params_t hparams = {
        .alpha = 1e-1,
        .epoch = 10,
        .threshold = 1e-6,
        .show_tqdm = false
    };

    my_nn_t neuro = {.name = "neuro"};

    uint32_t dims[] = {features.m, 2, targets.m};
    neuro.size = sizeof(dims) / sizeof(dims[0]);

    neuro.dims = dims;

    neuro.acti_type = base_type;
    neuro.funcs.af = my_nn_gelu;
    neuro.funcs.grad_af = my_nn_gelu_grad;

    my_nn_create(&neuro);

    sfVideoMode mode = {1000, 1000, 32};

    // free
    my_nn_free(&neuro);

    my_matrix_free(4, &features, &targets, &targets_tr, &features_tr);
    sfRenderWindow_destroy(window);
    return 0;
}
