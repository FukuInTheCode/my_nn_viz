#include "../../includes/my.h"

static uint32_t train(my_nn_t *nn, uint32_t h, my_nn_dat_t *data,\
                                                    uint32_t steps)
{
    if (h < data->hp->epoch) {
        my_nn_train(nn, data->x, data->y, &hp);
        usleep(100000);
    }
    return h + steps;
}

static void show(my_nn_t *nn, sfRenderWindow *window)
{
    sfRenderWindow_clear(window, sfBlack);
    my_nn_viz_arch(nn, window);
    sfRenderWindow_display(window);
}

void my_nn_viz_repr_train(my_nn_t *nn, sfVideoMode mode,\
                                my_nn_dat_t *data, uint32_t steps)
{
    sfRenderWindow *window = sfRenderWindow_create(mode,\
                                nn->name, sfDefaultStyle, NULL);
    sfEvent event;
    my_params_t hp = {
        .epoch = steps,
        .threshold = data->hp->threshold,
        .alpha = data->hp->alpha,
        .show_tqdm = false
    };
    uint32_t h = 0;
    while (sfRenderWindow_isOpen(window)) {
        while (sfRenderWindow_pollEvent(window, &event)) {
            if (event.type == sfEvtClosed)
                sfRenderWindow_close(window);
        }
        h = train(nn, h, data, steps);
        show(nn, window);
    }
    sfRenderWindow_destroy(window);
}
