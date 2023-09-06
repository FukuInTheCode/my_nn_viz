#include "../../includes/my.h"

typedef struct {
    double layer_vpad;
    double layer_hpad;
    double radius;
    uint32_t i;
    uint32_t j;
    my_nn_t *nn;
} my_shit_t;

static void draw_connections(sfRenderWindow *window, my_shit_t *s)
{
    sfVector2u window_size = sfRenderWindow_getSize(window);
    sfColor start = COLOR1_W;
    sfColor end = COLOR2_W;
    if (s->i != s->nn->size - 1) {
        sfVector2f pos = {
            .x = s->i * s->layer_hpad + PAD_X + s->radius * 2,
            .y = s->j * s->layer_vpad + PAD_Y + s->layer_vpad / 2
        };
        double nlayer_vpad = (window_size.y - PAD_Y * 2) / (double)(s->nn->dims[s->i + 1]);
        for (uint32_t k = 0; k < s->nn->dims[s->i + 1]; ++k) {
            sfVector2f pos2 = {
                .x = (s->i + 1) * s->layer_hpad + PAD_X + s->radius * 2,
                .y = k * nlayer_vpad + PAD_Y + nlayer_vpad / 2
            };
            sfVertex connection[] = {
                {pos, interpolate_color(start, end, s->nn->theta_arr[s->i].arr[k][s->j]), {0, 0}},
                {pos2, interpolate_color(start, end, s->nn->theta_arr[s->i].arr[k][s->j]), {0, 0}}
            };
            sfRenderWindow_drawPrimitives(window, connection, 2, sfLines, NULL);
        }
    }
}

static void plot_neuron(sfRenderWindow *window, my_shit_t *s, my_nn_t *nn)
{
    sfColor start = COLOR1_B;
    sfColor end = COLOR2_B;
    sfVector2f pos = {
        .x = s->i * s->layer_hpad + PAD_X + s->radius * 2,
        .y = s->j * s->layer_vpad + PAD_Y + s->layer_vpad / 2
    };
    sfColor color = COLOR_0;
    if (s->i != 0)
        color = interpolate_color(start, end,\
                    nn->bias_arr[s->i - 1].arr[s->j][0]);
    pos.x -= s->radius;
    pos.y -= s->radius;
    sfCircleShape *pt = sfCircleShape_create();
    sfCircleShape_setFillColor(pt, color);
    sfCircleShape_setRadius(pt, s->radius);
    sfCircleShape_setPosition(pt, pos);
    sfRenderWindow_drawCircleShape(window, pt, NULL);
    sfCircleShape_destroy(pt);
}

void my_nn_viz_arch(my_nn_t *nn, sfRenderWindow *window)
{
    sfVector2u window_size = sfRenderWindow_getSize(window);
    double layer_hpad = (window_size.x - PAD_X * 2) / (double)nn->size;
    double radius = 1. / 4. * layer_hpad;
    for (uint32_t i = 0; i < nn->size; ++i) {
        double layer_vpad = (window_size.y - PAD_Y * 2) /\
                                        (double)(nn->dims[i]);
        for (uint32_t j = 0; j < nn->dims[i]; ++j) {
            my_shit_t s = {
                .layer_hpad = layer_hpad,
                .layer_vpad = layer_vpad,
                .radius = radius,
                .i = i,
                .j = j,
                .nn = nn
            };
            draw_connections(window, &s);
            plot_neuron(window, &s, nn);
        }
    }
}
