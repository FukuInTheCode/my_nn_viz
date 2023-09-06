#include "../../includes/my.h"

static void draw_connections(my_nn_t *nn, sfRenderWindow *window, uint32_t i, uint32_t j)
{
    sfColor start = COLOR1_W;
    sfColor end = COLOR2_W;
    if (i != nn->size - 1) {
        double nlayer_vpad = (window_size.y - PAD_Y * 2) / (double)(nn->dims[i + 1]);
        // for each next nn->
        for (uint32_t k = 0; k < nn->dims[i + 1]; ++k) {
            sfVector2f pos2 = {
                .x = (i + 1) * layer_hpad + PAD_X + radius * 2,
                .y = k * nlayer_vpad + PAD_Y + nlayer_vpad / 2
            };
            sfVertex connection[] = {
                {pos, interpolate_color(start, end, nn->theta_arr[i].arr[k][j]), {0, 0}},
                {pos2, interpolate_color(start, end, nn->theta_arr[i].arr[k][j]), {0, 0}}
            };
            sfRenderWindow_drawPrimitives(window, connection, 2, sfLines, NULL);
        }
    }
}

void my_nn_viz_arch(my_nn_t *nn, sfRenderWindow *window)
{
    sfColor start = COLOR1_B;
    sfColor end = COLOR2_B;
    sfVector2u window_size = sfRenderWindow_getSize(window);
    double layer_hpad = (window_size.x - padding.x * 2) / (double)nn->size;
    double radius = 1. / 4. * layer_hpad;
    for (uint32_t i = 0; i < nn->size; ++i) {
        double layer_vpad = (window_size.y - padding.y * 2) / (double)(nn->dims[i]);
        // for each nn->of each layer
        for (uint32_t j = 0; j < nn->dims[i]; ++j) {
            sfVector2f pos = {
                .x = i * layer_hpad + PAD_X + radius * 2,
                .y = j * layer_vpad + PAD_Y + layer_vpad / 2
            };

            draw_connections(nn, window, i, j);
            pos.x -= radius;
            pos.y -= radius;
            sfCircleShape *pt = sfCircleShape_create();
            if (i == 0)
                sfCircleShape_setFillColor(pt, sfRed);
            else
                sfCircleShape_setFillColor(pt, interpolate_color(start, ends, nn->bias_arr[i - 1].arr[j][0]));
            sfCircleShape_setRadius(pt, radius);
            sfCircleShape_setPosition(pt, pos);
            sfRenderWindow_drawCircleShape(window, pt, NULL);
            sfCircleShape_destroy(pt);
        }
    }
}