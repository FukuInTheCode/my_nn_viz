#include "../../includes/my.h"

void my_nn_viz_repr(my_nn_t *nn, sfVideoMode mode)
{
    sfRenderWindow *window = sfRenderWindow_create(mode,\
                                "test", sfDefaultStyle, NULL);
    sfEvent event;
    while (sfRenderWindow_isOpen(window)) {
        while (sfRenderWindow_pollEvent(window, &event)) {
            if (event.type == sfEvtClosed)
                sfRenderWindow_close(window);
        }

        sfRenderWindow_clear(window, sfBlack);
        my_nn_viz_arch(nn, window);
        sfRenderWindow_display(window);
    }
}
