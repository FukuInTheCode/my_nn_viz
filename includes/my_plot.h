#ifndef MY_CPLOT_H
#define MY_CPLOT_H

#include "SFML/Graphics.h"
#include "my_math.h"

#define PLOT_DECLA(nvar, ti) my_plot_t nvar = {.title = #ti, .gs_n = 0}

#define GRAPH_DECLA(name) my_graph_t name = {.max_pts_n = 0};

typedef double (*func)(double);
typedef double (*func2)(double, void *);

typedef enum {
    points,
    function,
    function2
} graph_type;

typedef union {
    struct {
        sfColor bg;
        sfColor axis;
    };
    struct {
        sfColor point;
        uint32_t radius;
    };
} my_theme_t;

typedef struct {
    graph_type type;
    double *xs;
    double *ys;
    uint32_t pts_n;
    uint32_t max_pts_n;
    union {
        func f;
        struct {
            void *params;
            func2 f2;
        };
    };
    my_theme_t *th;
} my_graph_t;

typedef struct {
    sfRenderWindow *window;
    sfEvent event;
    char *title;
    my_theme_t *th;
    uint32_t gs_n;
    my_graph_t **gs;
    sfVector2f ratio;
    sfVector2f shift;
    sfVector2f zoom;
    bool is_pressed;
    sfVector2i mouse_save;
} my_plot_t;

void my_plot_create(my_plot_t *plt, my_theme_t *th);
void my_plot_show(my_plot_t *plt);
void my_plot_handle_events(my_plot_t *plt);
void my_plot_handle_mouse(my_plot_t *plt);
void my_plot_handle_zoom(my_plot_t *plt);
void my_plot_handle_shift(my_plot_t *plt);
void my_plot_axis(my_plot_t *plt);
void my_plot_pts(my_plot_t *plt, my_graph_t *g);
void my_plot_func(my_plot_t *plt, my_graph_t *g);
void my_plot_func2(my_plot_t *plt, my_graph_t *g);
void my_plot_free(my_plot_t *plt);
void my_plot_append(my_plot_t *plt, my_graph_t *g);
void calc_ratio(my_plot_t *plt);
void my_graph_create_f2(my_graph_t *g, uint32_t n, my_theme_t *th, func2 f);
void my_graph_create_f(my_graph_t *g, uint32_t n, my_theme_t *th, func f);
void my_graph_create_pts(my_graph_t *g, uint32_t n, my_theme_t *th);
void my_graph_set_pts(my_graph_t *g, double *xs, double *ys);

#endif