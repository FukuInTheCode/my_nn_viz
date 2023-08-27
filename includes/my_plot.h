#pragma once
#define MY_PLOT_H

#include "../includes/SFML/Graphics.h"
#include "../includes/SFML/Window.h"
#include "my_math.h"

#define SCALE 2

typedef enum {
    plot_th,
    graph_th
} my_th_type_t;

typedef enum {
    pts,
    axis
} my_obj_type_t;

typedef enum {
    static_pts,
    dynamic_pts,
    static_func,
    dynamic_func
} my_graph_type_t;

typedef struct {
    my_th_type_t type;
    union {
        struct {
            sfColor bg;
            sfColor axis;
        } plot;
        struct {
            double radius;
            sfColor pt;
        } graph;
    };
} my_theme_t;

typedef struct {
    my_graph_type_t type;

    sfVector2f max_values;
    sfVector2f min_values;

    sfVector2f ratio;
    size_t data_num;
    struct {
        size_t max_pts;
        my_func_t func;
    } st_func;
    sfVector2f *points;

    sfVector2f *computed_pts;
    my_theme_t *theme;
} my_graph_t;

typedef struct {
    sfRenderWindow *window;
    sfEvent *event;

    sfBool is_dragged;
    sfVector2f shift;
    sfVector2i last_shift;

    uint32_t graph_n;
    my_graph_t **graph;

    my_theme_t *theme;
} my_plot_t;

static inline __attribute__((always_inline)) void copy_vec(sfVector2f *to,\
                                                    sfVector2f *from, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        to[i].x = from[i].x;
        to[i].y = from[i].y;
    }
}

static inline __attribute__((always_inline)) void compute_pts(my_plot_t *plt,\
                                                                my_graph_t *g)
{
    sfVector2u tmp_vec = sfRenderWindow_getSize(plt->window);
    copy_vec(g->computed_pts, g->points, g->data_num);
    for (size_t i = 0; i < g->data_num; ++i) {
        g->computed_pts[i].x *= g->ratio.x;
        g->computed_pts[i].y *= g->ratio.y;
        g->computed_pts[i].y = (tmp_vec.y - g->theme->graph.radius*2) -\
                                    g->computed_pts[i].y;
        g->computed_pts[i].x += tmp_vec.x / 2.f;
        g->computed_pts[i].y -= tmp_vec.y / 2.f;
    }
}

void my_plot_create(my_plot_t *plt, char *title, sfVideoMode *md, sfEvent *evt);
void my_plot_show(my_plot_t *plt);
void my_plot_handle_event(my_plot_t *plt);
void my_plot_handle_mouse(my_plot_t *plt);
void my_plot_points(my_plot_t *plt, my_graph_t *g);
void my_plot_axis(my_plot_t *plt);
sfBool my_plot_is_onscreen(my_plot_t *plt, sfVector2f coords,\
                            my_obj_type_t type, my_graph_t *g);
void my_plot_func(my_plot_t *plt, my_graph_t *g);
