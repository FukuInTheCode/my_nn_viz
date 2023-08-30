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
    // ...
    return 0;
}
