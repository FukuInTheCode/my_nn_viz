NAME       = my_nn_viz

CFLAGS     = -W -Wall -Wextra

LIB        = -l$(NAME)

CSFML	   = -lcsfml-graphics -lcsfml-system -lcsfml-audio -lcsfml-window

DEPENDENCIES = -lmy_plot -lmy_neuralnetwork -lmy_matrix -lmy_math $(CSFML)

SRC        = $(wildcard ./src/*.c)

LIB_SRC    = $(wildcard ./src/func/*.c)

LIB_OBJS    = $(wildcard ./src/func/*.o)

all: archive build clean

archive: build_obj $(LIB_OBJS)
	@ar -rcs C:\Users\ADMIN\Code\my_libs\lib$(NAME).a $(LIB_OBJS)

build:
	@if not exist "./out" mkdir "./out"
	@gcc $(CFLAGS) $(SRC) -LC:\Users\ADMIN\Code\my_libs $(LIB) -I./includes -o ./out/$(NAME) $(DEPENDENCIES)

clean:
	@if exist ./src/*.o del /Q src\*.o
	@if exist ./src/func/*.o del /Q src\func\*.o
	@if exist ./*.o del /Q *.o

build_obj:
	@gcc $(CFLAGS) $(DEPENDENCIES) -c -LC:\Users\ADMIN\Code\my_libs -I./includes $(LIB_SRC)
	@move *.o ./src/func