OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: main.cpp
	g++ -Wall -g  main.cpp -o main $(OPTIONS)


clean:
	rm -rf main
