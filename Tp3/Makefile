CC=g++
CFLAGS= -c -Wall -std=c++11

all: tp3Sequential
tp3Sequential: main.o Matrix.o
	$(CC) main.o Matrix.o -o tp3Sequential

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

Matrix.o: Matrix.cpp
	$(CC) $(CFLAGS) Matrix.cpp

clean:
	rm -rf *.o *.out tp3*
