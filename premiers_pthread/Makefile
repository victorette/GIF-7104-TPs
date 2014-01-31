CC=g++
CFLAGS= -c -Wall -std=c++11

all: Main
Main: premiers_pthread.o
	$(CC) *.o -o tp1 -lpthread

premiers_pthread.o: premiers_pthread.cpp
	$(CC) $(CFLAGS) premiers_pthread.cpp

clean:
	rm -rf *.o *.out tp1
