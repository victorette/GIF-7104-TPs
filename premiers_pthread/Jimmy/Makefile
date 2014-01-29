CC=g++
CFLAGS= -c -Wall -std=c++11

all: Main
Main: trouverPremiers.o
	$(CC) *.o -o tp1 -lpthread

trouverPremiers.o: trouverPremiers.cc
	$(CC) $(CFLAGS) trouverPremiers.cc

clean:
	rm -rf *.o *.out tp1
