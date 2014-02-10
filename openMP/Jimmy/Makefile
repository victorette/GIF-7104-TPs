CC=g++
CFLAGS= -c -Wall -std=c++11

all: tp2Original tp2OpenMp tp2Sequential
tp2Original: original.o
	$(CC) original.o -o tp2Original

tp2Sequential: sequential.o
	$(CC) sequential.o -o tp2Sequential

tp2OpenMp: trouverPremiersOpenMP.o
	$(CC) trouverPremiersOpenMP.o -o tp2OpenMp -fopenmp

original.o: original.cc
	$(CC) $(CFLAGS) original.cc

sequential.o: sequential.cc
	$(CC) $(CFLAGS) sequential.cc

trouverPremiersOpenMP.o: trouverPremiersOpenMP.cc
	$(CC) $(CFLAGS) trouverPremiersOpenMP.cc -fopenmp

clean:
	rm -rf *.o *.out tp2*
