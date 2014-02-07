CC=g++
CFLAGS= -c -Wall -std=c++11

all: tp2Seq tp2OpenMP
tp2Seq: premiers.o
	$(CC) premiers.o -o tp2Seq

tp2OpenMP: premiersOpenMP.o
	$(CC) premiersOpenMP.o -o tp2OpenMp -lgomp

premiers.o: premiers.cpp
	$(CC) $(CFLAGS) premiers.cpp

premiersOpenMP.o: premiersOpenMP.cpp
	$(CC) $(CFLAGS) premiersOpenMP.cpp

clean:
	rm -rf *.o *.out tp2*
