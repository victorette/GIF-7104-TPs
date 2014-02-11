CC=g++
CFLAGS= -c -Wall -std=c++11

all: tp2Original tp2OpenMp tp2Sequential tp2OpenMpManual
tp2Original: original.o
	$(CC) original.o -o tp2Original

tp2Sequential: sequential.o
	$(CC) sequential.o -o tp2Sequential

tp2OpenMp: trouverPremiersOpenMP.o
	$(CC) trouverPremiersOpenMP.o -o tp2OpenMp -fopenmp

tp2OpenMpManual: trouverPremiersOpenMPManual.o
	$(CC) trouverPremiersOpenMPManual.o -o tp2OpenMpManual -fopenmp

original.o: original.cc
	$(CC) $(CFLAGS) original.cc

sequential.o: sequential.cc
	$(CC) $(CFLAGS) sequential.cc

trouverPremiersOpenMP.o: trouverPremiersOpenMP.cc
	$(CC) $(CFLAGS) trouverPremiersOpenMP.cc -fopenmp

trouverPremiersOpenMPManual.o: trouverPremiersOpenMPManual.cc
	$(CC) $(CFLAGS) trouverPremiersOpenMPManual.cc -fopenmp

clean:
	rm -rf *.o *.out tp2*
