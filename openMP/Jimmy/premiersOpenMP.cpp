//============================================================================
// Name        : premiers_pthread.cpp
// Author      : Victorette
// Version     :
// Copyright   : 
// Description : Programme qui trouve à l'aide de la passoire d'Ératosthène,
// tous les nombres premiers inférieurs à un certain seuil
// spécifié sur la ligne de commande.
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include "Chrono.hpp"
#include <omp.h>

int numThreads, maxLimit, nextbase;
char *lArrayPrimes;
long lSquareRoot;

int main(int argc, char *argv[]) {

	int numThreads;

	if (argc < 3 || argc > 3) {
		printf("Usage> %s limite_prime_number number_of_threads\n", argv[0]);
		exit(-1);
	}
	maxLimit = atol(argv[1]);
	numThreads = atol(argv[2]);

	lArrayPrimes = (char *) calloc(maxLimit, sizeof(char *));
	// Multiples de 2
	lArrayPrimes[1]++;
	for (int i = 4; i <= maxLimit; i += 2) {
		lArrayPrimes[i]++;
	}
	int base;
	nextbase = 3;
	lSquareRoot = sqrt(maxLimit);
	int i, np;
	// Démarrer le chronomètre
	Chrono lChrono(true);
	
	#pragma omp parallel num_threads(numThreads) shared(lArrayPrimes, nextbase) private(base,i)
    {
    	#pragma omp for schedule(static)
	//#pragma omp parallel for ordered schedule(dynamic)
		for (base = nextbase; base <= lSquareRoot; base +=2){
			if ((int)lArrayPrimes[base] == 0) {
				for (i = base; i * base <= maxLimit; i += 2){
					lArrayPrimes[i * base]++;
				}
			}
		}
		np = omp_get_num_threads();
	}
	//np = omp_get_num_threads();
	// Arrêter le chronomètre
	lChrono.pause();

	// Afficher les nombres trouvés à la console
	int count = 0;
	for (int i = 1; i <= maxLimit; i++) {
		if ((int) lArrayPrimes[i] == 0) {
			//printf("%i ", i);
			count++;
		}
	}
	printf("\n");
	printf("Limite Max : %i\nnumThreads : %i\n", maxLimit, np);
	printf("Primes numbers found : %i\n", count);

	// Afficher le temps d'exécution dans le stderr
	printf("Temps d'execution = %f sec\n", lChrono.get());
	return 0;
}
