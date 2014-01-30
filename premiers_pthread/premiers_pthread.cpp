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
#include <pthread.h>

int numThreads, maxLimit, nextbase;
char *lArrayPrimes;
long lSquareRoot;

pthread_mutex_t nextbaselock = PTHREAD_MUTEX_INITIALIZER;

void *getPrimeNumber(void *threadarg) {
	int base;
	do {
		pthread_mutex_lock(&nextbaselock);
		base = nextbase;
		nextbase += 2;
		pthread_mutex_unlock(&nextbaselock);

		if ((int)lArrayPrimes[base] == 0) {
			for (int i = base; i * base <= maxLimit; i += 2){
				lArrayPrimes[i * base]++;
			}
		}

	} while(base <= lSquareRoot);

	pthread_exit(NULL);
}

int main(int argc, char *argv[]) {

	int numThreads;
	int returnCode;
	char buffer[256];

	if (argc >= 2) {
		maxLimit = atol(argv[1]);
		numThreads = atol(argv[2]);
	} else {
		printf("Déterminer la limite supérieure pour la recherche: ");
		fgets(buffer, 256, stdin);
		maxLimit = atol(buffer);
		if (maxLimit == 0) {
			printf("limite non valide, par défaut, prendre 1000 \n");
			maxLimit = 1000;
		}
		printf("Déterminer le nombre de fils d'exécution désiré: ");
		fgets(buffer, 256, stdin);
		numThreads = atoi(buffer);
		if (numThreads == 0) {
			printf("nombre de threads non valide, par défaut, prendre 1 \n");
			numThreads = 1;
		}
	}


	lArrayPrimes = (char *) calloc(maxLimit, sizeof(char *));
	// Multiples de 2
	lArrayPrimes[1]++;
	for (unsigned long i = 4; i <= maxLimit; i += 2) {
		lArrayPrimes[i]++;
	}
	nextbase = 3;
	lSquareRoot = sqrt(maxLimit);
	pthread_t threads[numThreads];

	// Démarrer le chronomètre
	Chrono lChrono(true);

	for (int t = 0; t < numThreads; t++) {
		returnCode = pthread_create(&threads[t], NULL, getPrimeNumber, (void *) &t);
		if (returnCode) {
			printf("ERROR; return code from pthread_create() is %d\n", returnCode);
			exit(-1);
		}
	}
	for (int t = 0; t < numThreads; t++) {
		returnCode = pthread_join(threads[t], NULL);
		if (returnCode) {
			printf("ERROR; return code from pthread_join() is %d\n", returnCode);
			exit(-1);
		}
	}
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
	printf("Primes numbers found : %i ", count);

	// Afficher le temps d'exécution dans le stderr
	printf("Temps d'execution = %f sec\n", lChrono.get());
	return 0;
}
