//============================================================================
// Name        : premiers_pthread.cpp
// Author      : Victorette
// Version     :
// Copyright   : 
// Description : Programme qui trouve à l'aide de la passoire d'Ératosthène,
// tous les nombres premiers inférieurs à un certain seuil
// spécifié sur la ligne de commande.
// Attention, ce programme n'est aucunement optimisé!
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <pthread.h>
#include "Chrono.hpp"
struct thread_data {
	int id;
	int begin;
	int end;
};

void *getPrimeNumber(void *threadarg) {
	struct thread_data *my_data;
	my_data = (struct thread_data *) threadarg;

	// Allouer le tableau des drapeaux (flags) d'invalidation
	char *lFlags = (char*) calloc(my_data->end, sizeof(*lFlags));
	assert(lFlags != 0);

	// Appliquer la passoire d'Ératosthène
	for (unsigned long p = 2; p < my_data->end; p++) {
		if (lFlags[p] == 0) {
			// invalider tous les multiples
			for (unsigned long i = 2; i * p < my_data->end; i++) {
				lFlags[i * p]++;
			}
		}
	}
	pthread_exit((void*) lFlags);
}

int main(int argc, char *argv[]) {

	unsigned long limiteMax;
	int numThreads;
	int searchRangeSize;
	void* flagedNumbers;
	int returnCall;
	char buffer[256];

	if (argc >= 2) {
		limiteMax = atol(argv[1]);
	} else {
		printf("Déterminer la limite supérieure pour la recherche: ");
		fgets(buffer, 256, stdin);
		limiteMax = atol(buffer);
		if (limiteMax == 0) {
			printf("limite non valide, par défaut, prendre 1000 \n");
			limiteMax = 1000;
		}
	}

	printf("Déterminer le nombre de fils d'exécution désiré: ");
	fgets(buffer, 256, stdin);
	numThreads = atoi(buffer);
	if (numThreads == 0) {
		printf("nombre de threads non valide, par défaut, prendre 1 \n");
		numThreads = 1;
	}

	searchRangeSize = limiteMax / numThreads;

	pthread_t threads[numThreads];
	pthread_attr_t attr;
	struct thread_data thread_data_array[numThreads];

	// Démarrer le chronomètre
	Chrono lChrono(true);

	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	int startPoint = 0;
	for (int t = 0; t < numThreads; t++) {
		thread_data_array[t].id = t;
		thread_data_array[t].begin = startPoint;
		thread_data_array[t].end = startPoint + searchRangeSize;
		startPoint = startPoint + searchRangeSize;

		returnCall = pthread_create(&threads[t], &attr, getPrimeNumber, (void *) &thread_data_array[t]);
		if (returnCall) {
			printf("ERROR; return code from pthread_create() is %d\n", returnCall);
			exit(-1);
		}
	}
	char *lFlags;
	for (int t = 0; t < numThreads; t++) {
		returnCall = pthread_join(threads[t], &flagedNumbers);
		if (returnCall) {
			printf("ERROR; return code from pthread_join() is %d\n", returnCall);
			exit(-1);
		}
		lFlags = (char *) flagedNumbers;
	}

	// Arrêter le chronomètre
	lChrono.pause();

	// Afficher les nombres trouvés à la console
	for (unsigned long p = 2; p < limiteMax; p++) {
		if (lFlags[p] == 0)
			printf("%ld ", p);
	}
	printf("\n");


	// Afficher le temps d'exécution dans le stderr
	fprintf(stderr, "Temps d'execution = %f sec\n", lChrono.get());

	return 0;
}
