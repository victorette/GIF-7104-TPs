#include <iostream>
#include <cstdlib>
#include <cmath>

#include <pthread.h>

#include "Chrono.hpp"

char *gArrayPrimes;
unsigned int gNbPremierASupp;
unsigned int gMaxLimit;
long gLimiteSuppSqrt;

pthread_mutex_t gLockIncrement = PTHREAD_MUTEX_INITIALIZER;

void usage (char *iProgram) {
	std::cout << "Usage> " << iProgram << " limite_prime_number number_of_threads" << std::endl;
	exit(-1);
}

void printArray(char* iArray, unsigned int iSize) {
	for (unsigned int i = 0 ; i <= iSize ; i++) {
		std::cout << (int)iArray[i];
	}

	std::cout << std::endl;
}

void printPrimes(char* iArray, unsigned int iSize) {
	for (unsigned int i = 1 ; i <= iSize ; i++) {
		if ((int)iArray[i] == 0) {
			std::cout << i << " ";
		}
	}

	std::cout << std::endl;
}

int countPrimes(char* iArray, unsigned int iSize) {
	int nbPrime = 0;
	for (unsigned int i = 1 ; i <= iSize ; i++) {
		if ((int)iArray[i] == 0) {
			nbPrime++;
		}
	}

	return nbPrime;
}

void * epurerNombrePremier(void * iParam) {
	unsigned int currentNumber = 0;

	do {
		pthread_mutex_lock(&gLockIncrement);
		currentNumber = gNbPremierASupp;
		gNbPremierASupp += 2;
//		std::cout << *lThreadId << " : " << currentNumber << std::endl;
		pthread_mutex_unlock(&gLockIncrement);

		if (currentNumber > gLimiteSuppSqrt) {
			return NULL;

		}

		if (gArrayPrimes[currentNumber] == 0) {

			for (unsigned int i = currentNumber * currentNumber ; i <= gMaxLimit ; i += 2 * currentNumber) {
				gArrayPrimes[i]++;

			}
		}
	}
	while (true);
}

int main(int argc, char **argv) {

	if (argc < 3 || argc > 3) {
		usage(argv[0]);
	}
	
	gMaxLimit = atoi(argv[1]);
	unsigned short lNbThreads = atoi(argv[2]);

	pthread_t lThreads[lNbThreads];

	gNbPremierASupp = 3;
	gArrayPrimes = (char *) calloc(gMaxLimit, sizeof(char *));

	if (gArrayPrimes == NULL) {
		std::cout << "Not enough memory to allocate array space." << std::endl;
		exit(-2);
	}

	std::cout << "Travail Pratique 1 : Trouver les " << gMaxLimit << " premiers nombres premiers a l'aide de " << lNbThreads << " threads." << std::endl;

	Chrono lChrono(true);

	for (unsigned long long i = 4 ; i <= gMaxLimit ; i += 2) {
		gArrayPrimes[i]++;
	}

	gLimiteSuppSqrt = sqrt(gMaxLimit);
	
	for (int i = 0 ; i <= lNbThreads ; i++) {
		pthread_create(&lThreads[i], NULL, epurerNombrePremier, NULL);
	}

	for (int i = 0 ; i <= lNbThreads ; i++) {
		pthread_join(lThreads[i], NULL);
	}

// 	epurerNombrePremier(&lThreadId);


	/*
	for (unsigned int i = 3 ; i <= lSquareRoot ; i += 2) {
		if ((int)lArrayPrimes[i] == 0) {
			for (unsigned int j = i*i ; j <= gMaxLimit ; j += 2 * i) {
				lArrayPrimes[j]++;
				epurerNombrePremier(NULL);
				
			}
		}
	}
	*/

	lChrono.pause();

//	printPrimes(gArrayPrimes, gMaxLimit);
//	printArray(gArrayPrimes, gMaxLimit);

	std::cout << countPrimes(gArrayPrimes, gMaxLimit) << " primes found in " << lChrono.get() << " seconds." << std::endl;

	free(gArrayPrimes);

	return 0;
}
