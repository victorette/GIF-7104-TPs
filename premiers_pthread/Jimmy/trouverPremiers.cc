#include <iostream>
#include <cstdlib>
#include <cmath>
#include "Chrono.hpp"

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

int main(int argc, char **argv) {

	if (argc < 3 || argc > 3) {
		usage(argv[0]);
		std::cout << "Default values of 1000 as the upper limit and 1 for the number of threads will be used." << std::endl;
	}
	
	unsigned int lMaxLimit = atoi(argv[1]);
	unsigned int lNbThreads = atoi(argv[2]);

	char *lArrayPrimes = (char *) calloc(lMaxLimit, sizeof(char *));

	if (lArrayPrimes == NULL) {
		std::cout << "Not enough memory to allocate array space." << std::endl;
		exit(-2);
	}

	std::cout << "Travail Pratique 1 : Trouver les " << lMaxLimit << " premiers nombres premiers a l'aide de " << lNbThreads << " threads." << std::endl;

	Chrono lChrono(true);

	for (unsigned long long i = 4 ; i <= lMaxLimit ; i += 2) {
		lArrayPrimes[i]++;
	}

	long lSquareRoot = sqrt(lMaxLimit);

	for (unsigned int i = 3 ; i <= lSquareRoot ; i += 2) {
		if ((int)lArrayPrimes[i] == 0) {
			for (unsigned int j = i*i ; j <= lMaxLimit ; j += 2 * i) {
				lArrayPrimes[j]++;
			}
		}
	}

	lChrono.pause();

	printPrimes(lArrayPrimes, lMaxLimit);
	printArray(lArrayPrimes, lMaxLimit);

	std::cout << "Travail effectue en " << lChrono.get() << " secondes." << std::endl;

	free(lArrayPrimes);

	return 0;
}
