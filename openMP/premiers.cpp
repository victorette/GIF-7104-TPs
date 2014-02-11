#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#import <dispatch/dispatch.h>
#include <cmath>
//#include "Chrono.hpp"
#include <time.h>

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
    clock_t start, stop;
    start = clock();

    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_group_async(group, queue, ^{
        for (base = nextbase; base <= lSquareRoot; base +=2){
            if ((int)lArrayPrimes[base] == 0) {
                for (i = base; i * base <= maxLimit; i += 2){
                    lArrayPrimes[i * base]++;
                }
            }
        }
    });

    // Arrêter le chronomètre
    //lChrono.pause();
    stop = clock();
    double tm = (double) (stop-start)/CLOCKS_PER_SEC;

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
    //fprintf(stderr, "Temps d'execution = %f sec\n", lChrono.get());
    printf("Temps d'execution : %lf s\n", tm); 
 
    return 0;
}
