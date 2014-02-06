#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
//#include "Chrono.hpp"
#include <time.h>

// Programme qui trouve à l'aide de la passoire d'Ératosthène,
// tous les nombres premiers inférieurs à un certain seuil 
// spécifié sur la ligne de commande.
// Attention, ce programme n'est aucunement optimisé!
int main(int argc, char *argv[])
{
    // Déterminer la limite supérieure pour la recherche;
    // par défaut, prendre 1000
    unsigned long lMax = 1000000000;
    if (argc >= 2) {
        lMax = atol(argv[1]);
    }

    // Démarrer le chronomètre
    //Chrono lChrono(true);
    clock_t start, stop;
    start = clock();
 
    // Allouer le tableau des drapeaux (flags) d'invalidation
    char *lFlags = (char*) calloc(lMax, sizeof(*lFlags));
    assert(lFlags != 0);
    unsigned long p;
    #pragma omp parallel shared(lFlags) private(p)
    {
        // Appliquer la passoire d'Ératosthène
        #pragma omp for schedule(static)
        for (p=2; p < lMax; p++) {
            if (lFlags[p] == 0) {
                // invalider tous les multiples
                for (unsigned long i=2; i*p < lMax; i++) {
                    lFlags[i*p]++;
                }
            }
        }
    }
    // Arrêter le chronomètre
    //lChrono.pause();
    stop = clock();
    double tm = (double) (stop-start)/CLOCKS_PER_SEC;

    // Afficher les nombres trouvés à la console
    int count = 0;
    for (unsigned long p=2; p<lMax; p++) {
        if (lFlags[p] == 0) count++;//printf("%ld ", p);
    }
    printf("Total : %i",count);
    printf("\n");

    // Afficher le temps d'exécution dans le stderr
    //fprintf(stderr, "Temps d'execution = %f sec\n", lChrono.get());
    printf("Temps d'execution : %lf s\n", tm); 
 
    return 0;
}