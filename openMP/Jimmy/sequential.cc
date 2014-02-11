#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Chrono.hpp"

// Programme qui trouve à l'aide de la passoire d'Ératosthène,
// tous les nombres premiers inférieurs à un certain seuil 
// spécifié sur la ligne de commande.
// Attention, ce programme n'est aucunement optimisé!
int main(int argc, char *argv[])
{
    // Déterminer la limite supérieure pour la recherche;
    // par défaut, prendre 1000
    unsigned long lMax = 1000;
    if (argc >= 2) {
        lMax = atol(argv[1]);
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
 
    // Allouer le tableau des drapeaux (flags) d'invalidation
    char *lFlags = (char*) calloc(lMax, sizeof(*lFlags));
    assert(lFlags != 0);

    // Appliquer la passoire d'Ératosthène
    for (unsigned long p=2; p < lMax; p++) {
        if (lFlags[p] == 0) {
            // invalider tous les multiples
            for (unsigned long i=2; i*p < lMax; i++) {
                lFlags[i*p]++;
            }
        }
    }

    // Arrêter le chronomètre
    lChrono.pause();

    // Afficher les nombres trouvés à la console
    int somme = 0;
    for (unsigned long p=2; p<lMax; p++) {
        if (lFlags[p] == 0) somme++;
    }
    printf("%d nombre premiers trouves", somme);
    printf("\n");
    printf("Limite Max : %i\nnumThreads : %i\n", maxLimit, np);
    printf("Primes numbers found : %i\n", count);

    // Afficher le temps d'exécution dans le stderr
    fprintf(stderr, "Temps d'execution = %f sec\n", lChrono.get());
 
    return 0;
}
