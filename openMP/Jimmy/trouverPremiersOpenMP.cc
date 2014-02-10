#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include "Chrono.hpp"
#include <omp.h>

// Programme qui trouve à l'aide de la passoire d'Ératosthène,
// tous les nombres premiers inférieurs à un certain seuil 
// spécifié sur la ligne de commande.
// Attention, ce programme n'est aucunement optimisé!
int main(int argc, char *argv[])
{
    // Déterminer la limite supérieure pour la recherche;
    // par défaut, prendre 1000
    unsigned long lMax = 1000, lNbThreads = 8;
    if (argc >= 2) {
        lMax = atol(argv[1]);
    }
    if (argc >= 3) {
	lNbThreads = atol(argv[2]);
    }

 
    // Allouer le tableau des drapeaux (flags) d'invalidation
    char *lFlags = (char*) calloc(lMax, sizeof(*lFlags));
    assert(lFlags != 0);
    unsigned long root = sqrt(lMax);
    unsigned long p, i;

	// Multiples de 2
	lFlags[1]++;
	for (unsigned int i = 4; i <= lMax; i += 2) {
		lFlags[i]++;
	}

    // Démarrer le chronomètre
    Chrono lChrono(true);
    // Appliquer la passoire d'Ératosthène
	omp_set_num_threads(lNbThreads);
#pragma omp parallel shared(lFlags) private(p, i)
{
#pragma omp for schedule(dynamic)
    for (p = 3 ; p <= root ; p += 2) {
	//printf("%d : %lu\n", omp_get_thread_num(), p);
        if ((int)lFlags[p] == 0) {
            for (i = p ; i * p < lMax ; i += 2) {
		//printf("  %d : %lu - %lu\n", omp_get_thread_num(), i, i * p);
                lFlags[i*p]++;
            }
        }
    }
#pragma omp single
printf("Nombre de fils utilises : %d\n", omp_get_num_threads());
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

    // Afficher le temps d'exécution dans le stderr
    printf("Temps d'execution = %f sec\n", lChrono.get());
 
    return 0;
}
