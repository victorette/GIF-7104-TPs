//
//  mainGauss.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <mpi.h>

using namespace std;
using namespace MPI;

// Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
void invertParallel(Matrix& iA, int lRank, int lProcSize) {

 	// ✓ dataSize désigne la dimension de la matrice (n lignes par n colonnes) 
 	// ✓ i et j sont des indices de ligne et de colonne respectivement
	// ✓ k est l’indice d’étape de l’algorithme (n étapes au total)
	// ✓ q est l’indice du max (en valeur absolue) dans la colonne k
	// ✓ lRank (r) est le rang du processus, et lProcSize le nombre total de processus
	// ✓ la ligne i appartient au processus r si i%p=r (décomposition «row-cyclic»)

	int i,j,k,lDataSize;
	int q;
	struct
	{
		double value;
		int node;
	} in, out;

	// vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    lDataSize = iA.cols();

	// construire la matrice [A I]
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

	// 2. Rapatrier (Gatherv) toutes les lignes sur le processus 0

	// Pour chaque étape k: 0..n-1 de l’algorithme
	for (k = 0; k < lDataSize; ++k) {
		// if (k % lProcSize == lRank) {
	 //    }
		// a. Déterminer localement le q parmi les lignes qui appartiennent à r, puis
		// faire une reduction (Allreduce avec MAXLOC) pour déterminer le q global
		q = k;
        in.value = fabs(lAI(k,k));
        for(i = k; i < lAI.rows(); ++i) {
            if(fabs(lAI(i,k)) > in.value && i % lProcSize == lRank) {
                in.value = fabs(lAI(i,k));
                q = i;
                in.node = q;
            }
        }
		COMM_WORLD.Reduce(&in,&out,1,MPI_DOUBLE_INT,MPI_MAXLOC,0);
        if (lRank == 0) {
        	cout << "Indice: " << out.node << " - Value: " << out.value << endl;
			
			// b. Si la valeur du max est nulle, la matrice est singulière (ne peut être
			// inversée)
			if (lAI(out.node, k) == 0) throw runtime_error("Matrix not invertible");
			q = out.node;
			// c. Diffuser (Bcast) la ligne q appartenant au processus r=q%p (r est root)
			// FIXME ça ne marche pas...chaque fil possede un q diff
			COMM_WORLD.Bcast(&q,1,MPI_INT,0);
        }
        // MPI_Bcast(&q, 1, MPI_INT, 0, MPI_COMM_WORLD);
        COMM_WORLD.Barrier();

		// // d. Permuter localement les lignes q et k
		if (q != k) lAI.swapRows(q, k);

		// e. Normaliser la ligne k afin que l’élément (k,k) égale 1
		double lValue = lAI(k, k);
		for (j = 0; j < lAI.cols(); ++j) {
			// On divise les éléments de la rangée k
			// par la valeur du pivot.
			// Ainsi, lAI(k,k) deviendra égal à 1.
			lAI(k, j) /= lValue;
		}

		// f. Éliminer les éléments (i,k) pour toutes les lignes i qui appartiennent au processus r, sauf pour la ligne k
		for (size_t i=0; i<lAI.rows(); ++i) {
			if (i != k && i % lProcSize == lRank) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
				double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
			}
		}

	}

	// On copie la partie droite de la matrice AI ainsi transformée
	// dans la matrice courante (this).
	// for (unsigned int i=0; i<iA.rows(); ++i) {
	// 	iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
	// }

}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {
    
    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for(size_t i=0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for(size_t j=0; j < lRes.cols(); ++j) {
            lRes(i,j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char** argv) {
    
	srand((unsigned)time(NULL));
    
	int lS = 5;
	if (argc == 2) {
		lS = atoi(argv[1]);
	}

	MPI::Init();
	int lProcSize = MPI::COMM_WORLD.Get_size();
	int lRank = MPI::COMM_WORLD.Get_rank();
	
	MatrixRandom lA(lS, lS);
	Matrix lB(lA);
	if (lRank == 0)
	{
		COMM_WORLD.Bcast(&lB,lS*lS,MPI::DOUBLE,0);
	}
	cout << "Matrice random:\n" << lB.str() << endl;
    
	invertParallel(lB, lRank, lProcSize);
	cout << "Matrice inverse:\n" << lB.str() << endl;
    
 //    Matrix lRes = multiplyMatrix(lA, lB);
 //    cout << "Produit des deux matrices:\n" << lRes.str() << endl;
    
 //    cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;
	MPI::Finalize(); 
    
	return 0;
}

