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
	int q, lNBlocks = 0;
	struct
	{
		double value;
		int ligne;
	} send, recv;

	// vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    lDataSize = iA.cols();

	// construire la matrice [A I]
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

	// Pour chaque étape k: 0..n-1 de l’algorithme
	for (k = 0; k < lDataSize; ++k) {
		// a. Déterminer localement le q parmi les lignes qui appartiennent à r, puis
		// faire une reduction (Allreduce avec MAXLOC) pour déterminer le q global
		q = k;
        double lMax = fabs(lAI(k,k));
        for(i = lRank; i < lAI.rows(); ++i) {
        	// cout << "i : " << i << " lProcSize : " << lProcSize << " i % lProcSize : " << i % lProcSize << endl;
        	if (i % lProcSize == (unsigned)lRank)
        	{
	            if(fabs(lAI(i,k)) > lMax) {
	                lMax = fabs(lAI(i,k));
	                q = i;
	            }
        	}
        }
        send.value = lAI(q, k);
        send.ligne = q;

		COMM_WORLD.Allreduce((void *)&send,(void *)&recv,1,MPI::DOUBLE_INT,MPI::MAXLOC);

        if (lRank == 0) {
			
			// b. Si la valeur du max est nulle, la matrice est singulière (ne peut être
			// inversée)
			if (lAI(recv.ligne, k) == 0) throw runtime_error("Matrix not invertible");

        } 
		// c. Diffuser (Bcast) la ligne q appartenant au processus r=q%p (r est root)
    	valarray<double> copieLigne = lAI.getRowCopy(recv.ligne);
        double * tableauConverti = &copieLigne[0];
		MPI::COMM_WORLD.Bcast(tableauConverti, lAI.cols(), MPI::DOUBLE, recv.ligne % lProcSize);

		for (size_t i = 0 ; i < lAI.cols() ; i++) {
			// if (lRank != recv.ligne % lProcSize)
			// std::cout << "Rank " << lRank << " : " << lAI(k, i) << "; ";
			lAI(recv.ligne, i) = tableauConverti[i];
		}
		// delete[] tableauConverti;
		
		// d. Permuter localement les lignes q et k
		if (recv.ligne != k) lAI.swapRows(recv.ligne, k);
		
		MPI::COMM_WORLD.Barrier();

		// e. Normaliser la ligne k afin que l’élément (k,k) égale 1
		double lValue = lAI(k, k);
		for (j = 0; j < lAI.cols(); ++j) {
			// On divise les éléments de la rangée k
			// par la valeur du pivot.
			// Ainsi, lAI(k,k) deviendra égal à 1.
			lAI(k, j) /= lValue;
		}

		// f. Éliminer les éléments (i,k) pour toutes les lignes i qui appartiennent au processus r, sauf pour la ligne k
		for (size_t i = 0; i<lAI.rows(); ++i) {
			if (i != k) { // ...différente de k
				if (i % lProcSize == lRank)
				{
	                // On soustrait la rangée k
	                // multipliée par l'élément k de la rangée courante
					double lValue = lAI(i, k);
	                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
				}
			}
		}
	}

	// int allnLignes, allstarts;

	// int nLignes = (lAI.rows() + lRank)/lProcSize;
 //    int start = 0;
	int gsize,sendarray[lAI.rows()][lAI.cols()],*sptr;
	double lData[lAI.rows()][lAI.cols()];
	for (i = 0; i<lAI.rows(); ++i) {
		if (i % lProcSize == lRank){
			lNBlocks += 1;
		}
		for (j = 0; j<lAI.cols(); ++j) {
			lData[i][j] = lAI(i,j);
			sendarray[i][j] = lAI(i,j);
		}
	}

    int root, *stride, myrank, bufsize;
    double *rbuf;
    MPI_Datatype stype;
    int *displs,*rcounts,offset;

    root = 0;
    gsize = lProcSize;
    myrank = lRank;
    stride = (int *)malloc(gsize*sizeof(int));

    displs = (int *)malloc(gsize*sizeof(int));
	rcounts = (int *)malloc(gsize*sizeof(int));
	offset = 0;
	for (i=0; i<gsize; ++i) {
		displs[i] = offset;
		offset += stride[i];
		rcounts[i] = lAI.rows()-i;
	}
	
	bufsize = lNBlocks * lAI.cols() * lProcSize;//displs[gsize-1]+rcounts[gsize-1];
    rbuf = (double *)malloc(bufsize*sizeof(double));
      /* Create datatype for the column we are sending
       */
    // cout << "lNBlocks: " << lNBlocks << endl;
	MPI_Type_vector(lNBlocks, 1, lAI.cols(), MPI::DOUBLE, &stype);
	MPI_Type_commit( &stype );
	sptr = &sendarray[0][myrank];
	MPI_Gatherv(sptr, 1, stype, rbuf, rcounts, displs, MPI::DOUBLE, root, MPI_COMM_WORLD);

/*
	MPI_Datatype lType;
	MPI_Type_vector(lAI.rows(), 1, lAI.cols(), MPI::INT, &lType);
	MPI_Type_commit(&lType);
	// lType = MPI::INT.create_vector(lAI.cols(), lAI.cols(), 0);
	//lType.commit();
	int lStride = 20;//(lProcSize - 1)*lAI.cols();
	// double * lRecvBuf = (double *)malloc(lProcSize * lAI.rows() * lAI.cols() * sizeof(double));
	// int recvcounts = lAI.rows() * lAI.cols();
	double* lRecvBuf = new double[lProcSize * lStride]; 
	int* lDisps = new int[lProcSize];
	int* lCounts = new int[lProcSize]; 
	
	for (int i=0; i<lProcSize; ++i) {
		lCounts[i] = lAI.rows(); 
		lDisps[i] = i*lStride; 
	}
	MPI_Gatherv(&lData, 1, lType, lRecvBuf, lCounts, lDisps, MPI::DOUBLE, 0, MPI_COMM_WORLD);
	if (lRank == 0) {
		cout << "lRecvBuf:\n" << endl;
		for ( i = 0; i < lProcSize * lStride; i++ ) {
			cout << lRecvBuf[i] << ", ";
		}
		// cout << "Matrice inverse:\n" << lAI.str() << endl;
	} else {
		cout << "Matrice inverse:\n" << lAI.str() << endl;
	}
	*/
	// free(lData);
	// MPI_Type_free(&lType);
	// MPI_Gatherv(&lAI, 1, lType, lRecvBuf, lCounts, lDisps, MPI::DOUBLE, 0, MPI_COMM_WORLD);
	// MPI_Gatherv(lAI, 1, lType, lRecvBuf, recvcounts, allstarts, MPI::INT, 0, MPI_COMM_WORLD);
	// On copie la partie droite de la matrice AI ainsi transformée
	// dans la matrice courante (this).
	for (unsigned int i=0; i<iA.rows(); ++i) {
		iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
	}

	// 2. Rapatrier (Gatherv) toutes les lignes sur le processus 0
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
		// COMM_WORLD.Bcast(&lB,lS*lS,MPI::DOUBLE,0);
	}
	cout << "Matrice random:\n" << lB.str() << endl;
    
	invertParallel(lB, lRank, lProcSize);
	// cout << "Matrice inverse:\n" << lB.str() << endl;
    
 //    Matrix lRes = multiplyMatrix(lA, lB);
 //    cout << "Produit des deux matrices:\n" << lRes.str() << endl;
    
 //    cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;
	MPI::Finalize(); 
    
	return 0;
}

