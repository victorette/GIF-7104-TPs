//
//  main.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <mpi.h>

using namespace std;

double * convertirArray(std::valarray<double> tablo) {
	double * tableauRetour = new double[tablo.size()];
	for (size_t i = 0 ; i < tablo.size() ; i++)
		tableauRetour[i] = tablo[i];
	return tableauRetour;
}

std::string afficherArray(std::valarray<double> tablo) {
	std::ostringstream oss;
	oss.precision(2);
	oss << "[";
	for (size_t i = 0 ; i < tablo.size() ; i++)
		oss << tablo[i] << ", ";
	oss << "]";
	return oss.str();
}

std::string convertirArrayEnString(double * tablo, int size) {
	std::ostringstream oss;
	oss.precision(2);
	oss << "[";
	for (int i = 0 ; i < size ; i++)
		oss << tablo[i] << ", ";
	oss << "]";
	return oss.str();
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {
	// vérifier que la matrice est carrée
	assert(iA.rows() == iA.cols());
	// construire la matrice [A I]
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

	// traiter chaque rangée
	for (size_t k = 0 ; k < iA.rows() ; ++k) {
		// trouver l'index p du plus grand pivot de la colonne k en valeur absolue
		// (pour une meilleure stabilité numérique).
		size_t p = k;
		double lMax = fabs(lAI(k,k));
		for(size_t i = k; i < lAI.rows(); ++i) {
			if(fabs(lAI(i,k)) > lMax) {
				lMax = fabs(lAI(i,k));
				p = i;
			}
		}
		// vérifier que la matrice n'est pas singulière
		if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

		// échanger la ligne courante avec celle du pivot
		if (p != k) lAI.swapRows(p, k);

		double lValue = lAI(k, k);
		for (size_t j=0; j<lAI.cols(); ++j) {
			// On divise les éléments de la rangée k
			// par la valeur du pivot.
			// Ainsi, lAI(k,k) deviendra égal à 1.
			lAI(k, j) /= lValue;
		}

		// Pour chaque rangée...
		for (size_t i=0; i<lAI.rows(); ++i) {
			if (i != k) { // ...différente de k
				// On soustrait la rangée k
				// multipliée par l'élément k de la rangée courante
				double lValue = lAI(i, k);
				lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
			}
		}
	}

	// On copie la partie droite de la matrice AI ainsi transformée
	// dans la matrice courante (this).
	for (unsigned int i=0; i<iA.rows(); ++i) {
		iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
	}
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
void invertParallel(Matrix& iA) {
	struct {double val; int ligne;} send, recv;
	// vérifier que la matrice est carrée
	assert(iA.rows() == iA.cols());
	// construire la matrice [A I]
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));
	
	int lRank = MPI::COMM_WORLD.Get_rank();
	int lSize = MPI::COMM_WORLD.Get_size();
	// traiter chaque rangée
	for (size_t k = 0 ; k < iA.rows() ; ++k) {
		// trouver l'index p du plus grand pivot de la colonne k en valeur absolue
		// (pour une meilleure stabilité numérique).
		size_t p = k;
	//	double lMax = fabs(lAI(k,k));
        double lMax = std::numeric_limits<double>::lowest();
	//	std::cout << std::numeric_limits<double>::min() << std::endl;
	//	std::cout << std::numeric_limits<double>::lowest() << std::endl;
		for(size_t i = k ; i < lAI.rows() ; i++) {
			if (i%(lSize) == (unsigned)lRank) { 
				if(fabs(lAI(i,k)) > lMax) {
					lMax = fabs(lAI(i,k));
					p = i;
				}
			}
		}
		send.val = lMax;
		send.ligne = p;

		// On trouve qui detient la plus grande valeur pour le pivot.
		MPI::COMM_WORLD.Allreduce((void *)&send, (void *)&recv, 1, MPI::DOUBLE_INT, MPI::MAXLOC);
		
		std::valarray<double> copieLigneMax = lAI.getRowCopy(recv.ligne);
		std::valarray<double> copieLigneK = lAI.getRowCopy(k);

		double * tableauConvertiMax = convertirArray(copieLigneMax);
		double * tableauConvertiK = convertirArray(copieLigneK);

		MPI::COMM_WORLD.Bcast(tableauConvertiMax, copieLigneMax.size(), MPI::DOUBLE, recv.ligne%lSize);
		MPI::COMM_WORLD.Bcast(tableauConvertiK, copieLigneK.size(), MPI::DOUBLE, k%lSize);
		//std::cout << lRank << "(" << recv.ligne << ")[" << lSize << "]{" << recv.ligne%lSize << "} : " << convertirArrayEnString(tableauConverti, copieLigne.size()) << std::endl;
		for (size_t i = 0 ; i < copieLigneMax.size() ; i++) {
			lAI(recv.ligne, i) = tableauConvertiMax[i];
		}
		for (size_t i = 0 ; i < copieLigneK.size() ; i++) {
			lAI(k, i) = tableauConvertiK[i];
		}

		delete[] tableauConvertiMax;

		// vérifier que la matrice n'est pas singulière
		if (lAI(p, k) == 0) {
			std::cout << "Processus " << lRank << " lance exception." << std::endl;
			throw runtime_error("Matrix not invertible");
		}

		// échanger la ligne courante avec celle du pivot
		if ((unsigned)recv.ligne != k) lAI.swapRows(recv.ligne, k);

		double lValue = lAI(k, k);
		for (size_t j = 0 ; j < lAI.cols() ; ++j) {
			// On divise les éléments de la rangée k
			// par la valeur du pivot.
			// Ainsi, lAI(k,k) deviendra égal à 1.
			lAI(k, j) /= lValue;
		}
		//std::cout << "Apres dimin : " << lRank << std::endl << lAI.str() << std::endl;

		// Pour chaque rangée...
		for (size_t i=0; i<lAI.rows(); ++i) {
			if (i != k) { // ...différente de k
				if ( i%lSize == (unsigned)lRank ) {
					// On soustrait la rangée k
					// multipliée par l'élément k de la rangée courante
					double lValue = lAI(i, k);
					lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
				}
			}
		}
		std::cout << "Apres down : " << lRank << std::endl << lAI.str() << std::endl;
		//std::cout << "Apres Processus " << lRank << std::endl << lAI.str() << std::endl;
	}

	// On copie la partie droite de la matrice AI ainsi transformée
	// dans la matrice courante (this).
	for (unsigned int i = 0 ; i < iA.rows() ; ++i) {
		iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
	}
	
	double * recept = (double *)malloc(lSize * iA.rows() * iA.rows() * sizeof(double));
	int recvcounts = iA.rows() * iA.rows();
	MPI::COMM_WORLD.Barrier();

	double * tablo = convertirArray(iA.getDataArray());
	MPI::COMM_WORLD.Gather(tablo, iA.rows() * iA.rows(), MPI::DOUBLE, recept, recvcounts, MPI::DOUBLE, 0);
	
	if (lRank == 0) {
		int processus = 0;
		int ligne = 0;
		int indiceTableau = 0;
		std::cout.precision(2);
		for (size_t i = 0 ; i < lSize * lAI.rows() * lAI.rows() ; i++) {
			processus = i / (lAI.rows() * lAI.rows());
			ligne = (i % (lAI.rows() * lAI.rows()) / lAI.rows());

			if ((ligne % lSize) == processus)
				iA.getDataArray()[indiceTableau++] = recept[i];
		}
	}

	//double * tablo = convertirArray(lAI.getDataArray());
	//std::cout << "Processus " << lRank << " " << afficherArray(lAI.getDataArray()) << std::endl;
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
	MPI::Init();

	int lRank = MPI::COMM_WORLD.Get_rank();
	int lSize = MPI::COMM_WORLD.Get_size();

	double tempsDebut, seed;

	if (lRank == 0) {
		tempsDebut = MPI_Wtime();
		seed = tempsDebut;
		seed = 0;
		std::cout << lSize << " processus demarre." << std::endl;
	}

	MPI::COMM_WORLD.Bcast(&seed, 1, MPI::DOUBLE, 0);
	srand((unsigned)seed);

	unsigned int lDimension = 5;
	if (argc == 2) {
		lDimension = atoi(argv[1]);
	}

	MatrixRandom matrice(lDimension, lDimension);
	Matrix matriceInverse(matrice);

	invertParallel(matriceInverse);
	//invertSequential(matriceInverse);


	Matrix lDot = multiplyMatrix(matrice, matriceInverse);

	MPI::COMM_WORLD.Barrier();

	if (lRank == 0) {
		Matrix matriceInverseSeq(matrice);
		invertSequential(matriceInverseSeq);
		std::cout << std::endl << std::endl << "Sortie : " << std::endl;
		std::cout << "Matrice aleatoire : " << std::endl << matrice.str() << std::endl;
		std::cout << "Matrice Inverse : " << std::endl << matriceInverse.str() << std::endl;
		std::cout << "Matrice Inverse (solution) : " << std::endl << matriceInverseSeq.str() << std::endl;
		//std::cout << "Produit des matrices : " << std::endl << lDot.str() << std::endl;

		std::cout << "Erreur " << lDot.getDataArray().sum() - lDimension << std::endl;
	}

	if (lRank == 0) {
		std::cout << MPI_Wtime() - tempsDebut << " secondes ecoule." << std::endl;
	}

	MPI::Finalize();
/*
	srand((unsigned)time(NULL));
    
	unsigned int lS = 5;
	if (argc == 2) {
		lS = atoi(argv[1]);
	}
    
	MatrixRandom lA(lS, lS);
	cout << "Matrice random:\n" << lA.str() << endl;
    
    Matrix lB(lA); 
    invertSequential(lB);
	cout << "Matrice inverse:\n" << lB.str() << endl;
   
    Matrix lRes = multiplyMatrix(lA, lB);
    cout << "Produit des deux matrices:\n" << lRes.str() << endl;
    
    cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;
*/    
	return 0;
}

