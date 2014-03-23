
__kernel void invertParallel(__global double* A, unsigned int a, __global double* B) {
	
	int idx = get_global_id(0);

	// turm B into I
	for (unsigned int i = 0; i < a; i++) {
		B[i * a + i] = 1;
	}

	for (int i = 0; i < a; i++) {
		if (A[i * a + i] == 0) {
			double maxval=0;
			int kmax=0;
			for (int k = i + 1; k < a; k++) {
				if ((A[k * a + i] > maxval)) {
					maxval=A[k*a+i];
					kmax=k;
				}
			}
			// échanger la ligne courante avec celle du pivot
			for (int l = 0; l < a; l++) {
				double tempv = A[kmax * a + l];
				A[kmax * a + l] = A[i * a + l];
				A[i * a + l] = tempv;
				tempv = B[kmax * a + l];
				B[kmax * a + l] = B[i * a + l];
				B[i * a + l] = tempv;
			}
		}
		{
			// On divise les éléments de la rangée i
			// par la valeur du pivot.
			// Ainsi, A(i,i) deviendra égal à 1.
			double div = A[i * a + i];

			for (int k = 0; k < a; k++) {
				A[i * a + k] = A[i * a + k] / div;
				B[i * a + k] = B[i * a + k] / div;
			}

			// Pour chaque rangée...(chaque couer est responsable pour une rangée)
			div = A[idx * a + i]/A[i*a+i];
			if(idx!=i) {
				for (int l = 0; l < a; l++) {
					if(l>=i) {
						A[idx * a + l] -= A[i * a + l] * div;
					}
					B[idx * a + l] -= B[i * a + l] * div;

				}
			}

			// Pour chaque rangée...
			// for (int k = 0; k < a; k++) {
			// 	div = A[k * a + i]/A[i*a+i];
			// 	if(k!=i) {
			// 		// On soustrait la rangée i
   //               	// multipliée par l'élément i de la rangée courante (chaque couer est responsable pour une colonne)
   //              	if(idx>=i) {
			// 			A[k * a + idx] -= A[i * a + idx] * div;
			// 		}
			// 		B[k * a + idx] -= B[i * a + idx] * div;
			// 	}
			// }
		}
	}

}