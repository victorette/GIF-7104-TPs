
__kernel void invMatr(__global float* A, const unsigned int a, __global float* B) {
	
	// turm B into I
	for (int i = 0; i < a; i++) {
		B[i * a + i] = 1;
	}

	for (int i = 0; i < a; i++) {
		if (A[i * a + i] == 0) {
			float maxval=0;
			int kmax=0;
			for (int k = i + 1; k < a; k++) {
				if ((A[k * a + i] > maxval)||(A[k * a + i]<maxval)) {
					maxval=A[k*a+i];
					kmax=k;
				}
			}
			for (int l = 0; l < a; l++) {
				float tempv = A[kmax * a + l];
				A[kmax * a + l] = A[i * a + l];
				A[i * a + l] = tempv;
				tempv = B[kmax * a + l];
				B[kmax * a + l] = B[i * a + l];
				B[i * a + l] = tempv;
			}
		}
		{
			float div = A[i * a + i];

			for (int k = 0; k < a; k++) {
				A[i * a + k] = A[i * a + k] / div;
				B[i * a + k] = B[i * a + k] / div;
			}

			for (int k = 0; k < a; k++) {
				div = A[k * a + i]/A[i*a+i];
				if(k!=i) {
					for (int l = 0; l < a; l++) {
						if(l>=i) {
							A[k * a + l] -= A[i * a + l] * div;
						}
						B[k * a + l] -= B[i * a + l] * div;

					}
				}
			}
		}
	}

}