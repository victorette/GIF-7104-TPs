__kernel void reduce(__global float* buffer,
            __local float* scratch,
            __const int length,
            __global float* result) {
    
    int global_index = get_global_id(0);
    float accumulator = -INFINITY;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float element = buffer[global_index];
        accumulator = (accumulator > element) ? accumulator : element;
        global_index += get_global_size(0);
    }
    
    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset = offset / 2) {
        if (local_index < offset) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = (mine > other) ? mine : other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}

__kernel void invMatr(__global float* A, const unsigned int a, __global float* B) {
	
	// turm B into I
	//for (int i = 0; i < a*a; i++) {
		//B[i] = 0
        B[get_global_id(0)] = get_local_id(0);
        //B[get_local_id(0)] = get_local_id(0);
	//}
    /*
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
     */
}