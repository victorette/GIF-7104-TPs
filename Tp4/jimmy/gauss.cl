__kernel void reduce(__global float* buffer,
                     __local float* scratch,
                     __const int length,
                     __global float* result) {
    
    int global_index = get_global_id(0);
    
    //buffer[global_index] = fabs(buffer[global_index]);
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
    
    for(int offset = get_local_size(0) / 2 ; offset > 0 ; offset = offset / 2) {
        if (local_index < offset) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = (mine > other) ? mine : other;
            //scratch[local_index] = (fabs(mine) > fabs(other)) ? 1 : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}


__kernel void divideVector(__global float* buffer,
                           __const float divide) {
    int global_index = get_global_id(0);
    buffer[global_index] /= divide;
    
}

__kernel void eliminateColumnVector(__global float* buffer,
                                    __global float* bufferIdent,
                                    __const int bufferLineSize,
                                    __const int maxPos,
                                    __const int currentLine,
                                    __global float *divide,
                                    __global float *divideIdent) {
    int global_index = get_global_id(0);
    
    int k = (global_index / bufferLineSize);
    float max = buffer[maxPos + k * bufferLineSize];
    
    buffer[global_index] -=     (max * divide[global_index % bufferLineSize]);
    bufferIdent[global_index] -= (max * divideIdent[global_index % bufferLineSize]);
}
