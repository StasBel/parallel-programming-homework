__kernel void scan(__global const float * input, __global float * output, __local float * a, __local float * b) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int block_size = get_local_size(0);
    a[local_id] = b[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 1; s < block_size; s <<= 1) {
        if (local_id > s - 1) {
            b[local_id] = a[local_id] + a[local_id - s];
        } else {
            b[local_id] = a[local_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __local float* tmp = a;
        a = b;
        b = tmp;
    }
    output[global_id] = a[local_id];
}

__kernel void inc(__global const float * input, __global const float * add, __global float * output) {
    int global_id = get_global_id(0);
    int block_size = get_local_size(0);
    output[global_id] = input[global_id] + add[global_id / block_size];
}