__kernel void simple_add(__global const int* A, __global const int* B, __global int* C){
    C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];              
}

__kernel void simple_minus(__global const int* a, __global const int* b, __global int* c){
    size_t x = get_global_id(0);

    c[x] = a[x] - b[x];
}