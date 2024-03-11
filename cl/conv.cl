__kernel void conv(__global const uchar* input_mat, __global const float* ker, __global uchar* output, int mat_w, int mat_h, int k_w, int k_h) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int tmp = 0;
    for(int i = 0; i < k_h; ++i){
        for(int j = 0; j < k_w; ++j){
            int input_x = x + j;
            int input_y = y + i;

            tmp += input_mat[input_y * mat_w + input_x] * ker[i * k_w + j];
        }
    }
     output[y * (mat_w - k_w + 1) + x] = (tmp / k_h / k_w);
}