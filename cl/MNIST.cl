const size_t kWidth = 28;
const size_t kHeight = 28;
const size_t kImageSize = kHeight * kWidth;
const size_t kOutputSize = 10;

__kernel void eval(
    __global float* image,
    __global float* weight,
    __global float* bias, 
    __global float* res){

    size_t x = get_global_id(0);

    float tmp;

    for(size_t i = 0; i < kImageSize; ++i){
        tmp += (image[i] * weight[x * kImageSize + i]); 
    }

    res[x] = (tmp + bias[x]);
}

__kernel void cvtColor(__global uchar* image_8b, __global float* image_32b){
    size_t x = get_global_id(0);

    image_32b[x] = ((image_8b[x] * 1.) / 255.);        
}