#include <iostream>
#include <CL/opencl.hpp>

#include "Utils.hpp"

int main(){
    auto device = getDevice(CL_DEVICE_TYPE_GPU);
    cl::Context context({*device});

    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    const auto p_simple_add = cl2KernelFunc("D:/Fragments/DLTest/cl/simple_add.cl");

    const auto program = str2Program(context, *device, *p_simple_add);

    // create buffers on the device
    auto buffers = createBuffers(
        "A", context, CL_MEM_READ_WRITE, sizeof(int) * 10,
        "B", context, CL_MEM_READ_WRITE, sizeof(int) * 10,
        "C", context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, *device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(*(*buffers)["A"], CL_TRUE, 0, sizeof(int) * 10, A);
    queue.enqueueWriteBuffer(*(*buffers)["B"],CL_TRUE,0,sizeof(int)*10,B);


    //run the kernel
    cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(*program, "simple_add"));
    cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
    simple_add(eargs, *(*buffers)["A"], *(*buffers)["B"], *(*buffers)["C"]).wait();

    //alternative way to run the kernel
    /*cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
    queue.finish();*/

    int C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(*(*buffers)["C"],CL_TRUE,0,sizeof(int)*10,C);

    std::cout<<" result: \n";
    for(int i=0;i<10;i++){
        std::cout<<C[i]<<" ";
    }

    return 0;
}