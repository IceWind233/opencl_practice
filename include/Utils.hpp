#ifndef UTILS_HPP
#define UTILS_HPP

#include <istream>
#include <map>
#include <memory>

#include <CL/opencl.hpp>

using namespace std;

using buffers_t = std::shared_ptr<map<string, unique_ptr<cl::Buffer>>>;

bool selfChecking();

unique_ptr<cl::Device> getDevice(int device_type = CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU);

bool isClFile(const string &cl_path);

// reading .cl file
shared_ptr<string> cl2KernelFunc(const string &cl_path);

unique_ptr<cl::Program> str2Program(const cl::Context &ctx, const cl::Device device, const string &kernel_fun);

template <typename ...Arg>
void createBuffersHelper(
    buffers_t buffers,
    const string& name,
    const cl::Context ctx,
    int flag,
    size_t size,
    Arg... args) {

    cl::Buffer buffer(ctx, flag, size);

    (*buffers)[name] = make_unique<cl::Buffer>(buffer);
    createBuffersHelper(buffers, args...);
}

template <>
inline void createBuffersHelper(
    buffers_t buffers,
    const string& name,
    const cl::Context ctx,
    int flag,
    size_t size) {
    cl::Buffer buffer(ctx, flag, size);

    (*buffers)[name] = make_unique<cl::Buffer>(buffer);
}

 /**
 * \brief a function to create buffers
 * \tparam Arg: string, cl::Context, int, size_t
 * \param args: name, ctx, type, size 
 * \return 
 */
template <typename ...Arg>
buffers_t createBuffers(Arg... args) {
    buffers_t buffers = make_shared<map<string, unique_ptr<cl::Buffer>>>();
    createBuffersHelper(buffers, args...);
    return buffers;
}

#endif // UTILS_HPP