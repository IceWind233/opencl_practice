#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <Utils.hpp>
#include <__msvc_filebuf.hpp>

bool selfChecking() {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.empty()){
        throw runtime_error(" No platforms found. Check OpenCL installation!\n");
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.empty()){
        throw runtime_error(" No devices found. Check OpenCL installation!\n");
    }
    std::cout<< "Using device: "<<all_devices[0].getInfo<CL_DEVICE_NAME>() << "\n";

    return true;
}

unique_ptr<cl::Device> getDevice(int device_type) {
    std::vector<cl::Device> all_devices;

    selfChecking();

    cl::Platform default_platform = cl::Platform::getDefault();
    default_platform.getDevices(device_type, &all_devices);

    auto device = make_unique<cl::Device>(all_devices[0]);

	return device;
}

bool isClFile(const std::string& filePath) {
    std::size_t dotIndex = filePath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        std::string extension = filePath.substr(dotIndex + 1);
        return extension == "cl";
    }

    return false;
}

// using file name for sigle kernel function probably is a good idea?
shared_ptr<string> cl2KernelFunc(const string& cl_path){

    std::ifstream file(cl_path);
    // check file is .cl file 
    if(!isClFile(cl_path)) {
        throw runtime_error(cl_path + "is not .cl file");
    }

    if (!file.is_open()) {
        // print full path of cl file
        throw std::runtime_error("Cannot open file: " + cl_path);
    }

    std::istreambuf_iterator<char> begin(file);
    std::istreambuf_iterator<char> end;

    auto cl_contents = make_shared<string>(string(begin, end));

    return cl_contents;
}

unique_ptr<cl::Program> str2Program(
    const cl::Context &ctx,
    const cl::Device device,
    const string &kernel_fun) {

    cl::Program::Sources src;
    src.emplace_back(kernel_fun.c_str(), kernel_fun.length());

    auto program = make_unique<cl::Program>(cl::Program(ctx, src));
    if (program->build({device}) != CL_SUCCESS) {
	    throw std::runtime_error("Error building: " + program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
	}

    return program;
}