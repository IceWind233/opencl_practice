#include <iostream>
#include <CL/opencl.hpp>

#include "Utils.hpp"
#include "Conv.hpp"

int main(){
    auto device = getDevice(CL_DEVICE_TYPE_GPU);
    cl::Context context({*device});

	cv::Mat mat = cv::imread("D:/DeepSORT/asset/bus.jpg", cv::IMREAD_GRAYSCALE);

    Kernel kernel = {new float[9], 3, 3};

    for (size_t i = 0; i < kernel.height(); i++) {
        for (size_t j = 0; j < kernel.width(); j++) {
	        kernel.data()[i * kernel.width() + j] = 1.0f;
		}
    }

    cv::Mat output;
    Conv()(context, device, mat, kernel, output);

    cv::namedWindow("output", cv::WINDOW_NORMAL);
    imshow("output", output);
    cv::waitKey(0); 

    return 0;
}