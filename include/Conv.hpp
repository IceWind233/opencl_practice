#ifndef CONV_HPP
#define CONV_HPP

#include <CL/opencl.hpp>

#include "Utils.hpp"
#include "opencv2/opencv.hpp"

class Kernel {
public:

	Kernel(float* data, int width, int height) :
	_data(data),
	_width(width),
	_height(height) { }

	int width() const {
		return _width;
	}

	int height() const {
		return _height;
	}

	float* data() const{
		return _data;
	}

private:
	float* _data;
	int _width;
	int _height;

};



// gray scale only
class Conv {
public:
	void operator ()(cl::Context &ctx, unique_ptr<cl::Device> &device, cv::Mat &mat, const Kernel &kernel, cv::Mat &output) {
		auto size = mat.rows * mat.cols * mat.channels();

		if (mat.type() == CV_8UC1) {
			output = cv::Mat(mat.rows - kernel.height() + 1, mat.cols - kernel.width() + 1, CV_8UC1);
			auto output_size = output.rows * output.cols * output.channels();

			auto buffers = 
				createBuffers(
					"input", ctx, CL_MEM_READ_WRITE, size * sizeof(uchar),
					"kernel", ctx, CL_MEM_READ_WRITE, kernel.width() * kernel.height() * sizeof(float),
					"output", ctx, CL_MEM_READ_WRITE, output_size * sizeof(uchar));
			auto cl_kernel = cl2KernelFunc("D:/Fragments/DLTest/cl/conv.cl");
			auto conv = str2Program(ctx, *device, *cl_kernel);

			cl::CommandQueue queue(ctx, *device);

			queue.enqueueWriteBuffer(*(*buffers)["input"], CL_TRUE, 0, size * sizeof(uchar), mat.data);
			queue.enqueueWriteBuffer(*(*buffers)["kernel"], CL_TRUE, 0, kernel.width() * kernel.height() * sizeof(float), kernel.data());

			cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int> conv_kernel(cl::Kernel(*conv, "conv"));
			cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(output.cols, output.rows), cl::NullRange);

			conv_kernel(eargs, *(*buffers)["input"], *(*buffers)["kernel"], *(*buffers)["output"], mat.cols, mat.rows, kernel.width(), kernel.height()).wait();

			queue.enqueueReadBuffer(*(*buffers)["output"], CL_TRUE, 0, output_size * sizeof(uchar), output.data);
		}
	}

private:
};


#endif // CONV_HPP