#include "MNIST.hpp"

Mnist::Mnist(const std::string& weight_path, const std::string& bias_path) :
	weight_data(readWeight(weight_path)),
	bias_data(readBias(bias_path)) {

	init();
}

size_t Mnist::predict(const std::string& image_path) {
	uchar *image = readImage(image_path);
	const string cl_path = "D:/Fragments/DLTest/cl/MNIST.cl";

	size_t eval_y = doPredict(image, cl_path);

	return eval_y;
}

size_t Mnist::predict(const uchar* bytes) {
	const string cl_path = "D:/Fragments/DLTest/cl/MNIST.cl";

	size_t eval_y = doPredict(bytes, cl_path);

	return eval_y;
}

/*************************/

void Mnist::init() {
	selfChecking();

	device_ = getDevice(CL_DEVICE_TYPE_GPU);
	context_ = cl::Context(*device_);
}

std::string Mnist::readWeight(const std::string& weight_path) {
	std::ifstream weight_file(weight_path, std::ios::binary);

	if (!weight_file.is_open()) {
		throw std::runtime_error("Could not open file: " + weight_path);
	}

	auto str = std::string((std::istreambuf_iterator<char>(weight_file)), std::istreambuf_iterator<char>());

	return str;
}

std::string Mnist::readBias(const std::string& bias_path) {
	std::ifstream bias_file(bias_path, std::ios::binary);

	if (!bias_file.is_open()) {
		throw std::runtime_error("Could not open file: " + bias_path);
	}

	return std::string((std::istreambuf_iterator<char>(bias_file)), std::istreambuf_iterator<char>());
}

buffers_t Mnist::buffers() const {
	return createBuffers(
		"image_8b", context_, CL_MEM_READ_WRITE, kImageSize * sizeof(uchar),
		"image_32b", context_, CL_MEM_READ_WRITE, kImageSize * sizeof(float),
		"weight", context_, CL_MEM_READ_WRITE, kImageSize * kOutputSize * sizeof(float),
		"bias", context_, CL_MEM_READ_WRITE, kOutputSize * sizeof(float),
		"result", context_, CL_MEM_READ_WRITE, kOutputSize * sizeof(float)
	);
}

size_t Mnist::doPredict(const uchar* image, const string& cl_path) const {
	auto buffer = buffers();
	auto program = getKernel(cl_path);

	cvtColor(*program, buffer, image);
	float* res = predictHelper(*program, buffer);
	// print output tensor
	cout << "Output tensor: [";
	for (size_t i = 0; i < 10; i++) {
		cout << res[i] << ", ";
	}
	cout << "]" << endl;

	size_t max = 0;
	for(size_t i = 0; i < kOutputSize; ++i) {
		max = res[max] > res[i] ? max : i;
	}

	delete[] res;

	return max;
}

uchar* Mnist::readImage(const std::string& image_path) {
	image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	return image.data;
}

void Mnist::cvtColor(const cl::Program& program, const buffers_t& buffers, const uchar* image) const {
	cl::CommandQueue queue{context_, *device_};
	queue.enqueueWriteBuffer(*(*buffers)["image_8b"], CL_TRUE, 0, kImageSize * sizeof(uchar), image);

	cl::compatibility::make_kernel<cl::Buffer, cl::Buffer> cvtColor(program, "cvtColor");
	cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(kImageSize), cl::NDRange(1));

	cvtColor(eargs, *(*buffers)["image_8b"], *(*buffers)["image_32b"]).wait();

	float *a = new float[kImageSize];

	queue.enqueueReadBuffer(*(*buffers)["image_32b"], CL_TRUE, 0, kImageSize * sizeof(float), a);

	for (size_t i = 0; i < kImageSize; i++) {
		cout << a[i] << ", ";
	}
}

float* Mnist::predictHelper(cl::Program& program, buffers_t& buffers) const {
	cl::CommandQueue queue{context_, *device_};

	cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> eval(program, "eval");
	cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(kOutputSize), cl::NDRange(1));

	queue.enqueueWriteBuffer(*(*buffers)["weight"], CL_TRUE, 0, kImageSize * kOutputSize * sizeof(float), weight_data.c_str());
	queue.enqueueWriteBuffer(*(*buffers)["bias"], CL_TRUE, 0, kOutputSize * sizeof(float), bias_data.c_str());

	float* result = new float[kOutputSize];

	eval(eargs, *(*buffers)["image_32b"], *(*buffers)["weight"], *(*buffers)["bias"], *(*buffers)["result"]).wait();

	queue.enqueueReadBuffer(*(*buffers)["result"], CL_TRUE, 0, kOutputSize * sizeof(float), result);

	return result;
}

unique_ptr<cl::Program> Mnist::getKernel(const string& path) const {
	auto str = cl2KernelFunc(path);
	return str2Program(context_, *device_, *str);
}
