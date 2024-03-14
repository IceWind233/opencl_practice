#pragma once

#include <string>
#include <fstream>

#include <CL/opencl.hpp>

#include <opencv2/opencv.hpp>

#include "Utils.hpp"

constexpr size_t kWidth = 28;
constexpr size_t kHeight = 28;
constexpr size_t kImageSize = kHeight * kWidth;
constexpr size_t kOutputSize = 10;

class Mnist{
public:
    Mnist() = delete;

    Mnist(const std::string& weight_path, const std::string& bias_path);

    ~Mnist() = default;

    size_t predict(const std::string& image_path);

    size_t predict(const uchar* bytes);

private:

    void init();

    std::string readWeight(const std::string& weight_path);

    std::string readBias(const std::string& bias_path);

    buffers_t buffers() const;

    size_t doPredict(const uchar* image, const string &cl_path) const;

    uchar* readImage(const std::string& image_path);

    unique_ptr<cl::Program> getKernel(const string& path) const;

    void cvtColor(const cl::Program& program, const buffers_t& buffers, const uchar* image) const;

    float* predictHelper(cl::Program& program, buffers_t& buffers) const;

private:

    std::string weight_data;
    std::string bias_data;

    std::unique_ptr<cl::Device> device_;
    cl::Context context_;

    cv::Mat image{};
};
