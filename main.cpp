#include <iostream>
#include <chrono>

#include <CL/opencl.hpp>

#include "Utils.hpp"
#include "Conv.hpp"
#include "MNIST.hpp"

struct data_loader {
    std::vector<int> label{};
    std::vector<std::array<int, kImageSize>> image{};
};


int main(){
    const string weight = "D:/Fragments/DLTest/asset/weight.txt";
    const string bias = "D:/Fragments/DLTest/asset/bias.txt";

	Mnist mnist(weight, bias);

    for (int i = 0; i < 10; i++) {
        stringstream ss;
        ss << "D:/Fragments/DLTest/asset/" << i << ".jpg";

        auto start = std::chrono::high_resolution_clock::now();

        auto res = mnist.predict(ss.str());

		std::cout << "Test " << i << " : " << res << std::endl;

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << " time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << std::endl;
	}

    return 0;
}