#include <iostream>
#include <vector>
#include "cl.hpp"

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    cl::Platform::get(&platforms);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}