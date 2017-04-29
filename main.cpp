#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

void read_input(int &n, int &m, float *&a, float *&b) {
    FILE *const input_file = fopen("input.txt", "r");

    fscanf(input_file, "%d %d\n", &n, &m);

    a = new float[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fscanf(input_file, "%f", &a[i * n + j]);
        }
    }

    b = new float[m * m];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            fscanf(input_file, "%f", &b[i * m + j]);
        }
    }

    fclose(input_file);
}

void write_output(const float *const c, const int n) {
    FILE *const output_file = fopen("output.txt", "w");

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(output_file, "%0.3f ", c[i * n + j]);
        }
        fprintf(output_file, "\n");
    }

    fflush(output_file);
    fclose(output_file);
}

int main() {
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Context context(devices);

        cl::CommandQueue queue(context, devices.front(), CL_QUEUE_PROFILING_ENABLE);

        ifstream cl_file("matrix_conv.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

        cl::Program program(context, source);

        try {
            program.build(devices, "-D BLOCK_SIZE=16");
        } catch (cl::Error const &e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 1;
        }

        int n, m;
        float *a, *b;
        read_input(n, m, a, b);
        float c[n * n];

        const int block_size = 16;
        const int matrix_size_a = n * n;
        const int matrix_size_b = m * m;
        const int matrix_size_c = n * n;

        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size_a);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size_b);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size_c);

        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * matrix_size_a, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * matrix_size_b, b);

        cl::Kernel kernel(program, "matrix_conv");
        kernel.setArg(0, dev_a);
        kernel.setArg(1, dev_b);
        kernel.setArg(2, dev_c);
        kernel.setArg(3, n);
        kernel.setArg(4, m);
        const int N = block_size * ((n + block_size - 1) / block_size);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange((size_t) N, (size_t) N),
                                   cl::NDRange(block_size, block_size));

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * matrix_size_c, c);

        write_output(c, n);
    } catch (cl::Error const &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}