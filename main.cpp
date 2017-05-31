#define __CL_ENABLE_EXCEPTIONS
#define SIZE(sz) BLOCK_SIZE * ((sz + BLOCK_SIZE - 1) / BLOCK_SIZE)

#include "cl.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

const size_t BLOCK_SIZE = 512;

void read_input(size_t &n, vector<float> &vec) {
    FILE *file = fopen("input.txt", "r");

    int d;
    fscanf(file, "%d\n", &d);
    n = (size_t) d;

    vec.resize(n);
    for (int i = 0; i < n; ++i) {
        fscanf(file, "%f", &vec[i]);
    }

    fclose(file);
}

void write_output(const vector<float> &vec, const size_t n) {
    FILE *file = fopen("output.txt", "w");

    for (int i = 0; i < n; ++i) {
        fprintf(file, "%0.3f ", vec[i]);
    }

    fflush(file);
    fclose(file);
}

vector<float> prefix_sum(cl::Context &context, cl::CommandQueue &queue,
                         cl::Program &program, vector<float> &input) {
    cl::Buffer input_buf(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
    queue.enqueueWriteBuffer(input_buf, CL_TRUE, 0, sizeof(float) * input.size(), &input[0]);

    cl::Kernel kernel(program, "scan");
    cl::KernelFunctor scan(kernel, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
    cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * input.size());
    cl::Event event = scan(input_buf, output_buf,
                           cl::__local(sizeof(float) * BLOCK_SIZE), cl::__local(sizeof(float) * BLOCK_SIZE));
    event.wait();

    vector<float> output(input.size(), 0);
    queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);

    if (output.size() == BLOCK_SIZE) {
        return output;
    } else {
        vector<float> tails(SIZE(output.size() / BLOCK_SIZE), 0);
        for (int i = 1; i * BLOCK_SIZE - 1 < output.size() && i < tails.size(); ++i) {
            tails[i] = output[i * BLOCK_SIZE - 1];
        }
        vector<float> tails_prefix = prefix_sum(context, queue, program, tails);
        cl::Buffer inc_input_buf(context, CL_MEM_READ_ONLY, sizeof(float) * tails_prefix.size());
        queue.enqueueWriteBuffer(input_buf, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);
        queue.enqueueWriteBuffer(inc_input_buf, CL_TRUE, 0, sizeof(float) * tails_prefix.size(), &tails_prefix[0]);
        cl::Kernel kernel_inc(program, "inc");
        cl::KernelFunctor inc(kernel_inc, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
        cl::Event inc_event = inc(input_buf, inc_input_buf, output_buf);
        inc_event.wait();

        vector<float> result(input.size(), 0);
        queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float) * result.size(), &result[0]);

        return result;
    }
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

        ifstream cl_file("matrix_scan.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));
        cl::Program program(context, source);

        try {
            ostringstream ost;
            ost << "-D BLOCK_SIZE=" << BLOCK_SIZE;
            program.build(devices, ost.str().c_str());
        } catch (const cl::Error &e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 1;
        }

        size_t n;
        std::vector<float> input_vec;
        read_input(n, input_vec);

        input_vec.resize(SIZE(n));
        vector<float> output_vec = prefix_sum(context, queue, program, input_vec);

        write_output(output_vec, n);
    } catch (cl::Error const &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }
}

