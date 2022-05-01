// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <cmath>
#include "gputimer.h"
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <cuda_profiler_api.h>
using namespace std;

// Function to read data from file
float* read_from_file(string file_name){
  // lets get filesize
  struct stat results;
  if (stat(file_name.c_str(), &results) != 0){
    // An error occurred
    std::cout << "ERROR: unable to get filesize" << std::endl;
    return NULL;
  }
  // The size of the file in bytes is in results.st_size
  // Lets allocate an array to contain the binary file
  //std::cout << "Filename: " << file_name << "Size:" <<  results.st_size << std::endl;
  float* data = (float *)malloc(results.st_size);


  // lets write it to binary file
  ifstream infile;

  // open a binary file
  infile.open(file_name, ios::binary | ios::in);

  //read data from file
  infile.read((char*) data, results.st_size);

  // close the file
  infile.close();

  return data;
}



// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Randomization helpers
// adapted from https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/rocm-3.0/clients/include/rocblas_init.hpp#L42

void fill_sin(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = sin(float(i + j * nr_rows_A));
}


void fill_cos(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = cos(float(i + j * nr_rows_A));
}

int def(int value, int device) {

  cudaSetDevice(device);
  cudaStream_t computeStream;
  cudaError_t result;
  result = cudaStreamCreate(&computeStream);

	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = value;

        float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	float *d_A;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));

  string fname_A = string("host_A_") + to_string(value) +string(".bin");
  float *h_A = read_from_file(fname_A);

  GpuTimer timer;
    timer.Start();
	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpyAsync(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice, computeStream);
	//std::cout << "A =" << std::endl;
	//print_matrix(h_A, nr_rows_A, nr_cols_A);
    timer.Stop();
	std::cout <<"CudaMemCpy " << i << " Runtime = " << timer.Elapsed() << std::endl;
  }
  cudaProfilerStop();

	// Copy (and print) the result on host memory
	cudaMemcpyAsync(h_C,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost, computeStream);
	//std::cout << "C =" << std::endl;
	//print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	cudaFree(d_A);

  result = cudaStreamDestroy(computeStream);

	// Free CPU memory
	free(h_A);
	free(h_C);

	return 0;
}

int main(int argc, char* argv[]){
	// for (int i=100; i <= 100000; i = i*10){
	// 	std::cout << "\n\n\n" << i << "\n";
	// 	def(1024, i);
	// }
	if (argc != 4){
		std::cout << "Usage: mul <dim> <target-device num>" << std::endl;
		exit(-1);
	}
	int dim = atoi(argv[1]);
	int device = atoi(argv[2]);
	//cout << dim <<
	def(dim, device);
	return 0;
}
