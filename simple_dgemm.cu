// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <cmath>


// Randomization helpers 
// adapted from https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/rocm-3.0/clients/include/rocblas_init.hpp#L42

void fill_sin(double *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = sin(float(i + j * nr_rows_A));
}


void fill_cos(double *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = cos(float(i + j * nr_rows_A));
}

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul( cublasHandle_t handle, const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Do the actual multiplication
  cublasStatus_t err = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  if (err != CUBLAS_STATUS_SUCCESS)
  	std::cout << "Error: " <<  _cudaGetErrorEnum(err) << std::endl;
  
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const double *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int def(int value, int reps, int device) {

  cudaSetDevice(device);
  cudaStream_t computeStream;
  cudaError_t result;
  result = cudaStreamCreate(&computeStream);

	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = value;
	
	double *h_A = (double *)malloc(nr_rows_A * nr_cols_A * sizeof(double));
	double *h_B = (double *)malloc(nr_rows_B * nr_cols_B * sizeof(double));
	double *h_C = (double *)malloc(nr_rows_C * nr_cols_C * sizeof(double));

	// Allocate 3 arrays on GPU
	double *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(double));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(double));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(double));

	// cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	fill_sin(h_A, nr_rows_A, nr_cols_A);
	fill_cos(h_B, nr_rows_B, nr_cols_B);

	// We have useful values in A and B you can copy them in GPU:
	cudaMemcpyAsync(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice, computeStream);
	cudaMemcpyAsync(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(double),cudaMemcpyHostToDevice, computeStream);
	std::cout << "A =" << std::endl;
	std::cout << "B =" << std::endl;

  // Create a handle for CUBLAS
	cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, computeStream);

  for (int i=0; i< reps; i++){
	// Multiply A and B on GPU
  gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  
  cudaStreamSynchronize(computeStream);  
  }

	// Destroy the handle
	cublasDestroy(handle);

	// Copy (and print) the result on host memory
	cudaMemcpyAsync(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost, computeStream);
	std::cout << "C =" << std::endl;
	// print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
    cudaFree(d_C);
  
  result = cudaStreamDestroy(computeStream);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

int main(int argc, char* argv[]){
	// for (int i=100; i <= 100000; i = i*10){
	// 	std::cout << "\n\n\n" << i << "\n"; 
	// 	def(1024, i);
	// }
	if (argc != 4){
		std::cout << "Usage: mul <dim> <reps> <target-device num>" << std::endl;
		exit(-1);
	}
	int dim = atoi(argv[1]);
	int reps = atoi(argv[2]);
	int device = atoi(argv[3]);
	//cout << dim <<
	def(dim, reps, device);
	return 0;
}
