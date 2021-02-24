#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Header.h"


#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];


int main() {
	const int N = 1 << 20, M = 7;
	size_t bytes = N * sizeof(int);

	int* h_a, * h_m, * h_r;
	h_a = new int[N];
	h_r = new int[N];
	h_m = new int[M];

	for (int i = 0; i < N; i++) {
		h_a[i] = rand() % 100;
	}
	
	for (int i = 0; i < M; i++) {
		h_m[i] = rand() % 10;
	}

	int* d_a, * d_m, * d_r;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_m, M * sizeof(int));
	cudaMalloc(&d_r, bytes);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, h_m, M*sizeof(int), cudaMemcpyHostToDevice);

	int thread_size = 256;
	int grid_size = (N + thread_size - 1) / thread_size;

	BaseLine << <grid_size, thread_size >> > (d_a, d_m, d_r, N, M);

	cudaMemcpy(h_r, d_r, bytes, cudaMemcpyDeviceToHost);

	verify_result(h_a, h_m, h_r, M, N);

	std::cout << "COMPLETED SUCCESSFUL" << std::endl;
	
	int* d_r_2;
	cudaMalloc(&d_r_2, M * sizeof(int));
	
	cudaMemcpyToSymbol(mask, h_r, M * sizeof(int));

	ConstantMem << <grid_size, thread_size >> > (d_a, d_r_2, N);

	cudaMemcpy(h_r, d_r_2, bytes, cudaMemcpyDeviceToHost);

	verify_result(h_a, h_m, h_r, M, N);

	std::cout << "COMPLETED SUCCESSFUL" << std::endl;
	

	cudaFree(d_a);
	cudaFree(d_r);
	cudaFree(d_m);
	cudaFree(d_r_2);

	delete[] h_a;
	delete[] h_r;
	delete[] h_m;
	 
	return 0;
}

