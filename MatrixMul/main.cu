#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include "Header.h"

using namespace std;

main() {
	const int N = 1 << 10;
	size_t bytes = N * N * sizeof(int);

	vector<int> h_a(N * N);
	vector<int> h_b(N * N);
	vector<int> h_c(N * N);

	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	int* d_a, * d_b, * d_c;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	int THREADS = 32;
	int BLOCKS = N / THREADS;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	baseLine << <blocks, threads >> > (d_a, d_b, d_c, N);
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	verify_result(h_a, h_b, h_c, N);
	cout << "COMPLETED BASELINE SUCCESSFULLY\n";
	
	TiledMatrixMul << <blocks, threads >> > (d_a, d_b, d_c);
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	verify_result(h_a, h_b, h_c, N);
	cout << "COMPLETED TILEING SUCCESSFULLY\n";

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}