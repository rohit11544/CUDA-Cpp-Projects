#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "Header.h"
using namespace std;

int main() {
	
	const int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	int* h_a, * h_a_v;

	h_a = new int[N];
	h_a_v = new int[N];
	
	generateArray(h_a, N);

	int* d_a, * d_a_v;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_a_v, bytes);
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

	const int TB_SIZE = 256;
	int GRID_SIZE = N / TB_SIZE;

	BaseLine << < GRID_SIZE, TB_SIZE >> > (d_a, d_a_v);
	
	BaseLine << < 1, TB_SIZE >> > (d_a_v, d_a_v);
	
	cudaMemcpy(h_a_v, d_a_v, bytes, cudaMemcpyDeviceToHost);

	verifyResult(h_a, h_a_v, N);
	
	cout << "COMPLETED BaseLine SUCCESSFULLY" << endl;
	
	int* d_a_2, * d_a_v_2;
	cudaMalloc(&d_a_2, bytes);
	cudaMalloc(&d_a_v_2, bytes);
	cudaMemcpy(d_a_2, h_a, bytes, cudaMemcpyHostToDevice);
	
	NoWarpDiversion << < GRID_SIZE , TB_SIZE >> > (d_a_2, d_a_v_2);

	NoWarpDiversion << < 1, TB_SIZE >> > (d_a_v_2, d_a_v_2);

	cudaMemcpy(h_a_v, d_a_v_2, bytes, cudaMemcpyDeviceToHost);

	verifyResult(h_a, h_a_v, N);
	
	cout << "COMPLETED NoWarpDiversion SUCCESSFULLY" << endl;

	int* d_a_3, * d_a_v_3;
	cudaMalloc(&d_a_3, bytes);
	cudaMalloc(&d_a_v_3, bytes);
	cudaMemcpy(d_a_3, h_a, bytes, cudaMemcpyHostToDevice);

	NoBankConflits << < GRID_SIZE , TB_SIZE >> > (d_a_3, d_a_v_3);

	NoBankConflits << < 1, TB_SIZE >> > (d_a_v_3, d_a_v_3);

	cudaMemcpy(h_a_v, d_a_v_3, bytes, cudaMemcpyDeviceToHost);

	verifyResult(h_a, h_a_v, N);

	cout << "COMPLETED NoBankConflits SUCCESSFULLY" << endl;

	int* d_a_4, * d_a_v_4;
	cudaMalloc(&d_a_4, bytes);
	cudaMalloc(&d_a_v_4, bytes);
	cudaMemcpy(d_a_4, h_a, bytes, cudaMemcpyHostToDevice);

	LaunchHalfThreads << < GRID_SIZE / 2, TB_SIZE >> > (d_a_4, d_a_v_4);

	LaunchHalfThreads << < 1, TB_SIZE >> > (d_a_v_4, d_a_v_4);

	cudaMemcpy(h_a_v, d_a_v_4, bytes, cudaMemcpyDeviceToHost);

	verifyResult(h_a, h_a_v, N);

	cout << "COMPLETED LaunchHalfThreads SUCCESSFULLY" << endl;


	delete[] h_a;
	delete[] h_a_v;
	
	cudaFree(d_a);
	cudaFree(d_a_v);
	cudaFree(d_a_2);
	cudaFree(d_a_v_2);
	cudaFree(d_a_3);
	cudaFree(d_a_v_3);
	cudaFree(d_a_4);
	cudaFree(d_a_v_4);



	return 0;

}