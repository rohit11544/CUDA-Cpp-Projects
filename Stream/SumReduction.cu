#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

#define SHMEM_SIZE 256

__global__ void BaseLine(int* a, int* a_v) {
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = a[tid];
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		a_v[blockIdx.x] = partial_sum[0];
	}

}

void generateArray(int* a, const int N) {
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
	}
}


void verifyResult(int* a, int* a_v, const int n) {
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += a[i];
	}
	if (a_v[0] != sum) {
		cout << "NOT SUCCESSFUL" << endl;
		exit(0);
	}
	cout << "COMPLETED SUCCESSFULLY" << endl;
}


int main() {

	const int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	int* h_1, * h_v1;
	int* h_2, * h_v2;
	int* h_3, * h_v3;

	h_1 = new int[N];
	h_v1 = new int[N];
	h_2 = new int[N];
	h_v2 = new int[N];
	h_3 = new int[N];
	h_v3 = new int[N];

	generateArray(h_1, N);
	generateArray(h_2, N);
	generateArray(h_3, N);

	int* d_1, * d_v1;
	int* d_2, * d_v2;
	int* d_3, * d_v3;

	cudaMalloc(&d_1, bytes);
	cudaMalloc(&d_v1, bytes);
	cudaMalloc(&d_2, bytes);
	cudaMalloc(&d_v2, bytes);
	cudaMalloc(&d_3, bytes);
	cudaMalloc(&d_v3, bytes);

	const int TB_SIZE = 256;
	int GRID_SIZE = N / TB_SIZE;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start recording
	cudaEventRecord(start);

	cudaMemcpy(d_1, h_1, bytes, cudaMemcpyHostToDevice);
	BaseLine << < GRID_SIZE, TB_SIZE >> > (d_1, d_v1);
	BaseLine << < 1, TB_SIZE >> > (d_v1, d_v1);
	cudaMemcpy(h_v1, d_v1, bytes, cudaMemcpyDeviceToHost);

	cudaMemcpy(d_2, h_2, bytes, cudaMemcpyHostToDevice);
	BaseLine << < GRID_SIZE, TB_SIZE >> > (d_2, d_v2);
	BaseLine << < 1, TB_SIZE >> > (d_v2, d_v2);
	cudaMemcpy(h_v2, d_v2, bytes, cudaMemcpyDeviceToHost);

	cudaMemcpy(d_3, h_3, bytes, cudaMemcpyHostToDevice);
	BaseLine << < GRID_SIZE, TB_SIZE >> > (d_3, d_v3);
	BaseLine << < 1, TB_SIZE >> > (d_v3, d_v3);
	cudaMemcpy(h_v3, d_v3, bytes, cudaMemcpyDeviceToHost);

	
	cudaDeviceSynchronize();
	//stop recording
	cudaEventRecord(stop);

	verifyResult(h_1, h_v1, N);
	verifyResult(h_2, h_v2, N);
	verifyResult(h_3, h_v3, N);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Total time in ms taken to complete kernels : " << milliseconds << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delete[] h_1;
	delete[] h_v1;
	delete[] h_2;
	delete[] h_v2;
	delete[] h_3;
	delete[] h_v3;

	cudaFree(d_1);
	cudaFree(d_v1);
	cudaFree(d_2);
	cudaFree(d_v2);
	cudaFree(d_3);
	cudaFree(d_v3);

	return 0;

}