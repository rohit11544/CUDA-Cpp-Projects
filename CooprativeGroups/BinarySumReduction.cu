#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using std::cout;
using std::endl;


__device__ int reduce_block_sum(int* workspace, int val) {
	int id = threadIdx.x;

	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		workspace[id] = val;
		__syncthreads();

		if (id < i) val += workspace[id + i];
		__syncthreads();
	}
	return val;
}


__global__ void SumReduction(int* a, int* r, const int N) {
	extern __shared__ int workspace[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int block_sum = reduce_block_sum(workspace, a[tid]);

	if (threadIdx.x == 0) atomicAdd(&r[0], block_sum);
	
}


void verify_result(int* a, int* r, const int N) {
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += a[i];
	}
	
	if (sum != r[0]) {
		cout << "NOT SUCCESSEFUL" << endl;
		exit(0);
	}
	else {
		cout << "COMPLETED SUCCESSEFULLY" << endl;
	}
	
}


int main() {
	const int N = 1 << 10;

	int* h_a, * h_r;
	size_t bytes = N * sizeof(int);
	h_a = new int[N];
	h_r = new int[N];

	for (int i = 0; i < N; i++) {
		h_a[i] = rand() % 100;
	}

	int* d_a, * d_r;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_r, bytes);

	int thread = 256;
	int grid = (N + thread - 1) / thread;

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	SumReduction << <grid, thread, thread*sizeof(int) >> > (d_a, d_r, N);
	
	cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy(h_r, d_r, bytes, cudaMemcpyDeviceToHost);

	verify_result(h_a, h_r, N);


	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Total time in ms taken to complete kernels : " << milliseconds << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_a);
	cudaFree(d_r);

	delete[] h_a;
	delete[] h_r;

	return 0;
}