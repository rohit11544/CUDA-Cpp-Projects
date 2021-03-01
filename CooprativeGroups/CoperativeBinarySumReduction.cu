#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using std::cout;
using std::endl;
using namespace cooperative_groups;
#define THREADS 32 

__device__ int reduce_block_sum(thread_group g, int* workspace, int val) {
	int id = g.thread_rank();

	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		workspace[id] = val;
		g.sync();

		if (id < i) val += workspace[id + i];
		g.sync();
	}
	return val;
}


__global__ void SumReduction(int* a, int* ps, int* r) {
	
	extern __shared__ int workspace[THREADS*sizeof(int)];
	int total_sum, block_sum;
	grid_group grid = this_grid();
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	thread_group g = this_thread_block();
	block_sum = reduce_block_sum(g, workspace, a[i]);

	if (threadIdx.x == 0) ps[blockIdx.x] = block_sum;

	grid.sync();

	thread_group tile = tiled_partition(g, THREADS);
	
	if (blockIdx.x == 0 && threadIdx.x < THREADS) {
		total_sum = reduce_block_sum(tile, workspace, ps[i]);
	}

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		r[0] = total_sum;
	}
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

	int* d_a, * d_r, * ps;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_r, bytes);
	cudaMalloc(&ps, bytes);

	dim3 thread = THREADS;
	dim3 grid = N / thread.x;

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	
	void* kernalArgs[] = {
		&d_a ,&ps ,&d_r
	};

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaLaunchCooperativeKernel((void*)SumReduction, grid, thread, kernalArgs);

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