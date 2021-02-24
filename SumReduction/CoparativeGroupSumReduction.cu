#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

using namespace cooperative_groups;

__device__ int reduce_sum(thread_group g, int* temp, int val) {
	int lane = g.thread_rank();

	for (int i = g.size() / 2; i > 0; i /= 2) {
		temp[lane] = val;

		g.sync();
		if (lane < i) {
			val += temp[lane + i];
		}
		g.sync();
	}
	return val;
}


__device__ int thread_sum(int* input, int n) {
	int sum = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) {
		int4 in = ((int4*)input)[i];
		sum += in.x + in.y + in.z + in.w;
	}
	return sum;
}

__global__ void sum_reduction(int* sum, int* input, int n) {
	int my_sum = thread_sum(input, n);

	extern __shared__ int temp[];

	auto g = this_thread_block();

	int block_sum = reduce_sum(g, temp, my_sum);

	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum);
	}
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;
	}
}

int main() {
	int n = 1 << 13;
	size_t bytes = n * sizeof(int);

	int* sum;
	int* data;

	cudaMallocManaged(&sum, sizeof(int));
	cudaMallocManaged(&data, bytes);

	initialize_vector(data, n);

	int TB_SIZE = 256;

	int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;

	sum_reduction << <GRID_SIZE, TB_SIZE, n * sizeof(int) >> > (sum, data, n);

	cudaDeviceSynchronize();

	assert(*sum == 8192);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}