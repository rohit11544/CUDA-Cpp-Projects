#include "Header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

#define SHMEM_SIZE 256

__global__ void NoWarpDiversion(int* a, int* a_v) {
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int index;
	partial_sum[threadIdx.x] = a[tid];
	
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		
		index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		a_v[blockIdx.x] = partial_sum[0];
	}

}