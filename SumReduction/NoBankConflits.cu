#include "Header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

#define SHMEM_SIZE 256

__global__ void NoBankConflits(int* a, int* a_v) {
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	partial_sum[threadIdx.x] = a[tid];

	__syncthreads();

	for (int s = blockDim.x/2; s > 0; s >>= 1){
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		a_v[blockIdx.x] = partial_sum[0];
	}

}