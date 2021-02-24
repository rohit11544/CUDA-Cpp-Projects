
#include "Header.h"
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
}
