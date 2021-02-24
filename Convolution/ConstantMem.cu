#include "Header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];

__global__ void ConstantMem(int* a, int* result, const int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = MASK_LENGTH / 2;
	int start = tid - r;
	int temp = 0;

	for (int i = 0; i < MASK_LENGTH; i++) {
		if (((start + i) >= 0) && ((start + i) < N)) {
			temp += a[start + i] * mask[i];
		}
	}

	result[tid] = temp;
}
