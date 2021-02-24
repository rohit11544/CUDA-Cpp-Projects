
#include "Header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>


__global__ void BaseLine(int* a, int* m, int* result, const int N, const int M) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = M / 2;
	int start = tid - r;
	int temp = 0;

	for (int i = 0; i < M; i++) {
		if (((start + i) >= 0) && ((start + i) < N)) {
			temp += a[start + i] * m[i];
		}
	}

	result[tid] = temp;
}


void verify_result(int* a, int* m, int* r, const int M, const int N) {
	int temp = 0, start;

	for (int i = 0; i < N; i++) {
		temp = 0;
		start = i - (M / 2);
		for (int j = 0; j < M; j++) {
			if (((start + j) >= 0) && ((start + j) < N)) {
				temp += a[start + j] * m[j];
			}
		}

		if (temp != r[i]) {
			std::cout << "NOT SUCCESSFUL" << std::endl;
			exit(0);
		}
	}
}
