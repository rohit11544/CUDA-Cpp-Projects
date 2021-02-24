#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include "Header.h"

using namespace std;


__global__ void baseLine(const int* a, const int* b, int* c, int N) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	c[row * N + col] = 0;

	for (int i = 0; i < N; i++) {
		c[row * N + col] += a[row * N + i] * b[col + i * N];
	}
}


int verify_result(vector<int>& a, vector<int>& b, vector<int>& c, int N) {

	for (int i = 0; i < N; i++) {

		int flag = 0;
		for (int j = 0; j < N; j++) {

			int tmp = 0;
			for (int k = 0; k < N; k++) {

				tmp += a[i * N + k] * b[k * N + j];
			}

			if (tmp == c[i * N + j]) { flag = 1; }

			if (flag == 0) {
				cout << "Wronge" << endl;
				exit(0);
			}
		}
	}
}