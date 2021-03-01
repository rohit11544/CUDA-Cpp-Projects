#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>

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

	cout << "COMPLETED SUCCESSFULLY\n";
}


main() {
	const int N = 1 << 7;
	size_t bytes = N * N * sizeof(int);

	vector<int> h_1(N * N), h_2(N * N), h_r1(N * N);
	vector<int> h_3(N * N), h_4(N * N), h_r2(N * N);
	vector<int> h_5(N * N), h_6(N * N), h_r3(N * N);

	generate(h_1.begin(), h_1.end(), []() { return rand() % 100; });
	generate(h_2.begin(), h_2.end(), []() { return rand() % 100; });

	generate(h_3.begin(), h_3.end(), []() { return rand() % 100; });
	generate(h_4.begin(), h_4.end(), []() { return rand() % 100; });

	generate(h_5.begin(), h_5.end(), []() { return rand() % 100; });
	generate(h_6.begin(), h_6.end(), []() { return rand() % 100; });

	int* d_1, * d_2, * d_r1;
	int* d_3, * d_4, * d_r2;
	int* d_5, * d_6, * d_r3;

	cudaMalloc(&d_1, bytes);
	cudaMalloc(&d_2, bytes);
	cudaMalloc(&d_r1, bytes);

	cudaMalloc(&d_3, bytes);
	cudaMalloc(&d_4, bytes);
	cudaMalloc(&d_r2, bytes);

	cudaMalloc(&d_5, bytes);
	cudaMalloc(&d_6, bytes);
	cudaMalloc(&d_r3, bytes);

	int THREADS = 32;
	int BLOCKS = N / THREADS;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	cudaStream_t stream[3];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start recording
	cudaEventRecord(start);

	cudaMemcpyAsync(d_1, h_1.data(), bytes, cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(d_3, h_3.data(), bytes, cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(d_5, h_5.data(), bytes, cudaMemcpyHostToDevice, stream[2]);

	cudaMemcpyAsync(d_2, h_2.data(), bytes, cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(d_4, h_4.data(), bytes, cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(d_6, h_6.data(), bytes, cudaMemcpyHostToDevice, stream[2]);

	baseLine << <blocks, threads, 0, stream[0] >> > (d_1, d_2, d_r1, N);
	baseLine << <blocks, threads, 0, stream[1] >> > (d_3, d_4, d_r2, N);
	baseLine << <blocks, threads, 0, stream[2] >> > (d_5, d_6, d_r3, N);

	cudaMemcpyAsync(h_r1.data(), d_r1, bytes, cudaMemcpyDeviceToHost, stream[0]);
	cudaMemcpyAsync(h_r2.data(), d_r2, bytes, cudaMemcpyDeviceToHost, stream[1]);
	cudaMemcpyAsync(h_r3.data(), d_r3, bytes, cudaMemcpyDeviceToHost, stream[2]);

	//stop recording
	cudaEventRecord(stop);

	verify_result(h_1, h_2, h_r1, N);
	verify_result(h_3, h_4, h_r2, N);
	verify_result(h_5, h_6, h_r3, N);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Total time in ms taken to complete kernels : " << milliseconds << endl;

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);

	cudaFree(d_1);
	cudaFree(d_2);
	cudaFree(d_3);
	cudaFree(d_4);
	cudaFree(d_5);
	cudaFree(d_6);
	cudaFree(d_r1);
	cudaFree(d_r2);
	cudaFree(d_r3);

	return 0;
}