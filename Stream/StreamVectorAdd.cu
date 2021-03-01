#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using std::cout;
using std::endl;


__global__ void VectorAdd(int* a, int* b, int* r, const int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N) {
		r[tid] = a[tid] + b[tid];
	}
}


void verify_result(int* a, int* b, int* r, const int N) {
	for (int i = 0; i < N; i++) {
		if (r[i] != (a[i] + b[i])) {
			cout << "NOT SUCCESSFUL" << endl;
			exit(0);
		}
	}
	cout << "COMPLETED SUCCESSFULLY" << endl;
}




void GenArray(int* a, const int N) {
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
	}
}

int main() {
	const int N = 1 << 12;
	size_t bytes = N * sizeof(int);

	int* h_1, * h_2, * h_r1;
	int* h_3, * h_4, * h_r2;
	int* h_5, * h_6, * h_r3;
	
	h_1 = new int[N];
	h_2 = new int[N];
	h_r1 = new int[N];
	
	h_3 = new int[N];
	h_4 = new int[N];
	h_r2 = new int[N];
	
	h_5 = new int[N];
	h_6 = new int[N];
	h_r3 = new int[N];

	GenArray(h_1, N);
	GenArray(h_2, N);

	GenArray(h_3, N);
	GenArray(h_4, N);

	GenArray(h_5, N);
	GenArray(h_6, N);

	int* d_1, * d_2, * d_r1;
	cudaMalloc(&d_1, bytes);
	cudaMalloc(&d_2, bytes);
	cudaMalloc(&d_r1, bytes);
	
	int* d_3, * d_4, * d_r2;
	cudaMalloc(&d_3, bytes);
	cudaMalloc(&d_4, bytes);
	cudaMalloc(&d_r2, bytes);
	
	int* d_5, * d_6, * d_r3;
	cudaMalloc(&d_5, bytes);
	cudaMalloc(&d_6, bytes);
	cudaMalloc(&d_r3, bytes);

	const int Thread_Size = 32;
	const int Grid_Size = (N + Thread_Size - 1) / Thread_Size;

	cudaStream_t stream[3];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// start recording
	cudaEventRecord(start);
	
	// copying arrays from host to device 
	cudaMemcpyAsync(d_1, h_1, bytes, cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(d_2, h_2, bytes, cudaMemcpyHostToDevice, stream[0]);
	// kernel 1 launch
	VectorAdd << < Grid_Size, Thread_Size, 0, stream[0] >> > (d_1, d_2, d_r1, N);
	// copying result from device to host 
	cudaMemcpyAsync(h_r1, d_r1, bytes, cudaMemcpyDeviceToHost, stream[0]);

	// copying arrays from host to device 
	cudaMemcpyAsync(d_3, h_3, bytes, cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(d_4, h_4, bytes, cudaMemcpyHostToDevice, stream[1]);
	// kernel 2 launch
	VectorAdd << < Grid_Size, Thread_Size, 0, stream[1] >> > (d_3, d_4, d_r2, N);
	// copying result from device to host 
	cudaMemcpyAsync(h_r2, d_r2, bytes, cudaMemcpyDeviceToHost, stream[1]);

	// copying arrays from host to device 
	cudaMemcpyAsync(d_5, h_5, bytes, cudaMemcpyHostToDevice, stream[2]);
	cudaMemcpyAsync(d_6, h_6, bytes, cudaMemcpyHostToDevice, stream[2]);
	// kernel 3 launch
	VectorAdd << < Grid_Size, Thread_Size, 0, stream[2] >> > (d_5, d_6, d_r3, N);
	// copying result from device to host 
	cudaMemcpyAsync(h_r3, d_r3, bytes, cudaMemcpyDeviceToHost, stream[2]);

	cudaDeviceSynchronize();
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
	
	delete[] h_1;
	delete[] h_2;
	delete[] h_3;
	delete[] h_4;
	delete[] h_5;
	delete[] h_6;
	delete[] h_r1;
	delete[] h_r2;
	delete[] h_r3;

	return 0;

}

