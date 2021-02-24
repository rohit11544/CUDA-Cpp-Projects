#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void BaseLine(int* a, int* m, int* result, const int N, const int M);

__global__ void ConstantMem(int* a, int* result, const int N);


void verify_result(int* a, int* m, int* r, const int M, const int N);




