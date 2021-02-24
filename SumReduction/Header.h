#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>


using namespace std;

__global__ void BaseLine(int* a, int* a_v);

__global__ void NoWarpDiversion(int* a, int* a_v);

__global__ void NoBankConflits(int* a, int* a_v);

__global__ void LaunchHalfThreads(int* a, int* a_v);

void generateArray(int* a, const int N);

void verifyResult(int* a, int* a_v, const int n);
