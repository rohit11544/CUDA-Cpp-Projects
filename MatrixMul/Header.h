#pragma once
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

int verify_result(vector<int>& a, vector<int>& b, vector<int>& c, int N);

__global__ void baseLine(const int* a, const int* b, int* c, int N);

__global__ void TiledMatrixMul(const int* a, const int* b, int* c);
