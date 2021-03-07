# CUDA-Cpp-Projects
CUDA/C++ optimization programs 

Here are the implimentation of CUDA/C++ programs which are optimized to a have a good Kernel excution time.
These are the improvments in excution time for the specific optimization steps.

## Matrix Multiplaction 

| Optimization | Kernel Time (msec)        |
| -------------|:-------------------------:|
| BaseLine     |         24.44             |            
| cuBLAS       |         7.81              |            
| Tiling       |         7.80              | 

There is an improvement from 24.44 ms to 7.80 ms (68%).

## Sum Reduction

| Optimization        | Kernel Time (msec)        |
| --------------------|:-------------------------:|
| BaseLine            |         13.41             |  
| NoWarpDivergance    |         11.26             |            
| NoBankConflits      |         9.38              |     
| LaunchHalfThreads   |         7.80              |
| CoparativeGroup     |         3.14              | 

There is an improvement from 13.41 ms to 3.14 ms (76.58%).

## Convolution

| Optimization | Kernel Time (msec)        |
| -------------|:-------------------------:|
| BaseLine     |         108.67            |            
| Tiling       |          88.45            |            
| CacheSimplification |   75.68            |            
| 2DConvolution |        320.93            |

There is an improvement from 108.67 ms to 75.68 ms (30.35%).

## Streams 

| Operation    | Without streams (msec)    |  With streams (msec)      |
| -------------|:-------------------------:|:-------------------------:|
| MatrixMul    |         6.885             |         5.3015            |
| SumReduction |         1.9601            |         1.5442            |
| VectorAdd    |         0.9377            |         0.5717            |


There is an improvement in time of 23% for MatrixMul, 21.21% in SumReduction and 39% in VectorAdd.

## Cooprative Groups

| Operation          | Without Cooprative Groups (msec)    |  With Cooprative Groups (msec)      |
| -------------------|:-----------------------------------:|:-----------------------------------:|
| BinarySumReduction |         0.1088                      |         0.0875                      |

There is an improvement from 0.1088 ms to 0.0875 ms (19.57%).

* This repository contains the cuda code in c++ for  the above optimizations. 
* And also the kernal report using Nsight Compute profiler.
