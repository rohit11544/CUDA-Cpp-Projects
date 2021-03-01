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
       

## Sum Reduction

| Optimization        | Kernel Time (msec)        |
| --------------------|:-------------------------:|
| BaseLine            |         13.41             |  
| NoWarpDiversion     |         11.26             |            
| NoBankConflits      |         9.38              |     
| LaunchHalfThreads   |         7.80              |
| CoparativeGroup     |         3.14              | 


## Convolution

| Optimization | Kernel Time (msec)        |
| -------------|:-------------------------:|
| BaseLine     |         108.67            |            
| Tiling       |          88.45            |            
| CacheSimplification |   75.68            |            
| 2DConvolution |        320.93            |


## Streams 

| Operation    | Without streams (msec)    |  With streams (msec)      |
| -------------|:-------------------------:|:-------------------------:|
| MatrixMul    |         6.885             |         5.3015            |
| SumReduction |         1.9601            |         1.5442            |
| VectorAdd    |         0.9377            |         0.5717            |


## Cooprative Groups

| Operation          | Without Cooprative Groups (msec)    |  With Cooprative Groups (msec)      |
| -------------------|:-----------------------------------:|:-----------------------------------:|
| BinarySumReduction |         0.1088                      |         0.0875                      |


* This repository contains the cuda code in c++ for  the above optimizations. 
* And also the kernal report using Nsight Compute profiler.
