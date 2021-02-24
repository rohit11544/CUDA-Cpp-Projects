# CUDA-Cpp-Projects
CUDA/C++ optimization programs 

Here are the implimentation of some CUDA/C++ programs which are optimized to a good performance

## Matrix Multiplaction 

| Optimization | Kernal Time (msec)        |
| -------------|:-------------------------:|
| BaseLine     |         24.44             |            
| cuBLAS       |         7.81              |            
| Tiling       |         7.80              |            
       

## Sum Reduction

| Optimization        | Kernal Time (msec)        |
| --------------------|:-------------------------:|
| BaseLine            |         13.41             |  
| NoWarpDiversion     |         11.26             |            
| NoBankConflits      |         9.38              |     
| LaunchHalfThreads   |         7.80              |
| CoparativeGroup     |         3.14              | 


## Matrix Multiplaction 

| Optimization | Kernal Time (msec)        |
| -------------|:-------------------------:|
| BaseLine     |         24.44             |            
| cuBLAS       |         7.81              |            
| Tiling       |         7.80              |            
       
       
