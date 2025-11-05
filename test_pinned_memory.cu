#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

using namespace std;

float cuda_host_alloc_test(int size, bool up) {
  // 耗时统计
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *a, *dev_a;

  // 在主机上分配页锁定内存
  cudaError_t cudaStatus =
      cudaHostAlloc((void **)&a, size * sizeof(*a), cudaHostAllocDefault);
  if (cudaStatus != cudaSuccess) {
    printf("host alloc fail!\n");
    return -1;
  }

  // 在设备上分配内存空间
  cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(*dev_a));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!\n");
    return -1;
  }

  // 计时开始
  cudaEventRecord(start, 0);

  for (int i = 0; i < 1; i++) {
    // 从主机到设备复制数据
    cudaStatus =
        cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy Host to Device failed!\n");
      return -1;
    }

    // 从设备到主机复制数据
    cudaStatus =
        cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy Device to Host failed!\n");
      return -1;
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaFreeHost(a);
  cudaFree(dev_a);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return (float)elapsedTime / 1000;
}

float cuda_host_Malloc_test(int size, bool up) {
  // 耗时统计
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *a, *dev_a;

  // 在主机上分配可分页内存
  a = (int *)malloc(size * sizeof(*a));

  // 在设备上分配内存空间
  cudaError_t cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(*dev_a));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!\n");
    return -1;
  }

  // 计时开始
  cudaEventRecord(start, 0);

  for (int i = 0; i < 1; i++) {
    // 从主机到设备复制数据
    cudaStatus =
        cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy Host to Device failed!\n");
      return -1;
    }

    // 从设备到主机复制数据
    cudaStatus =
        cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy Device to Host failed!\n");
      return -1;
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  free(a);
  cudaFree(dev_a);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return (float)elapsedTime / 1000;
}

int main() {
  int size = 1024 * 1024 * 1024;
  float allocTime = cuda_host_alloc_test(size, true);
  cout << "页锁定内存: " << allocTime << " s" << endl;
  float mallocTime = cuda_host_Malloc_test(size, true);
  cout << "可分页内存: " << mallocTime << " s" << endl;
  return 0;
}
