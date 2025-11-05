#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

/* Outputs some information on CUDA-enabled devices on your computer,
 * including compute capability and current memory usage.
 *
 * On Linux, compile with: nvcc -o cuda_check cuda_check.c -lcuda
 * On Windows, compile with: nvcc -o cuda_check.exe cuda_check.c -lcuda
 *
 * Authors: Thomas Unterthiner, Jan Schlüter
 */

int ConvertSMVer2Cores(int major, int minor) {
  // Returns the number of CUDA cores per multiprocessor for a given
  // Compute Capability version. There is no way to retrieve that via
  // the API, so it needs to be hard-coded.
  // See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
  switch ((major << 4) + minor) {
    case 0x10:
      return 8;  // Tesla
    case 0x11:
      return 8;
    case 0x12:
      return 8;
    case 0x13:
      return 8;
    case 0x20:
      return 32;  // Fermi
    case 0x21:
      return 48;
    case 0x30:
      return 192;  // Kepler
    case 0x32:
      return 192;
    case 0x35:
      return 192;
    case 0x37:
      return 192;
    case 0x50:
      return 128;  // Maxwell
    case 0x52:
      return 128;
    case 0x53:
      return 128;
    case 0x60:
      return 64;  // Pascal
    case 0x61:
      return 128;
    case 0x62:
      return 128;
    case 0x70:
      return 64;  // Volta
    case 0x72:
      return 64;  // Xavier
    case 0x75:
      return 64;  // Turing
    case 0x80:
      return 64;  // Ampere
    case 0x86:
      return 128;
    case 0x87:
      return 128;
    case 0x89:
      return 128;  // Ada
    case 0x90:
      return 129;  // Hopper
    default:
      return 0;
  }
}

int main() {
  int nGpus = 0;
  cudaError_t err = cudaGetDeviceCount(&nGpus);
  if (err != cudaSuccess) {
    printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  printf("Found %d CUDA device(s).\n", nGpus);

  for (int i = 0; i < nGpus; ++i) {
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) {
      printf("cudaGetDeviceProperties failed for device %d: %s\n", i,
             cudaGetErrorString(err));
      continue;
    }

    printf("\nDevice %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);

    int cores_per_sm = ConvertSMVer2Cores(prop.major, prop.minor);
    int total_cores = prop.multiProcessorCount * cores_per_sm;
    if (total_cores > 0)
      printf("  CUDA Cores: %d\n", total_cores);
    else
      printf("  CUDA Cores: Unknown\n");

    printf("  Max Threads per Multiprocessor/SM: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Max Blocks per Multiprocessor/SM: %d\n",
           prop.maxBlocksPerMultiProcessor);
    printf("  Shared Memory per Multiprocessor/SM: %.2f KB\n",
           prop.sharedMemPerMultiprocessor / (1024.0));
    printf("  Shared Memory per Block/SM: %.2f KB\n",
           prop.sharedMemPerBlock / (1024.0));
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max regs per SM: %d\n", prop.regsPerBlock);
    printf("  GPU Clock: %.2f MHz\n", prop.clockRate / 1000.0f);
    printf("  Memory Clock: %.2f MHz\n", prop.memoryClockRate / 1000.0f);
    printf("  Total Memory: %.2f MiB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));

    size_t freeMem = 0, totalMem = 0;
    err = cudaSetDevice(i);  // 选择设备
    if (err == cudaSuccess) {
      err = cudaMemGetInfo(&freeMem, &totalMem);
      if (err == cudaSuccess) {
        printf("  Free Memory: %.2f MiB\n", freeMem / (1024.0 * 1024.0));
      } else {
        printf("  cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
      }
    }
  }

  return 0;
}
