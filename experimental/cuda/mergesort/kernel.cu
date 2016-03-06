
#include "cuda_runtime.h"
#include "curand.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>

using namespace std;

__global__ void merge_sort_kernal(float *d_unsorted_arr, float *d_sorted_arr,
                                  uint64_t length, uint64_t chunk) {
  uint64_t start = (blockIdx.x * blockDim.x + threadIdx.x) * chunk;
  if (start >= length) {
    return;
  }

  uint64_t middle = min(start + chunk / 2, length);
  uint64_t end = min(start + chunk, length);
  uint64_t left = start;
  uint64_t right = middle;
  uint64_t index = start;

  while (left < middle || right < end) {
    float result;
    if (left < middle && right < end) {
      result = d_unsorted_arr[left] <= d_unsorted_arr[right]
                   ? d_unsorted_arr[left++]
                   : d_unsorted_arr[right++];
    } else {
      result = left < middle ? d_unsorted_arr[left++] : d_unsorted_arr[right++];
    }
    d_sorted_arr[index++] = result;
  }

  for (uint64_t i = start; i < end; i++) {
    d_unsorted_arr[i] = d_sorted_arr[i];
  }
}

void print_array(float *arr, const uint64_t length) {
  std::stringstream ss;
  ss << "[ ";
  for (uint64_t i = 0; i < length; i++) {
    ss << arr[i] << ", ";
  }
  std::string str = ss.str();
  str = str.substr(0, str.length() - 2);
  std::cout << str << " ]" << std::endl;
}

string timed_operation;
chrono::system_clock::time_point start_time;

void start_timer(string operation) {
  timed_operation = operation;
  start_time = chrono::steady_clock::now();
}

void stop_timer() {
  auto end = chrono::steady_clock::now();
  auto diff = end - start_time;
  cout << timed_operation << " took "
       << chrono::duration<double, milli>(diff).count() << " ms" << endl;
}

void cuda_merge_sort(float *d_unsorted_array, float *d_sorted_array,
                     uint64_t length) {
  uint64_t chunk = 2;
  bool isSorted = false;
  const int threads_per_block = 512;
  // const int threads_per_block = 256;
  // const int threads_per_block = 32;
  while (!isSorted) {
    uint64_t threads = ceilf(length / float(chunk));
    uint64_t grids = ceilf(threads / float(threads_per_block));
    if (grids > 0) {
      merge_sort_kernal<<<grids, threads_per_block>>>(
          d_unsorted_array, d_sorted_array, length, chunk);
    } else {
      merge_sort_kernal<<<1, threads>>>(d_unsorted_array, d_sorted_array,
                                        length, chunk);
    }
    if (chunk >= length) {
      isSorted = true;
    }
    chunk *= 2;
  }
}

void cpu_merge_sort(float *h_unsorted_array, float *h_sorted_array,
                    uint64_t start, uint64_t chunk, uint64_t length) {
  uint64_t middle = min(start + chunk / 2, length);
  uint64_t end = min(start + chunk, length);
  uint64_t left = start;
  uint64_t right = middle;
  uint64_t index = start;

  while (left < middle || right < end) {
    float result;
    if (left < middle && right < end) {
      result = h_unsorted_array[left] <= h_unsorted_array[right]
                   ? h_unsorted_array[left++]
                   : h_unsorted_array[right++];
    } else {
      result =
          left < middle ? h_unsorted_array[left++] : h_unsorted_array[right++];
    }
    h_sorted_array[index++] = result;
  }

  for (uint64_t i = start; i < end; i++) {
    h_unsorted_array[i] = h_sorted_array[i];
  }
}

void cpu_merge_sort(float *h_unsorted_array, float *h_sorted_array,
                    uint64_t length) {
  uint64_t chunk = 2;
  bool isSorted = false;
  while (!isSorted) {
    uint64_t threads = ceilf(length / float(chunk));
    for (uint64_t i = 0; i < threads; i++) {
      cpu_merge_sort(h_unsorted_array, h_sorted_array, i * chunk, chunk,
                     length);
    }
    if (chunk >= length) {
      isSorted = true;
    }
    chunk *= 2;
  }
}

void check_sorted_array(float *sorted_array, uint64_t length) {
  bool isCorrect = true;
  uint64_t i = 0;
  while (isCorrect && i < length - 1) {
    isCorrect = sorted_array[i] <= sorted_array[i + 1];
    i++;
  }
  cout << "List size: " << length
       << ", Is correct: " << (isCorrect ? "Yes" : "No") << endl;
}

// TODO(domenicd): implement a third version in OpenMP and run on Chapman
// numa server to compare CUDA and OpenMP.
int main() {
  // This is the largest we can do without having to do more
  // sofisticated memory management. Becuase this takes up all
  // the avaliable memory on the GPU (6GB).
  // Takes the GPU 2,283 ms and the CPU 323,186 ms.
  // uint64_t length = 1610612736 / 2;

  // A quicker experiment.
  // Takes the GPU 2,161 ms and the CPU 15,630 ms.
  uint64_t length = 10000000 * 5;

  uint64_t size = length * sizeof(float);

  cudaError_t cudaStatus;
  curandStatus_t curandStatus;
  curandGenerator_t gen;
  curandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(gen, time(0));
  chrono::system_clock::time_point end;

  float *d_unsorted_array;
  float *d_sorted_array;
  cudaStatus = cudaMalloc(&d_unsorted_array, size);
  cudaStatus = cudaMalloc(&d_sorted_array, size);

  start_timer("Random number generation");
  curandStatus = curandGenerateUniform(gen, d_unsorted_array, length);
  cudaDeviceSynchronize();
  stop_timer();

  // Store the same sequence of random numbers to use in all tests
  float *h_unsorted_array = new float[length];
  cudaMemcpy(h_unsorted_array, d_unsorted_array, size, cudaMemcpyDeviceToHost);

  start_timer("CUDA sorting");
  cuda_merge_sort(d_unsorted_array, d_sorted_array, length);
  // Copy from device and check result
  float *h_sorted_array = new float[length];
  cudaMemcpy(h_sorted_array, d_unsorted_array, size, cudaMemcpyDeviceToHost);
  stop_timer();
  check_sorted_array(h_sorted_array, length);

  // Clean up
  cudaFree(d_sorted_array);
  cudaFree(d_unsorted_array);
  delete[] h_sorted_array;

  // Reallocate h_sorted_array and perform sorting on CPU
  h_sorted_array = new float[length];
  start_timer("CPU sorting");
  cpu_merge_sort(h_unsorted_array, h_sorted_array, length);
  stop_timer();
  check_sorted_array(h_sorted_array, length);

  // Clean up
  delete[] h_sorted_array;
  delete[] h_unsorted_array;

  return 0;
}
