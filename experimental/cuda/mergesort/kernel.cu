
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

__global__ void merge_sort(float *d_unsorted_arr, float *d_sorted_arr,
                           unsigned long long length,
                           unsigned long long chunk) {
  unsigned long long start = (blockIdx.x * blockDim.x + threadIdx.x) * chunk;
  if (start >= length) {
    return;
  }

  unsigned long long middle = min(start + chunk / 2, length);
  unsigned long long end = min(start + chunk, length);
  unsigned long long left = start;
  unsigned long long right = middle;
  unsigned long long index = start;

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

  for (unsigned long long i = start; i < end; i++) {
    d_unsorted_arr[i] = d_sorted_arr[i];
  }
}

void printArray(float *arr, const unsigned long long length) {
  std::stringstream ss;
  ss << "[ ";
  for (unsigned long long i = 0; i < length; i++) {
    ss << arr[i] << ", ";
  }
  std::string str = ss.str();
  str = str.substr(0, str.length() - 2);
  std::cout << str << " ]" << std::endl;
}

string timed_operation;
chrono::system_clock::time_point start_time;

void startTimer(string operation) {
  timed_operation = operation;
  start_time = chrono::steady_clock::now();
}

void stopTimer() {
  auto end = chrono::steady_clock::now();
  auto diff = end - start_time;
  cout << timed_operation << " took "
       << chrono::duration<double, milli>(diff).count() << " ms" << endl;
}

void cudaMergeSort(float *d_unsorted_array, float *d_sorted_array,
                   unsigned long long length) {
  unsigned long long chunk = 2;
  bool isSorted = false;
  const int threads_per_block = 512;
  // const int threads_per_block = 256;
  // const int threads_per_block = 32;
  while (!isSorted) {
    unsigned long long threads = ceilf(length / float(chunk));
    unsigned long long grids = ceilf(threads / float(threads_per_block));
    if (grids > 0) {
      merge_sort<<<grids, threads_per_block>>>(d_unsorted_array, d_sorted_array,
                                               length, chunk);
    } else {
      merge_sort<<<1, threads>>>(d_unsorted_array, d_sorted_array, length,
                                 chunk);
    }
    if (chunk >= length) {
      isSorted = true;
    }
    chunk *= 2;
  }
}

void cpuMergeSort(float *h_unsorted_array, float *h_sorted_array,
                  unsigned long long start, unsigned long long chunk,
                  unsigned long long length) {
  unsigned long long middle = min(start + chunk / 2, length);
  unsigned long long end = min(start + chunk, length);
  unsigned long long left = start;
  unsigned long long right = middle;
  unsigned long long index = start;

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

  for (unsigned long long i = start; i < end; i++) {
    h_unsorted_array[i] = h_sorted_array[i];
  }
}

void cpuMergeSort(float *h_unsorted_array, float *h_sorted_array,
                  unsigned long long length) {
  unsigned long long chunk = 2;
  bool isSorted = false;
  while (!isSorted) {
    unsigned long long threads = ceilf(length / float(chunk));
    for (unsigned long long i = 0; i < threads; i++) {
      cpuMergeSort(h_unsorted_array, h_sorted_array, i * chunk, chunk, length);
    }
    if (chunk >= length) {
      isSorted = true;
    }
    chunk *= 2;
  }
}

void checkSortedArray(float *sorted_array, unsigned long long length) {
  bool isCorrect = true;
  unsigned long long i = 0;
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
  // unsigned long long length = 1610612736/2;
  unsigned long long length =
      10000000 * 20; // CPU takes 15 seconds at this length
  unsigned long long size = length * sizeof(float);

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

  startTimer("Random number generation");
  curandStatus = curandGenerateUniform(gen, d_unsorted_array, length);
  cudaDeviceSynchronize();
  stopTimer();

  // Store the same sequence of random numbers to use in all tests
  float *h_unsorted_array = new float[length];
  cudaMemcpy(h_unsorted_array, d_unsorted_array, size, cudaMemcpyDeviceToHost);

  startTimer("CUDA sorting");
  cudaMergeSort(d_unsorted_array, d_sorted_array, length);
  // Copy from device and check result
  float *h_sorted_array = new float[length];
  cudaMemcpy(h_sorted_array, d_unsorted_array, size, cudaMemcpyDeviceToHost);
  stopTimer();
  checkSortedArray(h_sorted_array, length);

  // Clean up
  cudaFree(d_sorted_array);
  cudaFree(d_unsorted_array);
  delete[] h_sorted_array;

  // Reallocate h_sorted_array and perform sorting on CPU
  h_sorted_array = new float[length];
  startTimer("CPU sorting");
  cpuMergeSort(h_unsorted_array, h_sorted_array, length);
  stopTimer();
  checkSortedArray(h_sorted_array, length);

  // Clean up
  delete[] h_sorted_array;
  delete[] h_unsorted_array;

  getchar();
  return 0;
}