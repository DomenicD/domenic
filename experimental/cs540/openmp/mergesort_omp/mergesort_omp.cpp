#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <ctime>

void benchmark(std::string label, std::function<void(void)> lambda) {
	// Source: http://en.cppreference.com/w/cpp/chrono/c/clock
	auto wall_clock_start = std::chrono::high_resolution_clock::now();
	auto cpu_clock_start = std::clock();
  lambda();
  auto wall_clock_end = std::chrono::high_resolution_clock::now();
  auto cpu_clock_end = std::clock();
  auto wall_diff =
      std::chrono::duration<double>(wall_clock_end - wall_clock_start).count();
  auto cpu_diff = (cpu_clock_end - cpu_clock_start) / (double)CLOCKS_PER_SEC;  
  std::cout << label << " wall time: " << wall_diff << " sec" << std::endl;
  std::cout << label << " cpu  time: " << cpu_diff << " sec" << std::endl << std::endl;
}

void print_array(float *arr, const int64_t length) {
  std::stringstream ss;
  ss << "[ ";
  for (int64_t i = 0; i < length; i++) {
    ss << arr[i] << ", ";
  }
  std::string str = ss.str();
  str = str.substr(0, str.length() - 2);
  std::cout << str << " ]" << std::endl;
}

void cpu_merge_sort(float *arr, float *buffer, int64_t start, int64_t chunk,
                    int64_t length) {
  int64_t middle = std::min(start + chunk / 2, length);
  int64_t end = std::min(start + chunk, length);
  int64_t left = start;
  int64_t right = middle;
  int64_t index = start;

  while (left < middle || right < end) {
    float result;
    if (left < middle && right < end) {
      result = arr[left] <= arr[right] ? arr[left++] : arr[right++];
    } else {
      result = left < middle ? arr[left++] : arr[right++];
    }
    buffer[index++] = result;
  }

  for (int64_t i = start; i < end; i++) {
    arr[i] = buffer[i];
  }
}

void print_thread_count() {
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
    }
  }
}

void cpu_merge_sort(float *arr, float *buffer, int64_t length) {
  int64_t chunk = 2;
  bool isSorted = false;
  while (!isSorted) {
    int64_t threads = static_cast<int64_t>(ceilf(length / float(chunk)));
#pragma omp parallel for
    for (int64_t i = 0; i < threads; i++) {
      cpu_merge_sort(arr, buffer, i * chunk, chunk, length);
    }
    if (chunk >= length) {
      isSorted = true;
    }
    chunk *= 2;
  }
}

void fill_with_random_numbers(float *arr, int64_t length) {
#pragma omp parallel
  {
    std::mt19937_64 generator(
        (omp_get_thread_num() + 1) *
        static_cast<uint64_t>(std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now())));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
#pragma omp for
    for (int64_t i = 0; i < length; i++) {
      arr[i] = distribution(generator);
    }
  }
}

int main(int argc, char *argv[]) {
  // With 1 thread: 89,500 ms.
  // With 8 threads: 23,212 ms.
  int64_t length = 10000000 * 5;

  float *arr = new float[length];
  float *buffer = new float[length];

  //omp_set_num_threads(1);

  std::cout << "Starting random number generation" << std::endl;
  benchmark("Random generation", [length, &arr]() {
    print_thread_count();
    fill_with_random_numbers(arr, length);
  });

  std::cout << "Starting merge sort" << std::endl;
  benchmark("Merge sort", [length, &arr, &buffer]() {
    print_thread_count();
    cpu_merge_sort(arr, buffer, length);
  });

  delete[] arr;
  delete[] buffer;
  return 0;
}
