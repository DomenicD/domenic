#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>

void print_thread_count() {
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
    }
  }
}

void benchmark(std::string label, std::function<void(void)> lambda,
               int samples) {
  // Source: http://en.cppreference.com/w/cpp/chrono/c/clock
  double wall_time = 0.0;
  double cpu_time = 0.0;
  for (size_t i = 0; i < samples; i++) {
    auto wall_clock_start = std::chrono::high_resolution_clock::now();
    auto cpu_clock_start = std::clock();
    lambda();
    auto wall_clock_end = std::chrono::high_resolution_clock::now();
    auto cpu_clock_end = std::clock();
    wall_time +=
        (std::chrono::duration<double>(wall_clock_end - wall_clock_start)
             .count()) /
        samples;
    cpu_time +=
        ((cpu_clock_end - cpu_clock_start) / (double)CLOCKS_PER_SEC) / samples;
  }

  std::cout << label << " samples: " << samples << std::endl;
  std::cout << label << " avg wall time: " << wall_time << " sec" << std::endl;
  std::cout << label << " avg cpu  time: " << cpu_time << " sec" << std::endl
            << std::endl;
}

void benchmark(std::string label, std::function<void(void)> lambda) {
  benchmark(label, lambda, 1);
}

void benchmark(std::string label, std::function<void(void)> lambda,
               std::vector<int> processors, int samples) {
  for (int i : processors) {
    omp_set_num_threads(i);
    print_thread_count();
    benchmark(label, lambda, samples);
  }
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

void omp_merge_sort(float *arr, float *buffer, int64_t start, int64_t chunk,
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

void omp_merge_sort(float *arr, float *buffer, int64_t length) {
  int64_t chunk = 2;
  bool isSorted = false;
  while (!isSorted) {
    int64_t threads = static_cast<int64_t>(ceilf(length / float(chunk)));
#pragma omp parallel for
    for (int64_t i = 0; i < threads; i++) {
      omp_merge_sort(arr, buffer, i * chunk, chunk, length);
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
  int64_t length = 10000000;

  float *random_numbers = new float[length];

  std::cout << "Starting random number generation" << std::endl;
  print_thread_count();
  benchmark("Random generation", [length, &random_numbers]() {
    fill_with_random_numbers(random_numbers, length);
  });

  float *array_to_sort = new float[length];
  float *buffer = new float[length];

  auto merge_sort_operation = [length, &random_numbers, &array_to_sort,
                               &buffer]() {
    // Reset the array back to the inital set of random numbers before each run.
    std::copy(random_numbers, random_numbers + length, array_to_sort);
    // Sort the random numbers using merge sort.
    omp_merge_sort(array_to_sort, buffer, length);
  };

  std::cout << "Starting merge sort" << std::endl;
  benchmark("Merge sort", merge_sort_operation, {1, 2, 4, 8}, 10);

  delete[] random_numbers;
  delete[] array_to_sort;
  delete[] buffer;
  return 0;
}
