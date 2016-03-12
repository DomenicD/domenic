// mergesort_omp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


void benchmark(std::string label, std::function<void(void)> lambda)
{
	auto start_time = std::chrono::steady_clock::now();
	lambda();
	auto end = std::chrono::steady_clock::now();
	auto diff = std::chrono::duration<double, std::milli>(end - start_time).count();
	std::cout << label << " took " << diff << " ms" << std::endl;
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

void cpu_merge_sort(float *arr, float *buffer,
	int64_t start, int64_t chunk, int64_t length) {
	int64_t middle = std::min(start + chunk / 2, length);
	int64_t end = std::min(start + chunk, length);
	int64_t left = start;
	int64_t right = middle;
	int64_t index = start;

	while (left < middle || right < end) {
		float result;
		if (left < middle && right < end) {
			result = arr[left] <= arr[right]
				? arr[left++]
				: arr[right++];
		}
		else {
			result =
				left < middle ? arr[left++] : arr[right++];
		}
		buffer[index++] = result;
	}

	for (int64_t i = start; i < end; i++) {
		arr[i] = buffer[i];
	}
}


void cpu_merge_sort(float *arr, float *buffer, int64_t length) {
	int64_t chunk = 2;
	bool isSorted = false;
	while (!isSorted) {
		int64_t threads = static_cast<int64_t>(ceilf(length / float(chunk)));
		#pragma omp parallel for
		for (int64_t i = 0; i < threads; i++) {
			cpu_merge_sort(arr, buffer, i * chunk, chunk,
				length);
		}
		if (chunk >= length) {
			isSorted = true;
		}
		chunk *= 2;
	}
}

int main(int argc, char* argv[])
{
	

	// Takes the GPU 2,283 ms and the CPU 323,186 ms.
	// int64_t length = 1610612736 / 2;

	// A quicker experiment.
	// Takes the GPU 2,161 ms and the CPU 15,630 ms.
	// int64_t length = 10000000 * 5;
	int64_t length = 10;

	float *arr = new float[length];
	float *buffer = new float[length];
	benchmark("Random generation", [length, &arr]()
	{	
		#pragma omp parallel
		{
			std::mt19937_64 generator((omp_get_thread_num() + 1) * static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
			std::uniform_real_distribution<double> distribution(0.0, 1.0);
			#pragma omp parallel for
			for (int64_t i = 0; i < length; i++)
			{
				arr[i] = distribution(generator);
			}
		}
	});
	print_array(arr, length);

	benchmark("Omp mergesort", [length, &arr, &buffer]()
	{
		cpu_merge_sort(arr, buffer, length);
	});
	
	print_array(arr, length);

	delete [] arr;
	delete [] buffer;
	return 0;
}

