
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

using namespace std;

__global__ void merge_sort(float *arr, long length, long chunk)
{
	long start = (blockIdx.x * blockDim.x + threadIdx.x) * chunk;
	if (start >= length)
	{
		return;
	}

	long middle = min(start + chunk / 2, length);
	long end = min(start + chunk, length);
	long left = start;
	long right = middle;
	long i = 0;
	float* temp = new float[chunk]{0.f};

	while (left < middle || right < end)
	{
		float result;
		if (left < middle && right < end)
		{
			result = arr[left] <= arr[right] ? arr[left++] : arr[right++];
		} else 
		{
			result = left < middle ? arr[left++] : arr[right++];
		}
		temp[i++] = result;
	}	

	for (long index = 0; start + index < end; index++)
	{
		arr[start + index] = temp[index];
	}
	delete [] temp;
}

void printArray(float* arr, const long length)
{
	std::stringstream ss;
	ss << "[ ";
	for (long i = 0; i < length; i++)
	{
		ss << arr[i] << ", ";
	}
	std::string str = ss.str();
	str = str.substr(0, str.length() - 2);
	std::cout << str << " ]" << std::endl;
}

int main()
{
	long length = 2147483640;
	long size = length * sizeof(float);

	cudaError_t cudaStatus;
	curandStatus_t curandStatus;
	curandGenerator_t gen;
	curandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));

	float* arr;
	cudaStatus = cudaMalloc(&arr, size);
	curandStatus = curandGenerateUniform(gen, arr, length);

	long chunk = 2;
	bool isSorted = false;
	while (!isSorted)
	{
		long threads = ceilf(length / float(chunk));
		long grids = ceilf(threads / 32.f);
		if (grids > 0)
		{
			merge_sort << <grids, 32 >> >(arr, length, chunk);
		} else
		{
			merge_sort << <1, threads >> >(arr, length, chunk);
		}
		if (chunk >= length)
		{
			isSorted = true;
		}
		chunk *= 2;
	}

	float* sorted = new float[length]{0.f};
	cudaMemcpy(sorted, arr, size, cudaMemcpyDeviceToHost);	
	bool isCorrect = true;
	long i = 0;
	while (isCorrect && i < length - 1)
	{
		isCorrect = sorted[i] <= sorted[i + 1];
		i++;
	}	
	cout << "List size: " << length << ", Is correct: " << isCorrect << endl;
	// printArray(sorted, length);
	delete [] sorted;
	getchar();
    return 0;
}

