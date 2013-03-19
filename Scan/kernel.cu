
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "timer.h"

#define MAX_THREAD_BLOCK_SIZE 512

// #define ADD
// #define SCAN
#define SCAN_EXCLUSIVE

#if defined(ADD)

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#endif

#if defined(SCAN)

cudaError_t scanWithCuda(int *c, const int *a, size_t size);

// Inclusive Scan (Hillis/Steele)

__global__ void scanKernelInclusive(int *c, const int *a, size_t size, size_t offset)
{
    int myId = 
		threadIdx.x;

	if (((myId - offset) < size) &&
		(myId >= offset))
	{
		c[myId] = a[myId];

		__syncthreads();

		size_t _stepsLeft =
			size;

		unsigned int _neighbor =
			1;

		while (_stepsLeft)
		{
			int op1 = c[myId];
			int op2 = 0;

			if ((myId - offset) >= _neighbor)
			{
				op2 =
					c[myId - _neighbor];
			}
			else
			{
				break;
			}

			__syncthreads();

			c[myId] =
				op1 + op2;

			__syncthreads();

			_stepsLeft >>= 1;
			_neighbor <<= 1;
		}

		if (offset > 0)
		{
			c[myId] +=
				c[offset - 1];
		}
	}
}

#endif

#if defined(SCAN_EXCLUSIVE)

cudaError_t scanExclusiveWithCuda(const unsigned int *h_In, unsigned int *h_Out, size_t size);

// Exclusive Scan (Blelloch)

__global__
	void scanKernelExclusive(const unsigned int *d_In,
							 unsigned int *d_Out,
							 size_t size,
							 size_t offset,
							 bool isLastCall)
{
	// Stores boundary values to account for sizes that are not powers of 2
	__shared__ unsigned int _boundaryValueCurrent;
	__shared__ unsigned int _finalAdd;
	unsigned int _finalRemember;

	int myId = 
		threadIdx.x;

	if (myId == 0)
	{
		_boundaryValueCurrent = 0;

		_finalRemember =
			d_In[offset + size - 1];

		if (offset > 0)
		{
			_finalAdd =
				d_Out[0] + d_Out[offset - 1];
		}
	}

	__syncthreads();

	if (myId < size)
	{
		// Initial data fetch
		d_Out[myId + offset] =
			d_In[myId + offset];

		__syncthreads();

		// Used to track how many steps are left by right-shifting its value
		// (i.e. implicitely calculating log2 of the size)
		size_t _stepsLeft =
			size;

		// Which neighbor to the left has to be added?
		unsigned int _neighbor =
			1;

		// Is it my turn to add?
		unsigned int _selfMask =
			1;

		// Step 1: Adding neighbors

		while (_stepsLeft)
		{
			if ((_selfMask & myId) == _selfMask)
			{
				d_Out[myId + offset] +=
					d_Out[(myId + offset) - _neighbor];
			}

			_stepsLeft >>= 1;
			_neighbor <<= 1;
			_selfMask <<= 1;
			_selfMask++;

			__syncthreads();
		}

		// Step 2: Down-sweep and adding neighbors again

		// Adjustment to properly start
		_selfMask--;
		_selfMask >>= 1;
		_neighbor >>= 1;
		_stepsLeft = size;

		while (_stepsLeft)
		{
			bool _fillInBoundaryValue =
				true;

			if ((_selfMask & myId) == _selfMask)
			{
				unsigned int _tmp =
					d_Out[myId + offset];

				d_Out[myId + offset] +=
					d_Out[(myId + offset) - _neighbor];

				d_Out[(myId + offset) - _neighbor] =
					_tmp;

				_fillInBoundaryValue =
					false;
			}

			__syncthreads();

			// Cross-sweep of boundary value

			unsigned int _selfMaskCrossSweep =
				_selfMask >> 1;

			if (_fillInBoundaryValue)
			{
				if (((_selfMask & myId) ^ _selfMaskCrossSweep) == 0)
				{
					if ((myId + _neighbor) >= size)
					{
						unsigned int _boundaryValueTmp =
							_boundaryValueCurrent + d_Out[(myId + offset)];

						d_Out[myId + offset] =
							_boundaryValueCurrent;

						_boundaryValueCurrent =
							_boundaryValueTmp;
					}
				}
			}
			
			_selfMask--;
			_selfMask >>= 1;
			_neighbor >>= 1;
			_stepsLeft >>= 1;

			__syncthreads();
		}

		if (offset > 0)
		{
			d_Out[(myId + offset)] +=
				_finalAdd;
		}

		__syncthreads();
	}

	if (myId == 0)
	{
		if (isLastCall)
		{
			d_Out[0] =
				0;
		}
		else
		{
			d_Out[0] =
				_finalRemember;
		}
	}
}

#endif

int main()
{
    const int arraySize = 13;
    const unsigned int a[arraySize] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    const unsigned int b[arraySize] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 };
    unsigned int c[arraySize] = { 0 };

	cudaError_t cudaStatus =
		cudaSuccess;

	float _elapsed = 0;

#if defined(ADD)

    // Add vectors in parallel.
    cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

#endif

#if defined(SCAN)

	// Timing
	{
		GpuTimer timer;
		timer.Start();
		// Scan vector in parallel.
		cudaStatus = scanWithCuda(c, a, arraySize);
		timer.Stop();
		_elapsed = timer.Elapsed();
	}
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "scanWithCuda failed!");
        return 1;
    }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
	else
	{
		printf("%f msecs.\n\n", _elapsed);

		for (int i = 0; i < arraySize; i++)
		{
			printf(
				"element %i -> in: %d\t out: %d\n",
				i,
				a[i],
				c[i]);
		}
	}

#endif

#if defined(SCAN_EXCLUSIVE)

	// Timing
	{
		GpuTimer timer;
		timer.Start();
		// Scan vector in parallel.
		cudaStatus = scanExclusiveWithCuda(a, c, arraySize);
		timer.Stop();
		_elapsed = timer.Elapsed();
	}
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "scanExclusiveWithCuda failed!");
        return 1;
    }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
	else
	{
		printf("%f msecs.\n\n", _elapsed);

		for (int i = 0; i < arraySize; i++)
		{
			printf(
				"element %i -> in: %d\t out: %d\n",
				i,
				a[i],
				c[i]);
		}
	}

#endif

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	getchar();

    return 0;
}

#if defined(ADD)

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

#endif

#if defined(SCAN)

cudaError_t scanWithCuda(int *c, const int *a, size_t size)
{
	int *dev_a = 0;
	int *dev_c = 0;

    cudaError_t cudaStatus =
		size < MAX_THREAD_BLOCK_SIZE ?
			cudaSuccess :
			cudaErrorUnknown;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Size cannot exceed maximum thread block size of %d", MAX_THREAD_BLOCK_SIZE);
        goto Error;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	 // Launch a kernel on the GPU with one thread for each element.
	scanKernelInclusive<<<1, size>>>(dev_c, dev_a, size >> 1, 0);
	scanKernelInclusive<<<1, size>>>(dev_c, dev_a, size >> 1 - 1, size >> 1);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);

	return cudaStatus;
}

#endif

#if defined(SCAN_EXCLUSIVE)

cudaError_t scanExclusiveWithCuda(const unsigned int *h_In, unsigned int *h_Out, size_t size)
{
	unsigned int *d_In = 0;
	unsigned int *d_Out = 0;

    cudaError_t cudaStatus =
		size < MAX_THREAD_BLOCK_SIZE ?
			cudaSuccess :
			cudaErrorUnknown;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Size cannot exceed maximum thread block size of %d", MAX_THREAD_BLOCK_SIZE);
        goto Error;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_Out, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&d_In, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_In, h_In, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	 // Launch a kernel on the GPU with one thread for each element.
	scanKernelExclusive<<<1, size>>>(d_In, d_Out, size >> 1, 0, false);
	scanKernelExclusive<<<1, size>>>(d_In, d_Out, size - (size >> 1), size >> 1, true);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_Out, d_Out, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_Out);
    cudaFree(d_In);

	return cudaStatus;
}

#endif