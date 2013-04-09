
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include "timer.h"

cudaError_t transposeWithCuda(const int *h_In, int *h_Out);

const int N = 1024;
const int K = 16;

const int displayMax = 6;

#define USE_FLOAT4

__global__ void transposeKernelCoalescedReadGlobalMemory(const int *d_In, int *d_Out)
{
    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myIdIn = (blockId * threadsPerBlock) + threadId;

    int myRowIn =
        myIdIn / N;

	int myColIn =
        myIdIn - (myRowIn * N);
    
    int myRowOut =
        myColIn;
    
    int myColOut =
        myRowIn;
    
    int myIdOut =
        myRowOut * N + myColOut;
    
    d_Out[myIdOut] =
        d_In[myIdIn];
}

#if !defined(USE_FLOAT4)

__global__ void transposeKernelCoalescedReadWriteGlobalMemory(const int *d_In, int *d_Out)
{
    __shared__ float _shared[K * K];
    
    int i = blockIdx.x * K + threadIdx.x;
    int j = blockIdx.y * K + threadIdx.y;

    int myBlockId =
        threadIdx.y * K + threadIdx.x;
    int myBlockIdTransposed =
        threadIdx.x * K + threadIdx.y;
    
    int myDataOffsetIn =
        j * N + i;

	_shared[myBlockIdTransposed] =
        d_In[myDataOffsetIn];

	__syncthreads();
    
    int myDataOffsetOut =
        (blockIdx.x * N * K) + (blockIdx.y * K);

    myDataOffsetOut +=
        threadIdx.y * N + threadIdx.x;

    d_Out[myDataOffsetOut] =
        _shared[myBlockId];

}

#else

__global__ void transposeKernelCoalescedReadWriteGlobalMemoryFloat4(const int *d_In, int *d_Out)
{
    __shared__ float4 _shared[K / 4][K];
    
    int i = blockIdx.x * K / 4 + threadIdx.x;
    int j = blockIdx.y * K + threadIdx.y;

    int myDataOffsetIn =
        j * N / 4 + i;

	// Fetch 4 values at once from global memory into local (fastest) memory
	float4 _tmp =
		((float4*)d_In)[myDataOffsetIn];

	unsigned int _threadIdxDotyDivBy4 =
		threadIdx.y / 4;

	unsigned int _threadIdxDotxTimes4 =
		threadIdx.x * 4;

	unsigned int _threadIdxDotyMod4 =
		threadIdx.y % 4;

	// Store local value transposed into shared memory

	float *_target =
		(float*)(&_shared[_threadIdxDotyDivBy4][_threadIdxDotxTimes4 + 0]);

	_target[_threadIdxDotyMod4] =
		_tmp.x;

	_target =
		(float*)(&_shared[_threadIdxDotyDivBy4][_threadIdxDotxTimes4 + 1]);

	_target[_threadIdxDotyMod4] =
		_tmp.y;

	_target =
		(float*)(&_shared[_threadIdxDotyDivBy4][_threadIdxDotxTimes4 + 2]);

	_target[_threadIdxDotyMod4] =
		_tmp.z;

	_target =
		(float*)(&_shared[_threadIdxDotyDivBy4][_threadIdxDotxTimes4 + 3]);

	_target[_threadIdxDotyMod4] =
		_tmp.w;

	__syncthreads();

	// Transpose the tiles and write back out to global memory

	int myDataOffsetOut =
        (blockIdx.x * N / 4 * K) + (blockIdx.y * K / 4);

    myDataOffsetOut +=
        threadIdx.y * N / 4 + threadIdx.x;

	((float4*)d_Out)[myDataOffsetOut] =
		_shared[threadIdx.x][threadIdx.y];
}

#endif

int main()
{
	int *a =
		new int[N * N];
	int *b =
		new int[N * N];

	memset(a, 0x0, N * N * sizeof(int));
	memset(b, 0x0, N * N * sizeof(int));

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] =
				i * N + j + 1;
		}
	}

    // Transpose matrizes in parallel
    cudaError_t cudaStatus =
		transposeWithCuda(a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "transposeWithCuda failed!");
        return 1;
    }

	printf("Input\n");
	printf("===============================\n\n");

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d\t", a[i * N + j]);

			if ((j + 1) >= displayMax)
			{
				printf("...");
				break;
			}
		}

		printf("\n");

		if ((i + 1) >= displayMax)
		{
			printf("...\n");
			break;
		}
	}

	printf("\n\nOutput\n");
	printf("===============================\n\n");

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d\t", b[i * N + j]);

			if ((j + 1) >= displayMax)
			{
				printf("...");
				break;
			}
		}

		printf("\n");

		if ((i + 1) >= displayMax)
		{
			printf("...\n");
			break;
		}
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	getchar();

	delete []a;
	delete []b;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t transposeWithCuda(const int *h_In, int *h_Out)
{
    int *d_In = 0;
    int *d_Out = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&d_In, N * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_Out, N * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_In, h_In, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	dim3 blockSize(

#if !defined(USE_FLOAT4)

		K,

#else

		K / 4,

#endif

		K,
		1);

	dim3 gridSize(

#if !defined(USE_FLOAT4)

		(N - 1) / (blockSize.x) + 1,

#else
		((N / 4) - 1) / (blockSize.x) + 1,

#endif

		(N - 1) / (blockSize.y) + 1,
		1);

	float _elapsed = 0;

	// Timing
	{
		GpuTimer timer;
		timer.Start();
		// Launch a kernel on the GPU with one thread for each element.

#if !defined(USE_FLOAT4)

		transposeKernelCoalescedReadWriteGlobalMemory<<<gridSize, blockSize>>>(d_In, d_Out);

#else

		transposeKernelCoalescedReadWriteGlobalMemoryFloat4<<<gridSize, blockSize>>>(d_In, d_Out);

#endif
		timer.Stop();
		_elapsed = timer.Elapsed();
	}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	printf("%f msecs.\n\n", _elapsed);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_Out, d_Out, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_In);
    cudaFree(d_Out);
    
    return cudaStatus;
}
