
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE_MAX 4				// i.e. maximum number of threads per block
#define GRID_SIZE_MAX 1					// i.e. maximum number of blocks
#define NUMBER_OF_ELEMS_PER_THREAD 2	// Number of elements (values) to be processed per thread

const int N = 1024;					// Number of values
const int B	= 8;					// Number of bins
const int displayValuesMax = 16;	// Maximum number of input values to display
const int displayBinsMax = 8;		// Maximum number of bins to display

cudaError_t histogramWithCuda(
	const unsigned int *h_In,
	unsigned int *h_Out,
	const unsigned int numBins,
	const unsigned int numElems);

__global__ void 
	histogramKernel(
		const unsigned int * const d_In,
		unsigned int *d_Out,
		unsigned int numVals)
{
    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

	int myTrueIdUpperLimit = (myId * NUMBER_OF_ELEMS_PER_THREAD) + NUMBER_OF_ELEMS_PER_THREAD;

	for (
		int myTrueId = myId * NUMBER_OF_ELEMS_PER_THREAD;
		myTrueId < myTrueIdUpperLimit;
		myTrueId++)
	{
		if ( myTrueId >= numVals )
		{
			return;
		}
		else
		{
			atomicAdd(&(d_Out[d_In[myTrueId]]), 1);
		}
	}
}

int main()
{
    unsigned int *h_In =
		new unsigned int[N];

	unsigned int *h_Out =
		new unsigned int[B];

	unsigned int *h_OutVerify =
		new unsigned int[B];

	memset(
		h_In,
		0x0,
		N * sizeof(unsigned int));

	memset(
		h_Out,
		0x0,
		B * sizeof(unsigned int));

	memset(
		h_OutVerify,
		0x0,
		B * sizeof(unsigned int));

	// Initialize (seed) random number generator
	srand(time(NULL));

	// Generate random values (bins)
	for (int i = 0; i < N; i++)
	{
		h_In[i] =
			rand() % B;
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus =
		histogramWithCuda(
			h_In,
			h_Out,
			B,
			N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramWithCuda failed!");
        return 1;
    }

    printf("Input values\n");
	printf("===============================\n\n");

	for (int i = 0; i < N; i++)
	{
		if (i < displayValuesMax)
		{
			printf("%d ", h_In[i]);
		}
		else
		{
			printf("...");
			break;
		}
	}

	printf("\n");

	printf("\n\nHistogram (bins)\n");
	printf("===============================\n\n");

	for (int i = 0; i < B; i++)
	{
		if (i < displayBinsMax)
		{
			printf("%d ", h_Out[i]);
		}
		else
		{
			printf("...");
			break;
		}
	}

	printf("\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	// Verification

	for (int i = 0; i < N; i++)
	{
		h_OutVerify[h_In[i]]++;
	}

	if (memcmp(h_Out, h_OutVerify, B * sizeof(unsigned int)))
	{
		printf("\n\n\t>>> Histogram verification failed!!! <<<\n");
	}

	if (h_In)
	{
		delete []h_In;
	}

	if (h_Out)
	{
		delete []h_Out;
	}

	if (h_OutVerify)
	{
		delete []h_OutVerify;
	}

	getchar();

    return 0;
}

// Helper function for using CUDA to generate a histogram in parallel.
cudaError_t histogramWithCuda(
	const unsigned int *h_In,
	unsigned int *h_Out,
	const unsigned int numBins,
	const unsigned int numElems)
{
    unsigned int *d_In = 0;
    unsigned int *d_Out = 0;

	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers

	// Input values
    cudaStatus =
		cudaMalloc((void**)&d_In, N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Histogram (bins)
    cudaStatus =
		cudaMalloc((void**)&d_Out, B * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus =
		cudaMemcpy(d_In, h_In, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Initialize histogram
    cudaStatus =
		cudaMemset(d_Out, 0x0, B * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

	unsigned int _numElemsProcessed = 0;

	dim3 _block(BLOCK_SIZE_MAX);

	while (_numElemsProcessed < numElems)
	{
		int numElemGroupsLeft =
			(numElems - _numElemsProcessed - 1) / NUMBER_OF_ELEMS_PER_THREAD + 1;

		int _gridSize =
			(numElemGroupsLeft - 1) / BLOCK_SIZE_MAX + 1;

		_gridSize =
			_gridSize < GRID_SIZE_MAX ?
				_gridSize :
				GRID_SIZE_MAX;

		dim3 _grid(_gridSize);

		// Launch a kernel on the GPU with one thread for each element.
		histogramKernel<<<_grid, _block>>>
			(&d_In[_numElemsProcessed],
			 d_Out,
			 N);

		_numElemsProcessed +=
			_gridSize * BLOCK_SIZE_MAX * NUMBER_OF_ELEMS_PER_THREAD;
	}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus =
		cudaMemcpy(h_Out, d_Out, B * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_In);
    cudaFree(d_Out);
    
    return cudaStatus;
}
