// Radix Sort as per http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html (section 39.3.3 Radix Sort)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#include "timer.h"

#define MAX_THREAD_BLOCK_SIZE 512

#define BLOCK_SIZE_MAX_X 20
#define BLOCK_SIZE_MAX_Y 20

// #define DEBUG

cudaError_t
	radixSortWithCuda(
		const unsigned int *h_In,
		unsigned int *h_Out,
		size_t size);

// Calculates the split based on the least significant bit 'bitPos'.
// Returns the last value set in 'lastValue'.
//
// Sets '1' for each '0' input, otherwise '0'
__global__ void 
	splitKernel(
		const unsigned int *d_In,
		unsigned int *d_Out,
		size_t size,
		unsigned int bitPos,
		unsigned int *lastValue)
{
    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

	if (myId < size)
	{
		unsigned int _tmp =
			(d_In[myId] >> bitPos) & 0x1;
		_tmp =
			_tmp ? 0 : 1;

		d_Out[myId] =
			_tmp;

		if ((myId == (size - 1)) &&
			(lastValue))
		{
			*lastValue =
				_tmp;
		}
	}
}

// Exclusive Scan (Blelloch)

__global__
	void scanKernelExclusive_Phase1(const unsigned int *d_In,
							        unsigned int *d_Out,
							        size_t size)
{
	// Stores boundary values to account for sizes that are not powers of 2
	__shared__ unsigned int _boundaryValueCurrent;
	unsigned int _finalRemember;
	__shared__ unsigned int _sharedVals[MAX_THREAD_BLOCK_SIZE];

    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

    unsigned int _offset =
        blockId * threadsPerBlock;

    unsigned int _size =
        threadsPerBlock;

	_size =
        _offset + _size > size ?
            size - _offset :
            _size;

    if (threadId == 0)
	{
		_boundaryValueCurrent = 0;

		_finalRemember =
			d_In[_offset + _size - 1];
    }

    __syncthreads();

	if (myId < size)
	{
        // Initial data fetch
		_sharedVals[threadId] =
			d_In[myId];

		__syncthreads();

        // Used to track how many steps are left by right-shifting its value
		// (i.e. implicitely calculating log2 of the size)
		size_t _stepsLeft =
			_size;

		// Which neighbor to the left has to be added?
		unsigned int _neighbor =
			1;

		// Is it my turn to add?
		unsigned int _selfMask =
			1;

        // Step 1: Adding neighbors

		while (_stepsLeft)
		{
			if ((_selfMask & threadId) == _selfMask)
			{
				_sharedVals[threadId] +=
					_sharedVals[threadId - _neighbor];
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
		_stepsLeft = _size;

		while (_stepsLeft)
		{
			bool _fillInBoundaryValue =
				true;

			if ((_selfMask & threadId) == _selfMask)
			{
				unsigned int _tmp =
					_sharedVals[threadId];

				_sharedVals[threadId] +=
					_sharedVals[threadId - _neighbor];

				_sharedVals[threadId - _neighbor] =
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
				if (((_selfMask & threadId) ^ _selfMaskCrossSweep) == 0)
				{
					if ((threadId + _neighbor) >= _size)
					{
						unsigned int _boundaryValueTmp =
							_boundaryValueCurrent + _sharedVals[(threadId)];

						_sharedVals[threadId] =
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

        d_Out[myId] =
				_sharedVals[threadId];

        if (threadId == 0)
		{
			d_Out[_offset] =
				_finalRemember;
		}
    }
}

__global__
	void scanKernelExclusive_Phase2(const unsigned int *d_In,
							        unsigned int *d_Out,
							        size_t size,
									unsigned int origBlockSize)
{
    // Stores boundary values to account for sizes that are not powers of 2
	__shared__ unsigned int _boundaryValueCurrent;
	__shared__ unsigned int _sharedVals[MAX_THREAD_BLOCK_SIZE];

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

	int myId = threadId;

    myId *=
        origBlockSize;

    if (myId < size)
	{
        // Initial data fetch
		_sharedVals[threadId] =
			d_In[myId];

		__syncthreads();
    }

    if ((myId < size) &&
        (myId > 0))
    {
        unsigned int _tmp =
            _sharedVals[threadId - 1];

	__syncthreads();

        _sharedVals[threadId] =
             d_In[myId - 1] + _tmp;

        __syncthreads();
    }

	__syncthreads();

    if (myId == 0)
	{
        _sharedVals[0] = 0;
        _boundaryValueCurrent = 0;
    }

    __syncthreads();

    unsigned int _size =
        (size - 1) / origBlockSize + 1;

	if (myId < size)
	{
        // Used to track how many steps are left by right-shifting its value
		// (i.e. implicitely calculating log2 of the size)
		size_t _stepsLeft =
			_size;

		// Which neighbor to the left has to be added?
		unsigned int _neighbor =
			1;

		// Is it my turn to add?
		unsigned int _selfMask =
			1;

        // Step 1: Adding neighbors

		while (_stepsLeft)
		{
			if ((_selfMask & threadId) == _selfMask)
			{
				_sharedVals[threadId] +=
					_sharedVals[threadId - _neighbor];
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
		_stepsLeft = _size;

		while (_stepsLeft)
		{
			bool _fillInBoundaryValue =
				true;

			if ((_selfMask & threadId) == _selfMask)
			{
				unsigned int _tmp =
					_sharedVals[threadId];

				_sharedVals[threadId] +=
					_sharedVals[threadId - _neighbor];

				_sharedVals[threadId - _neighbor] =
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
				if (((_selfMask & threadId) ^ _selfMaskCrossSweep) == 0)
				{
					if ((threadId + _neighbor) >= _size)
					{
						unsigned int _boundaryValueTmp =
							_boundaryValueCurrent + _sharedVals[(threadId)];

						_sharedVals[threadId] =
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

        // Inclusive scan
        if ((threadId + 1) < _size)
        {
            d_Out[myId] =
                _sharedVals[threadId + 1];
        }
        else
        {
            d_Out[myId] =
                _boundaryValueCurrent;
        }
    }
}

__global__
	void scanKernelExclusive_Phase3(const unsigned int *d_In,
							        unsigned int *d_Out,
							        size_t size,
                                    unsigned int *total)
{
    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

    unsigned int _offset =
        blockId * threadsPerBlock;

    __shared__ unsigned int _finalAdd;

    if (threadId == 0)
	{
        _finalAdd =
            d_In[_offset];
    }

    __syncthreads();

    if ((myId < size) &&
        (threadId > 0))
	{
        d_Out[myId] +=
            _finalAdd;
    }

    if ((myId == (size - 1)) &&
		(total))
	{
		*total =
			d_Out[myId];
	}
}

// Scatters elements in 'd_In' to 'd_Out' using the (previously) calculated address
// 'd_FalseKeyAddresses' for the false keys (i.e. where the least significant bit
// 'bitPos' in the input 'd_In' was '0').
__global__ void
	scatterKernel(
		const unsigned int *d_In,
		const unsigned int *d_FalseKeyAddresses,
		unsigned int *d_Out,
		const unsigned int totalFalses,
		size_t size,
		unsigned int bitPos)
{
	int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

    if (myId < size)
    {
		unsigned int _myFalseKeyAddress =
			d_FalseKeyAddresses[myId];

		// Calculate true sort key address
		int _trueSortKeyAddress =
			myId - _myFalseKeyAddress + totalFalses;

		// True sort key?
		unsigned int _trueSortKey =
			(d_In[myId] >> bitPos) & 0x1;
		
		int _destinationAddress =
			_trueSortKey ?
				_trueSortKeyAddress :
				_myFalseKeyAddress;

		d_Out[_destinationAddress] =
			d_In[myId];

	}
}

int main()
{
    const unsigned int arraySize = 10;
    const unsigned int h_In[arraySize] = { 252, 88, 114, 148, 245, 211, 96, 13, 131, 38 };
    unsigned int h_Out[arraySize] = { 0 };

	cudaError_t cudaStatus =
		cudaSuccess;

	float _elapsed = 0;

	{
		GpuTimer timer;
		timer.Start();
		// Radix sort in parallel.
		cudaError_t cudaStatus =
			radixSortWithCuda(h_In, h_Out, arraySize);
		timer.Stop();
		_elapsed = timer.Elapsed();
	}

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "radixSortWithCuda failed!");
        return 1;
    }
	else
	{
		printf("%f msecs.\n\n", _elapsed);

		for (int i = 0; i < arraySize; i++)
		{
			printf(
				"element %i -> in: %d\t out: %d\n",
				i,
				h_In[i],
				h_Out[i]);
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

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t 
	radixSortWithCuda(
		const unsigned int *h_In,
		unsigned int *h_Out, 
		size_t size)
{
    unsigned int *d_In = 0;
    unsigned int *d_Out = 0;
	unsigned int *d_FalseKeyAddresses = 0;
	unsigned int h_TotalFalsesInput[] = { 0, 0 };

	cudaError_t cudaStatus =
		cudaSuccess;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate and initialize GPU buffers.
    cudaStatus = cudaMalloc((void**)&d_Out, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemset(d_Out, 0x0, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_In, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// d_FalseKeyAddresses[0]: Last value set in split operation
	// d_FalseKeyAddresses[1]: Number of least significant bits 'bitPos' that were 'false' minus 1
	// d_FalseKeyAddresses[2 ... 2 + size - 1]: Addresses for the 'false' keys
    cudaStatus = cudaMalloc((void**)&d_FalseKeyAddresses, 2 * sizeof(unsigned int) + size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemset(d_FalseKeyAddresses, 0x0, 2 * sizeof(unsigned int) + size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_In, h_In, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	unsigned int *_src = d_In;
	unsigned int *_dst = d_Out;

	const int numBits = 1;

	for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
	{
		// Launch a kernel on the GPU with one thread for each element.
		splitKernel<<<1, size>>>(
			_src,
			_dst,
			size,
			i,
			&d_FalseKeyAddresses[0]);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_Out, _dst, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

#endif

		// Block size (i.e., number of threads per block)
		dim3 blockSize(
			BLOCK_SIZE_MAX_X,
			BLOCK_SIZE_MAX_Y,
			1);

		int gridSizeX =
			(ceil(sqrt((double)size)) - 1) / BLOCK_SIZE_MAX_X + 1;

		int gridSizeY =
			(ceil(sqrt((double)size)) - 1) / BLOCK_SIZE_MAX_Y + 1;
    
		dim3 gridSize(
			gridSizeX,
			gridSizeY,
			1);

		scanKernelExclusive_Phase1<<<gridSize, blockSize>>>(
			_dst,
			&d_FalseKeyAddresses[2],
			size);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_Out, &d_FalseKeyAddresses[2], size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

#endif

		// Block size (i.e., number of threads per block)
		dim3 blockSizePhase2(
			min(max(gridSize.x, blockSize.x), BLOCK_SIZE_MAX_X),
			min(max(gridSize.y, blockSize.y), BLOCK_SIZE_MAX_Y),
			1);

		scanKernelExclusive_Phase2<<<1, blockSizePhase2>>>(
			&d_FalseKeyAddresses[2],
			&d_FalseKeyAddresses[2],
			size,
			blockSize.x * blockSize.y);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_Out, &d_FalseKeyAddresses[2], size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

#endif

		scanKernelExclusive_Phase3<<<gridSize, blockSize>>>(
			&d_FalseKeyAddresses[2],
			&d_FalseKeyAddresses[2],
			size,
			&d_FalseKeyAddresses[1]);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_Out, &d_FalseKeyAddresses[2], size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

#endif

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_TotalFalsesInput, d_FalseKeyAddresses, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		scatterKernel<<<1, size>>>(
			_src,
			&d_FalseKeyAddresses[2],
			_dst,
			h_TotalFalsesInput[0] + h_TotalFalsesInput[1],
			size,
			i);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(h_Out, _dst, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

#endif

		//swap the buffers (pointers only)
		std::swap(_dst, _src);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_Out, _dst, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_In);
    cudaFree(d_Out);
    
    return cudaStatus;
}
