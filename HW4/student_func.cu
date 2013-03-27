//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define MAX_THREAD_BLOCK_SIZE 512

#define BLOCK_SIZE_MAX_X 22
#define BLOCK_SIZE_MAX_Y 22

// #define DEBUG

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
        const unsigned int *d_InPos,
		const unsigned int *d_FalseKeyAddresses,
		unsigned int *d_Out,
        unsigned int *d_OutPos,
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

        d_OutPos[_destinationAddress] =
			d_InPos[myId];
	}
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int *d_FalseKeyAddresses = 0;
	unsigned int h_TotalFalsesInput[] = { 0, 0 };

    unsigned int *vals_src = d_inputVals;
    unsigned int *pos_src  = d_inputPos;

    unsigned int *vals_dst = d_outputVals;
    unsigned int *pos_dst  = d_outputPos;

#if defined(DEBUG)

    unsigned int *h_inputVals =
        new unsigned int[numElems];
    unsigned int *h_inputPos =
        new unsigned int[numElems];

    checkCudaErrors(
        cudaMemcpy(
            h_inputVals,
            d_inputVals,
            numElems * sizeof(unsigned int),
            cudaMemcpyDeviceToHost));

    checkCudaErrors(
        cudaMemcpy(
            h_inputPos,
            d_inputPos,
            numElems * sizeof(unsigned int),
            cudaMemcpyDeviceToHost));

    unsigned int *h_outputVals =
        new unsigned int[numElems];
    unsigned int *h_outputPos =
        new unsigned int[numElems];

#endif

    // Block size (i.e., number of threads per block)
	dim3 blockSizeSplitAndScatter(
        BLOCK_SIZE_MAX_X,
        BLOCK_SIZE_MAX_Y,
        1);

    int gridSizeSplitAndScatterX =
        (ceil(sqrt((double)numElems)) - 1) / BLOCK_SIZE_MAX_X + 1;

    int gridSizeSplitAndScatterY =
        (ceil(sqrt((double)numElems)) - 1) / BLOCK_SIZE_MAX_Y + 1;
    
    dim3 gridSizeSplitAndScatter(
        gridSizeSplitAndScatterX,
        gridSizeSplitAndScatterY,
        1);

    // d_FalseKeyAddresses[0]: Last value set in split operation
	// d_FalseKeyAddresses[1]: Number of least significant bits 'bitPos' that were 'false' (minus 1)
	// d_FalseKeyAddresses[2 ... 2 + numElems - 1]: Addresses for the 'false' keys
    checkCudaErrors(
        cudaMalloc(
            (void**)&d_FalseKeyAddresses,
            2 * sizeof(unsigned int) + numElems * sizeof(unsigned int)));

	checkCudaErrors(
        cudaMemset(
            d_FalseKeyAddresses,
            0x0, 2 * sizeof(unsigned int) + numElems * sizeof(unsigned int)));

    const int numBits = 1;

	for (
        unsigned int i = 0;
        i < 8 * sizeof(unsigned int);
        i += numBits)
	{
        // Launch a kernel on the GPU with one thread for each element.
        splitKernel<<<gridSizeSplitAndScatter, blockSizeSplitAndScatter>>>(
			vals_src,
			vals_dst,
			numElems,
			i,
			&d_FalseKeyAddresses[0]);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_outputVals,
                vals_dst,
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#endif

        // Block size (i.e., number of threads per block)
	    dim3 blockSizeScan(
            BLOCK_SIZE_MAX_X,
            BLOCK_SIZE_MAX_Y,
            1);

        int gridSizeScanX =
            (ceil(sqrt((double)numElems)) - 1) / BLOCK_SIZE_MAX_X + 1;

        int gridSizeScanY =
            (ceil(sqrt((double)numElems)) - 1) / BLOCK_SIZE_MAX_Y + 1;
    
        dim3 gridSizeScan(
            gridSizeScanX,
            gridSizeScanY,
            1);

        scanKernelExclusive_Phase1<<<gridSizeScan, blockSizeScan>>>(
            vals_dst,
            &d_FalseKeyAddresses[2],
            numElems);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_outputVals,
                &d_FalseKeyAddresses[2],
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#endif

        // Block size (i.e., number of threads per block)
		dim3 gridSizeScanPhase2(
			min(max(gridSizeScan.x, blockSizeScan.x), BLOCK_SIZE_MAX_X),
			min(max(gridSizeScan.y, blockSizeScan.y), BLOCK_SIZE_MAX_Y),
			1);

        scanKernelExclusive_Phase2<<<1, gridSizeScanPhase2>>>(
            &d_FalseKeyAddresses[2],
            &d_FalseKeyAddresses[2],
            numElems,
            blockSizeScan.x * blockSizeScan.y);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_outputVals,
                &d_FalseKeyAddresses[2],
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#endif

        scanKernelExclusive_Phase3<<<gridSizeScan, blockSizeScan>>>(
            &d_FalseKeyAddresses[2],
            &d_FalseKeyAddresses[2],
            numElems,
            &d_FalseKeyAddresses[1]);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_outputVals,
                &d_FalseKeyAddresses[2],
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#endif

        // Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_TotalFalsesInput,
                d_FalseKeyAddresses,
                2 * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

        scatterKernel<<<gridSizeSplitAndScatter, blockSizeSplitAndScatter>>>(
			vals_src,
            pos_src,
			&d_FalseKeyAddresses[2],
			vals_dst,
            pos_dst,
			h_TotalFalsesInput[0] + h_TotalFalsesInput[1],
			numElems,
			i);

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_outputVals,
                vals_dst,
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

        checkCudaErrors(
            cudaMemcpy(
                h_outputPos,
                pos_dst,
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#endif

        //swap the buffers (pointers only)
		std::swap(vals_dst, vals_src);
        std::swap(pos_dst, pos_src);

    }

    checkCudaErrors(
		cudaMemcpy(
			d_outputVals,
			d_inputVals,
            sizeof(unsigned int) *  numElems,
			cudaMemcpyDeviceToDevice));
    
    checkCudaErrors(
		cudaMemcpy(
			d_outputPos,
			d_inputPos,
            sizeof(unsigned int) *  numElems,
			cudaMemcpyDeviceToDevice));

#if defined(DEBUG)

		// Copy output vector from GPU buffer to host memory.
		checkCudaErrors(
            cudaMemcpy(
                h_outputVals,
                d_outputVals,
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

        checkCudaErrors(
            cudaMemcpy(
                h_outputPos,
                d_outputPos,
                numElems * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));

#endif

#if defined(DEBUG)

    delete []h_outputVals;
    delete []h_outputPos;

    delete []h_inputVals;
    delete []h_inputPos;

#endif

    checkCudaErrors(
        cudaFree(
            d_FalseKeyAddresses));
}
