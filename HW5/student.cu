/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference_calc.cpp"

#define BLOCK_SIZE_MAX 512				// i.e. maximum number of threads per block
#define GRID_SIZE_MAX 512				// i.e. maximum number of blocks
#define NUMBER_OF_ELEMS_PER_THREAD 8	// Number of elements (values) to be processed per thread

// #define HISTO2

__global__
void consolidateKernel(const unsigned int* const d_In,	//INPUT: values
               unsigned int* const d_Out,				//OUPUT: histogram
			   int numVals,
			   unsigned int valsOffset,
			   unsigned int numBins)
{
	int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

	if ( ((valsOffset + myId) < numVals) &&
		 (myId < numBins) )
	{
		d_Out[myId] +=
			d_In[myId];
	}
}

__global__
void histogramKernel(const unsigned int* const d_In,	//INPUT: values
               unsigned int* const d_Out,				//OUPUT: histogram
               int numVals,
			   unsigned int valsOffset,
			   unsigned int numBins)
{
	int threadsPerBlock = blockDim.x * blockDim.y;

	int threadsPerGrid = threadsPerBlock * gridDim.x * gridDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

	for (
		int _step = 0;
		_step < NUMBER_OF_ELEMS_PER_THREAD;
		_step++)
	{
		int _myTrueId =
			myId + _step * threadsPerGrid;

		if ( (_myTrueId + valsOffset) >= numVals )
		{
			break;
		}
		else
		{
			unsigned int _in =
				d_In[_myTrueId];

			atomicAdd(&(d_Out[_in]), 1);
		}
	}
}

#if defined(HISTO2)

__global__
void histogramKernel2(const unsigned int* const d_In, //INPUT: values
               unsigned int* const d_Out,      //OUPUT: histogram
               int numVals,
			   unsigned int valsOffset,
			   unsigned int numBins)
{
	extern __shared__ unsigned int s_histogramKernel_Out[];

	int threadsPerBlock = blockDim.x * blockDim.y;

	int threadsPerGrid = threadsPerBlock * gridDim.x * gridDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

	for (
		int i = 0;
		i < (numBins / threadsPerBlock);
		i++)
	{
		int _index =
			i * threadsPerBlock + threadId;

		if (_index < numBins)
		{
			s_histogramKernel_Out[_index] =
				d_Out[_index];
		}
	}

	__syncthreads();

    int myId = (blockId * threadsPerBlock) + threadId;

	for (
		int _step = 0;
		_step < NUMBER_OF_ELEMS_PER_THREAD;
		_step++)
	{
		int _myTrueId =
			myId + _step * threadsPerGrid;

		if ( (_myTrueId + valsOffset) >= numVals )
		{
			break;
		}
		else
		{
			unsigned int _in =
				d_In[_myTrueId];

			atomicAdd(&(s_histogramKernel_Out[_in]), 1);
		}
	}

	__syncthreads();

	for (
		int i = 0;
		i < (numBins / threadsPerBlock);
		i++)
	{
		int _index =
			i * threadsPerBlock + threadId;

		if (_index < numBins)
		{
			d_Out[_index] =
				s_histogramKernel_Out[_index];
		}
	}
}

#define PARALLEL_HISTOS 2

#endif

void computeHistogram(const unsigned int* const d_In, //INPUT: values
                      unsigned int* const d_Out,      //OUTPUT: histogram
                      const unsigned int numBins,
                      const unsigned int numElems)
{
	unsigned int _numElemsProcessed = 0;

	dim3 _block(BLOCK_SIZE_MAX);

#if defined(HISTO2)

	unsigned int* d_OutTmp = 0;

	checkCudaErrors(
		cudaMalloc(
			&d_OutTmp,
			numBins * sizeof(unsigned int) * PARALLEL_HISTOS));

	if (d_OutTmp)
	{
		checkCudaErrors(
			cudaMemset(
				d_OutTmp,
				0x0,
				numBins * sizeof(unsigned int) * PARALLEL_HISTOS));
	}

	bool ping = true;

#endif

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

#if !defined(HISTO2)

		// Launch a kernel on the GPU with one thread for each element.
		histogramKernel<<<_grid, _block, (numBins * sizeof(unsigned int))>>>
			(&d_In[_numElemsProcessed],
			 d_Out,
			 numElems,
			 _numElemsProcessed,
			 numBins);

#else
		unsigned int *_d_OutTmp =
			ping ?
				&d_OutTmp[0]:
				&d_OutTmp[(PARALLEL_HISTOS >> 1) * numBins];

		// Launch a kernel on the GPU with one thread for each element.
		histogramKernel<<<_grid, _block, (numBins * sizeof(unsigned int))>>>
			(&d_In[_numElemsProcessed],
			 _d_OutTmp,
			 numElems,
			 _numElemsProcessed,
			 numBins);

		dim3 _blockConsolidate(BLOCK_SIZE_MAX);

		int _gridSizeConsolidate =
			(numBins - 1) / BLOCK_SIZE_MAX + 1;

		dim3 _gridConsolidate(_gridSizeConsolidate);

		consolidateKernel<<<_gridConsolidate, _blockConsolidate>>>
			(_d_OutTmp,
			d_Out,
			numElems,
			_numElemsProcessed,
			numBins);

		checkCudaErrors(
			cudaMemset(
				_d_OutTmp,
				0x0,
				numBins * sizeof(unsigned int)));

		ping = !ping;
#endif

		_numElemsProcessed +=
			_gridSize * BLOCK_SIZE_MAX * NUMBER_OF_ELEMS_PER_THREAD;
	}

  //if you want to use/launch more than one kernel,
  //feel free
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#if defined(HISTO2)

  if (d_OutTmp)
  {
	cudaFree(d_OutTmp);
  }

#endif

    /*
  delete[] h_vals;
  delete[] h_histo;
  delete[] your_histo;*/
}
