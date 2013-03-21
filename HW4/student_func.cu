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


// #define USE_REFERENCE_CALCULATION

#if !defined(USE_REFERENCE_CALCULATION)

#define HISTOGRAM_ON_GPU

#if defined(HISTOGRAM_ON_GPU)

// #define HISTOGRAM_ON_GPU_DBG

#endif

#define SCAN_ON_GPU
// #define GATHER_ON_GPU

#endif

#define BLOCK_SIZE_MAX_X 20
#define BLOCK_SIZE_MAX_Y 20

__global__
    void simple_histo_binary(
        unsigned int *d_Bins,
        const unsigned int *d_In,
        const unsigned int mask,
        const unsigned int index,
        const size_t numElems)
{
    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

    if (myId < numElems)
    {
        unsigned int myItem = d_In[myId];

        int myBin = (myItem & mask) >> index;

        atomicAdd(&(d_Bins[myBin]), 1);
    }
}

__global__
	void scan_naive_exclusive(unsigned int *d_In,
							  unsigned int *d_Out,
							  const size_t numElems)
{
	int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

    if (myId < numElems)
    {
		for (int i = 0; i < myId; i++)
		{
			d_Out[myId] +=
				d_In[i];
		}
	}
}

__global__
    void gather(
        unsigned int *d_InVal,
        unsigned int *d_InPos,
        unsigned int *d_OutVal,
        unsigned int *d_OutPos,
        unsigned int *d_Bins,
        const unsigned int mask,
        const unsigned int index,
        const size_t numElems)
{
    int threadsPerBlock = blockDim.x * blockDim.y;

    int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

    int myId = (blockId * threadsPerBlock) + threadId;

    if (myId < numElems)
    {
        unsigned int myItem = d_InVal[myId];

        int myBin = (myItem & mask) >> index;
        
        d_OutVal[d_Bins[myBin]] = d_InVal[myId];
        d_OutPos[d_Bins[myBin]]  = d_InPos[myId];

        atomicAdd(&(d_Bins[myBin]), 1);
    }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{

#if defined(USE_REFERENCE_CALCULATION) || !defined(GATHER_ON_GPU)

    unsigned int *h_inputVals =
		new unsigned int[sizeof(unsigned int) * numElems];
    
    unsigned int *h_inputPos =
		new unsigned int[sizeof(unsigned int) * numElems];
    
    unsigned int *h_outputVals =
		new unsigned int[sizeof(unsigned int) * numElems];
    
    unsigned int *h_outputPos =
		new unsigned int[sizeof(unsigned int) * numElems];

    checkCudaErrors(
		cudaMemcpy(
			h_inputVals,
			d_inputVals,
            sizeof(unsigned int) * numElems,
			cudaMemcpyDeviceToHost));
    
    checkCudaErrors(
		cudaMemcpy(
			h_inputPos,
			d_inputPos,
            sizeof(unsigned int) * numElems,
			cudaMemcpyDeviceToHost));

    memset(
        h_outputVals,
        0x0,
        sizeof(unsigned int) * numElems);

    memset(
        h_outputPos,
        0x0,
        sizeof(unsigned int) * numElems);

#endif

#if defined(USE_REFERENCE_CALCULATION)
    
    reference_calculation(
        h_inputVals,
        h_inputPos,
        h_outputVals,
        h_outputPos,
        numElems);

    checkCudaErrors(
		cudaMemcpy(
			d_outputVals,
			h_outputVals,
			sizeof(unsigned int) * numElems,
			cudaMemcpyHostToDevice));
    
    checkCudaErrors(
		cudaMemcpy(
			d_outputPos,
			h_outputPos,
			sizeof(unsigned int) * numElems,
			cudaMemcpyHostToDevice));

#else

  const int numBits = 8;
  const int numBins = 1 << numBits;
    
  unsigned int *binHistogram = new unsigned int[numBins];
    
#if defined(HISTOGRAM_ON_GPU)
    
    unsigned int *d_binHisto;
    
    checkCudaErrors(
        cudaMalloc(
            &d_binHisto,
            sizeof(unsigned int) *  numBins));

    // Block size (i.e., number of threads per block)
	dim3 blockSizeHistogram(
        BLOCK_SIZE_MAX_X,
        BLOCK_SIZE_MAX_Y,
        1);

    int gridSizeHistogramX =
        (ceil(sqrt((double)numElems)) - 1) / BLOCK_SIZE_MAX_X + 1;

    int gridSizeHistogramY =
        (ceil(sqrt((double)numElems)) - 1) / BLOCK_SIZE_MAX_Y + 1;
    
    dim3 gridSizeHistogram(
        gridSizeHistogramX,
        gridSizeHistogramY,
        1);
    
#endif

#if defined(SCAN_ON_GPU)
    
    unsigned int *d_Scan;
    
    checkCudaErrors(
        cudaMalloc(
            &d_Scan,
            sizeof(unsigned int) *  numBins));

    checkCudaErrors(
        cudaMemset(
            d_Scan,
            0x0,
            sizeof(unsigned int) *  numBins));

    // Block size (i.e., number of threads per block)
	dim3 blockSizeScan(
        BLOCK_SIZE_MAX_X,
        BLOCK_SIZE_MAX_Y,
        1);

    int gridSizeScanX =
        (ceil(sqrt((double)numBins)) - 1) / BLOCK_SIZE_MAX_X + 1;

    int gridSizeScanY =
        (ceil(sqrt((double)numBins)) - 1) / BLOCK_SIZE_MAX_Y + 1;
    
    dim3 gridSizeScan(
        gridSizeScanX,
        gridSizeScanY,
        1);
    
#endif
    
  unsigned int *binScan      = new unsigned int[numBins];

#if defined(GATHER_ON_GPU)
    
    unsigned int *vals_src = d_inputVals;
    unsigned int *pos_src  = d_inputPos;

    unsigned int *vals_dst = d_outputVals;
    unsigned int *pos_dst  = d_outputPos;

#else

  unsigned int *vals_src = h_inputVals;
  unsigned int *pos_src  = h_inputPos;

  unsigned int *vals_dst = h_outputVals;
  unsigned int *pos_dst  = h_outputPos;

#endif

  //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    memset(binHistogram, 0, sizeof(unsigned int) * numBins); //zero out the bins
    memset(binScan, 0, sizeof(unsigned int) * numBins); //zero out the bins

#if defined (HISTOGRAM_ON_GPU)

    checkCudaErrors(
        cudaMemset(
            d_binHisto,
            0x0,
            sizeof(unsigned int) *  numBins));

    simple_histo_binary<<<gridSizeHistogram, blockSizeHistogram>>>(
        d_binHisto,
        d_inputVals,
        mask,
        i,
        numElems);

#if !defined(SCAN_ON_GPU)

    checkCudaErrors(
		cudaMemcpy(
			binHistogram,
			d_binHisto,
			sizeof(unsigned int) * numBins,
			cudaMemcpyDeviceToHost));

#elif defined(HISTOGRAM_ON_GPU_DBG)

    unsigned int *h_binHisto_dbg =
        new unsigned int[numBins];

    memset(
        h_binHisto_dbg,
        0x0,
        sizeof(unsigned int) * numBins);

    checkCudaErrors(
		cudaMemcpy(
			h_binHisto_dbg,
			d_binHisto,
			sizeof(unsigned int) * numBins,
			cudaMemcpyDeviceToHost));

    delete []h_binHisto_dbg;

#endif

#else
      
    //perform histogram of data & mask into bins
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      binHistogram[bin]++;
    }

#endif

#if defined(SCAN_ON_GPU)

    checkCudaErrors(
        cudaMemset(
            d_Scan,
            0x0,
            sizeof(unsigned int) *  numBins));

    scan_naive_exclusive<<<gridSizeScan, blockSizeScan>>>(
        d_binHisto,
        d_Scan,
        numBins);

    checkCudaErrors(
		cudaMemcpy(
			binScan,
			d_Scan,
            sizeof(unsigned int) *  numBins,
			cudaMemcpyDeviceToHost));
      
#else

    //perform exclusive prefix sum (scan) on binHistogram to get starting
    //location for each bin
    for (unsigned int j = 1; j < numBins; ++j) {
      binScan[j] = binScan[j - 1] + binHistogram[j - 1];
    }

#endif

#if defined(GATHER_ON_GPU)
      
      gather<<<gridSizeHistogram, blockSizeHistogram>>>(
        vals_src,
        pos_src,
        vals_dst,
        pos_dst,
        d_Scan,
        mask,
        i,
        numElems);

#else

    //Gather everything into the correct location
    //need to move vals and positions
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      vals_dst[binScan[bin]] = vals_src[j];
      pos_dst[binScan[bin]]  = pos_src[j];
      binScan[bin]++;
    }

#endif

    //swap the buffers (pointers only)
    std::swap(vals_dst, vals_src);
    std::swap(pos_dst, pos_src);
  }

#if defined(GATHER_ON_GPU)

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
    
#else

  //we did an even number of iterations, need to copy from input buffer into output
  std::copy(h_inputVals, h_inputVals + numElems, h_outputVals);
  std::copy(h_inputPos, h_inputPos + numElems, h_outputPos);

#endif

  delete[] binHistogram;
  delete[] binScan;

#if !defined(GATHER_ON_GPU)
    
    checkCudaErrors(
		cudaMemcpy(
			d_outputVals,
			h_outputVals,
			sizeof(unsigned int) * numElems,
			cudaMemcpyHostToDevice));
    
    checkCudaErrors(
		cudaMemcpy(
			d_outputPos,
			h_outputPos,
			sizeof(unsigned int) * numElems,
			cudaMemcpyHostToDevice));

#endif

#endif

#if defined(USE_REFERENCE_CALCULATION) || !defined(GATHER_ON_GPU)

    delete []h_outputPos;
    
    delete []h_outputVals;
    
    delete []h_inputPos;
    
    delete []h_inputVals;

#endif

#if defined (HISTOGRAM_ON_GPU)

    checkCudaErrors(cudaFree(d_binHisto));

#endif

#if defined (SCAN_ON_GPU)

    checkCudaErrors(cudaFree(d_Scan));

#endif

}
