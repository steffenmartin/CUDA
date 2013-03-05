/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include "cuda_runtime.h"
#include <math.h>

// #define USE_PRINTF_FOR_DEBUG

// #define ENABLE_REF_CHECK

#define BLOCK_SIZE_MAX_X 16						// i.e. maximum number of threads per block (x dimension)
												// Note: In this particular application it should be a power
												//       of 2
#define BLOCK_SIZE_MAX_Y 16						// i.e. maximum number of threads per block (y dimension)
												// Note: In this particular application it should be a power
												//       of 2

#define BLOCK_SIZE_HISTO_MAX_X 22				// i.e. maximum number of threads per block (x dimension)
#define BLOCK_SIZE_HISTO_MAX_Y 22				// i.e. maximum number of threads per block (y dimension)

#define BLOCK_SIZE_SCAN_MAX_X 512				// i.e. maximum number of threads per block (x dimension)
#define BLOCK_SIZE_SCAN_MAX_Y 1					// i.e. maximum number of threads per block (y dimension)

__global__
	void global_find_min(float *d_Out,
						 float *d_OutMax,
						 float *d_In,
						 float *d_InMax,
						 int numRows, int numCols)
{
	// Determine position in 2D (for boundary checking)
	const int2 thread_2D_pos =
		make_int2( blockIdx.x * blockDim.x + threadIdx.x,
				   blockIdx.y * blockDim.y + threadIdx.y);

	if ( thread_2D_pos.x >= numCols ||
		 thread_2D_pos.y >= numRows )
	{
		return;
	}
	else
	{
		// Determine position in 1D (walking along columns and jumping to the next row at the end of the colum).
		const int _thread_1D_pos =
			thread_2D_pos.y * numCols + thread_2D_pos.x;

		// Let's calculate total number of pixels (just once)
		const int numPixelTotal =
			numRows * numCols;

		// Let's determine the number of pixel this block is working on
		const int numPixelBlock =
			(blockDim.x * blockDim.y);

		int thread_1D_pos =
			blockIdx.x * numPixelBlock + threadIdx.y * blockDim.x + threadIdx.x;
		thread_1D_pos +=
			blockIdx.y * blockDim.y * numCols;

		// Let's determine the index inside of this block
		int tid =
			threadIdx.y * blockDim.x + threadIdx.x;

		// do reduction in global mem
		for (unsigned int s = numPixelBlock / 2; s > 0; s >>= 1)
		{
			if (tid < s &&
				(thread_1D_pos + s) < numPixelTotal)
			{
				d_In[thread_1D_pos] =
					min(d_In[thread_1D_pos], d_In[thread_1D_pos + s]);
				d_InMax[thread_1D_pos] =
					max(d_InMax[thread_1D_pos], d_InMax[thread_1D_pos + s]);
			}
			__syncthreads();        // make sure all min/max at one stage are done!
		}

		// only thread 0 writes result for this block back to global mem
		if (tid == 0)
		{
			d_Out[thread_1D_pos / numPixelBlock] = d_In[thread_1D_pos];
			d_OutMax[thread_1D_pos / numPixelBlock] = d_InMax[thread_1D_pos];
		}
	}
}

__global__
	void simple_histo(unsigned int *d_bins,
					  const float *d_In,
					  const unsigned int BIN_COUNT,
					  float _min,
					  float _range,
					  int numRows, int numCols)
{
	// Determine position in 2D (for boundary checking)
	const int2 thread_2D_pos =
		make_int2( blockIdx.x * blockDim.x + threadIdx.x,
				   blockIdx.y * blockDim.y + threadIdx.y);

	if ( thread_2D_pos.x >= numCols ||
		 thread_2D_pos.y >= numRows )
	{
		return;
	}
	else
	{
		// const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
		const int thread_1D_pos =
			blockIdx.y * blockDim.y * gridDim.x +
			blockIdx.x * blockDim.x * blockDim.y +
			threadIdx.y * blockDim.x +
			threadIdx.x;

		float myItem = d_In[thread_1D_pos];
		
		// int myBin = ((myItem - d_Min[0]) / (d_Max[0] - d_Min[0])) * BIN_COUNT;

		unsigned int myBin = 
			min(
				static_cast<unsigned int>(BIN_COUNT - 1),
				static_cast<unsigned int>((myItem - _min) / _range * BIN_COUNT));

		atomicAdd(&(d_bins[myBin]), 1);
	}
}

__global__
	void scan_naive_exclusive(unsigned int *d_In,
							  unsigned int *d_Out,
							  int numElements)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;

	if (myId < numElements)
	{
		for (int i = 0; i < myId; i++)
		{
			d_Out[myId] +=
				d_In[i];
		}
	}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
#if defined(USE_PRINTF_FOR_DEBUG)

	printf("Image is %i columns x %i rows\n",
		numCols,
		numRows);

	printf("Number of bins is %i\n",
		numBins);

#endif

  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
	   */

	int gridSizeX = (numCols - 1) / BLOCK_SIZE_MAX_X + 1;
	int gridSizeY = (numRows - 1) / BLOCK_SIZE_MAX_Y + 1;

	// Block size (i.e., number of threads per block)
	dim3 blockSize(BLOCK_SIZE_MAX_X, BLOCK_SIZE_MAX_Y, 1);

	// Compute grid size (i.e., number of blocks per kernel launch)
	// from the image size and and block size.
	dim3 gridSize(gridSizeX, gridSizeY, 1);

	float *d_IntermediateIn;
	float *d_IntermediateInMax;
	// Allocate memory on the device for storing the intermediate input values and copy them
	checkCudaErrors(
		cudaMalloc(
			&d_IntermediateIn,
			sizeof(float) * numRows * numCols));
	checkCudaErrors(
		cudaMalloc(
			&d_IntermediateInMax,
			sizeof(float) * numRows * numCols));
	checkCudaErrors(
		cudaMemcpy(
			d_IntermediateIn,
			d_logLuminance,
			sizeof(float) * numRows * numCols,
			cudaMemcpyDeviceToDevice));
	checkCudaErrors(
		cudaMemcpy(
			d_IntermediateInMax,
			d_logLuminance,
			sizeof(float) * numRows * numCols,
			cudaMemcpyDeviceToDevice));

	float *d_IntermediateOut;
	float *d_IntermediateOutMax;
	// Allocate memory on the device for storing the intermediate output values
	checkCudaErrors(
		cudaMalloc(
			&d_IntermediateOut,
			sizeof(float) * gridSizeX * gridSizeY));
	checkCudaErrors(
		cudaMalloc(
			&d_IntermediateOutMax,
			sizeof(float) * gridSizeX * gridSizeY));
	checkCudaErrors(
		cudaMemset(
			d_IntermediateOut,
			0x0,
			sizeof(float) * gridSizeX * gridSizeY));
	checkCudaErrors(
		cudaMemset(
			d_IntermediateOutMax,
			0x0,
			sizeof(float) * gridSizeX * gridSizeY));

	float *d_MinOut;
	float *d_MaxOut;
	float *d_MinMaxOut;
	// Allocate memory on the device for storing the output value
	checkCudaErrors(
		cudaMalloc(
			&d_MinOut,
			sizeof(float)));
	checkCudaErrors(
		cudaMalloc(
			&d_MaxOut,
			sizeof(float)));
	checkCudaErrors(
		cudaMalloc(
			&d_MinMaxOut,
			2 * sizeof(float)));
	checkCudaErrors(
		cudaMemset(
			d_MinOut,
			0x0,
			sizeof(float)));
	checkCudaErrors(
		cudaMemset(
			d_MaxOut,
			0x0,
			sizeof(float)));
	checkCudaErrors(
		cudaMemset(
			d_MinMaxOut,
			0x0,
			2 * sizeof(float)));

#if defined(USE_PRINTF_FOR_DEBUG)

	float *h_Intermediate =
		// new float[sizeof(float) * gridSizeX * gridSizeY];
		new float[sizeof(float) * numRows * numCols];
	memset(
		h_Intermediate,
		0x0,
		// sizeof(float) * gridSizeX * gridSizeY);
		sizeof(float) * numRows * numCols);

	checkCudaErrors(
		cudaMemcpy(
			h_Intermediate,
			d_logLuminance,
			// sizeof(float) * gridSizeX * gridSizeY,
			sizeof(float) * numRows * numCols,
			cudaMemcpyDeviceToHost));

	checkCudaErrors(
		cudaMemcpy(
			h_Intermediate,
			d_IntermediateIn,
			// sizeof(float) * gridSizeX * gridSizeY,
			sizeof(float) * numRows * numCols,
			cudaMemcpyDeviceToHost));

	float h_Out = 0;

	printf("Blocksize\tX: %i\tY: %i\tZ: %i\n",
		blockSize.x,
		blockSize.y,
		blockSize.z);

	printf("Gridsize\tX: %i\tY: %i\tZ: %i\n",
		gridSize.x,
		gridSize.y,
		gridSize.z);

#endif

	global_find_min<<<gridSize, blockSize>>>
		(d_IntermediateOut,
		 d_IntermediateOutMax,
		 d_IntermediateIn,
		 d_IntermediateInMax,
		 numRows,
		 numCols);

#if defined(USE_PRINTF_FOR_DEBUG)

	checkCudaErrors(
		cudaMemcpy(
			h_Intermediate,
			d_IntermediateOut,
			sizeof(float) * gridSizeX * gridSizeY,
			cudaMemcpyDeviceToHost));

	checkCudaErrors(
		cudaMemcpy(
			h_Intermediate,
			d_IntermediateOutMax,
			sizeof(float) * gridSizeX * gridSizeY,
			cudaMemcpyDeviceToHost));

#endif

	global_find_min<<<1, blockSize>>>
		(&d_MinMaxOut[0],
		 &d_MinMaxOut[1],
		 d_IntermediateOut,
		 d_IntermediateOutMax,
		 gridSizeX,
		 gridSizeY);

#if defined(USE_PRINTF_FOR_DEBUG)

	checkCudaErrors(
		cudaMemcpy(
			&h_Out,
			&d_MinMaxOut[0],
			sizeof(float),
			cudaMemcpyDeviceToHost));

	printf("Min: %f\n", h_Out);

	checkCudaErrors(
		cudaMemcpy(
			&h_Out,
			&d_MinMaxOut[1],
			sizeof(float),
			cudaMemcpyDeviceToHost));

	printf("Max: %f\n", h_Out);

#endif

	/*
    2) subtract them to find the range
	*/

	float h_MinMaxOut[2];

	checkCudaErrors(
		cudaMemcpy(
			&h_MinMaxOut[0],
			d_MinMaxOut,
			2 * sizeof(float),
			cudaMemcpyDeviceToHost));

	min_logLum =
		h_MinMaxOut[0];
	max_logLum =
		h_MinMaxOut[1];
	float _logLumRange = max_logLum - min_logLum;

	/*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	   */

	unsigned int *d_Bins;

	// Allocate memory on the device for storing the intermediate output values
	checkCudaErrors(
		cudaMalloc(
			&d_Bins,
			sizeof(unsigned int) * numBins));
	checkCudaErrors(
		cudaMemset(
			d_Bins,
			0x0,
			sizeof(unsigned int) * numBins));

#if defined(USE_PRINTF_FOR_DEBUG)

	unsigned int *h_Bins =
		new unsigned int[numBins];

	memset(h_Bins, 0x0, sizeof(unsigned int) * numBins);

#endif

	gridSizeX = (numCols - 1) / BLOCK_SIZE_HISTO_MAX_X + 1;
	gridSizeY = (numRows - 1) / BLOCK_SIZE_HISTO_MAX_Y + 1;

	// Block size (i.e., number of threads per block)
	blockSize.x = BLOCK_SIZE_HISTO_MAX_X;
	blockSize.y = BLOCK_SIZE_HISTO_MAX_Y;

	// Compute grid size (i.e., number of blocks per kernel launch)
	// from the image size and and block size.
	gridSize.x = gridSizeX;
	gridSize.y = gridSizeY;

	simple_histo<<<gridSize, blockSize>>>(
		d_Bins,
		d_logLuminance,
		numBins,
		h_MinMaxOut[0],
		_logLumRange,
		numRows,
		numCols);

#if defined(USE_PRINTF_FOR_DEBUG)

	checkCudaErrors(
		cudaMemcpy(
			h_Bins,
			d_Bins,
			sizeof(unsigned int) * numBins,
			cudaMemcpyDeviceToHost));
#endif

	/*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	gridSizeX = (numBins - 1) / BLOCK_SIZE_SCAN_MAX_X + 1;
	gridSizeY = 1;

	// Block size (i.e., number of threads per block)
	blockSize.x = BLOCK_SIZE_SCAN_MAX_X;
	blockSize.y = BLOCK_SIZE_SCAN_MAX_Y;

	// Compute grid size (i.e., number of blocks per kernel launch)
	// from the image size and and block size.
	gridSize.x = gridSizeX;
	gridSize.y = gridSizeY;

	scan_naive_exclusive<<<gridSize, blockSize>>>(
		d_Bins,
		d_cdf,
		numBins);

  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference cdf on the host by running the            *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated cdf back to the host and calls a function that compares the     *
  * the two and will output the first location they differ.                   *
  * ************************************************************************* */

#if defined(ENABLE_REF_CHECK)

  float *h_logLuminance = new float[numRows * numCols];
  unsigned int *h_cdf   = new unsigned int[numBins];
  unsigned int *h_your_cdf = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, numCols * numRows * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_your_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins);

  //compare the results of the CDF
  // checkResultsExact(h_cdf, h_your_cdf, numBins);
  checkResultsEps(h_cdf, h_your_cdf, numBins, 3, 10);
 
  delete[] h_logLuminance;
  delete[] h_cdf; 
  delete[] h_your_cdf;

#endif

	checkCudaErrors(cudaFree(d_IntermediateIn));
	checkCudaErrors(cudaFree(d_IntermediateInMax));
	checkCudaErrors(cudaFree(d_IntermediateOut));
	checkCudaErrors(cudaFree(d_IntermediateOutMax));
	checkCudaErrors(cudaFree(d_Bins));

#if defined(USE_PRINTF_FOR_DEBUG)

	delete []h_Intermediate;
	delete []h_Bins;

#endif

}
