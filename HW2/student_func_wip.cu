// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "reference_calc.cpp"
#include "utils.h"

// #define USE_PRINTF_FOR_DEBUG

#define BLOCK_SIZE_MAX_X 9						// i.e. maximum number of threads per block (x dimension)
#define BLOCK_SIZE_MAX_Y 9						// i.e. maximum number of threads per block (y dimension)

#define USE_SHARED_MEMORY_FOR_FILTER			// Optimization 1: Move filter array into shared memory

#if defined(USE_SHARED_MEMORY_FOR_FILTER)
#define MAX_FILTER_WIDTH 9
#endif

#define USE_OPTIMIZED_BLUR						// Optimization 2: Use optimzed version of the blurring kernel
												//                 which obsoletes the channel separation and
												//                 recombination kernels and using local memory
												//				   instead of global memory for that purpose

// #define USE_SHARED_MEMORY_FOR_IMAGE

__global__
	void gaussian_blur_optimized(const uchar4* const inputImageRGBA,
								 uchar4* const outputImageRGBA,
								 int numRows, int numCols,
								 const float* const filter, const int filterWidth)
{
#if defined(USE_SHARED_MEMORY_FOR_FILTER)

	__shared__ float filterShared[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];

#endif

#if defined(USE_SHARED_MEMORY_FOR_IMAGE)

	__shared__ uchar3 imageShared[(BLOCK_SIZE_MAX_X + MAX_FILTER_WIDTH / 2) * (BLOCK_SIZE_MAX_Y + MAX_FILTER_WIDTH / 2) * sizeof(uchar3)];

#endif

	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);

	if ( thread_2D_pos.x >= numCols ||
		 thread_2D_pos.y >= numRows )
	{
		return;
	}
	else
	{
		const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

#if defined(USE_SHARED_MEMORY_FOR_FILTER)

		if (filterWidth <= MAX_FILTER_WIDTH)
		{
			if (threadIdx.x < MAX_FILTER_WIDTH &&
				threadIdx.y < MAX_FILTER_WIDTH)
			{
				filterShared[threadIdx.y * filterWidth + threadIdx.x] =
					filter[threadIdx.y * filterWidth + threadIdx.x];
			}

			__syncthreads();
		}

#endif

#if defined(USE_SHARED_MEMORY_FOR_IMAGE)

		int blockDimXTotal = blockDim.x + (filterWidth / 2) * 2;
		int blockDimYTotal = blockDim.y + (filterWidth / 2) * 2;

		if (threadIdx.x == 0 &&
			threadIdx.y == 0)
		{
			for (int _row = (-filterWidth / 2); _row < (blockDimYTotal); _row++)
			{
				int _rowGlobal = (blockIdx.y * blockDim.y + _row);

				if ((_rowGlobal < 0) ||
					(_rowGlobal >= numRows))
					continue;

				for (int _col = (-filterWidth / 2); _col < blockDimXTotal; _col++)
				{
					int _colGlobal = (blockIdx.x * blockDim.x + _col);

					if ((_colGlobal < 0) ||
						(_colGlobal >= numCols))
						continue;

					if ((_row < 0) ||
						(_row >= blockDim.y) ||
						(_col < 0) ||
						(_col >= blockDim.x))
					{
						int _posLocal =
							(_row + filterWidth / 2) * (blockDim.x + 2 * (filterWidth / 2)) + (_col + filterWidth / 2);

						int _posGlobal =
							(_rowGlobal * numCols + _colGlobal);

						imageShared[_posLocal].x = inputImageRGBA[_posGlobal].x;
						imageShared[_posLocal].y = inputImageRGBA[_posGlobal].y;
						imageShared[_posLocal].z = inputImageRGBA[_posGlobal].z;
					}
				}
			}
		}

		const int thread_1D_pos_local =
			(threadIdx.y + filterWidth / 2) * (blockDimXTotal) + (threadIdx.x + filterWidth / 2);

		
		imageShared[thread_1D_pos_local].x = inputImageRGBA[thread_1D_pos].x;
		imageShared[thread_1D_pos_local].y = inputImageRGBA[thread_1D_pos].y;
		imageShared[thread_1D_pos_local].z = inputImageRGBA[thread_1D_pos].z;
		

		__syncthreads();
		

#endif

		int c = thread_2D_pos.x;
		int r = thread_2D_pos.y;

		float result_r = 0.f;
		float result_g = 0.f;
		float result_b = 0.f;

		//For every value in the filter around the pixel (c, r)
		for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
			for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
				//Find the global image position for this filter position
				//clamp to boundary of the image
				int image_r = min(max(r + filter_r, 0), static_cast<int>(numRows - 1));
				int image_c = min(max(c + filter_c, 0), static_cast<int>(numCols - 1));

#if defined(USE_SHARED_MEMORY_FOR_IMAGE)

				float image_value_r = 0.f;
				float image_value_g = 0.f;
				float image_value_b = 0.f;

				int image_r_local = image_r - blockIdx.y * blockDim.y + filterWidth / 2;
				int image_c_local = image_c - blockIdx.x * blockDim.x + filterWidth / 2;

				/*
				if (image_r_local >= (filterWidth / 2) &&
					image_r_local < (blockDim.y) &&
					image_c_local >= (filterWidth / 2) &&
					image_c_local < (blockDim.x))*/
				{
					image_value_r = static_cast<float>(imageShared[image_r_local * (blockDimXTotal) + image_c_local].x);
					image_value_g = static_cast<float>(imageShared[image_r_local * (blockDimXTotal) + image_c_local].y);
					image_value_b = static_cast<float>(imageShared[image_r_local * (blockDimXTotal) + image_c_local].z);
				}
				// else
				{
					/*
					image_value_r = static_cast<float>(inputImageRGBA[image_r * numCols + image_c].x);
					image_value_g = static_cast<float>(inputImageRGBA[image_r * numCols + image_c].y);
					image_value_b = static_cast<float>(inputImageRGBA[image_r * numCols + image_c].z);
					*/
				}

#else

				float image_value_r = static_cast<float>(inputImageRGBA[image_r * numCols + image_c].x);
				float image_value_g = static_cast<float>(inputImageRGBA[image_r * numCols + image_c].y);
				float image_value_b = static_cast<float>(inputImageRGBA[image_r * numCols + image_c].z);

#endif

#if defined(USE_SHARED_MEMORY_FOR_FILTER)

				float filter_value = filterShared[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
				
#else

				float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

#endif

				result_r += image_value_r * filter_value;
				result_g += image_value_g * filter_value;
				result_b += image_value_b * filter_value;
			}
		}

		outputImageRGBA[thread_1D_pos].x = result_r;
		outputImageRGBA[thread_1D_pos].y = result_g;
		outputImageRGBA[thread_1D_pos].z = result_b;
		outputImageRGBA[thread_1D_pos].w = inputImageRGBA[thread_1D_pos].w;
	}
}

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.

#if defined(USE_SHARED_MEMORY_FOR_FILTER)

	__shared__ float filterShared [MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];

#endif

	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);

	if ( thread_2D_pos.x >= numCols ||
		 thread_2D_pos.y >= numRows )
	{
		return;
	}
	else
	{
		const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

#if defined(USE_SHARED_MEMORY_FOR_FILTER)

		if (filterWidth <= MAX_FILTER_WIDTH)
		{
			if (threadIdx.x < MAX_FILTER_WIDTH &&
				threadIdx.y < MAX_FILTER_WIDTH)
			{
				filterShared[threadIdx.y * filterWidth + threadIdx.x] =
					filter[threadIdx.y * filterWidth + threadIdx.x];
			}

			__syncthreads();
		}

#endif

		int c = thread_2D_pos.x;
		int r = thread_2D_pos.y;

		float result = 0.f;

		//For every value in the filter around the pixel (c, r)
		for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
			for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
				//Find the global image position for this filter position
				//clamp to boundary of the image
				int image_r = min(max(r + filter_r, 0), static_cast<int>(numRows - 1));
				int image_c = min(max(c + filter_c, 0), static_cast<int>(numCols - 1));

				float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);

#if defined(USE_SHARED_MEMORY_FOR_FILTER)

				float filter_value = filterShared[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
				
#else

				float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

#endif

				result += image_value * filter_value;
			}
		}

		outputChannel[thread_1D_pos] = result;
	}
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }

	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);

	if ( thread_2D_pos.x >= numCols ||
		 thread_2D_pos.y >= numRows )
	{
		return;
	}
	else
	{
		const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

		redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
		greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
		blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
	}
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc

#if defined(USE_PRINTF_FOR_DEBUG)

  printf("Filter width is %i\n", filterWidth);
  printf("Size of float is %i bytes\n", sizeof(float));

#endif

  size_t _filterByteSize =
	  filterWidth * filterWidth * sizeof(float);

  checkCudaErrors(cudaMalloc(&d_filter, _filterByteSize));

#if defined(USE_PRINTF_FOR_DEBUG)

  printf("Allocated %i bytes (in global memory) on device for filter\n", _filterByteSize);

#endif

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, _filterByteSize, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{

#if defined(USE_PRINTF_FOR_DEBUG)

	printf("Image is %i columns x %i rows\n",
		numCols,
		numRows);

#endif

	int gridSizeX = (numCols - 1) / BLOCK_SIZE_MAX_X + 1;
	int gridSizeY = (numRows - 1) / BLOCK_SIZE_MAX_Y + 1;

  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(BLOCK_SIZE_MAX_X, BLOCK_SIZE_MAX_Y, 1);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(gridSizeX, gridSizeY, 1);

#if defined(USE_PRINTF_FOR_DEBUG)

	printf("Blocksize\tX: %i\tY: %i\tZ: %i\n",
		blockSize.x,
		blockSize.y,
		blockSize.z);

	printf("Gridsize\tX: %i\tY: %i\tZ: %i\n",
		gridSize.x,
		gridSize.y,
		gridSize.z);

#endif

#if defined(USE_OPTIMIZED_BLUR)

	gaussian_blur_optimized<<<gridSize, blockSize>>>(d_inputImageRGBA,
													 d_outputImageRGBA,
													 numRows,
													 numCols,
													 d_filter,
													 filterWidth);

#else

  //TODO: Launch a kernel for separating the RGBA image into different color channels
	separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                             numRows,
                                             numCols,
											 d_red,
											 d_green,
											 d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red,
									     d_redBlurred,
										 numRows,
										 numCols,
										 d_filter,
										 filterWidth);

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_green,
									     d_greenBlurred,
										 numRows,
										 numCols,
										 d_filter,
										 filterWidth);

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_blue,
									     d_blueBlurred,
										 numRows,
										 numCols,
										 d_filter,
										 filterWidth);

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#endif

  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference image on the host by running the          *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated image back to the host and calls a function that compares the   *
  * the two and will output the first location they differ by too much.       *
  * ************************************************************************* */

  /*uchar4 *h_outputImage     = new uchar4[numRows * numCols];
  uchar4 *h_outputReference = new uchar4[numRows * numCols];

  checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImageRGBA, 
                             numRows * numCols * sizeof(uchar4), 
                             cudaMemcpyDeviceToHost));

  referenceCalculation(h_inputImageRGBA, h_outputReference, numRows, numCols,
                       h_filter, filterWidth);

  //the 4 is because there are 4 channels in the image
  checkResultsExact((unsigned char *)h_outputReference,
                    (unsigned char *)h_outputImage,
                    numRows * numCols * 4); 
 
  delete [] h_outputImage;
  delete [] h_outputReference;*/
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_filter));
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}
