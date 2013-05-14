//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

#include "reference_calc_custom.h"

#define BLOCK_SIZE_CALC_MASK_MAX_X 22
#define BLOCK_SIZE_CALC_MASK_MAX_Y 22

#define ENABLE_DEBUG
// #define ENABLE_STRICT_ERROR_CHECKING

__global__
	void calculateMaskKernel(
		const uchar4 * const d_sourceImg,
		const size_t numRowsSource,
		const size_t numColsSource,
		unsigned char* d_mask

#if defined (ENABLE_DEBUG)

		, uchar4 * d_shared_for_debug

#endif

		)
{

#define MASK_KERNEL_USE_SHARED

#if defined (MASK_KERNEL_USE_SHARED)

	__shared__ uchar4 _shared[BLOCK_SIZE_CALC_MASK_MAX_X + 2][BLOCK_SIZE_CALC_MASK_MAX_Y + 2];

#endif

	const int2 threadPos2D =
		make_int2(
			blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);

	if ( threadPos2D.x >= numColsSource ||
		 threadPos2D.y >= numRowsSource )
	{
		return;
	}
	else
	{
		const int myId = 
			threadPos2D.y * numColsSource +
			threadPos2D.x;

#if defined (MASK_KERNEL_USE_SHARED)

		int myIdTop =
			(threadPos2D.y - 1) * numColsSource +
			threadPos2D.x;

		int myIdBottom =
			(threadPos2D.y + 1) * numColsSource +
			threadPos2D.x;

		int myIdLeft =
			threadPos2D.y * numColsSource +
			(threadPos2D.x - 1);

		int myIdRight =
			threadPos2D.y * numColsSource +
			(threadPos2D.x + 1);

		// Top left thread fetches top left neighbor (if available)
		if ((threadIdx.x == 0) &&
			(threadIdx.y == 0))
		{
			if ((threadPos2D.x > 0) &&
				(threadPos2D.y > 0))
			{
				_shared[threadIdx.x][threadIdx.y] =
					make_uchar4(
						(myIdTop - 1),
						(myIdTop - 1) >> 8,
						(myIdTop - 1) >> 16,
						(myIdTop - 1) >> 24);
					// d_sourceImg[myIdTop - 1];
			}
			else
			{
				_shared[threadIdx.x][threadIdx.y] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		__syncthreads();

		// Top row fetches all top neighbors
		if (threadIdx.y == 0)
		{
			if (threadPos2D.y > 0)
			{
				_shared[threadIdx.x + 1][threadIdx.y] =
					make_uchar4(
						(myIdTop),
						(myIdTop) >> 8,
						(myIdTop) >> 16,
						(myIdTop) >> 24);
					// d_sourceImg[myIdTop];
			}
			else
			{
				_shared[threadIdx.x + 1][threadIdx.y] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		__syncthreads();

		// Top right thread fetches top right neighbor (if available)
		if (((threadIdx.x == (blockDim.x - 1)) ||
			 (threadPos2D.x == (numColsSource - 1))) &&
			(threadIdx.y == 0))
		{
			if ((threadPos2D.x < numColsSource) &&
				(threadPos2D.y > 0))
			{
				_shared[threadIdx.x + 2][threadIdx.y] =
					make_uchar4(
						(myIdTop + 1),
						(myIdTop + 1) >> 8,
						(myIdTop + 1) >> 16,
						(myIdTop + 1) >> 24);
					// d_sourceImg[myIdTop + 1];
			}
			else
			{
				_shared[threadIdx.x + 2][threadIdx.y] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		__syncthreads();

		// Left column fetches all left neighbors
		if ((threadIdx.x == 0) &&
			(threadPos2D.y < (numRowsSource - 1)))
		{
			if (threadPos2D.x > 0)
			{
				_shared[threadIdx.x][threadIdx.y + 1] =
					make_uchar4(
						(myIdLeft),
						(myIdLeft) >> 8,
						(myIdLeft) >> 16,
						(myIdLeft) >> 24);
				// d_sourceImg[myIdLeft];
			}
			else
			{
				_shared[threadIdx.x][threadIdx.y + 1] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		_shared[threadIdx.x + 1][threadIdx.y + 1] =
			d_sourceImg[myId];

		__syncthreads();

		// Right column fetches all right neighbors
		if (((threadIdx.x == (blockDim.x - 1)) ||
			 (threadPos2D.x == (numColsSource - 1))) &&
			(threadPos2D.y < (numRowsSource - 1)))
		{
			if (threadPos2D.x < numColsSource)
			{
				_shared[threadIdx.x + 2][threadIdx.y + 1] =
					make_uchar4(
						(myIdRight),
						(myIdRight) >> 8,
						(myIdRight) >> 16,
						(myIdRight) >> 24);
				// d_sourceImg[myIdRight];
			}
			else
			{
				_shared[threadIdx.x][threadIdx.y + 1] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		d_mask[myId] =
			((_shared[threadIdx.x + 1][threadIdx.y + 1].x +
			  _shared[threadIdx.x + 1][threadIdx.y + 1].y +
			  _shared[threadIdx.x + 1][threadIdx.y + 1].z) < 3 * 255) ?
			   1 :
			   0;

		__syncthreads();

		// Bottom left thread fetches bottom left neighbor (if available)
		if ((threadIdx.x == 0) &&
			((threadIdx.y == (blockDim.y - 1)) ||
			(threadPos2D.y == (numRowsSource - 1))))
		{
			if ((threadPos2D.x > 0) &&
				(threadPos2D.y < numRowsSource))
			{
				_shared[threadIdx.x][threadIdx.y + 2] =
					make_uchar4(
						(myIdBottom - 1),
						(myIdBottom - 1) >> 8,
						(myIdBottom - 1) >> 16,
						(myIdBottom - 1) >> 24);
					// d_sourceImg[myIdBottom - 1];
			}
			else
			{
				_shared[threadIdx.x][threadIdx.y + 2] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		__syncthreads();

		// Bottom row fetches all top neighbors
		if ((threadIdx.y == (blockDim.y - 1)) ||
			(threadPos2D.y == (numRowsSource - 1)))
		{
			if (threadPos2D.y < numRowsSource)
			{
				_shared[threadIdx.x + 1][threadIdx.y + 2] =
					make_uchar4(
						(myIdBottom),
						(myIdBottom) >> 8,
						(myIdBottom) >> 16,
						(myIdBottom) >> 24);
					// d_sourceImg[myIdBottom];
			}
			else
			{
				_shared[threadIdx.x + 1][threadIdx.y + 2] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		__syncthreads();

		// Bottom right thread fetches bottom right neighbor (if available)
		if (((threadIdx.x == (blockDim.x - 1)) ||
			 (threadPos2D.x == (numColsSource - 1))) &&
			((threadIdx.y == (blockDim.y - 1)) ||
			 (threadPos2D.y == (numRowsSource - 1))))
		{
			if ((threadPos2D.x < numColsSource) &&
				(threadPos2D.y < numRowsSource))
			{
				_shared[threadIdx.x + 2][threadIdx.y + 2] =
					make_uchar4(
						(myIdBottom + 1),
						(myIdBottom + 1) >> 8,
						(myIdBottom + 1) >> 16,
						(myIdBottom + 1) >> 24);
					// d_sourceImg[myIdBottom + 1];
			}
			else
			{
				_shared[threadIdx.x + 2][threadIdx.y + 2] =
					make_uchar4(
						1,
						2,
						3,
						4);
			}
		}

		__syncthreads();

#if defined (ENABLE_DEBUG)

		if ((blockIdx.x == 1) &&
			(blockIdx.y == 1))
		{
			int _sharedId =
				((threadIdx.y + 1) * (BLOCK_SIZE_CALC_MASK_MAX_X + 2)) +
				 (threadIdx.x + 1);

			d_shared_for_debug[_sharedId] =
				_shared[threadIdx.x + 1][threadIdx.y + 1];

			// Top left
			if ((threadIdx.x == 0) &&
				(threadIdx.y == 0))
			{
				_sharedId =
					0;

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x][threadIdx.y];
			}

			// Top row
			if (threadIdx.y == 0)
			{
				_sharedId =
					threadIdx.x + 1;

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x + 1][threadIdx.y];
			}

			// Top right
			if (((threadIdx.x == (blockDim.x - 1)) ||
				 (threadPos2D.x == (numColsSource - 1))) &&
				(threadIdx.y == 0))
			{
				_sharedId =
					threadIdx.x + 2;

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x + 2][threadIdx.y];
			}

			// Left column
			if ((threadIdx.x == 0) &&
				(threadPos2D.y < (numRowsSource - 1)))
			{
				_sharedId =
					(threadIdx.y + 1) * (BLOCK_SIZE_CALC_MASK_MAX_X + 2);

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x][threadIdx.y + 1];
			}

			// Right column
			if (((threadIdx.x == (blockDim.x - 1)) ||
				 (threadPos2D.x == (numColsSource - 1))) &&
				(threadPos2D.y < (numRowsSource - 1)))
			{
				_sharedId =
					(threadIdx.y + 1) * (BLOCK_SIZE_CALC_MASK_MAX_X + 2) + threadIdx.x + 2;

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x + 2][threadIdx.y + 1];
			}

			// Bottom left
			if ((threadIdx.x == 0) &&
				((threadIdx.y == (blockDim.y - 1)) ||
				(threadPos2D.y == (numRowsSource - 1))))
			{
				_sharedId =
					(threadIdx.y + 2) * (BLOCK_SIZE_CALC_MASK_MAX_X + 2);

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x][threadIdx.y + 2];
			}


			// Bottom row
			if ((threadIdx.y == (blockDim.y - 1)) ||
				(threadPos2D.y == (numRowsSource - 1)))
			{
				_sharedId =
					(threadIdx.y + 2) * (BLOCK_SIZE_CALC_MASK_MAX_X + 2) + threadIdx.x + 1;

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x + 1][threadIdx.y + 2];
			}

			// Bottom right
			if (((threadIdx.x == (blockDim.x - 1)) ||
				 (threadPos2D.x == (numColsSource - 1))) &&
				((threadIdx.y == (blockDim.y - 1)) ||
				 (threadPos2D.y == (numRowsSource - 1))))
			{
				_sharedId =
					(threadIdx.y + 2) * (BLOCK_SIZE_CALC_MASK_MAX_X + 2) + threadIdx.x + 2;

				d_shared_for_debug[_sharedId] =
					_shared[threadIdx.x + 2][threadIdx.y + 2];
			}
		}

#endif

#else

		uchar4 _local =
			d_sourceImg[myId];

		d_mask[myId] =
			((_local.x +
			  _local.y +
			  _local.z) < 3 * 255) ?
			   1 :
			   0;

#endif

	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
		*/

	size_t srcSize = numRowsSource * numColsSource;

#if defined (ENABLE_DEBUG)

	unsigned char* h_mask_dbg = new unsigned char[srcSize];

	memset(
		h_mask_dbg,
		0x0,
		srcSize * sizeof(unsigned char));

	uchar4 *h_shared_for_debug = 
		new uchar4[(BLOCK_SIZE_CALC_MASK_MAX_X + 2) * (BLOCK_SIZE_CALC_MASK_MAX_Y + 2)];

	memset(
		h_shared_for_debug,
		0x0,
		(BLOCK_SIZE_CALC_MASK_MAX_X + 2) * (BLOCK_SIZE_CALC_MASK_MAX_Y + 2) * sizeof(uchar4));

	uchar4 *d_shared_for_debug;

	// Allocate memory on the device for storing the mask data
	checkCudaErrors(
		cudaMalloc(
			&d_shared_for_debug,
			(BLOCK_SIZE_CALC_MASK_MAX_X + 2) * (BLOCK_SIZE_CALC_MASK_MAX_Y + 2) * sizeof(uchar4)));

#endif

	unsigned char* d_mask;

	// Allocate memory on the device for storing the mask data
	checkCudaErrors(
		cudaMalloc(
			&d_mask,
			srcSize * sizeof(unsigned char)));

	uchar4* d_sourceImg;

	// Allocate memory on the device for storing the source image data
	checkCudaErrors(
		cudaMalloc(
			&d_sourceImg,
			srcSize * sizeof(uchar4)));

	// Copy source image data to device
	checkCudaErrors(
		cudaMemcpy(
			d_sourceImg,
			h_sourceImg,
			srcSize * sizeof(uchar4),
			cudaMemcpyHostToDevice));

	int gridSizeX =
		(numColsSource - 1) / BLOCK_SIZE_CALC_MASK_MAX_X + 1;
	int gridSizeY =
		(numRowsSource - 1) / BLOCK_SIZE_CALC_MASK_MAX_Y + 1;

	// Set block size (i.e., number of threads per block)
	const dim3 blockSize(
		BLOCK_SIZE_CALC_MASK_MAX_X,
		BLOCK_SIZE_CALC_MASK_MAX_Y,
		1);

	// Set grid size (i.e., number of blocks per kernel launch)
	const dim3 gridSize(
		gridSizeX,
		gridSizeY,
		1);

	calculateMaskKernel<<<gridSize, blockSize>>>(
		d_sourceImg,
		numRowsSource,
		numColsSource,
		d_mask

#if defined(ENABLE_DEBUG)

		, d_shared_for_debug

#endif

		);

#if defined (ENABLE_STRICT_ERROR_CHECKING)

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#endif

#if defined (ENABLE_DEBUG)

	// Copy mask data to host (debug)
	checkCudaErrors(
		cudaMemcpy(
			h_mask_dbg,
			d_mask,
			srcSize * sizeof(unsigned char),
			cudaMemcpyDeviceToHost));

	// Copy shared data to host (debug)
	checkCudaErrors(
		cudaMemcpy(
			h_shared_for_debug,
			d_shared_for_debug,
			(BLOCK_SIZE_CALC_MASK_MAX_X + 2) * (BLOCK_SIZE_CALC_MASK_MAX_Y + 2) * sizeof(uchar4),
			cudaMemcpyDeviceToHost));

#endif

	cudaFree(
		d_mask);

	cudaFree(
		d_sourceImg);

	/*

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

#if defined (ENABLE_DEBUG)

	reference_calc_custom(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_blendedImg, h_mask_dbg);

	delete []h_mask_dbg;

	cudaFree(
		d_shared_for_debug);

	delete []h_shared_for_debug;

#endif
}
