#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

int const r = 4;
__global__ void imageFilterKernelPartA(char3* inputPixels, char3* outputPixels, 
	uint width, uint height, int numPixels_thread)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = 0; i < numPixels_thread; i++){
		int idx = index * numPixels_thread + i;

		int offset_x = idx % width;
		int offset_y = idx / width;

		int3 sum = {0, 0, 0};
		int count = 0;
	    
		for(int dx = -r; dx <= r; dx++)
		{
			for(int dy = -r; dy <= r; dy++)
			{
				// check if the pixel exists
				if(((offset_x + dx) >= 0) && ((offset_x + dx) < width) && ((offset_y + dy) >= 0) 
					&& ((offset_y + dy) < height))
				{
					int idx_offset = idx + dy * width + dx;
					sum.x += (int)inputPixels[idx_offset].x;
					sum.y += (int)inputPixels[idx_offset].y;
					sum.z += (int)inputPixels[idx_offset].z;
					count++;
				}
			}
	    }	
		outputPixels[idx].x = sum.x/count;
		outputPixels[idx].y = sum.y/count;
		outputPixels[idx].z = sum.z/count;			
	}
}


__global__ void imageFilterKernelPartB(char3* inputPixels, char3* outputPixels, 
	uint width, uint height, int numPixels_thread, int numThreads)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = 0; i < numPixels_thread; i++)
	{
		int idx = i * numThreads + index;
		int3 sum = {0, 0, 0};
		int count = 0;

		int offset_x = idx % width;
		int offset_y = idx / width;
		
		for(int dx = -r; dx <= r; dx++)
		{
			for(int dy = -r; dy <= r; dy++)
			{
				// check if the pixel exists
				if(((offset_x + dx) >= 0) && ((offset_x + dx) < width) 
					&& ((offset_y + dy) >= 0) && ((offset_y + dy) < height))
				{
					int idx_offset = idx + dy * width + dx;
					sum.x += (int)inputPixels[idx_offset].x;
					sum.y += (int)inputPixels[idx_offset].y;
					sum.z += (int)inputPixels[idx_offset].z;
					count++;
				}
			}
	    }
		outputPixels[idx].x = sum.x/count;
		outputPixels[idx].y = sum.y/count;
		outputPixels[idx].z = sum.z/count;
	}
}

__global__ void imageFilterKernelPartC(char3* inputPixels, char3* outputPixels, 
	uint width, uint height, int nblocks_per_col, int nblocks_per_row, int nloops)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//each 32 threads load 1 row of data to shared memory in order to access memory coalescing 
	int threads_per_row = 32;
	int sm_pos_x = (index - blockIdx.x * blockDim.x) % threads_per_row; 
	int sm_pos_y = (index - blockIdx.x * blockDim.x) / threads_per_row;
	int pixel_per_thread = 128 / threads_per_row;

	__shared__ char3 s_data[128 * 128];

	for(int i = 0; i < nloops; i++)
	{
		//block position
		int gm_block_x = (blockIdx.x + i * 12) % nblocks_per_row;
		int gm_block_y = (blockIdx.x + i * 12) / nblocks_per_row;
		int gm_block_pos = gm_block_y * 120 * width + gm_block_x * 120;

		//load data from global memory to shared memory
		for(int k = 0; k < 4; k++)      //Since 1024 threads can hance 128/32 rows, hence we need for 4 loops in order to calculate the offset
		{
			for(int j = 0; j < pixel_per_thread; j++)
			{
				int sm_thread_pos_y = (sm_pos_y + k * 32);
				int sm_thread_pos_x = sm_pos_x + j * threads_per_row;
				int sm_thread_pos = sm_thread_pos_y * width + sm_thread_pos_x;
				int gm_address = gm_block_pos + sm_thread_pos;
				int sm_address = (sm_pos_y + k * 32) * 128 + sm_pos_x + j * threads_per_row;

				if(    ((gm_block_y * 120 + sm_pos_y + k * 32) >= 0) 
                    && ((gm_block_y * 120 + sm_pos_y + k * 32) < height) 
					&& ((gm_block_x * 120 + sm_pos_x + j * threads_per_row) >= 0) 
                    && ((gm_block_x * 120 + sm_pos_x + j * threads_per_row) < width))
				{
					s_data[sm_address] = inputPixels[gm_address];
				}
			}	
		}
		__syncthreads();

		for(int k = 0; k < 4; k++)
		{
			for(int j = 0; j < pixel_per_thread; j++)
			{
				int sm_thread_pos_y = (sm_pos_y + k * 32);
				int sm_thread_pos_x = sm_pos_x + j * threads_per_row;
				int sm_thread_pos = sm_thread_pos_y * width + sm_thread_pos_x;
				int gm_address = gm_block_pos + sm_thread_pos;
				int pixel_idx_y = gm_block_y * 120 + sm_thread_pos_y;
				int pixel_idx_x = gm_block_x * 120 + sm_thread_pos_x;
				// if the pixel is within the 120 * 120 area, just simply sum the 9 * 9 pixels and find the average
                if((sm_pos_x + j * threads_per_row >= 4) && (sm_pos_x + j * threads_per_row <= 123) 
                	&& (sm_pos_y + k * 32 >= 4) && (sm_pos_y + k * 32 <= 123))   
                {
                    int3 sum = {0, 0, 0};
				    for(int dx = -r; dx <= r; dx++)
				    {
					    for(int dy = -r; dy <= r; dy++)
					    {
							sum.x += (int)s_data[(sm_thread_pos_y + dy) * 128 + (sm_thread_pos_x + dx)].x; // read from the corresponding position in shared memory
							sum.y += (int)s_data[(sm_thread_pos_y + dy) * 128 + (sm_thread_pos_x + dx)].y;
							sum.z += (int)s_data[(sm_thread_pos_y + dy) * 128 + (sm_thread_pos_x + dx)].z;
					    }
				    }
	
				    if((pixel_idx_y >=0) && (pixel_idx_y < height) 
				    	&& (pixel_idx_x >= 0) && (pixel_idx_x < width))
				    {
					    outputPixels[gm_address].x = sum.x / 81;            // write to the corresponding position in global memory
					    outputPixels[gm_address].y = sum.y / 81;
					    outputPixels[gm_address].z = sum.z / 81;
				    }
			    }

			    // check if the pixel is in the leftmost/righmost/top/bottom, read data from global memory 
                if((pixel_idx_y <= 3) || ((pixel_idx_y >= height-4) && (pixel_idx_y < height)) || 
                	(pixel_idx_x <= 3) || ((pixel_idx_x >= width-4) && (pixel_idx_x < width)))
                {
                    int3 sum = {0, 0, 0};
			        int count = 0;
				    for(int dx = -r; dx <= r; dx++)
				    {
					    for(int dy = -r; dy <= r; dy++)
					    {
					    	// check if the pixel exists
                            if(((pixel_idx_y + dy) >=0) && ((pixel_idx_y + dy) < height) && 
                            	((pixel_idx_x + dx) >= 0) && ((pixel_idx_x + dx) < width))
						    {
							    sum.x += (int)inputPixels[gm_address + dy * width + dx].x;
							    sum.y += (int)inputPixels[gm_address + dy * width + dx].y;
							    sum.z += (int)inputPixels[gm_address + dy * width + dx].z;
							    count++;
						    }
					    }
				    }
	
				    if((pixel_idx_y >=0) && (pixel_idx_y < height) && 
				    	(pixel_idx_x >= 0) && (pixel_idx_x < width))
				    {
					    outputPixels[gm_address].x = sum.x / count;
					    outputPixels[gm_address].y = sum.y / count;
					    outputPixels[gm_address].z = sum.z / count;
				    }
                }
			}
		}
        __syncthreads();
	}
}

#endif // _IMAGEFILTER_KERNEL_H_
