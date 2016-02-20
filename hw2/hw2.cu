#include <stdio.h>
#include <time.h>

const int M = 1024 * 1024;
const int n_thread = 512;

//define the timer for GPU
#define time_record_begin(start){ \
	cudaEventCreate(&start);	  \
	cudaEventRecord(start, 0);	  \
}
#define time_record_end(start, stop, time){ \
	cudaEventCreate(&stop);		\
	cudaEventRecord(stop, 0);	\
	cudaEventSynchronize(stop);	\
	cudaEventElapsedTime(&time, start, stop);\
}

__global__ void findMin(int *A){

	int tid = threadIdx.x;
	int idx = blockIdx.x * (blockDim.x * 2) + tid;
   
    __shared__ int s_data[512];

	if(A[idx] < A[idx + blockDim.x]){
		s_data[tid] = A[idx];
	}
	else{
		s_data[tid] = A[idx + blockDim.x];
	}
	__syncthreads();

	for(int k = 256; k > 32; k /= 2){
		if(tid < k){
			if(s_data[tid] > s_data[tid + k])	
				s_data[tid] = s_data[tid + k];
		}
		__syncthreads();
	}

	// threads within one warp can work in completely parallel and __syncthreads() is not needed
	if(tid < 32){
		if(blockDim.x > 32)
			if(s_data[tid] > s_data[tid + 32])		s_data[tid] = s_data[tid + 32];
		if(blockDim.x > 16)
			if(s_data[tid] > s_data[tid + 16])		s_data[tid] = s_data[tid + 16];
		if(blockDim.x > 8)
			if(s_data[tid] > s_data[tid + 8])		s_data[tid] = s_data[tid + 8];
		if(blockDim.x > 4)
			if(s_data[tid] > s_data[tid + 4])		s_data[tid] = s_data[tid + 4];
		if(blockDim.x > 2)
			if(s_data[tid] > s_data[tid + 2])		s_data[tid] = s_data[tid + 2];
		if(blockDim.x > 1)
			if(s_data[tid] > s_data[tid + 1])		s_data[tid] = s_data[tid + 1];
	}

	if(tid == 0)
		A[blockIdx.x] = s_data[0];
}

__global__ void findMax(int *A){

	int tid = threadIdx.x;
	int idx = blockIdx.x * (blockDim.x * 2) + tid;
    
    __shared__ int s_data[512];

	if(A[idx] > A[idx + blockDim.x]){
		s_data[tid] = A[idx];
	}
	else{
		s_data[tid] = A[idx + blockDim.x];
	}
	__syncthreads();

	for(int k = 256; k > 32; k /= 2){
		if(tid < k){
			if(s_data[tid] < s_data[tid + k])	
				s_data[tid] = s_data[tid + k];
		}
		__syncthreads();
	}

	// threads within one warp can work in completely parallel and __syncthreads() is not needed
	if(tid < 32){
		if(blockDim.x > 32)
			if(s_data[tid] < s_data[tid + 32])		s_data[tid] = s_data[tid + 32];
		if(blockDim.x > 16)
			if(s_data[tid] < s_data[tid + 16])		s_data[tid] = s_data[tid + 16];
		if(blockDim.x > 8)
			if(s_data[tid] < s_data[tid + 8])		s_data[tid] = s_data[tid + 8];
		if(blockDim.x > 4)
			if(s_data[tid] < s_data[tid + 4])		s_data[tid] = s_data[tid + 4];
		if(blockDim.x > 2)
			if(s_data[tid] < s_data[tid + 2])		s_data[tid] = s_data[tid + 2];
		if(blockDim.x > 1)
			if(s_data[tid] < s_data[tid + 1])		s_data[tid] = s_data[tid + 1];
	}

	if(tid == 0)
		A[blockIdx.x] = s_data[0];
}

void random_number_generator(int *A, int size){
	time_t t;
	srand((unsigned) time(&t));
	for(int i = 0; i < size; i++){
		A[i] = rand();
	}
}

int findMax_cpu(int* A, int n_Elements){
	int max = INT_MIN;
	for(int i = 0; i < n_Elements; i++){
		if(A[i] > max)
			max = A[i];
	}
	return max;
}

int findMin_cpu(int* A, int n_Elements){
	int min = INT_MAX;
	for(int i = 0; i < n_Elements; i++){
		if(A[i] < min)
			min = A[i];
	}
	return min;
}

void findMinMax(int s, int* SIZE){
	int *A;
	int *d_A_MAX, *d_A_MIN;
	int max_gpu, min_gpu;
	int max_cpu, min_cpu;

	int size = SIZE[s] * M;

	A = (int*) malloc(size * sizeof(int));
	random_number_generator(A, size);

	//difine the grid size and block size for three kernel calls
    dim3 dimGrid[3] = {size/n_thread/2, size/n_thread/n_thread/4, 1};
    dim3 dimBlock[3] = {n_thread, n_thread, size/n_thread/n_thread/8};

    // Timer initialize
	float max_average_cpu = 0;
	float min_average_cpu = 0;

	cudaEvent_t start,stop;
	float time_findMax, time_findMin;
	float max_average_gpu = 0;
	float min_average_gpu = 0;

	// Loop for 10 times
	for(int j = 0; j < 10; j++){
		// Find MIN and MAX on CPU
		clock_t tiem_start, time_end;
		tiem_start = clock();
		max_cpu = findMax_cpu(A, size);
		time_end = clock();
		max_average_cpu += (float)(time_end - tiem_start) / 1000000;	// find_max_cpu_time in seconds

		tiem_start = clock();
		min_cpu = findMin_cpu(A, size);
		time_end = clock();
		min_average_cpu += (float)(time_end - tiem_start) / 1000000;	// find_min_cpu_time in seconds


		// Find MIN and MAX on GPU
		// Allocate memory on gpu and copy data from host to device
		cudaMalloc((void**)&d_A_MAX, size * sizeof(int));
		cudaMemcpy(d_A_MAX, A, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_A_MIN, size * sizeof(int));
		cudaMemcpy(d_A_MIN, A, size * sizeof(int), cudaMemcpyHostToDevice);

		// find min and max by 3 kernel calls
		time_record_begin(start);
		for(int k = 0; k < 3; k++){
			findMax<<<dimGrid[k], dimBlock[k]>>>(d_A_MAX);
		}
		time_record_end(start, stop, time_findMax);
		max_average_gpu += time_findMax / 1000;							// find_max_gpu_time in seconds
		cudaMemcpy(&max_gpu, d_A_MAX, sizeof(int), cudaMemcpyDeviceToHost);

		time_record_begin(start);
		for(int k = 0; k < 3; k++){
			findMin<<<dimGrid[k], dimBlock[k]>>>(d_A_MIN);
		}
		time_record_end(start, stop, time_findMin);
		min_average_gpu += time_findMin / 1000;							// find_min_gpu_time in seconds
		cudaMemcpy(&min_gpu, d_A_MIN, sizeof(int), cudaMemcpyDeviceToHost);

		// free memomy 
		cudaFree(d_A_MAX);
		cudaFree(d_A_MIN);
	}

	// Calculate the average time in ten runs and the speedup for GPU
	max_average_cpu /= 10;
	min_average_cpu /= 10;
	max_average_gpu /= 10;
	min_average_gpu /= 10;
	float speedup_max = max_average_cpu / max_average_gpu;
	float speedup_min = min_average_cpu / min_average_gpu;

	printf("N:%dM GPUmax:%d CPUmax:%d GPUtime:%f CPUtime:%f GPUSpeedup:%f N:%dM GPUmin:%d CPUmin:%d GPUtime:%f CPUtime:%f GPUSpeedup:%f\n",
		SIZE[s], max_gpu, max_cpu, max_average_gpu, max_average_cpu, speedup_max, SIZE[s], min_gpu, min_cpu, min_average_gpu, min_average_cpu, speedup_min);

	free(A);
}

int main(int argc, char *argv[]){
	int SIZE[3] = {2, 8, 32};
	printf("FIRSTNAME: Xi\nLASTNAME: Chen\n");
	for(int i = 0; i < 3; i++){
		findMinMax(i, SIZE);
	}
}

