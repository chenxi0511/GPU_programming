#include <stdio.h>
#include <stdint.h>

const int MILLION = 1000000; //define the constant million
const int thread_per_block = 1000;

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

__global__	void arradd(float *A, float target, int numElements){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numElements){
		A[idx] = A[idx] + target;
	}
}

__global__	void darradd(double *A, double target, int numElements){
	int idx= blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numElements){
		A[idx] = A[idx] + target;
	}
}

__global__	void iarradd(int32_t *A, int32_t target, int numElements){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numElements){
		A[idx] = A[idx] + target;
	}
}

__global__	void xarradd(float *A, float target, int times, int numElements){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numElements){
		for(int j = 0; j < times; j++){
			A[idx] = A[idx] + target;
		}
	}
}

/***  Time measure for single precision floating-point numbers  ***/
void measure_float(){
	printf("Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	float *A;
	float X = 10.0f;
	float *d_A;

	cudaEvent_t start,stop;
	float time_cpu_to_gpu, time_kernel, time_gpu_to_cpu;

	FILE* fp;
	fp = fopen("outA.txt", "w+");
	fprintf(fp,"Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	for(int j = 1; j <=256; j *= 2){
		//Malloc for the input array A on CPU
		int size = j * MILLION;
		A = (float *) malloc(size * sizeof(float));

		//Initialize the array
		for(int i = 0; i < size; i++){
			A[i] = (float)i / 3.0f;
		}

		//Malloc array on GPU
		cudaMalloc((void**)&d_A, size * sizeof(float));
		//Initialize the size of grid and block
		dim3 gridDim((size + thread_per_block - 1)/thread_per_block);
		dim3 blockDim(thread_per_block);

		//Measure time for Memcpy from CPU to GPU
		time_record_begin(start);
		cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
		time_record_end(start, stop, time_cpu_to_gpu);

		//Measure time for kernel
		time_record_begin(start);
		arradd<<<gridDim, blockDim>>>(d_A, X, size);
		cudaDeviceSynchronize();
		time_record_end(start, stop, time_kernel);

		//Measure time for Memcpy from GPU to CPU
		time_record_begin(start);
		cudaMemcpy(A, d_A, size * sizeof(float), cudaMemcpyDeviceToHost);
		time_record_end(start, stop, time_gpu_to_cpu);

		fprintf(fp,"%d\t\t %f\t %f\t %f\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);
		printf("%d\t\t %f\t %f\t %f\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);

		//free memory on GPU and CPU
		cudaFree(d_A);
		free(A);
	}
	fclose(fp);
}

/***  Time measure for double precision floating-point numbers  ***/
void measure_double(){
	printf("Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	double *A;
	double X = 10.0;
	double *d_A;

	cudaEvent_t start,stop;
	float time_cpu_to_gpu, time_kernel, time_gpu_to_cpu;

	FILE* fp;
	fp = fopen("outB.txt", "w+");
	fprintf(fp,"Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	for(int j = 1; j <= 256; j *= 2){
		int size = j * MILLION;
		A = (double *) malloc(size * sizeof(double));

		for(int i = 0; i < size; i++){
			A[i] = (double)i / 3.0f;
		}

		cudaMalloc((void**)&d_A, size * sizeof(double));
		//Initialize the size of grid and block
		dim3 gridDim((size + thread_per_block - 1)/thread_per_block);
		dim3 blockDim(thread_per_block);

		//Measure time for Memcpy from CPU to GPU
		time_record_begin(start);
		cudaMemcpy(d_A, A, size * sizeof(double), cudaMemcpyHostToDevice);
		time_record_end(start, stop, time_cpu_to_gpu);

		//Measure time for kernel
		time_record_begin(start);
		darradd<<<gridDim, blockDim>>>(d_A, X, size);
		cudaDeviceSynchronize();
		time_record_end(start, stop, time_kernel);

		//Measure time for Memcpy from GPU to CPU
		time_record_begin(start);
		cudaMemcpy(A, d_A, size * sizeof(double), cudaMemcpyDeviceToHost);
		time_record_end(start, stop, time_gpu_to_cpu);

		fprintf(fp,"%d\t\t %f\t %f\t %f\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);
		printf("%d\t\t %f\t %f\t %f\t\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);

		cudaFree(d_A);
		free(A);
	}
	fclose(fp);
}

/***  Time measure for 32-bit integers  ***/
void measure_int32(){
	printf("Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	int32_t *A;
	int32_t X = 10;
	int32_t *d_A;

	cudaEvent_t start,stop;
	float time_cpu_to_gpu, time_kernel, time_gpu_to_cpu;

	FILE* fp;
	fp = fopen("outC.txt", "w+");
	fprintf(fp,"Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	for(int j = 1; j <= 256; j *= 2){
		int size = j * MILLION;
		A = (int32_t *) malloc(size * sizeof(int32_t));

		for(int i = 0; i < size; i++){
			A[i] = (int32_t)(i / 3);
		}

		cudaMalloc((void**)&d_A, size * sizeof(int32_t));
		//Initialize the size of grid and block
		dim3 gridDim((size + thread_per_block - 1)/thread_per_block);
		dim3 blockDim(thread_per_block);

		//Measure time for Memcpy from CPU to GPU
		time_record_begin(start);
		cudaMemcpy(d_A, A, size * sizeof(int32_t), cudaMemcpyHostToDevice);
		time_record_end(start, stop, time_cpu_to_gpu);

		//Measure time for kernel
		time_record_begin(start);
		iarradd<<<gridDim, blockDim>>>(d_A, X, size);
		cudaDeviceSynchronize();
		time_record_end(start, stop, time_kernel);

		//Measure time for Memcpy from GPU to CPU
		time_record_begin(start);
		cudaMemcpy(A, d_A, size * sizeof(int32_t), cudaMemcpyDeviceToHost);
		time_record_end(start, stop, time_gpu_to_cpu);

		fprintf(fp,"%d\t\t %f\t %f\t %f\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);
		printf("%d\t\t %f\t %f\t %f\t\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);

		cudaFree(d_A);
		free(A);
	}
	fclose(fp);
}

/***  Time measure for diffrent adding times  ***/
void measure_xaddtimes(){
	printf("XaddedTimes\t Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	float *A;
	float X = 10.0;
	float *d_A;

	cudaEvent_t start,stop;
	float time_cpu_to_gpu, time_kernel, time_gpu_to_cpu;

	int size = 128 * MILLION;
	//Initialize the size of grid and block
	dim3 gridDim((size + thread_per_block - 1)/thread_per_block);
	dim3 blockDim(thread_per_block);

	FILE* fp;
	fp = fopen("outD.txt", "w+");
	fprintf(fp,"XaddedTimes\t Elements(M)\t CPUtoGPU(ms)\t Kernel(ms)\t GPUtoCPU(ms)\n");
	for(int j = 1; j <= 256; j *= 2){
		A = (float *) malloc(size * sizeof(float));
		for(int i = 0; i < size; i++){
			A[i] = (float) i / 3.0f;
		}

		cudaMalloc((void**)&d_A, size * sizeof(float));
		//Measure time for Memcpy from CPU to GPU
		time_record_begin(start);
		cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
		time_record_end(start, stop, time_cpu_to_gpu);

		//Measure time for kernel
		time_record_begin(start);
		xarradd<<<gridDim, blockDim>>>(d_A, X, j, size);
		cudaDeviceSynchronize();
		time_record_end(start, stop, time_kernel);

		//Measure time for Memcpy from GPU to CPU
		time_record_begin(start);
		cudaMemcpy(A, d_A, size * sizeof(float), cudaMemcpyDeviceToHost);
		time_record_end(start, stop, time_gpu_to_cpu);

		fprintf(fp,"%d\t\t .\t\t %f\t %f\t %f\t\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);
		printf("%d\t\t .\t\t %f\t %f\t %f\t\n", j, time_cpu_to_gpu, time_kernel, time_gpu_to_cpu);

		cudaFree(d_A);
		free(A);
	}
	fclose(fp);
}

int main(int argc, char ** argv){
	printf("Part A\n");
	measure_float();

	printf("Part B\n");
	measure_double();

	printf("Part C\n");
	measure_int32();

	printf("Part D\n");
	measure_xaddtimes();

	printf("Done.\n");
}

