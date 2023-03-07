/*Tramnsposicion de matrices*/

#include <stdio.h>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>




#define TILE_DIM 16

__global__ void transpose_no_SM(double* source, double* dest, int size) {
	int i = threadIdx.x + blockIdx.x + blockDim.x;
	int j = threadIdx.y + blockIdx.y + blockDim.y;

	if (i < size && j < size) {
		int dst_idx = i * size + j;
		int src_idx = j * size + i;
		dest[dst_idx] = source[src_idx];
	}
}


__global__ void transpose_shared(double* source, double* dest, int size) {

	__shared__ double tile[TILE_DIM][TILE_DIM + 1];

	//input threads idx

	int i_in = threadIdx.x + blockDim.x + blockIdx.x;
	int j_in = threadIdx.y + blockIdx.y + blockDim.y;

	//index
	int src_idx = j_in * size + i_in;

	//1D index calculation shared memory
	int _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

	//coordinate for transpose matrix
	int i_out = blockIdx.y * blockDim.y + threadIdx.x;
	int j_out = blockIdx.x * blockDim.y + threadIdx.y;

	//output index
	int dst_idx = j_out * size + i_out;

	if(i_in < size && j_in < size) {
		//Load from in array in row major and store to shared memory in row major
		tile[threadIdx.y][threadIdx.x] = source[src_idx];

		//wait untill all the threads load the values
		__syncthreads();

		dest[dst_idx] = tile[threadIdx.x][threadIdx.y];

	}

}

int main() {
	int mat_size = 4096;
	int byte_size = mat_size * mat_size * sizeof(double);

	//hostallocation
	double* mat_input = (double*)malloc(byte_size);
	double* mat_output = (double*)malloc(byte_size);
	memset(mat_output, 0, byte_size);

	//init array
	srand((unsigned)time(NULL));
	for (int i = 0; i < mat_size * mat_size; i++) {
		mat_input[i] = (double)(rand() % 10);
	}

	//allocate device pointers
	//cuda_ptr<double>in_gpu(mat_input, byte_size);
	//cuda_ptr<double>out_gpu(mat_output, byte_size);

	//launch kernel
	int block_size = TILE_DIM;
	int grid_size = (int)ceil((float)mat_size / block_size);//ceil redondea hacia arriba

	dim3 block(block_size, block_size);

	dim3 grid(grid_size, grid_size);

	transpose_shared << <grid, block >> > (mat_input, mat_output, mat_size);
	//CUDA_ERROR_HANDLER(cudaDeviceSynchronize());
	//out_gpu.to_host(mat_output, byte_size);


	return 0;
}