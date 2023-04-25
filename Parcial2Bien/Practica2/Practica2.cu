%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>

#define BLOCK_SIZE 1024
#define N 1000

__global__ void bubbleSort(int *array) {
    __shared__ int sharedArray[BLOCK_SIZE];
    int i, j;

    // Cargar datos en shared memory
    sharedArray[threadIdx.x] = array[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    __syncthreads();

    // BubbleSort en shared memory
    for (i = 0; i < BLOCK_SIZE; i++) {
        for (j = i + 1; j < BLOCK_SIZE; j++) {
            if (sharedArray[i] > sharedArray[j]) {
                int temp = sharedArray[i];
                sharedArray[i] = sharedArray[j];
                sharedArray[j] = temp;
            }
        }
    }

    // Copiar datos ordenados de vuelta a memoria global
    array[blockIdx.x * BLOCK_SIZE + threadIdx.x] = sharedArray[threadIdx.x];
}

int main() {
    int i;
    int *array, *d_array;

    // Inicializar datos
    array = (int*)malloc(N * sizeof(int));
    for (i = 0; i < N; i++) {
        array[i] = rand() % 100;
    }

    // Copiar datos a la memoria de la GPU
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    // Ejecutar kernel para ordenar los datos
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bubbleSort<<<numBlocks, BLOCK_SIZE>>>(d_array);
    cudaDeviceSynchronize();

    // Copiar datos ordenados de vuelta a memoria principal
    cudaMemcpy(array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir los datos ordenados
    for (i = 0; i < N; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    // Liberar memoria
    free(array);
    cudaFree(d_array);

    return 0;
}
