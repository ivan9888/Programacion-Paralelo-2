%%cu
#include stdio.h
#include stdlib.h
#include cuda_runtime.h

#define N 4  filas
#define M 5  columnas

__global__ void transpose(int input, int output) {
    __shared__ int tile[N][N+1];
    int x = blockIdx.x  N + threadIdx.x;
    int y = blockIdx.y  N + threadIdx.y;
    int index_in = x + y  M;
    x = blockIdx.y  N + threadIdx.x;
    y = blockIdx.x  N + threadIdx.y;
    int index_out = x + y  N;

    for (int i = 0; i  N; i += blockDim.y) {
        tile[threadIdx.y+i][threadIdx.x] = input[index_in+iM+threadIdx.x];
    }
    __syncthreads();
    
    for (int i = 0; i  N; i += blockDim.y) {
        output[index_out+iN+threadIdx.x] = tile[threadIdx.x][threadIdx.y+i];
    }
}

int main() {
    int input[NM] = {1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20};
    int output[MN];

    int d_input, d_output;
    cudaMalloc(&d_input, NMsizeof(int));
    cudaMalloc(&d_output, MNsizeof(int));

    cudaMemcpy(d_input, input, NMsizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((M+N-1)N, (N+N-1)N, 1);
    dim3 dimBlock(N, N, 1);

    transposedimGrid, dimBlock(d_input, d_output);

    cudaMemcpy(output, d_output, MNsizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    printf(Matriz transpuestan);
    for (int i = 0; i  M; i++) {
        for (int j = 0; j  N; j++) {
            printf(%d , output[iN+j]);
        }
        printf(n);
    }

    return 0;
}
