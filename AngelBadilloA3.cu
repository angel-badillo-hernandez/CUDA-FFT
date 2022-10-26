// CMPS-4563-101
// Homework 3
// Angel Badillo
// Due: 10/05/22
#include <stdio.h>

const int N = 10240;

__global__ void multiplyA(int *A, int *B, int *C)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = globalIdx; i < (N/2)*(blockIdx.x+1); i+=blockDim.x)
    C[i] = A[i]*B[i];
}

__global__ void multiplyC(int *A, int *B, int *C)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = globalIdx; i < (N/10)*(blockIdx.x+1); i+=blockDim.x)
    C[i] = A[i]*B[i];
}

int main()
{
    int A[N];
    int B[N];
    int C[N];
    int *A_d;
    int *B_d;
    int *C_d;
    const int isize = N * sizeof(int);

    // Grid and block dimensions for part A
    dim3 dimGridA(10, 1);
    dim3 dimBlockA(1024, 1);

    // Grid and block dimensions for part C
    dim3 dimGridC(10, 1);
    dim3 dimBlockC(1024, 1);
    

    // Inititialize to A to even numbers
    // and B to odd numbers
    for (int n = 0; n < N; n++)
    {
        A[n] = 2 * n;
        B[n] = 2 * n + 1;
    }

    // Allocate memory for A_d, B_d, and C_d
    cudaMalloc((void **)&A_d, isize);
    cudaMalloc((void **)&B_d, isize);
    cudaMalloc((void **)&C_d, isize);

    // Copy A to A_d and B to B_d
    cudaMemcpy(A_d, A, isize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, isize, cudaMemcpyHostToDevice);

    // Invoke kernel for part A
    multiplyA<<<dimGridA, dimBlockA>>>(A_d, B_d,C_d);

    // Copy C_d to C
    cudaMemcpy(C, C_d, isize, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // Print result of part A
    printf("2 Blocks - Not Cyclic (C[0], C[%d]) = (%d, %d)\n", N-1, C[0], C[N-1]);

    // Allocate memory for A_d, B_d, and C_d
    cudaMalloc((void **)&A_d, isize);
    cudaMalloc((void **)&B_d, isize);
    cudaMalloc((void **)&C_d, isize);

    // Copy A to A_d and B to B_d
    cudaMemcpy(A_d, A, isize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, isize, cudaMemcpyHostToDevice);

    // Invoke kernel for part C
    multiplyC<<<dimGridC, dimBlockC>>>(A_d, B_d,C_d);

    // Copy C_d to C
    cudaMemcpy(C, C_d, isize, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // Print result of part C
    printf("10 Blocks - (C[0], C[%d]) = (%d, %d)\n", N-1, C[0], C[N-1]);
    
    return EXIT_SUCCESS;
}