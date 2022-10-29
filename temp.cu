//**************************************************************
// Assignment #4
// Name: Angel Badillo, and James
// GPU Programming Date: Date of Submission (M/D/Y)
//***************************************************************
// Place your general program documentation here. It should
// be quite a few lines explaining the programs duty carefully.
// It should also indicate how to run the program and data
// input format, filenames etc
//*****************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// No. of samples
const int N_SAMPLES = 8192;

// Constant for PI, precision of 15 digits after decimal point
const long double pi = 3.14159265358979323846264338;

// Byte size of type long double
const int NBYTE_SIZE = N_SAMPLES * sizeof(long double);

/**
 * @brief Represents a Complex number.
 * Contains real and imaginary part.
 * 
 */
struct CmplxNum
{
    long double a;  // real part of complex number
    long double bi; // imaginary part of complex number
};

/**
 * CmplxAdd
 * @brief 
 * Calculates the sum of CmplxNum X and Y, then returns sum as CmplxNum.
 * @param X Augend, a CmplxNum
 * @param Y Addend, a CmplxNum
 * @return Sum as struct CmplxNum 
 */
__device__ struct CmplxNum CmplxAdd(struct CmplxNum X, struct CmplxNum Y)
{
    struct CmplxNum Z = {.a = X.a + Y.a, .bi = X.bi + Y.bi};
    return Z;
}

/**
 * CmplxSub
 * @brief 
 * Calculates the difference from CmplxNum X and Y, then returns difference as CmplxNum.
 * @param X Subtrahend, as a  CmplxNum
 * @param Y Minuend, as a CmplxNum
 * @return Difference, as a struct CmplxNum 
 */
struct CmplxNum CmplxSub(struct CmplxNum X, struct CmplxNum Y)
{
    struct CmplxNum Z = {.a = X.a - Y.a, .bi = X.bi - Y.bi};
    return Z;
}

/**
 * CmplxMult
 * @brief
 * Multiplies two CmplxNum together, then returns product as a CmplxNum.
 * @param X Multiplicand, as a CmplxNum
 * @param Y Multiplier, as a CmplxNum
 * @return struct CmplxNum 
 */
__device__ struct CmplxNum CmplxMult(struct CmplxNum X, struct CmplxNum Y)
{   
    // X = a + bi
    // Y = c + di
    // XY = (a+bi)(c+di) = (ac−bd) + (ad+bc)i
    struct CmplxNum Z = {.a = (X.a * Y.a) - (X.bi*Y.bi), .bi = (X.a * Y.bi) + (X.bi*Y.a)};
    return Z;
}

__device__ struct CmplxNum evenPartAtm(long double R[], long double I[], int m, int N, int k)
{
    struct CmplxNum funcX2m = {.a = R[2 * m], .bi = I[2 * m]};
    struct CmplxNum eulerPart = {.a = cos(2 * pi * 2 * m * k / N), .bi = -sin(2 * pi * 2 * m * k / N)};
    return CmplxMult(funcX2m, eulerPart);
}

__device__ void evenPartOfK(long double XR[], long double XI[], long double R[], long double I[], int N, int k)
{
    for (int n = 0; n <= (N / 2) - 1; n++)
    {
        struct CmplxNum result = evenPartAtm(R, I, n, N, k);
        printf("Even n=%d: %f + %fi\n", n, result.a, result.bi);
        XR[k] += result.a;
        XI[k] += result.bi;
    }
}

__device__ struct CmplxNum oddPartAtm(long double R[], long double I[], int m, int N, int k)
{
    struct CmplxNum funcX2mP1 = {.a = R[(2 * m) + 1], .bi = I[(2 * m) + 1]};
    struct CmplxNum eulerPart = {.a = cos(2 * pi * 2 * m * k / N), .bi = -sin(2 * pi * 2 * m * k / N)};
    return CmplxMult(funcX2mP1, eulerPart);
}

__device__ struct CmplxNum twiddleFactor(int N, int k)
{
    struct CmplxNum tFactor = {.a = cos(2 * pi * k / N), .bi = -sin(2 * pi * k / N)};
    return tFactor;
}

__device__ void oddPartOfK(long double XR[], long double XI[], long double R[], long double I[], int N, int k)
{
    for (int n = 0; n <= (N / 2) - 1; n++)
    {
        struct CmplxNum result = CmplxMult(twiddleFactor(N, k),oddPartAtm(R, I, n, N, k));
        printf("Odd n=%d: %f + %fi\n", n, result.a, result.bi);
        XR[k] += result.a;
        XI[k] += result.bi;
    }
}


/**
 * computeFFT
 * @brief Compute K no. of FFT coefficients with N no. of samples.
 *        Implements the Cooley-Turkey FFT algorithm (AKA Radix-2).
 * 
 * @param R real part of samples for x(n)
 * @param I imaginary part of samples for x(n)
 * @param XR real part of FTT coefficients
 * @param XI imaginary part of FFT coefficients
 */
__global__ void computeFFT(long double* XR, long double* XI, long double* R, long double* I)
{
    // Global index
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    struct CmplxNum tFactor = {.a = cos(2 * pi * k / N_SAMPLES), .bi = -sin(2 * pi * k / N_SAMPLES)};
    struct CmplxNum ntFactor = {.a = -tFactor.a, .bi = -tFactor.bi};
    struct CmplxNum evenPart = {.a = 0, .bi = 0};
    struct CmplxNum oddPart = {.a = 0, .bi = 0};
    for(int m = 0; m <= N_SAMPLES/2-1; m++)
    {   
        // Ek
        struct CmplxNum funcX2m = {.a = R[2 * m], .bi = I[2 * m]};
        struct CmplxNum eulerPart = {.a = cos(2 * pi * 2 * m * k / N_SAMPLES), .bi = -sin(2 * pi * 2 * m * k / N_SAMPLES)};
        struct CmplxNum resEven = CmplxMult(funcX2m, eulerPart);
        evenPart = CmplxAdd(evenPart, resEven);

        // Ok
        struct CmplxNum funcX2mP1 = {.a = R[(2 * m) + 1], .bi = I[(2 * m) + 1]};
        struct CmplxNum resOdd = CmplxMult(funcX2mP1,eulerPart);
        oddPart = CmplxAdd(oddPart, resOdd);
    }
    // Adding even parts
    XR[k] += evenPart.a;
    XI[k] += evenPart.bi;
    XR[k+N_SAMPLES/2] += evenPart.a;
    XI[k+N_SAMPLES/2] += evenPart.bi;

    // Adding odd parts
    struct CmplxNum temp = CmplxMult(tFactor, oddPart);
    XR[k] += temp.a;
    XI[k] += temp.bi;

    temp = CmplxMult(ntFactor, oddPart);
    XR[k+N_SAMPLES/2] += temp.a;
    XI[k+N_SAMPLES/2] += temp.bi;
}

int main()
{
    // Real and imaginary components of samples
    long double R[N_SAMPLES] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5.0, 4.3};
    long double I[N_SAMPLES] = {2.6, 6.3, 4.0, 9.1, 0.4, 4.8, 2.6, 4.1};
    long double* R_d;
    long double* I_d;

    //Output for Fourier coefficients
    long double XR[N_SAMPLES];
    long double XI[N_SAMPLES];
    long double* XR_d;
    long double* XI_d;
    
    // Allocate memory for R_d, I_d, XR_d, and XI_d
    cudaMalloc((void **)&R_d, NBYTE_SIZE);
    cudaMalloc((void **)&I_d, NBYTE_SIZE);
    cudaMalloc((void **)&XR_d, NBYTE_SIZE);
    cudaMalloc((void **)&XI_d, NBYTE_SIZE);

    // Copy arrays R & I to R_d & I_d, respectively
    cudaMemcpy(R_d, R, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(I_d, I, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(XR_d, XR, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(XI_d, XI, NBYTE_SIZE, cudaMemcpyHostToDevice);


    // Dimensions of grid & block
    dim3 dimGrid(4,1);
    dim3 dimBlock(1024,1);

    // Call kernel to compute FFT
    computeFFT<<<dimGrid, dimBlock>>>(XR_d, XI_d, R_d, I_d);
    
    // Copy XR_d & XI_d to XR & XI
    cudaMemcpy(XR, XR_d, NBYTE_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(XI, XI_d, NBYTE_SIZE, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(R_d);
    cudaFree(I_d);
    cudaFree(XR_d);
    cudaFree(XI_d);

    // Print the N Fourier Coefficients from X(0) to X(7)
    printf("==========================================================================\n");
    printf("TOTAL PROCESSED SAMPLES: %d\n", N_SAMPLES);
    printf("==========================================================================\n");
    for (int i = 0; i < 8; ++i)
    {
        printf("XR[%d]: %.6f          XI[%d]: %.6fi\n", i, XR[i], i, XI[i]);
        printf("==========================================================================\n");
    }
    
    return EXIT_SUCCESS;
}

// =====================================
// TOTAL PROCESSED SAMPLES :8192
// =====================================
// 35.400002 + 33.900002j  [K= 0]
// 35.485325 + 33.799137j  [K= 1]
// 35.570263 + 33.697975j  [K= 2]
// 35.654804 + 33.596500j  [K= 3]
// 35.738964 + 33.494720j  [K= 4]
// 35.822723 + 33.392647j  [K= 5]
// 35.906090 + 33.290264j  [K= 6]
// 35.989056 + 33.187588j  [K= 7]