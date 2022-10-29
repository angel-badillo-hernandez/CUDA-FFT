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
const double pi = 3.141592653589793;

// Byte size of type double
const int NBYTE_SIZE = N_SAMPLES * sizeof(double);

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
__global__ void computeFFT(double* XR, double* XI, double* R, double* I)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < N_SAMPLES/2; i++)
    {
        evenPartOfK(XR, XI, R, I, N_SAMPLES, globalIdx);
        evenPartOfK(XR, XI, R, I, N_SAMPLES, globalIdx+N_SAMPLES/2);
        oddPartOfK(XR, XI, R, I, N_SAMPLES, globalIdx);
        oddPartOfK(XR, XI, R, I, N_SAMPLES, globalIdx+N_SAMPLES/2);
    }
}

/**
 * @brief Represents a Complex number.
 * Contains real and imaginary part.
 * 
 */
struct CmplxNum
{
    double a;  // real part of complex number
    double bi; // imaginary part of complex number
};

/**
 * CmplxAdd
 * @brief 
 * Calculates the sum of CmplxNum X and Y, then returns sum as CmplxNum.
 * @param X Augend, a CmplxNum
 * @param Y Addend, a CmplxNum
 * @return Sum as struct CmplxNum 
 */
struct CmplxNum CmplxAdd(struct CmplxNum X, struct CmplxNum Y)
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
struct CmplxNum CmplxMult(struct CmplxNum X, struct CmplxNum Y)
{   
    // X = a + bi
    // Y = c + di
    // XY = (a+bi)(c+di) = (ac−bd) + (ad+bc)i
    struct CmplxNum Z = {.a = (X.a * Y.a) - (X.bi*Y.bi), .bi = (X.a * Y.bi) + (X.bi*Y.a)};
    return Z;
}

struct CmplxNum evenPartAtm(double R[], double I[], int m, int N, int k)
{
    struct CmplxNum funcX2n = {.a = R[2 * m], .bi = I[2 * m]};
    struct CmplxNum eulerPart = {.a = cos(2 * pi * 2 * m * k / N), .bi = -sin(2 * pi * 2 * m * k / N)};
    return CmplxMult(funcX2n, eulerPart);
}

void evenPartOfK(double XR[], double XI[], double R[], double I[], int N, int k)
{
    for (int n = 0; n <= (N / 2) - 1; n++)
    {
        struct CmplxNum result = evenPartAtm(R, I, n, N, k);
        printf("Even n=%d: %f + %fi\n", n, result.a, result.bi);
        XR[k] += result.a;
        XI[k] += result.bi;
    }
}

struct CmplxNum oddPartAtm(double R[], double I[], int m, int N, int k)
{
    struct CmplxNum funcX2n = {.a = R[(2 * m) + 1], .bi = I[(2 * m) + 1]};
    struct CmplxNum eulerPart = {.a = cos(2 * pi * 2 * m * k / N), .bi = -sin(2 * pi * 2 * m * k / N)};
    return CmplxMult(funcX2n, eulerPart);
}

struct CmplxNum twiddleFactor(int N, int k)
{
    struct CmplxNum tFactor = {.a = cos(2 * pi * k / N), .bi = -sin(2 * pi * k / N)};
    return tFactor;
}

void oddPartOfK(double XR[], double XI[], double R[], double I[], int N, int k)
{
    for (int n = 0; n <= (N / 2) - 1; n++)
    {
        struct CmplxNum result = CmplxMult(twiddleFactor(N, k),oddPartAtm(R, I, n, N, k));
        printf("Odd n=%d: %f + %fi\n", n, result.a, result.bi);
        XR[k] += result.a;
        XI[k] += result.bi;
    }
}

int main()
{
    // Real and imaginary components of samples
    double R[N_SAMPLES] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5.0, 4.3};
    double I[N_SAMPLES] = {2.6, 6.3, 4.0, 9.1, 0.4, 4.8, 2.6, 4.1};
    double* R_d;
    double* I_d;

    //Output for Fourier coefficients
    double XR[N_SAMPLES];
    double XI[N_SAMPLES];
    double* XR_d;
    double* XI_d;
    
    // Allocate memory for R_d, I_d, XR_d, and XI_d
    cudaMalloc((void **)&R_d, NBYTE_SIZE);
    cudaMalloc((void **)&I_d, NBYTE_SIZE);
    cudaMalloc((void **)&XR_d, NBYTE_SIZE);
    cudaMalloc((void **)&XI_d, NBYTE_SIZE);

    // Copy arrays R & I to R_d & I_d, respectively
    cudaMemcpy(R_d, R, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(I_d, I, NBYTE_SIZE, cudaMemcpyHostToDevice);

    // Dimensions of grid & block
    dim3 dimGrid(4,1);
    dim3 dimBlock(1024,1);

    // Call kernel to compute FFT
    computeFFT<<<dimGrid, dimBlock>>>(R_d, I_d, XR_d, XI_d);
    
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