//**************************************************************
// Assignment #2
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
#define N 8192

// No. of Fourier Coefficients
#define K 8

// Constant for PI, precision of 15 digits after decimal point
// 3.141592653589793
const double PI = 2*acos(0.0);

// Byte size of type double
const int DOUBLE_SIZE = sizeof(double);

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


int main()
{
    // Real and imaginary components of samples
    double R[N] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5.0, 4.3};
    double I[N] = {2.6, 6.3, 4.0, 9.1, 0.4, 4.8, 2.6, 4.1};
    double* R_d;
    double* I_d;

    //Output for Fourier coefficients
    double XR[K];
    double XI[K];
    double* XR_d;
    double* XI_d;
    
    // Dimensions of grid & block
    dim3 dimGrid(4,1);
    dim3 dimBlock(1024,1);

    printf("=====================================\n");
    printf("TOTAL PROCESSED SAMPLES: %d\n", N);
    printf("=====================================\n");
    for (int i = 0; i < K; ++i)
    {
        printf("%.6f + %.6fi  [K= %d]\n", XR[i], XI[i], i);
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