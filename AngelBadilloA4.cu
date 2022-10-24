#include <stdio.h>
#include <stdlib.h>
#include <math.h>
const int N = 8192;

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
    int R[N];
    int I[N];

    // Real and imaginary components of
    int* R_d;
    int* I_d;
    // Dimensions of grid & block
    dim3 dimGrid(4,1);
    dim3 dimBlock(1024,1);
}