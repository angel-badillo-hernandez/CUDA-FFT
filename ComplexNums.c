#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define K 8
#define N 8192
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
    // struct CmplxNum B = {25,25};
    // struct CmplxNum A = {25,-100};
    // struct CmplxNum Z = CmplxMult(A, B);
    // printf("%f + %fi", Z.a, Z.bi);

    // Only accurate to 15 digits after decimal point
    const double pi = 2*acos(0.0);
    printf("\n%.15f\n", pi);
    // 3.141592653589793
    
    
    // double XR[K];
    // double XI[K];

    // printf("==========================================================================\n");
    // printf("TOTAL PROCESSED SAMPLES: %d\n", N);
    // printf("==========================================================================\n");
    // for (int i = 0; i < K; ++i)
    // {
    //     printf("XR[%d]: %.6f          XI[%d]: %.6fi\n", i, XR[i], i, XI[i]);
    //     printf("==========================================================================\n");
    // }
}