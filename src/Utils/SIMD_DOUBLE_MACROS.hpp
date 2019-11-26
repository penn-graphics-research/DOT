//
// Created by mingg18 on 12/31/18.
//

#ifndef DOT_SIMD_DOUBLE_MACROS_HPP
#define DOT_SIMD_DOUBLE_MACROS_HPP


// A B C are double[3][3][4]
#define MATRIX_TIMES_MATRIX(A,B,C) {\
    for(int u=0;u<3;++u)\
    {\
        __m256d vARow0 = _mm256_load_pd(&(A[u][0][0]));\
        __m256d vARow1 = _mm256_load_pd(&(A[u][1][0]));\
        __m256d vARow2 = _mm256_load_pd(&(A[u][2][0]));\
        for(int v=0;v<3;++v)\
        {\
            __m256d vBColumn0 = _mm256_load_pd(&(B[0][v][0]));\
            __m256d vResult = _mm256_mul_pd(vARow0,vBColumn0);\
            __m256d vBColumn1 = _mm256_load_pd(&(B[1][v][0]));\
            vResult = _mm256_fmadd_pd(vARow1,vBColumn1,vResult);\
            __m256d vBColumn2 = _mm256_load_pd(&(B[2][v][0]));\
            vResult = _mm256_fmadd_pd(vARow2,vBColumn2,vResult);\
            _mm256_store_pd(&(C[u][v][0]), vResult);\
        }\
    }\
}\

#define MATRIX_TIMES_MATRIX_TRANSPOSE(A,B,C) {\
    for(int u=0;u<3;++u)\
    {\
        __m256d vARow0 = _mm256_load_pd(&(A[u][0][0]));\
        __m256d vARow1 = _mm256_load_pd(&(A[u][1][0]));\
        __m256d vARow2 = _mm256_load_pd(&(A[u][2][0]));\
        for(int v=0;v<3;++v)\
        {\
            __m256d vBColumn0 = _mm256_load_pd(&(B[v][0][0]));\
            __m256d vResult = _mm256_mul_pd(vARow0,vBColumn0);\
            __m256d vBColumn1 = _mm256_load_pd(&(B[v][1][0]));\
            vResult = _mm256_fmadd_pd(vARow1,vBColumn1,vResult);\
            __m256d vBColumn2 = _mm256_load_pd(&(B[v][2][0]));\
            vResult = _mm256_fmadd_pd(vARow2,vBColumn2,vResult);\
            _mm256_store_pd(&(C[u][v][0]), vResult);\
        }\
    }\
}\

// C=A*S*BT
#define MATRIX_TIMES_VECTOR_TIMES_MATRIX_TRANPOSE(A,S,B,C) {\
    __m256d vS0 = _mm256_load_pd(&(S[0][0]));\
    __m256d vS1 = _mm256_load_pd(&(S[1][0]));\
    __m256d vS2 = _mm256_load_pd(&(S[2][0]));\
    for(int u=0;u<3;++u)\
    {\
        __m256d vARow0 = _mm256_load_pd(&(A[u][0][0]));\
        vARow0 = _mm256_mul_pd(vARow0, vS0);\
        __m256d vARow1 = _mm256_load_pd(&(A[u][1][0]));\
        vARow1 = _mm256_mul_pd(vARow1, vS1);\
        __m256d vARow2 = _mm256_load_pd(&(A[u][2][0]));\
        vARow2 = _mm256_mul_pd(vARow2, vS2);\
        for(int v=0;v<3;++v)\
        {\
            __m256d vBColumn0 = _mm256_load_pd(&(B[v][0][0]));\
            __m256d vResult = _mm256_mul_pd(vARow0,vBColumn0);\
            __m256d vBColumn1 = _mm256_load_pd(&(B[v][1][0]));\
            vResult = _mm256_fmadd_pd(vARow1,vBColumn1,vResult);\
            __m256d vBColumn2 = _mm256_load_pd(&(B[v][2][0]));\
            vResult = _mm256_fmadd_pd(vARow2,vBColumn2,vResult);\
            _mm256_store_pd(&(C[u][v][0]), vResult);\
        }\
    }\
}\

#define ENERGY_FIXED_COROTATED(e, vOne, vOneHalf, mu, lambda, sigma0, sigma1, sigma2, vResult) {\
    \
    __m256d vSigma = _mm256_load_pd(&sigma0[e * 4]);\
    __m256d vJ = vSigma;\
    vSigma = _mm256_sub_pd(vSigma, vOne);\
    vResult = _mm256_mul_pd(vSigma, vSigma);\
    \
    vSigma = _mm256_load_pd(&sigma1[e * 4]);\
    vJ = _mm256_mul_pd(vJ, vSigma);\
    vSigma = _mm256_sub_pd(vSigma, vOne);\
    vResult = _mm256_fmadd_pd(vSigma, vSigma, vResult);\
    \
    vSigma = _mm256_load_pd(&sigma2[e * 4]);\
    vJ = _mm256_mul_pd(vJ, vSigma);\
    vSigma = _mm256_sub_pd(vSigma, vOne);\
    vResult = _mm256_fmadd_pd(vSigma, vSigma, vResult);\
    \
    __m256d vMu = _mm256_load_pd(&mu[e * 4]);\
    vResult = _mm256_mul_pd(vMu, vResult);\
    \
    vJ = _mm256_sub_pd(vJ, vOne);\
    vJ = _mm256_mul_pd(vJ, vJ);\
    vMu = _mm256_load_pd(&lambda[e * 4]);\
    vMu = _mm256_mul_pd(vMu, vOneHalf);\
    vResult = _mm256_fmadd_pd(vJ, vMu, vResult);\
}\

#define PHAT_FIXED_COROTATED(e, vOne, vTwo, mu, lambda, sigma0, sigma1, sigma2, vResult0, vResult1, vResult2) {\
    __m256d vMu = _mm256_load_pd(&mu[e * 4]);\
    vMu = _mm256_mul_pd(vMu, vTwo);\
    \
    __m256d vSigma0 = _mm256_load_pd(&sigma0[e * 4]);\
    vResult0 = _mm256_sub_pd(vSigma0, vOne);\
    vResult0 = _mm256_mul_pd(vResult0,vMu);\
    __m256d vSigma1 = _mm256_load_pd(&sigma1[e * 4]);\
    vResult1 = _mm256_sub_pd(vSigma1, vOne);\
    vResult1 = _mm256_mul_pd(vResult1,vMu);\
    __m256d vJ = _mm256_mul_pd(vSigma0, vSigma1);\
    __m256d vSigma2 = _mm256_load_pd(&sigma2[e * 4]);\
    vResult2 = _mm256_sub_pd(vSigma2, vOne);\
    vResult2 = _mm256_mul_pd(vResult2,vMu);\
    vJ = _mm256_mul_pd(vJ, vSigma2);\
\
    vMu = _mm256_load_pd(&lambda[e * 4]);\
    __m256d vTmp = _mm256_sub_pd(vJ, vOne);\
    vTmp = _mm256_mul_pd(vTmp, vMu);\
\
    vMu = _mm256_mul_pd(vSigma1,vSigma2);\
    vResult0 = _mm256_fmadd_pd(vTmp, vMu, vResult0);\
\
    vMu = _mm256_mul_pd(vSigma0,vSigma2);\
    vResult1 = _mm256_fmadd_pd(vTmp, vMu, vResult1);\
    \
    vMu = _mm256_mul_pd(vSigma0,vSigma1);\
    vResult2 = _mm256_fmadd_pd(vTmp, vMu, vResult2);\
}\



#define ENERGY_Stable_NeoHookean(e, vOne, vOneHalf, vThree, mu, lambda, sigma0, sigma1, sigma2, vResult) {\
    \
    __m256d vSigma0 = _mm256_load_pd(&sigma0[e * 4]);\
    __m256d vSigma1 = _mm256_load_pd(&sigma1[e * 4]);\
    __m256d vSigma2 = _mm256_load_pd(&sigma2[e * 4]);\
\
    __m256d vSquaredNorm = _mm256_mul_pd(vSigma0, vSigma0);\
    vSquaredNorm = _mm256_fmadd_pd(vSigma1, vSigma1, vSquaredNorm);\
    vSquaredNorm = _mm256_fmadd_pd(vSigma2, vSigma2, vSquaredNorm);\
    vResult = _mm256_sub_pd(vSquaredNorm, vThree);\
    __m256d vMu = _mm256_load_pd(&mu[e * 4]);\
    vResult = _mm256_mul_pd(vResult, vMu);\
    __m256d vLambda = _mm256_load_pd(&lambda[e * 4]);\
    vMu = _mm256_div_pd(vMu, vLambda);\
    __m256d vJ = _mm256_mul_pd(vSigma0, vSigma1);\
    vJ = _mm256_mul_pd(vJ, vSigma2);\
    vJ = _mm256_sub_pd(vJ, vOne);\
    vJ = _mm256_sub_pd(vJ, vMu);\
    vJ = _mm256_mul_pd(vJ, vJ);\
    vJ = _mm256_mul_pd(vJ, vLambda);\
    vResult = _mm256_add_pd(vResult, vJ);\
    vResult = _mm256_mul_pd(vResult, vOneHalf);\
}\

#define PHAT_Stable_NeoHookean(e, vOne, vTwo, mu, lambda, sigma0, sigma1, sigma2, vResult0, vResult1, vResult2) {\
    __m256d vMu = _mm256_load_pd(&mu[e * 4]);\
    \
    __m256d vSigma0 = _mm256_load_pd(&sigma0[e * 4]);\
    vResult0 = _mm256_mul_pd(vSigma0,vMu);\
    __m256d vSigma1 = _mm256_load_pd(&sigma1[e * 4]);\
    vResult1 = _mm256_mul_pd(vSigma1,vMu);\
    __m256d vJ = _mm256_mul_pd(vSigma0, vSigma1);\
    __m256d vSigma2 = _mm256_load_pd(&sigma2[e * 4]);\
    vResult2 = _mm256_mul_pd(vSigma2,vMu);\
    vJ = _mm256_mul_pd(vJ, vSigma2);\
\
    __m256d vLambda = _mm256_load_pd(&lambda[e * 4]);\
    vMu = _mm256_div_pd(vMu, vLambda);\
    vJ = _mm256_sub_pd(vJ, vOne);\
    vJ = _mm256_sub_pd(vJ, vMu);\
    vJ = _mm256_mul_pd(vJ, vLambda);\
    vLambda = _mm256_mul_pd(vSigma1, vSigma2);\
    vResult0 = _mm256_fmadd_pd(vJ, vLambda, vResult0);\
    vLambda = _mm256_mul_pd(vSigma0, vSigma2);\
    vResult1 = _mm256_fmadd_pd(vJ, vLambda, vResult1);\
    vLambda = _mm256_mul_pd(vSigma0, vSigma1);\
    vResult2 = _mm256_fmadd_pd(vJ, vLambda, vResult2);\
}\



#endif //DOT_SIMD_DOUBLE_MACROS_HPP
