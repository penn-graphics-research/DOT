//
//  FixedCoRotEnergy.cpp
//  DOT
//
//  Created by Minchen Li on 6/20/18.
//

#include "FixedCoRotEnergy.hpp"
#include "IglUtils.hpp"
#include "Timer.hpp"

#include <immintrin.h>
#include "SVD_EFTYCHIOS/PTHREAD_QUEUE.h"
#include "SIMD_DOUBLE_MACROS.hpp"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

extern double *Gmu, *Glambda, *Gsigma0, *Gsigma1, *Gsigma2;

extern Timer timer_temp;

namespace DOT {
    
    template<int dim>
    void FixedCoRotEnergy<dim>::computeEnergyVal(const Mesh<dim>& data, int redoSVD,
                                                 std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                                 std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                                 std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                                 std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                                 std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                                 double coef,
                                                 double& energyVal) const
    {
        Base::computeEnergyValBySVD(data, redoSVD, svd, F, U, V, Sigma, coef, energyVal);
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::computeGradient(const Mesh<dim>& data, bool redoSVD,
                                                std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                                std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                                std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                                std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                                std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                                double coef,
                                                Eigen::VectorXd& gradient) const
    {
        Base::computeGradientByPK(data, redoSVD, svd, F, U, V, Sigma, coef, gradient);
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::computeHessian(const Mesh<dim>& data, bool redoSVD,
                                               std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                               std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                               double coef,
                                               LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                               bool projectSPD) const
    {
        Base::computeHessianByPK(data, redoSVD, svd, F, coef, linSysSolver, projectSPD);
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::computeHessian(const Mesh<dim>& data, bool redoSVD,
                                               std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                               std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                               double coef,
                                               Eigen::MatrixXd& hessian,
                                               bool projectSPD) const
    {
        Base::computeHessianByPK(data, redoSVD, svd, F, coef, hessian, projectSPD);
    }
    
    template<int dim>
    void FixedCoRotEnergy<dim>::getEnergyValPerElem(const Mesh<dim>& data,
                                                    Eigen::VectorXd& energyValPerElem,
                                                    bool uniformWeight) const
    {
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd(data.F.rows());
        std::vector<Eigen::Matrix<double, dim, dim>> F(data.F.rows()), U(data.F.rows()), V(data.F.rows());
        std::vector<Eigen::Matrix<double, dim, 1>> Sigma(data.F.rows());
        Energy<dim>::getEnergyValPerElemBySVD(data, true, svd, F, U, V, Sigma, energyValPerElem, uniformWeight);
    }
    
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                                          double u, double lambda,
                                          double& E) const
    {
        const double sigmam12Sum = (singularValues -
                                    Eigen::Matrix<double, dim, 1>::Ones()).squaredNorm();
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        
        E = u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                      double u, double lambda,
                                                      Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const
    {
        const double sigmaProdm1lambda = lambda * (singularValues.prod() - 1.0);
        Eigen::Matrix<double, dim, 1> sigmaProd_noI;
        if(dim == 2) {
            sigmaProd_noI[0] = singularValues[1];
            sigmaProd_noI[1] = singularValues[0];
        }
        else {
            sigmaProd_noI[0] = singularValues[1] * singularValues[2];
            sigmaProd_noI[1] = singularValues[2] * singularValues[0];
            sigmaProd_noI[2] = singularValues[0] * singularValues[1];
        }
        
        double _2u = u * 2;
        dE_div_dsigma[0] = (_2u * (singularValues[0] - 1.0) +
                            sigmaProd_noI[0] * sigmaProdm1lambda);
        dE_div_dsigma[1] = (_2u * (singularValues[1] - 1.0) +
                            sigmaProd_noI[1] * sigmaProdm1lambda);
        if(dim == 3) {
            dE_div_dsigma[2] = (_2u * (singularValues[2] - 1.0) +
                                sigmaProd_noI[2] * sigmaProdm1lambda);
        }
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_d2E_div_dsigma2(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                        double u, double lambda,
                                                        Eigen::Matrix<double, dim, dim>& d2E_div_dsigma2) const
    {
        const double sigmaProd = singularValues.prod();
        Eigen::Matrix<double, dim, 1> sigmaProd_noI;
        if(dim == 2) {
            sigmaProd_noI[0] = singularValues[1];
            sigmaProd_noI[1] = singularValues[0];
        }
        else {
            sigmaProd_noI[0] = singularValues[1] * singularValues[2];
            sigmaProd_noI[1] = singularValues[2] * singularValues[0];
            sigmaProd_noI[2] = singularValues[0] * singularValues[1];
        }
        
        double _2u = u * 2;
        d2E_div_dsigma2(0, 0) = _2u + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
        d2E_div_dsigma2(1, 1) = _2u + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
        if(dim == 3) {
            d2E_div_dsigma2(2, 2) = _2u + lambda * sigmaProd_noI[2] * sigmaProd_noI[2];
        }
        
        if(dim == 2) {
            d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) =
                lambda * ((sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
        }
        else {
            d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) =
                lambda * (singularValues[2] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
            d2E_div_dsigma2(0, 2) = d2E_div_dsigma2(2, 0) =
                lambda * (singularValues[1] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[2]);
            d2E_div_dsigma2(2, 1) = d2E_div_dsigma2(1, 2) =
                lambda * (singularValues[0] * (sigmaProd - 1.0) + sigmaProd_noI[2] * sigmaProd_noI[1]);
        }
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_BLeftCoef(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                  double u, double lambda,
                                                  Eigen::Matrix<double, dim * (dim - 1) / 2, 1>& BLeftCoef) const
    {
        const double sigmaProd = singularValues.prod();
        const double halfLambda = lambda / 2.0;
        if(dim == 2) {
            BLeftCoef[0] = u - halfLambda * (sigmaProd - 1);
        }
        else {
            BLeftCoef[0] = u - halfLambda * singularValues[2] * (sigmaProd - 1);
            BLeftCoef[1] = u - halfLambda * singularValues[0] * (sigmaProd - 1);
            BLeftCoef[2] = u - halfLambda * singularValues[1] * (sigmaProd - 1);
        }
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                                  const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                                  double u, double lambda,
                                                  Eigen::Matrix<double, dim, dim>& dE_div_dF) const
    {
        Eigen::Matrix<double, dim, dim> JFInvT;
        IglUtils::computeCofactorMtr(F, JFInvT);
        dE_div_dF = (u * 2 * (F - svd.matrixU() * svd.matrixV().transpose()) +
                     lambda * (svd.singularValues().prod() - 1) * JFInvT);
    }
    
#ifdef USE_SIMD
    template<int dim>
    void FixedCoRotEnergy<dim>::computeElemUEnergyVal_SIMD(Eigen::VectorXd& energyValPerElem,
                                                           int ceiling_size) const
    {
        using T = double;
        
        const T __attribute__ ((aligned(64))) aOne[4] = {1., 1., 1., 1.};
        const T __attribute__ ((aligned(64))) aOneHalf[4] = {.5, .5, .5, .5};
        const T __attribute__ ((aligned(64))) aThree[4] = {3., 3., 3., 3.};
        
        __m256d vOne = _mm256_load_pd(aOne);
        __m256d vOneHalf = _mm256_load_pd(aOneHalf);
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int) ceiling_size / 4, 1, [&](int e)
#else
        for(int e = 0; e < ceiling_size / 4; ++e)
#endif
        {
            __m256d vResult;
            ENERGY_FIXED_COROTATED(e,vOne,vOneHalf,Gmu,Glambda,Gsigma0,Gsigma1,Gsigma2,vResult);
            
            double __attribute__ ((aligned(64))) buffer[4];
            _mm256_store_pd(buffer, vResult);
            energyValPerElem[e * 4 + 0] = buffer[0];
            energyValPerElem[e * 4 + 1] = buffer[1];
            energyValPerElem[e * 4 + 2] = buffer[2];
            energyValPerElem[e * 4 + 3] = buffer[3];
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    void FixedCoRotEnergy<dim>::computePHat_SIMD(std::vector<Eigen::Matrix<double, dim, 1>>& PHat,
                                                 int ceiling_size) const
    {
        using T = double;
        
        const T __attribute__ ((aligned(64))) aOne[4] = {1., 1., 1., 1.};
        const T __attribute__ ((aligned(64))) aTwo[4] = {2, 2, 2, 2};
        __m256d vOne = _mm256_load_pd(aOne);
        __m256d vTwo = _mm256_load_pd(aTwo);
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int) ceiling_size / 4, 1, [&](int e)
#else
        for(int e = 0; e < ceiling_size / 4; ++ e)
#endif
        {
            __m256d vResult0, vResult1, vResult2;
            PHAT_FIXED_COROTATED(e, vOne, vTwo, Gmu, Glambda, Gsigma0, Gsigma1, Gsigma2, vResult0, vResult1, vResult2);
              
            double __attribute__ ((aligned(64))) buffer[4];
              
            _mm256_store_pd(buffer, vResult0);
            PHat[e* 4 + 0][0] = buffer[0];
            PHat[e* 4 + 1][0] = buffer[1];
            PHat[e* 4 + 2][0] = buffer[2];
            PHat[e* 4 + 3][0] = buffer[3];
            
            _mm256_store_pd(buffer, vResult1);
            PHat[e* 4 + 0][1] = buffer[0];
            PHat[e* 4 + 1][1] = buffer[1];
            PHat[e* 4 + 2][1] = buffer[2];
            PHat[e* 4 + 3][1] = buffer[3];
            
            _mm256_store_pd(buffer, vResult2);
            PHat[e* 4 + 0][2] = buffer[0];
            PHat[e* 4 + 1][2] = buffer[1];
            PHat[e* 4 + 2][2] = buffer[2];
            PHat[e* 4 + 3][2] = buffer[3];
        }
#ifdef USE_TBB
        );
#endif
    }
#endif // #ifdef USE_SIMD
    
    template<int dim>
    void FixedCoRotEnergy<dim>::checkEnergyVal(const Mesh<dim>& data) const // check with isometric case
    {
        //TODO: move to super class, only provide a value
        
        std::cout << "check energyVal computation..." << std::endl;
        
        double err = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Eigen::Matrix<double, dim, dim>::Identity()); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
            
            double energyVal;
            compute_E(svd.singularValues(), data.u[triI], data.lambda[triI], energyVal);
            err += data.triWeight[triI] * data.triArea[triI] * energyVal;
        }
        
        std::cout << "energyVal computation error = " << err << std::endl;
    }
    
    template<int dim>
    FixedCoRotEnergy<dim>::FixedCoRotEnergy(void) :
        Energy<dim>(true, true)
    {
//        const double sigmam12Sum = 2;
//        const double sigmaProdm1 = 3;
//        
//        const double visRange_max = (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
//        Energy<dim>::setVisRange_energyVal(0.0, visRange_max);
        Energy<dim>::setVisRange_energyVal(0.0, 1.0);
    }
    
    template class FixedCoRotEnergy<DIM>;
    
}
