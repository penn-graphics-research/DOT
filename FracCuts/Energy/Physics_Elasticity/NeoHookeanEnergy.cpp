//
//  NeoHookeanEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/19/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "NeoHookeanEnergy.hpp"
#include "IglUtils.hpp"

namespace FracCuts {
    
    template<int dim>
    void NeoHookeanEnergy<dim>::computeEnergyVal(const TriangleSoup<dim>& data, bool redoSVD,
                                                 std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                                 std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                                 double& energyVal) const
    {
        Base::computeEnergyValBySVD(data, redoSVD, svd, F, energyVal);
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::computeGradient(const TriangleSoup<dim>& data, bool redoSVD,
                                                std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                                std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                                Eigen::VectorXd& gradient) const
    {
        Base::computeGradientByPK(data, redoSVD, svd, F, gradient);
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::computeHessian(const TriangleSoup<dim>& data, bool redoSVD,
                                               std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                               std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                               double coef,
                                               LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                               bool projectSPD) const
    {
        Base::computeHessianByPK(data, redoSVD, svd, F, coef, linSysSolver, projectSPD);
    }
    
    template<int dim>
    void NeoHookeanEnergy<dim>::getEnergyValPerElem(const TriangleSoup<dim>& data,
                                                    Eigen::VectorXd& energyValPerElem,
                                                    bool uniformWeight) const
    {
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd(data.F.rows());
        std::vector<Eigen::Matrix<double, dim, dim>> F(data.F.rows());
        Energy<dim>::getEnergyValPerElemBySVD(data, true, svd, F, energyValPerElem, uniformWeight);
    }
    
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                                          double& E) const
    {
        const double sigma2Sum = singularValues.squaredNorm();
        const double sigmaProd = singularValues.prod();
        const double log_sigmaProd = std::log(sigmaProd);
        
        E = u / 2.0 * (sigma2Sum - dim) - (u - lambda / 2.0 * log_sigmaProd) * log_sigmaProd;
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                      Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        
        const double inv0 = 1.0 / singularValues[0];
        dE_div_dsigma[0] = u * (singularValues[0] - inv0) + lambda * inv0 * log_sigmaProd;
        const double inv1 = 1.0 / singularValues[1];
        dE_div_dsigma[1] = u * (singularValues[1] - inv1) + lambda * inv1 * log_sigmaProd;
        if(dim == 3) {
            const double inv2 = 1.0 / singularValues[2];
            dE_div_dsigma[2] = u * (singularValues[2] - inv2) + lambda * inv2 * log_sigmaProd;
        }
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_d2E_div_dsigma2(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                        Eigen::Matrix<double, dim, dim>& d2E_div_dsigma2) const
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        
        const double inv2_0 = 1.0 / singularValues[0] / singularValues[0];
        d2E_div_dsigma2(0, 0) = u * (1.0 + inv2_0) - lambda * inv2_0 * (log_sigmaProd - 1.0);
        const double inv2_1 = 1.0 / singularValues[1] / singularValues[1];
        d2E_div_dsigma2(1, 1) = u * (1.0 + inv2_1) - lambda * inv2_1 * (log_sigmaProd - 1.0);
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda / singularValues[0] / singularValues[1];
        if(dim == 3) {
            const double inv2_2 = 1.0 / singularValues[2] / singularValues[2];
            d2E_div_dsigma2(2, 2) = u * (1.0 + inv2_2) - lambda * inv2_2 * (log_sigmaProd - 1.0);
            d2E_div_dsigma2(1, 2) = d2E_div_dsigma2(2, 1) = lambda / singularValues[1] / singularValues[2];
            d2E_div_dsigma2(2, 0) = d2E_div_dsigma2(0, 2) = lambda / singularValues[2] / singularValues[0];
        }
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_BLeftCoef(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                  Eigen::Matrix<double, dim * (dim - 1) / 2, 1>& BLeftCoef) const
    {
        //TODO: right coef also has analytical form
        const double sigmaProd = singularValues.prod();
        if(dim == 2) {
            BLeftCoef[0] = (u + (u - lambda * std::log(sigmaProd)) / sigmaProd) / 2.0;
        }
        else {
            const double middle = u - lambda * std::log(sigmaProd);
            BLeftCoef[0] = (u + middle / singularValues[0] / singularValues[1]) / 2.0;
            BLeftCoef[1] = (u + middle / singularValues[1] / singularValues[2]) / 2.0;
            BLeftCoef[2] = (u + middle / singularValues[2] / singularValues[0]) / 2.0;
        }
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                                  const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                                  Eigen::Matrix<double, dim, dim>& dE_div_dF) const
    {
        //TODO: optimize for 2D
        const double J = svd.singularValues().prod();
        Eigen::Matrix<double, dim, dim> FInvT;
        IglUtils::computeCofactorMtr(F, FInvT);
        FInvT /= J;
        dE_div_dF = u * (F - FInvT) + lambda * std::log(J) * FInvT;
    }
    
    template<int dim>
    void NeoHookeanEnergy<dim>::checkEnergyVal(const TriangleSoup<dim>& data) const // check with isometric case
    {
        //TODO: move to super class, only provide a value
        
        std::cout << "check energyVal computation..." << std::endl;
        
        double err = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Eigen::Matrix<double, dim, dim>::Identity()); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
            
            double energyVal;
            compute_E(svd.singularValues(), energyVal);
            err += data.triWeight[triI] * data.triArea[triI] * energyVal;
        }
        
        std::cout << "energyVal computation error = " << err << std::endl;
    }
    
    template<int dim>
    NeoHookeanEnergy<dim>::NeoHookeanEnergy(double YM, double PR) :
        Energy<dim>(true, true, true), u(YM / 2.0 / (1.0 + PR)), lambda(YM * PR / (1.0 + PR) / (1.0 - 2.0 * PR))
    {
        const double sigma2Sum = 8;
        const double sigmaProd = 4;
        const double log_sigmaProd = std::log(sigmaProd);
        
        const double visRange_max = u / 2.0 * (sigma2Sum - 2) - u * log_sigmaProd + lambda / 2.0 * log_sigmaProd * log_sigmaProd;
        Energy<dim>::setVisRange_energyVal(0.0, visRange_max);
    }
    
    template<int dim>
    void NeoHookeanEnergy<dim>::getBulkModulus(double& bulkModulus)
    {
        bulkModulus = lambda + (2.0/3.0) * u;
    }
    
    template class NeoHookeanEnergy<DIM>;
    
}
