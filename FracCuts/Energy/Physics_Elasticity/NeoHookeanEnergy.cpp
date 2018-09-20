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
    void NeoHookeanEnergy<dim>::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        std::vector<AutoFlipSVD<Eigen::Matrix2d>> svd(data.F.rows());
        std::vector<Eigen::Matrix2d> F(data.F.rows());
        Energy<dim>::getEnergyValPerElemBySVD(data, true, svd, F, energyValPerElem, uniformWeight);
    }
    
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_E(const Eigen::Vector2d& singularValues,
                                     double& E) const
    {
        const double sigma2Sum = singularValues.squaredNorm();
        const double sigmaProd = singularValues.prod();
        const double log_sigmaProd = std::log(sigmaProd);
        
        E = u / 2.0 * (sigma2Sum - singularValues.size()) - (u - lambda / 2.0 * log_sigmaProd) * log_sigmaProd;
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_dE_div_dsigma(const Eigen::Vector2d& singularValues,
                                                 Eigen::Vector2d& dE_div_dsigma) const
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        
        const double inv0 = 1.0 / singularValues[0];
        dE_div_dsigma[0] = u * (singularValues[0] - inv0) + lambda * inv0 * log_sigmaProd;
        const double inv1 = 1.0 / singularValues[1];
        dE_div_dsigma[1] = u * (singularValues[1] - inv1) + lambda * inv1 * log_sigmaProd;
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_d2E_div_dsigma2(const Eigen::Vector2d& singularValues,
                                                   Eigen::Matrix2d& d2E_div_dsigma2) const
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        
        const double inv2_0 = 1.0 / singularValues[0] / singularValues[0];
        d2E_div_dsigma2(0, 0) = u * (1.0 + inv2_0) - lambda * inv2_0 * (log_sigmaProd - 1.0);
        const double inv2_1 = 1.0 / singularValues[1] / singularValues[1];
        d2E_div_dsigma2(1, 1) = u * (1.0 + inv2_1) - lambda * inv2_1 * (log_sigmaProd - 1.0);
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda / singularValues[0] / singularValues[1];
    }
    template<int dim>
    void NeoHookeanEnergy<dim>::compute_dE_div_dF(const Eigen::Matrix2d& F,
                                             const AutoFlipSVD<Eigen::Matrix2d>& svd,
                                             Eigen::Matrix2d& dE_div_dF) const
    {
        //TODO: optimize for 2D
        const double J = svd.singularValues().prod();
        Eigen::Matrix2d FInvT;
        FInvT(0, 0) = F(1, 1) / J;
        FInvT(0, 1) = -F(1, 0) / J;
        FInvT(1, 0) = -F(0, 1) / J;
        FInvT(1, 1) = F(0, 0) / J;
        dE_div_dF = u * (F - FInvT) + lambda * std::log(J) * FInvT;
    }
    
    template<int dim>
    void NeoHookeanEnergy<dim>::checkEnergyVal(const TriangleSoup& data) const // check with isometric case
    {
        //TODO: move to super class, only provide a value
        
        std::cout << "check energyVal computation..." << std::endl;
        
        double err = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Vector3d x0_3D[3] = {
                data.V_rest.row(triVInd[0]),
                data.V_rest.row(triVInd[1]),
                data.V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x0[3];
            IglUtils::mapTriangleTo2D(x0_3D, x0);
            
            Eigen::Matrix2d X0, A;
            X0 << x0[1] - x0[0], x0[2] - x0[0];
            A = X0.inverse(); //TODO: this only need to be computed once
            
            AutoFlipSVD<Eigen::MatrixXd> svd(X0 * A); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
            
            const double sigma2Sum = svd.singularValues().squaredNorm();
            const double sigmaProd = svd.singularValues().prod();
            const double log_sigmaProd = std::log(sigmaProd);
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            const double energyVal = w * (u / 2.0 * (sigma2Sum - svd.singularValues().size()) - u * log_sigmaProd + lambda / 2.0 * log_sigmaProd * log_sigmaProd);
            err += energyVal;
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
    
    template class NeoHookeanEnergy<2>;
    
}
