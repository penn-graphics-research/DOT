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
    
    void NeoHookeanEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        Energy::getEnergyValPerElemBySVD(data, energyValPerElem, uniformWeight);
    }
    
    void NeoHookeanEnergy::compute_E(const Eigen::VectorXd& singularValues,
                                     double& E) const
    {
        const double sigma2Sum = singularValues.squaredNorm();
        const double sigmaProd = singularValues.prod();
        const double log_sigmaProd = std::log(sigmaProd);
        
        E = u / 2.0 * (sigma2Sum - singularValues.size()) - u * log_sigmaProd + lambda / 2.0 * log_sigmaProd * log_sigmaProd;
    }
    void NeoHookeanEnergy::compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                       Eigen::VectorXd& dE_div_dsigma) const
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        
        dE_div_dsigma.resize(singularValues.size());
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            const double inv = 1.0 / singularValues[sigmaI];
            dE_div_dsigma[sigmaI] = u * (singularValues[sigmaI] - inv) + lambda * inv * log_sigmaProd;
        }
    }
    void NeoHookeanEnergy::compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                                   Eigen::MatrixXd& d2E_div_dsigma2) const
    {
        const double log_sigmaProd = std::log(singularValues.prod());
        
        d2E_div_dsigma2.resize(singularValues.size(), singularValues.size());
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            const double inv2 = 1.0 / singularValues[sigmaI] / singularValues[sigmaI];
            d2E_div_dsigma2(sigmaI, sigmaI) = u * (1.0 + inv2) - lambda * inv2 * (log_sigmaProd - 1.0);
            for(int sigmaJ = sigmaI + 1; sigmaJ < singularValues.size(); sigmaJ++) {
                d2E_div_dsigma2(sigmaI, sigmaJ) = d2E_div_dsigma2(sigmaJ, sigmaI) = lambda / singularValues[sigmaI] / singularValues[sigmaJ];
            }
        }
    }
    void NeoHookeanEnergy::compute_dE_div_dF(const Eigen::MatrixXd& F,
                                             const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                             Eigen::MatrixXd& dE_div_dF) const
    {
        const double J = svd.singularValues().prod();
        Eigen::Matrix2d FInvT = svd.matrixU() *
            Eigen::DiagonalMatrix<double, 2>(1.0 / svd.singularValues()[0], 1.0 / svd.singularValues()[1]) *
            svd.matrixV().transpose();
        dE_div_dF = u * (F - FInvT) + lambda * std::log(J) * FInvT;
    }
    
    void NeoHookeanEnergy::checkEnergyVal(const TriangleSoup& data) const // check with isometric case
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
    
    NeoHookeanEnergy::NeoHookeanEnergy(double YM, double PR) :
        Energy(true, true, true), u(YM / 2.0 / (1.0 + PR)), lambda(YM * PR / (1.0 + PR) / (1.0 - 2.0 * PR))
    {
        const double sigma2Sum = 8;
        const double sigmaProd = 4;
        const double log_sigmaProd = std::log(sigmaProd);
        
        const double visRange_max = u / 2.0 * (sigma2Sum - 2) - u * log_sigmaProd + lambda / 2.0 * log_sigmaProd * log_sigmaProd;
        Energy::setVisRange_energyVal(0.0, visRange_max);
    }
    
    void NeoHookeanEnergy::getBulkModulus(double& bulkModulus)
    {
        bulkModulus = lambda + (2.0/3.0) * u;
    }
    
}
