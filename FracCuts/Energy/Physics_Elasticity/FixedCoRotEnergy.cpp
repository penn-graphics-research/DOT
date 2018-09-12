//
//  FixedCoRotEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "FixedCoRotEnergy.hpp"
#include "IglUtils.hpp"
#include "Timer.hpp"

#include <tbb/tbb.h>

extern Timer timer_temp;

namespace FracCuts {
    
    void FixedCoRotEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        std::vector<AutoFlipSVD<Eigen::MatrixXd>> svd(data.F.rows());
        Energy::getEnergyValPerElemBySVD(data, true, svd, energyValPerElem, uniformWeight);
    }
    
    void FixedCoRotEnergy::compute_E(const Eigen::VectorXd& singularValues,
                                     double& E) const
    {
        const double sigmam12Sum = (singularValues - Eigen::Vector2d::Ones()).squaredNorm();
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        
        E = u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
    }
    void FixedCoRotEnergy::compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                                 Eigen::VectorXd& dE_div_dsigma) const
    {
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        Eigen::Vector2d sigmaProd_noI = Eigen::Vector2d::Ones();
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            for(int sigmaJ = 0; sigmaJ < singularValues.size(); sigmaJ++) {
                if(sigmaJ != sigmaI) {
                    sigmaProd_noI[sigmaI] *= singularValues[sigmaJ];
                }
            }
        }
        
        dE_div_dsigma.resize(singularValues.size());
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            dE_div_dsigma[sigmaI] = 2.0 * u * (singularValues[sigmaI] - 1.0) +
                lambda * sigmaProd_noI[sigmaI] * sigmaProdm1;
        }
    }
    void FixedCoRotEnergy::compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                                   Eigen::MatrixXd& d2E_div_dsigma2) const
    {
        const double sigmaProd = singularValues.prod();
        Eigen::Vector2d sigmaProd_noI = Eigen::Vector2d::Ones();
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            for(int sigmaJ = 0; sigmaJ < singularValues.size(); sigmaJ++) {
                if(sigmaJ != sigmaI) {
                    sigmaProd_noI[sigmaI] *= singularValues[sigmaJ];
                }
            }
        }
        
        d2E_div_dsigma2.resize(singularValues.size(), singularValues.size());
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            d2E_div_dsigma2(sigmaI, sigmaI) = 2.0 * u +
                lambda * sigmaProd_noI[sigmaI] * sigmaProd_noI[sigmaI];
            for(int sigmaJ = sigmaI + 1; sigmaJ < singularValues.size(); sigmaJ++) {
                d2E_div_dsigma2(sigmaI, sigmaJ) = d2E_div_dsigma2(sigmaJ, sigmaI) =
                    lambda * ((sigmaProd - 1.0) + sigmaProd_noI[sigmaI] * sigmaProd_noI[sigmaJ]);
            }
        }
    }
    void FixedCoRotEnergy::compute_dE_div_dF(const Eigen::MatrixXd& F,
                                             const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                             Eigen::MatrixXd& dE_div_dF) const
    {
        // 2D
        const double J = svd.singularValues().prod();
        Eigen::Matrix2d JFInvT, VT = svd.matrixV().transpose();
        JFInvT.col(0) = svd.matrixU().col(0) * svd.singularValues()[1];
        JFInvT.col(1) = svd.matrixU().col(1) * svd.singularValues()[0];
        JFInvT *= VT;
        dE_div_dF = (2 * u * (F - svd.matrixU() * VT) + lambda * (J - 1) * JFInvT);
    }
    
    void FixedCoRotEnergy::checkEnergyVal(const TriangleSoup& data) const // check with isometric case
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
            
            const double sigmam12Sum = (svd.singularValues() - Eigen::Vector2d::Ones()).squaredNorm();
            const double sigmaProdm1 = svd.singularValues().prod() - 1.0;
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            const double energyVal = w * (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
            err += energyVal;
        }
        
        std::cout << "energyVal computation error = " << err << std::endl;
    }
    
    FixedCoRotEnergy::FixedCoRotEnergy(double YM, double PR) :
        Energy(true, false, true), u(YM / 2.0 / (1.0 + PR)), lambda(YM * PR / (1.0 + PR) / (1.0 - 2.0 * PR))
    {
        const double sigmam12Sum = 2;
        const double sigmaProdm1 = 3;
        
        const double visRange_max = (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
        Energy::setVisRange_energyVal(0.0, visRange_max);
    }
    
    void FixedCoRotEnergy::getBulkModulus(double& bulkModulus)
    {
        bulkModulus = lambda + (2.0/3.0) * u;
    }
    
}
