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
        std::vector<AutoFlipSVD<Eigen::Matrix2d>> svd(data.F.rows());
        Energy::getEnergyValPerElemBySVD(data, true, svd, energyValPerElem, uniformWeight);
    }
    
    void FixedCoRotEnergy::compute_E(const Eigen::Vector2d& singularValues,
                                     double& E) const
    {
        const double sigmam12Sum = (singularValues - Eigen::Vector2d::Ones()).squaredNorm();
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        
        E = u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
    }
    void FixedCoRotEnergy::compute_dE_div_dsigma(const Eigen::Vector2d& singularValues,
                                                 Eigen::Vector2d& dE_div_dsigma) const
    {
        const double sigmaProdm1lambda = lambda * (singularValues.prod() - 1.0);
        Eigen::Vector2d sigmaProd_noI(singularValues[1], singularValues[0]);
        
        dE_div_dsigma[0] = (_2u * (singularValues[0] - 1.0) +
                            sigmaProd_noI[0] * sigmaProdm1lambda);
        dE_div_dsigma[1] = (_2u * (singularValues[1] - 1.0) +
                            sigmaProd_noI[1] * sigmaProdm1lambda);
    }
    void FixedCoRotEnergy::compute_d2E_div_dsigma2(const Eigen::Vector2d& singularValues,
                                                   Eigen::Matrix2d& d2E_div_dsigma2) const
    {
        const double sigmaProd = singularValues.prod();
        Eigen::Vector2d sigmaProd_noI(singularValues[1], singularValues[0]);
        
        d2E_div_dsigma2(0, 0) = _2u + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
        d2E_div_dsigma2(1, 1) = _2u + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) =
            lambda * ((sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
    }
    void FixedCoRotEnergy::compute_dE_div_dF(const Eigen::Matrix2d& F,
                                             const AutoFlipSVD<Eigen::Matrix2d>& svd,
                                             Eigen::Matrix2d& dE_div_dF) const
    {
        Eigen::Matrix2d JFInvT;
        JFInvT(0, 0) = F(1, 1);
        JFInvT(0, 1) = -F(1, 0);
        JFInvT(1, 0) = -F(0, 1);
        JFInvT(1, 1) = F(0, 0);
        dE_div_dF = (_2u * (F - svd.matrixU() * svd.matrixV().transpose()) +
                     lambda * (svd.singularValues().prod() - 1) * JFInvT);
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
        Energy(true, false, true), u(YM / 2.0 / (1.0 + PR)), lambda(YM * PR / (1.0 + PR) / (1.0 - 2.0 * PR)), _2u(2.0 * u)
    {
        const double sigmam12Sum = 2;
        const double sigmaProdm1 = 3;
        
        const double visRange_max = (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
        Energy::setVisRange_energyVal(0.0, visRange_max);
    }
    
    void FixedCoRotEnergy::getBulkModulus(double& bulkModulus)
    {
        bulkModulus = lambda + _2u / 3.0;
    }
    
}
