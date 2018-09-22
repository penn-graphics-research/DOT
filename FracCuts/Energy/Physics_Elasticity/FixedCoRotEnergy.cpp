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

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

extern Timer timer_temp;

namespace FracCuts {
    
    template<int dim>
    void FixedCoRotEnergy<dim>::getEnergyValPerElem(const TriangleSoup<dim>& data,
                                                    Eigen::VectorXd& energyValPerElem,
                                                    bool uniformWeight) const
    {
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd(data.F.rows());
        std::vector<Eigen::Matrix<double, dim, dim>> F(data.F.rows());
        Energy<dim>::getEnergyValPerElemBySVD(data, true, svd, F, energyValPerElem, uniformWeight);
    }
    
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                                          double& E) const
    {
        const double sigmam12Sum = (singularValues -
                                    Eigen::Matrix<double, dim, 1>::Ones()).squaredNorm();
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        
        E = u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_dE_div_dsigma(const Eigen::Vector2d& singularValues,
                                                 Eigen::Vector2d& dE_div_dsigma) const
    {
        const double sigmaProdm1lambda = lambda * (singularValues.prod() - 1.0);
        Eigen::Vector2d sigmaProd_noI(singularValues[1], singularValues[0]);
        
        dE_div_dsigma[0] = (_2u * (singularValues[0] - 1.0) +
                            sigmaProd_noI[0] * sigmaProdm1lambda);
        dE_div_dsigma[1] = (_2u * (singularValues[1] - 1.0) +
                            sigmaProd_noI[1] * sigmaProdm1lambda);
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_d2E_div_dsigma2(const Eigen::Vector2d& singularValues,
                                                   Eigen::Matrix2d& d2E_div_dsigma2) const
    {
        const double sigmaProd = singularValues.prod();
        Eigen::Vector2d sigmaProd_noI(singularValues[1], singularValues[0]);
        
        d2E_div_dsigma2(0, 0) = _2u + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
        d2E_div_dsigma2(1, 1) = _2u + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) =
            lambda * ((sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
    }
    template<int dim>
    void FixedCoRotEnergy<dim>::compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                                  const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                                  Eigen::Matrix<double, dim, dim>& dE_div_dF) const
    {
        Eigen::Matrix<double, dim, dim> JFInvT;
        IglUtils::computeCofactorMtr(F, JFInvT);
        dE_div_dF = (_2u * (F - svd.matrixU() * svd.matrixV().transpose()) +
                     lambda * (svd.singularValues().prod() - 1) * JFInvT);
    }
    
    template<int dim>
    void FixedCoRotEnergy<dim>::checkEnergyVal(const TriangleSoup<dim>& data) const // check with isometric case
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
    
    template<int dim>
    FixedCoRotEnergy<dim>::FixedCoRotEnergy(double YM, double PR) :
        Energy<dim>(true, false, true), u(YM / 2.0 / (1.0 + PR)), lambda(YM * PR / (1.0 + PR) / (1.0 - 2.0 * PR)), _2u(2.0 * u)
    {
        const double sigmam12Sum = 2;
        const double sigmaProdm1 = 3;
        
        const double visRange_max = (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
        Energy<dim>::setVisRange_energyVal(0.0, visRange_max);
    }
    
    template<int dim>
    void FixedCoRotEnergy<dim>::getBulkModulus(double& bulkModulus)
    {
        bulkModulus = lambda + _2u / 3.0;
    }
    
    template class FixedCoRotEnergy<2>;
    
}
