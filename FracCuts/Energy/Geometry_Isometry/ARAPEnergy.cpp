//
//  ARAPEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/5/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "ARAPEnergy.hpp"
#include "Energy.hpp"
#include "IglUtils.hpp"

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

namespace FracCuts {
    
    //TODO: precomputation to accelerate optimization process
    
    template<int dim>
    void ARAPEnergy<dim>::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        energyValPerElem.resize(data.F.rows());
        for(int triI = 0; triI < data.F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector3d x_3D[3] = {
                data.V_rest.row(triVInd[0]),
                data.V_rest.row(triVInd[1]),
                data.V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x[3];
            IglUtils::mapTriangleTo2D(x_3D, x);
            
            const Eigen::Vector2d u[3] = {
                data.V.row(triVInd[0]),
                data.V.row(triVInd[1]),
                data.V.row(triVInd[2])
            };
            
            Eigen::Matrix2d triMtrX, triMtrU;
            triMtrX << x[1], x[2];
            triMtrU << u[1] - u[0], u[2] - u[0];
            //TODO: change to AutoFlipSVD
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(triMtrU * triMtrX.inverse(), Eigen::ComputeFullU | Eigen::ComputeFullV);
            const double w = x[1][0] * x[2][1] / 2.0;
            //!!! currently does not support triangle weight
            if(triMtrU.determinant() < 0.0) {
                energyValPerElem[triI] = w * ((svd.singularValues()[0] - 1.0) * (svd.singularValues()[0] - 1.0) +
                                              (-svd.singularValues()[1] - 1.0) * (-svd.singularValues()[1] - 1.0));
            }
            else {
                energyValPerElem[triI] = w * ((svd.singularValues()[0] - 1.0) * (svd.singularValues()[0] - 1.0) +
                                              (svd.singularValues()[1] - 1.0) * (svd.singularValues()[1] - 1.0));
            }
        }
    }
    
    template<int dim>
    void ARAPEnergy<dim>::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        Eigen::MatrixXd cotVals;
        igl::cotmatrix_entries(data.V_rest, data.F, cotVals);
        
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector3d x_3D[3] = {
                data.V_rest.row(triVInd[0]),
                data.V_rest.row(triVInd[1]),
                data.V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x[3];
            IglUtils::mapTriangleTo2D(x_3D, x);
            
            const Eigen::Vector2d u[3] = {
                data.V.row(triVInd[0]),
                data.V.row(triVInd[1]),
                data.V.row(triVInd[2])
            };
            
            Eigen::Matrix2d crossCov = Eigen::Matrix2d::Zero();
            for(int vI = 0; vI < 3; vI++)
            {
                int vI_post = (vI + 1) % 3;
                int vI_pre = (vI + 2) % 3;
                crossCov += cotVals(triI, vI_pre) * (u[vI] - u[vI_post]) * ((x[vI] - x[vI_post]).transpose());
            }
            //TODO: change to AutoFlipSVD
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(crossCov, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d targetRotMtr = svd.matrixU() * svd.matrixV().transpose();
            if(targetRotMtr.determinant() < 0.0) { //!!! deciding by determinant is only valid in 2D, change to use eigen value decomposition!
                Eigen::Matrix2d V = svd.matrixV();
                V.col(1) *= -1.0;
                targetRotMtr = svd.matrixU() * V.transpose();
            }
            
            for(int vI = 0; vI < 3; vI++)
            {
                int vI_post = (vI + 1) % 3;
                int vI_pre = (vI + 2) % 3;
                const Eigen::Vector2d vec = cotVals(triI, vI_pre) * ((targetRotMtr * (x[vI] - x[vI_post])) -
                    (u[vI] - u[vI_post])); // this makes the solve to give search direction rather than new configuration as in [Liu et al. 2008]
                //!!! currently does not support triangle weight
                gradient.block(triVInd[vI] * 2, 0, 2, 1) -= vec;
                gradient.block(triVInd[vI_post] * 2, 0, 2, 1) += vec;
            }
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    
    template<int dim>
    void ARAPEnergy<dim>::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight) const
    {
        precondMtr = data.LaplacianMtr;
        //!!! currently does not support triangle weight
    }
    
    template<int dim>
    void ARAPEnergy<dim>::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight) const
    {
        assert(0 && "no hessian computation for this energy");
    }
    
    template<int dim>
    void ARAPEnergy<dim>::checkEnergyVal(const TriangleSoup& data) const
    {
        // not quite necessary
    }
    
    template<int dim>
    ARAPEnergy<dim>::ARAPEnergy(void) :
        Energy<dim>(false)
    {
        
    }
    
    template class ARAPEnergy<2>;
    
}
