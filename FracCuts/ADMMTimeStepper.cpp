//
//  ADMMTimeStepper.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/28/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "ADMMTimeStepper.hpp"

#include "IglUtils.hpp"

#include <tbb/tbb.h>

namespace FracCuts {
    
    ADMMTimeStepper::ADMMTimeStepper(const TriangleSoup& p_data0,
                                     const std::vector<Energy*>& p_energyTerms,
                                     const std::vector<double>& p_energyParams,
                                     int p_propagateFracture,
                                     bool p_mute,
                                     bool p_scaffolding,
                                     const Eigen::MatrixXd& UV_bnds,
                                     const Eigen::MatrixXi& E,
                                     const Eigen::VectorXi& bnd) :
        Optimizer(p_data0, p_energyTerms, p_energyParams,
                  p_propagateFracture, p_mute, p_scaffolding,
                  UV_bnds, E, bnd)
    {}
    
    void ADMMTimeStepper::precompute(void)
    {
        Optimizer::precompute();
        
        
        z.resize(result.F.rows(), 4);
        
        u.resize(result.F.rows(), 4);
        
        double bulkModulus;
        energyTerms[0]->getBulkModulus(bulkModulus);
        double wi = dt * std::sqrt(bulkModulus);
        weights.resize(result.F.rows());
        for(int triI = 0; triI < result.F.rows(); triI++) {
            weights[triI] = wi * std::sqrt(result.triArea[triI]);
        }
        weights2 = weights.cwiseProduct(weights);
        
        // initialize D_array
        D_array.resize(result.F.rows());
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            
            Eigen::Vector3d x0_3D[3] = {
                result.V_rest.row(triVInd[0]),
                result.V_rest.row(triVInd[1]),
                result.V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x0[3];
            IglUtils::mapTriangleTo2D(x0_3D, x0);
            
            Eigen::Matrix2d X0, A;
            X0 << x0[1] - x0[0], x0[2] - x0[0];
            A = X0.inverse();
            
            const double mA11mA21 = -A(0, 0) - A(1, 0);
            const double mA12mA22 = -A(0, 1) - A(1, 1);
            D_array[triI].resize(4, 6);
            D_array[triI] <<
                mA11mA21, 0.0, A(0, 0), 0.0, A(1, 0), 0.0,
                mA12mA22, 0.0, A(0, 1), 0.0, A(1, 1), 0.0,
                0.0, mA11mA21, 0.0, A(0, 0), 0.0, A(1, 0),
                0.0, mA12mA22, 0.0, A(0, 1), 0.0, A(1, 1);
        }
        
        rhs_xUpdate.resize(result.V.rows() * 2);
        M_mult_xHat.resize(result.V.rows() * 2);
        
        // construct and prefactorize the linear system
        std::vector<Eigen::Triplet<double>> triplet(result.F.rows() * 4 * 3);
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            for(int FRowI = 0; FRowI < 2; FRowI++) {
                for(int localVI = 0; localVI < 3; localVI++) {
                    triplet.emplace_back(triI * 4 + FRowI * 2, triVInd[localVI] * 2 + FRowI,
                                         weights[triI] * D_array[triI](FRowI * 2, localVI * 2 + FRowI));
                    triplet.emplace_back(triI * 4 + FRowI * 2 + 1, triVInd[localVI] * 2 + FRowI,
                                         weights[triI] * D_array[triI](FRowI * 2 + 1, localVI * 2 + FRowI));
                }
            }
        }
        Eigen::SparseMatrix<double> WD;
        WD.resize(result.F.rows() * 4, result.V.rows() * 2);
        WD.setFromTriplets(triplet.begin(), triplet.end());
        
        Eigen::SparseMatrix<double> coefMtr = WD.transpose() * WD;
        for(int vI = 0; vI < result.V.rows(); vI++) {
            double massI = result.massMatrix.coeffRef(vI, vI);
            coefMtr.coeffRef(vI * 2, vI * 2) += massI;
            coefMtr.coeffRef(vI * 2 + 1, vI * 2 + 1) += massI;
        }
        offset_fixVerts.clear();
        for (int k = 0; k < coefMtr.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(coefMtr, k); it; ++it)
            {
                bool fixed_rowV = (result.fixedVert.find(it.row() / 2) != result.fixedVert.end());
                bool fixed_colV = (result.fixedVert.find(it.col() / 2) != result.fixedVert.end());
                if(fixed_rowV || fixed_colV) {
                    if(!fixed_rowV) {
                        offset_fixVerts[std::pair<int, int>(it.row(), it.col())] = it.value();
                    }
                    it.valueRef() = 0.0;
                }
            }
        }
        for(const auto& fVI : result.fixedVert) {
            coefMtr.coeffRef(fVI * 2, fVI * 2) = 1.0;
            coefMtr.coeffRef(fVI * 2 + 1, fVI * 2 + 1) = 1.0;
        }
        coefMtr.makeCompressed();
        
        linSysSolver_xUpdate.compute(coefMtr);
        assert(linSysSolver_xUpdate.info() == Eigen::Success);
        
        D_mult_x.resize(result.F.rows(), 4);
    }
    
    bool ADMMTimeStepper::fullyImplicit(void)
    {
        // initialize x with xHat, M_mult_xHat, u with 0, and D_mult_x and z with Dx
        for(int vI = 0; vI < result.V.rows(); vI++) {
            if(result.fixedVert.find(vI) == result.fixedVert.end()) {
                result.V.row(vI) += (dt * velocity.segment(vI * 2, 2) + dtSq * gravity).transpose();
            }
            M_mult_xHat.segment(vI * 2, 2) = result.massMatrix.coeffRef(vI, vI) *
                result.V.row(vI).transpose();
        }
        u.setZero();
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI) {
            compute_Di_mult_xi(triI);
            z.row(triI) = D_mult_x.row(triI);
        });
        
        // ADMM iterations
        int ADMMIterAmt = 10000;
        for(int ADMMIterI = 0; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            zuUpdate();
            xUpdate();
            
            computeGradient(result, scaffold, gradient);
            double sqn_g = gradient.squaredNorm();
            std::cout << "Step" << globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            if(sqn_g < targetGRes * 1000.0) { //!!!
                break;
            }
        }
        
        return false;
    }
    
    void ADMMTimeStepper::zuUpdate(void)
    {
        int localMaxIter = __INT_MAX__;
        double localTol = targetGRes / result.F.rows();
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI) {
//        for(int triI = 0; triI < result.F.rows(); triI++) {
            Eigen::VectorXd zi = z.row(triI).transpose();
            Eigen::VectorXd g;
            for(int j = 0; j < localMaxIter; j++) {
                computeGradient_zUpdate(triI, zi.transpose(), g);
//                std::cout << "  " << triI << "-" << j << " ||g_local||^2 = "
//                    << g.squaredNorm() << std::endl;
                if(g.squaredNorm() < localTol) {
                    break;
                }
                
                Eigen::MatrixXd P;
                computeHessianProxy_zUpdate(triI, zi.transpose(), P);
                //NOTE: for z being the deformation gradient,
                // no need to consider position constraints
                
                // solve for search direction
                Eigen::VectorXd p = P.ldlt().solve(-g);
                
                // line search init
                double alpha = 1.0;
                energyTerms[0]->initStepSize(zi, p, alpha);
                alpha *= 0.99;
                
                // Armijo's rule:
                const double m = p.dot(g);
                const double c1m = 1.0e-4 * m;
                const Eigen::VectorXd zi0 = zi;
                double E0;
                computeEnergyVal_zUpdate(triI, zi0.transpose(), E0);
                zi = zi0 + alpha * p;
                double E;
                computeEnergyVal_zUpdate(triI, zi.transpose(), E);
                while(E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    zi = zi0 + alpha * p;
                    computeEnergyVal_zUpdate(triI, zi.transpose(), E);
                }
            }
            
            z.row(triI) = zi.transpose();
            
            u.row(triI) += D_mult_x.row(triI) - z.row(triI);
        });
//        }
    }
    void ADMMTimeStepper::xUpdate(void)
    {
        // compute rhs
        rhs_xUpdate = M_mult_xHat;
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::VectorXd& rhs_right_triI = D_array[triI].transpose() * ((z.row(triI) - u.row(triI)) * weights2[triI]).transpose();
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            rhs_xUpdate.segment(triVInd[0] * 2, 2) += rhs_right_triI.segment(0, 2);
            rhs_xUpdate.segment(triVInd[1] * 2, 2) += rhs_right_triI.segment(2, 2);
            rhs_xUpdate.segment(triVInd[2] * 2, 2) += rhs_right_triI.segment(4, 2);
        }
        //TODO: arrange by row and parallelize
        for(const auto& entryI : offset_fixVerts) {
            int vI = entryI.first.second / 2;
            int dimI = entryI.first.second % 2;
            rhs_xUpdate[entryI.first.first] -= entryI.second * result.V(vI, dimI);
        }
        for(const auto& fVI : result.fixedVert) {
            rhs_xUpdate.segment(fVI * 2, 2) = result.V.row(fVI).transpose();
        }
        
        // solve linear system with pre-factorized info and update x
        Eigen::VectorXd x = linSysSolver_xUpdate.solve(rhs_xUpdate);
        assert(linSysSolver_xUpdate.info() == Eigen::Success);
//        for(int vI = 0; vI < result.V.rows(); vI++) {
        tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI) {
            result.V.row(vI) = x.segment(vI * 2, 2).transpose();
//        }
        });
        
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI) {
            compute_Di_mult_xi(triI);
        });
    }
    
    void ADMMTimeStepper::compute_Di_mult_xi(int triI)
    {
        assert(triI < result.F.rows());
        
        const Eigen::RowVector3i& triVInd = result.F.row(triI);
        
        Eigen::VectorXd Xt;
        Xt.resize(6);
        Xt <<
            result.V.row(triVInd[0]).transpose(),
            result.V.row(triVInd[1]).transpose(),
            result.V.row(triVInd[2]).transpose();
        
        D_mult_x.row(triI) = (D_array[triI] * Xt).transpose();
    }
    
    void ADMMTimeStepper::computeEnergyVal_zUpdate(int triI,
                                                   const Eigen::RowVectorXd& zi,
                                                   double& Ei) const
    {
        energyTerms[0]->computeEnergyValBySVD_F(result, triI, zi, Ei);
        Ei *= dtSq;
        Ei += (D_mult_x.row(triI) - zi + u.row(triI)).squaredNorm() * weights2[triI] / 2.0;
    }
    void ADMMTimeStepper::computeGradient_zUpdate(int triI,
                                                  const Eigen::RowVectorXd& zi,
                                                  Eigen::VectorXd& g) const
    {
        energyTerms[0]->computeGradientBySVD_F(result, triI, zi, g);
        g *= dtSq;
        g -= ((D_mult_x.row(triI) - zi + u.row(triI)) * weights2[triI]).transpose() ;
    }
    void ADMMTimeStepper::computeHessianProxy_zUpdate(int triI,
                                                      const Eigen::RowVectorXd& zi,
                                                      Eigen::MatrixXd& P) const
    {
        energyTerms[0]->computeHessianBySVD_F(result, triI, zi, P);
        P *= dtSq;
        P.diagonal().array() += weights2[triI];
    }
    
}
