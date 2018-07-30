//
//  ADMMTimeStepper.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/28/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "DADMMTimeStepper.hpp"

#include "IglUtils.hpp"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

namespace FracCuts {
    
    DADMMTimeStepper::DADMMTimeStepper(const TriangleSoup& p_data0,
                                       const std::vector<Energy*>& p_energyTerms,
                                       const std::vector<double>& p_energyParams,
                                       int p_propagateFracture,
                                       bool p_mute,
                                       bool p_scaffolding,
                                       const Eigen::MatrixXd& UV_bnds,
                                       const Eigen::MatrixXi& E,
                                       const Eigen::VectorXi& bnd,
                                       AnimScriptType animScriptType) :
    Optimizer(p_data0, p_energyTerms, p_energyParams,
    p_propagateFracture, p_mute, p_scaffolding,
              UV_bnds, E, bnd, animScriptType)
    {
        z.resize(result.F.rows(), 6);
        u.resize(result.F.rows(), 6);
        dz.resize(result.F.rows(), 6);
        rhs_xUpdate.resize(result.V.rows() * 2);
        M_mult_xHat.resize(result.V.rows() * 2);
        coefMtr_diag.resize(result.V.rows() * 2);
        D_mult_x.resize(result.F.rows(), 6);
        
        // initialize weights
        double bulkModulus;
        energyTerms[0]->getBulkModulus(bulkModulus);
        double wi = dt * std::sqrt(bulkModulus);
        weights.resize(result.F.rows());
        for(int triI = 0; triI < result.F.rows(); triI++) {
            weights[triI] = wi * std::sqrt(result.triArea[triI]) * 20; //TODO: figure out parameters
        }
        weights2 = weights.cwiseProduct(weights);
    }
    
    void DADMMTimeStepper::precompute(void)
    {
        // construct and prefactorize the linear system
        std::vector<Eigen::Triplet<double>> triplet(result.F.rows() * 6);
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            for(int localVI = 0; localVI < 3; localVI++) {
                triplet.emplace_back(triI * 6 + localVI * 2, triVInd[localVI] * 2, weights[triI]);
                triplet.emplace_back(triI * 6 + localVI * 2 + 1, triVInd[localVI] * 2 + 1, weights[triI]);
            }
        }
        Eigen::SparseMatrix<double> WD;
        WD.resize(result.F.rows() * 6, result.V.rows() * 2);
        WD.setFromTriplets(triplet.begin(), triplet.end());
        
        Eigen::SparseMatrix<double> coefMtr = WD.transpose() * WD;
        for(int vI = 0; vI < result.V.rows(); vI++) {
            double massI = result.massMatrix.coeffRef(vI, vI);
            coefMtr.coeffRef(vI * 2, vI * 2) += massI;
            coefMtr.coeffRef(vI * 2 + 1, vI * 2 + 1) += massI;
        }
        for(const auto& fVI : result.fixedVert) {
            coefMtr.coeffRef(fVI * 2, fVI * 2) = 1.0;
            coefMtr.coeffRef(fVI * 2 + 1, fVI * 2 + 1) = 1.0;
        }
        coefMtr.makeCompressed();
        
        for (int k = 0; k < coefMtr.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(coefMtr, k); it; ++it)
            {
                if(it.row() == it.col()) {
                    coefMtr_diag[it.row()] = it.value();
                }
            }
        }
    }
    
    void DADMMTimeStepper::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field = Eigen::VectorXd::LinSpaced(result.F.rows(), 0, result.F.rows() - 1);
    }
    
    bool DADMMTimeStepper::fullyImplicit(void)
    {
        // initialize x with xHat
        initX(2);
        
        // initialize M_mult_xHat
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < result.V.rows(); vI++)
#endif
        {
            if(result.fixedVert.find(vI) == result.fixedVert.end()) {
                M_mult_xHat.segment(vI * 2, 2) = (result.massMatrix.coeffRef(vI, vI) *
                                                  (resultV_n.row(vI).transpose() + dt * velocity.segment(vI * 2, 2) + dtSq * gravity));
            }
            else {
                M_mult_xHat.segment(vI * 2, 2) = (result.massMatrix.coeffRef(vI, vI) *
                                                  resultV_n.row(vI).transpose());
            }
        }
#ifdef USE_TBB
        );
#endif
        
        u.setZero();
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < result.F.rows(); triI++)
#endif
        {
            compute_Di_mult_xi(triI);
            z.row(triI) = D_mult_x.row(triI);
        }
#ifdef USE_TBB
        );
#endif
        
        // ADMM iterations
        int ADMMIterAmt = 1000, ADMMIterI = 0;
        for(; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            file_iterStats << globalIterNum << " ";
            
            zuUpdate();
            checkRes();
            xUpdate();
            
            computeGradient(result, scaffold, gradient);
            double sqn_g = gradient.squaredNorm();
            std::cout << "Step" << globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            file_iterStats << sqn_g << std::endl;
            if(sqn_g < targetGRes) {
                break;
            }
        }
        innerIterAmt += ADMMIterI;
        
        return (ADMMIterI == ADMMIterAmt);
    }
    
    void DADMMTimeStepper::zuUpdate(void)
    {
        int localMaxIter = __INT_MAX__;
        double localTol = targetGRes / result.F.rows() / 1000.0;
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < result.F.rows(); triI++)
#endif
        {
            Eigen::VectorXd zi = z.row(triI).transpose();
            Eigen::VectorXd g;
            for(int j = 0; j < localMaxIter; j++) {
                computeGradient_zUpdate(triI, zi.transpose(), g);
                if(g.squaredNorm() < localTol) {
                    break;
                }
                
                Eigen::MatrixXd P;
                computeHessianProxy_zUpdate(triI, zi.transpose(), P);
                
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
            
            dz.row(triI) = zi.transpose() - z.row(triI);
            z.row(triI) = zi.transpose();
            
            u.row(triI) += D_mult_x.row(triI) - z.row(triI);
        }
#ifdef USE_TBB
        );
#endif
    }
    void DADMMTimeStepper::checkRes(void)
    {
        Eigen::VectorXd s = Eigen::VectorXd::Zero(result.V.rows() * 2);
        double sqn_r = 0.0;
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::VectorXd& s_triI = (dz.row(triI) * weights2[triI]).transpose();
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            s.segment(triVInd[0] * 2, 2) += s_triI.segment(0, 2);
            s.segment(triVInd[1] * 2, 2) += s_triI.segment(2, 2);
            s.segment(triVInd[2] * 2, 2) += s_triI.segment(4, 2);
            
            sqn_r += (D_mult_x.row(triI) - z.row(triI)).squaredNorm() * weights2[triI];
        }
        double sqn_s = s.squaredNorm();
        std::cout << "||s||^2 = " << sqn_s << ", ||r||^2 = " << sqn_r << ", ";
        file_iterStats << sqn_s << " " << sqn_r << " ";
    }
    void DADMMTimeStepper::xUpdate(void)
    {
        // compute rhs
        rhs_xUpdate = M_mult_xHat;
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::VectorXd& rhs_right_triI = ((z.row(triI) - u.row(triI)) * weights2[triI]).transpose();
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            rhs_xUpdate.segment(triVInd[0] * 2, 2) += rhs_right_triI.segment(0, 2);
            rhs_xUpdate.segment(triVInd[1] * 2, 2) += rhs_right_triI.segment(2, 2);
            rhs_xUpdate.segment(triVInd[2] * 2, 2) += rhs_right_triI.segment(4, 2);
        }
        for(const auto& fVI : result.fixedVert) {
            rhs_xUpdate.segment(fVI * 2, 2) = result.V.row(fVI).transpose();
        }
        
        // solve linear system with pre-factorized info and update x
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < result.V.rows(); vI++)
#endif
        {
            result.V(vI, 0) = rhs_xUpdate[vI * 2] / coefMtr_diag[vI * 2];
            result.V(vI, 1) = rhs_xUpdate[vI * 2 + 1] / coefMtr_diag[vI * 2 + 1];
        }
#ifdef USE_TBB
        );
#endif

#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < result.F.rows(); triI++)
#endif
        {
            compute_Di_mult_xi(triI);
        }
#ifdef USE_TBB
        );
#endif
    }
    
    void DADMMTimeStepper::compute_Di_mult_xi(int triI)
    {
        assert(triI < result.F.rows());
        
        const Eigen::RowVector3i& triVInd = result.F.row(triI);
        
        D_mult_x.row(triI) <<
            result.V.row(triVInd[0]),
            result.V.row(triVInd[1]),
            result.V.row(triVInd[2]);
    }
    
    void DADMMTimeStepper::computeEnergyVal_zUpdate(int triI,
                                                   const Eigen::RowVectorXd& zi,
                                                   double& Ei) const
    {
        energyTerms[0]->computeEnergyValBySVD(result, triI, zi, Ei);
        Ei *= dtSq;
        Ei += (D_mult_x.row(triI) - zi + u.row(triI)).squaredNorm() * weights2[triI] / 2.0;
    }
    void DADMMTimeStepper::computeGradient_zUpdate(int triI,
                                                  const Eigen::RowVectorXd& zi,
                                                  Eigen::VectorXd& g) const
    {
        energyTerms[0]->computeGradientBySVD(result, triI, zi, g);
        g *= dtSq;
        g -= ((D_mult_x.row(triI) - zi + u.row(triI)) * weights2[triI]).transpose();
        
        const Eigen::RowVector3i& triVInd = result.F.row(triI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            if(result.fixedVert.find(vI) != result.fixedVert.end()) {
                g.segment(localVI * 2, 2).setZero();
            }
        }
    }
    void DADMMTimeStepper::computeHessianProxy_zUpdate(int triI,
                                                      const Eigen::RowVectorXd& zi,
                                                      Eigen::MatrixXd& P) const
    {
        energyTerms[0]->computeHessianBySVD(result, triI, zi, P);
        P *= dtSq;
        P.diagonal().array() += weights2[triI];
        
        const Eigen::RowVector3i& triVInd = result.F.row(triI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            if(result.fixedVert.find(vI) != result.fixedVert.end()) {
                P.row(localVI * 2).setZero();
                P.row(localVI * 2 + 1).setZero();
                P.col(localVI * 2).setZero();
                P.col(localVI * 2 + 1).setZero();
                P.block(localVI * 2, localVI * 2, 2, 2).setIdentity();
            }
        }
    }
    
}
