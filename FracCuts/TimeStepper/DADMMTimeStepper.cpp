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
    
    template<int dim>
    DADMMTimeStepper<dim>::DADMMTimeStepper(const TriangleSoup<dim>& p_data0,
                                            const std::vector<Energy<dim>*>& p_energyTerms,
                                            const std::vector<double>& p_energyParams,
                                            int p_propagateFracture,
                                            bool p_mute,
                                            bool p_scaffolding,
                                            const Eigen::MatrixXd& UV_bnds,
                                            const Eigen::MatrixXi& E,
                                            const Eigen::VectorXi& bnd,
                                            const Config& animConfig) :
    Base(p_data0, p_energyTerms, p_energyParams,
         p_propagateFracture, p_mute, p_scaffolding,
         UV_bnds, E, bnd, animConfig)
    {
        z.resize(Base::result.F.rows(), 6);
        u.resize(Base::result.F.rows(), 6);
        dz.resize(Base::result.F.rows(), 6);
        rhs_xUpdate.resize(Base::result.V.rows() * 2);
        M_mult_xHat.resize(Base::result.V.rows() * 2);
        coefMtr_diag.resize(Base::result.V.rows() * 2);
        D_mult_x.resize(Base::result.F.rows(), 6);
        
        // initialize weights
        double bulkModulus;
        Base::energyTerms[0]->getBulkModulus(bulkModulus);
        double wi = Base::dt * std::sqrt(bulkModulus);
        weights.resize(Base::result.F.rows());
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            weights[triI] = wi * std::sqrt(Base::result.triArea[triI]) * 20; //TODO: figure out parameters
        }
        weights2 = weights.cwiseProduct(weights);
    }
    
    template<int dim>
    void DADMMTimeStepper<dim>::precompute(void)
    {
        // construct and prefactorize the linear system
        std::vector<Eigen::Triplet<double>> triplet(Base::result.F.rows() * 6);
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
            for(int localVI = 0; localVI < 3; localVI++) {
                triplet.emplace_back(triI * 6 + localVI * 2, triVInd[localVI] * 2, weights[triI]);
                triplet.emplace_back(triI * 6 + localVI * 2 + 1, triVInd[localVI] * 2 + 1, weights[triI]);
            }
        }
        Eigen::SparseMatrix<double> WD;
        WD.resize(Base::result.F.rows() * 6, Base::result.V.rows() * 2);
        WD.setFromTriplets(triplet.begin(), triplet.end());
        
        Eigen::SparseMatrix<double> coefMtr = WD.transpose() * WD;
        for(int vI = 0; vI < Base::result.V.rows(); vI++) {
            double massI = Base::result.massMatrix.coeffRef(vI, vI);
            coefMtr.coeffRef(vI * 2, vI * 2) += massI;
            coefMtr.coeffRef(vI * 2 + 1, vI * 2 + 1) += massI;
        }
        for(const auto& fVI : Base::result.fixedVert) {
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
    
    template<int dim>
    void DADMMTimeStepper<dim>::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field = Eigen::VectorXd::LinSpaced(Base::result.F.rows(), 0, Base::result.F.rows() - 1);
    }
    
    template<int dim>
    bool DADMMTimeStepper<dim>::fullyImplicit(void)
    {
        // initialize x with xHat
        Base::initX(2);
        
        // initialize M_mult_xHat
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < Base::result.V.rows(); vI++)
#endif
        {
            if(Base::result.fixedVert.find(vI) == Base::result.fixedVert.end()) {
                M_mult_xHat.segment(vI * 2, 2) = (Base::result.massMatrix.coeffRef(vI, vI) *
                                                  (Base::resultV_n.row(vI).transpose() +
                                                   Base::dt * Base::velocity.segment(vI * 2, 2) +
                                                   Base::dtSq * Base::gravity));
            }
            else {
                M_mult_xHat.segment(vI * 2, 2) = (Base::result.massMatrix.coeffRef(vI, vI) *
                                                  Base::resultV_n.row(vI).transpose());
            }
        }
#ifdef USE_TBB
        );
#endif
        
        u.setZero();
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
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
            Base::file_iterStats << Base::globalIterNum << " ";
            
            zuUpdate();
            checkRes();
            xUpdate();
            
            Base::computeGradient(Base::result, Base::scaffold, true, Base::gradient);
            double sqn_g = Base::gradient.squaredNorm();
            std::cout << "Step" << Base::globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            Base::file_iterStats << sqn_g << std::endl;
            if(sqn_g < Base::targetGRes) {
                break;
            }
        }
        Base::innerIterAmt += ADMMIterI + 1;
        
        return (ADMMIterI == ADMMIterAmt);
    }
    
    template<int dim>
    void DADMMTimeStepper<dim>::zuUpdate(void)
    {
        int localMaxIter = __INT_MAX__;
        double localTol = Base::targetGRes / Base::result.F.rows() / 1000.0;
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
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
                Base::energyTerms[0]->initStepSize(zi, p, alpha);
                
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
    template<int dim>
    void DADMMTimeStepper<dim>::checkRes(void)
    {
        Eigen::VectorXd s = Eigen::VectorXd::Zero(Base::result.V.rows() * 2);
        double sqn_r = 0.0;
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::VectorXd& s_triI = (dz.row(triI) * weights2[triI]).transpose();
            const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
            s.segment(triVInd[0] * 2, 2) += s_triI.segment(0, 2);
            s.segment(triVInd[1] * 2, 2) += s_triI.segment(2, 2);
            s.segment(triVInd[2] * 2, 2) += s_triI.segment(4, 2);
            
            sqn_r += (D_mult_x.row(triI) - z.row(triI)).squaredNorm() * weights2[triI];
        }
        double sqn_s = s.squaredNorm();
        std::cout << "||s||^2 = " << sqn_s << ", ||r||^2 = " << sqn_r << ", ";
        Base::file_iterStats << sqn_s << " " << sqn_r << " ";
    }
    template<int dim>
    void DADMMTimeStepper<dim>::xUpdate(void)
    {
        // compute rhs
        rhs_xUpdate = M_mult_xHat;
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::VectorXd& rhs_right_triI = ((z.row(triI) - u.row(triI)) * weights2[triI]).transpose();
            const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
            rhs_xUpdate.segment(triVInd[0] * 2, 2) += rhs_right_triI.segment(0, 2);
            rhs_xUpdate.segment(triVInd[1] * 2, 2) += rhs_right_triI.segment(2, 2);
            rhs_xUpdate.segment(triVInd[2] * 2, 2) += rhs_right_triI.segment(4, 2);
        }
        for(const auto& fVI : Base::result.fixedVert) {
            rhs_xUpdate.segment(fVI * 2, 2) = Base::result.V.row(fVI).transpose();
        }
        
        // solve linear system with pre-factorized info and update x
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < Base::result.V.rows(); vI++)
#endif
        {
            Base::result.V(vI, 0) = rhs_xUpdate[vI * 2] / coefMtr_diag[vI * 2];
            Base::result.V(vI, 1) = rhs_xUpdate[vI * 2 + 1] / coefMtr_diag[vI * 2 + 1];
        }
#ifdef USE_TBB
        );
#endif

#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
#endif
        {
            compute_Di_mult_xi(triI);
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    void DADMMTimeStepper<dim>::compute_Di_mult_xi(int triI)
    {
        assert(triI < Base::result.F.rows());
        
        const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
        
        D_mult_x.row(triI) <<
            Base::result.V.row(triVInd[0]),
            Base::result.V.row(triVInd[1]),
            Base::result.V.row(triVInd[2]);
    }
    
    template<int dim>
    void DADMMTimeStepper<dim>::computeEnergyVal_zUpdate(int triI,
                                                   const Eigen::RowVectorXd& zi,
                                                   double& Ei) const
    {
        Base::energyTerms[0]->computeEnergyValBySVD(Base::result, triI, zi, Ei);
        Ei *= Base::dtSq;
        Ei += (D_mult_x.row(triI) - zi + u.row(triI)).squaredNorm() * weights2[triI] / 2.0;
    }
    template<int dim>
    void DADMMTimeStepper<dim>::computeGradient_zUpdate(int triI,
                                                  const Eigen::RowVectorXd& zi,
                                                  Eigen::VectorXd& g) const
    {
        Base::energyTerms[0]->computeGradientBySVD(Base::result, triI, zi, g);
        g *= Base::dtSq;
        g -= ((D_mult_x.row(triI) - zi + u.row(triI)) * weights2[triI]).transpose();
        
        const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            if(Base::result.fixedVert.find(vI) != Base::result.fixedVert.end()) {
                g.segment(localVI * 2, 2).setZero();
            }
        }
    }
    template<int dim>
    void DADMMTimeStepper<dim>::computeHessianProxy_zUpdate(int triI,
                                                      const Eigen::RowVectorXd& zi,
                                                      Eigen::MatrixXd& P) const
    {
        Base::energyTerms[0]->computeHessianBySVD(Base::result, triI, zi, P);
        P *= Base::dtSq;
        P.diagonal().array() += weights2[triI];
        
        const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            if(Base::result.fixedVert.find(vI) != Base::result.fixedVert.end()) {
                P.row(localVI * 2).setZero();
                P.row(localVI * 2 + 1).setZero();
                P.col(localVI * 2).setZero();
                P.col(localVI * 2 + 1).setZero();
                P.block(localVI * 2, localVI * 2, 2, 2).setIdentity();
            }
        }
    }
    
    template class DADMMTimeStepper<DIM>;
    
}
