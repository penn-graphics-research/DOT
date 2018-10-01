//
//  ADMMTimeStepper.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/28/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "ADMMTimeStepper.hpp"

#include "IglUtils.hpp"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

namespace FracCuts {
    
    template<int dim>
    ADMMTimeStepper<dim>::ADMMTimeStepper(const TriangleSoup<dim>& p_data0,
                                     const std::vector<Energy<dim>*>& p_energyTerms,
                                     const std::vector<double>& p_energyParams,
                                     int p_propagateFracture,
                                     bool p_mute,
                                     bool p_scaffolding,
                                     const Eigen::MatrixXd& UV_bnds,
                                     const Eigen::MatrixXi& E,
                                     const Eigen::VectorXi& bnd,
                                     const Config& animConfig) :
        Optimizer<dim>(p_data0, p_energyTerms, p_energyParams,
                       p_propagateFracture, p_mute, p_scaffolding,
                       UV_bnds, E, bnd, animConfig)
    {
        z.resize(Base::result.F.rows(), 4);
        u.resize(Base::result.F.rows(), 4);
        dz.resize(Base::result.F.rows(), 4);
        rhs_xUpdate.resize(Base::result.V.rows() * 2);
        M_mult_xHat.resize(Base::result.V.rows() * 2);
        D_mult_x.resize(Base::result.F.rows(), 4);
        
        // initialize weights
        double bulkModulus;
        Base::energyTerms[0]->getBulkModulus(bulkModulus);
//        std::cout << bulkModulus << std::endl;
//        Eigen::MatrixXd d2E_div_dF2_rest;
//        energyTerms[0]->compute_d2E_div_dF2_rest(d2E_div_dF2_rest);
//        bulkModulus = d2E_div_dF2_rest.norm();
//        std::cout << bulkModulus << std::endl;
        double wi = Base::dt * std::sqrt(bulkModulus);
        weights.resize(Base::result.F.rows());
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            weights[triI] = wi * std::sqrt(Base::result.triArea[triI]) * 2;
        }
        weights2 = weights.cwiseProduct(weights);
        
        // initialize D_array
        D_array.resize(Base::result.F.rows());
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
            
            Eigen::Vector3d x0_3D[3] = {
                Base::result.V_rest.row(triVInd[0]),
                Base::result.V_rest.row(triVInd[1]),
                Base::result.V_rest.row(triVInd[2])
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
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::precompute(void)
    {
        // construct and prefactorize the linear system
        std::vector<Eigen::Triplet<double>> triplet(Base::result.F.rows() * 4 * 3);
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
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
        WD.resize(Base::result.F.rows() * 4, Base::result.V.rows() * 2);
        WD.setFromTriplets(triplet.begin(), triplet.end());
        
        Eigen::SparseMatrix<double> coefMtr = WD.transpose() * WD;
        for(int vI = 0; vI < Base::result.V.rows(); vI++) {
            double massI = Base::result.massMatrix.coeffRef(vI, vI);
            coefMtr.coeffRef(vI * 2, vI * 2) += massI;
            coefMtr.coeffRef(vI * 2 + 1, vI * 2 + 1) += massI;
        }
        offset_fixVerts.resize(0);
        offset_fixVerts.resize(Base::result.V.rows() * 2);
        for (int k = 0; k < coefMtr.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(coefMtr, k); it; ++it)
            {
                bool fixed_rowV = (Base::result.fixedVert.find(it.row() / 2) !=
                                   Base::result.fixedVert.end());
                bool fixed_colV = (Base::result.fixedVert.find(it.col() / 2) !=
                                   Base::result.fixedVert.end());
                if(fixed_rowV || fixed_colV) {
                    if(!fixed_rowV) {
                        offset_fixVerts[it.row()][it.col()] = it.value();
                    }
                    it.valueRef() = 0.0;
                }
            }
        }
        for(const auto& fVI : Base::result.fixedVert) {
            coefMtr.coeffRef(fVI * 2, fVI * 2) = 1.0;
            coefMtr.coeffRef(fVI * 2 + 1, fVI * 2 + 1) = 1.0;
        }
        coefMtr.makeCompressed();
        
        Base::linSysSolver->set_type(1, 2);
        Base::linSysSolver->set_pattern(coefMtr);
        Base::linSysSolver->analyze_pattern();
        Base::linSysSolver->factorize(); //TODO: error check
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field = Eigen::VectorXd::LinSpaced(Base::result.F.rows(), 0, Base::result.F.rows() - 1);
    }
    
    template<int dim>
    bool ADMMTimeStepper<dim>::fullyImplicit(void)
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
        
        // initialize D_mult_x and z with Dx
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
        int ADMMIterAmt = 300, ADMMIterI = 0;
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
    void ADMMTimeStepper<dim>::zuUpdate(void)
    {
        int localMaxIter = __INT_MAX__;
        double localTol = Base::targetGRes / Base::result.F.rows() / 1000.0; //TODO: needs to be more adaptive to global tol
        //TODO: needs to be in F space!!!
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
#endif
        {
            Eigen::RowVectorXd zi = z.row(triI);
            Eigen::Matrix<double, dim * dim, 1> g;
            for(int j = 0; j < localMaxIter; j++) {
                computeGradient_zUpdate(triI, zi, g);
//                std::cout << "  " << triI << "-" << j << " ||g_local||^2 = "
//                    << g.squaredNorm() << std::endl;
                if(g.squaredNorm() < localTol) {
                    break;
                }
                
                Eigen::Matrix<double, dim * dim, dim * dim> P;
                computeHessianProxy_zUpdate(triI, zi, P);
                //NOTE: for z being the deformation gradient,
                // no need to consider position constraints
                
                // solve for search direction
                Eigen::Matrix<double, dim * dim, 1> p = P.ldlt().solve(-g);
                
                // line search init
                double alpha = 1.0;
                Base::energyTerms[0]->initStepSize(zi.transpose(), p, alpha); //TODO: different in F space
                
                // Armijo's rule:
                const double m = p.dot(g);
                const double c1m = 1.0e-4 * m;
                const Eigen::RowVectorXd zi0 = zi;
                double E0;
                computeEnergyVal_zUpdate(triI, zi0, E0);
                zi = zi0 + alpha * p.transpose();
                double E;
                computeEnergyVal_zUpdate(triI, zi, E);
                while(E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    zi = zi0 + alpha * p.transpose();
                    computeEnergyVal_zUpdate(triI, zi, E);
                }
            }
            
            dz.row(triI) = zi - z.row(triI);
            z.row(triI) = zi;
            
            u.row(triI) += D_mult_x.row(triI) - z.row(triI);
        }
#ifdef USE_TBB
        );
#endif
    }
    template<int dim>
    void ADMMTimeStepper<dim>::checkRes(void)
    {
        Eigen::VectorXd s = Eigen::VectorXd::Zero(Base::result.V.rows() * 2);
        double sqn_r = 0.0;
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::VectorXd& s_triI = D_array[triI].transpose() * (dz.row(triI) * weights2[triI]).transpose();
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
    void ADMMTimeStepper<dim>::xUpdate(void)
    {
        // compute rhs
        rhs_xUpdate = M_mult_xHat;
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::VectorXd& rhs_right_triI = D_array[triI].transpose() * ((z.row(triI) - u.row(triI)) * weights2[triI]).transpose();
            const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
            rhs_xUpdate.segment(triVInd[0] * 2, 2) += rhs_right_triI.segment(0, 2);
            rhs_xUpdate.segment(triVInd[1] * 2, 2) += rhs_right_triI.segment(2, 2);
            rhs_xUpdate.segment(triVInd[2] * 2, 2) += rhs_right_triI.segment(4, 2);
        }
#ifdef USE_TBB
        tbb::parallel_for(0, (int)offset_fixVerts.size(), 1, [&](int rowI)
#else
        for(int rowI = 0; rowI < offset_fixVerts.size(); rowI++)
#endif
        {
            for(const auto& entryI : offset_fixVerts[rowI]) {
                int vI = entryI.first / 2;
                int dimI = entryI.first % 2;
                rhs_xUpdate[rowI] -= entryI.second * Base::result.V(vI, dimI);
            }
        }
#ifdef USE_TBB
        );
#endif
        for(const auto& fVI : Base::result.fixedVert) {
            rhs_xUpdate.segment(fVI * 2, 2) = Base::result.V.row(fVI).transpose();
        }
        
        // solve linear system with pre-factorized info and update x
        Base::linSysSolver->solve(rhs_xUpdate, x_solved); //TODO: error check
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < Base::result.V.rows(); vI++)
#endif
        {
            Base::result.V.row(vI) = x_solved.segment(vI * 2, 2).transpose();
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
    void ADMMTimeStepper<dim>::compute_Di_mult_xi(int triI)
    {
        assert(triI < Base::result.F.rows());
        
        const Eigen::RowVector3i& triVInd = Base::result.F.row(triI);
        
        Eigen::VectorXd Xt;
        Xt.resize(6);
        Xt <<
            Base::result.V.row(triVInd[0]).transpose(),
            Base::result.V.row(triVInd[1]).transpose(),
            Base::result.V.row(triVInd[2]).transpose();
        
        D_mult_x.row(triI) = (D_array[triI] * Xt).transpose();
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::computeEnergyVal_zUpdate(int triI,
                                                   const Eigen::RowVectorXd& zi,
                                                   double& Ei) const
    {
        Base::energyTerms[0]->computeEnergyValBySVD_F(Base::result, triI, zi, Ei);
        Ei *= Base::dtSq;
        Ei += (D_mult_x.row(triI) - zi + u.row(triI)).squaredNorm() * weights2[triI] / 2.0;
    }
    template<int dim>
    void ADMMTimeStepper<dim>::computeGradient_zUpdate(int triI,
                                                       const Eigen::RowVectorXd& zi,
                                                       Eigen::Matrix<double, dim * dim, 1>& g) const
    {
        Base::energyTerms[0]->computeGradientBySVD_F(Base::result, triI, zi, g);
        g *= Base::dtSq;
        g -= ((D_mult_x.row(triI) - zi + u.row(triI)) * weights2[triI]).transpose() ;
    }
    template<int dim>
    void ADMMTimeStepper<dim>::computeHessianProxy_zUpdate(int triI,
                                                           const Eigen::RowVectorXd& zi,
                                                           Eigen::Matrix<double, dim * dim, dim * dim>& P) const
    {
        Base::energyTerms[0]->computeHessianBySVD_F(Base::result, triI, zi, P);
        P *= Base::dtSq;
        P.diagonal().array() += weights2[triI];
    }
    
    template class ADMMTimeStepper<DIM>;
    
}
