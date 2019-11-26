//
//  ADMMTimeStepper.cpp
//  DOT
//
//  Created by Minchen Li on 6/28/18.
//

#include "ADMMTimeStepper.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#include "IglUtils.hpp"

#include "Timer.hpp"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

extern std::ofstream logFile;
extern Timer timer_temp3;

namespace DOT {
    
    template<int dim>
    ADMMTimeStepper<dim>::ADMMTimeStepper(const Mesh<dim>& p_data0,
                                          const std::vector<Energy<dim>*>& p_energyTerms,
                                          const std::vector<double>& p_energyParams,
                                          bool p_mute,
                                          const Config& animConfig) :
        Optimizer<dim>(p_data0, p_energyTerms, p_energyParams, p_mute, animConfig)
    {
        timer_temp3.start(0);
        
        z.resize(Base::result.F.rows(), dim * dim);
        u.resize(Base::result.F.rows(), dim * dim);
        dz.resize(Base::result.F.rows(), dim * dim);
        du.resize(Base::result.F.rows(), dim * dim);
        rhs_xUpdate.resize(Base::result.V.rows() * dim);
        M_mult_xHat.resize(Base::result.V.rows() * dim);
        D_mult_x.resize(Base::result.F.rows(), dim * dim);
        
        // initialize weights
        GW.resize(Base::result.F.rows());
        
        // initialize D_array
        //TODO: simpify!!
        D_array.resize(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
#endif
        {
            const Eigen::Matrix<double, dim, dim>& A = Base::result.restTriInv[triI];
            if(dim == 2) {
                const double mA11mA21 = -A(0, 0) - A(1, 0);
                const double mA12mA22 = -A(0, 1) - A(1, 1);
                D_array[triI].resize(4, 6);
                D_array[triI] <<
                    mA11mA21, 0.0, A(0, 0), 0.0, A(1, 0), 0.0,
                    mA12mA22, 0.0, A(0, 1), 0.0, A(1, 1), 0.0,
                    0.0, mA11mA21, 0.0, A(0, 0), 0.0, A(1, 0),
                    0.0, mA12mA22, 0.0, A(0, 1), 0.0, A(1, 1);
            }
            else {
                const double mA11mA21mA31 = -A(0, 0) - A(1, 0) - A(2, 0);
                const double mA12mA22mA32 = -A(0, 1) - A(1, 1) - A(2, 1);
                const double mA13mA23mA33 = -A(0, 2) - A(1, 2) - A(2, 2);
                D_array[triI].resize(9, 12);
                D_array[triI] <<
                    mA11mA21mA31, 0.0, 0.0, A(0, 0), 0.0, 0.0, A(1, 0), 0.0, 0.0, A(2, 0), 0.0, 0.0,
                    mA12mA22mA32, 0.0, 0.0, A(0, 1), 0.0, 0.0, A(1, 1), 0.0, 0.0, A(2, 1), 0.0, 0.0,
                    mA13mA23mA33, 0.0, 0.0, A(0, 2), 0.0, 0.0, A(1, 2), 0.0, 0.0, A(2, 2), 0.0, 0.0,
                    0.0, mA11mA21mA31, 0.0, 0.0, A(0, 0), 0.0, 0.0, A(1, 0), 0.0, 0.0, A(2, 0), 0.0,
                    0.0, mA12mA22mA32, 0.0, 0.0, A(0, 1), 0.0, 0.0, A(1, 1), 0.0, 0.0, A(2, 1), 0.0,
                    0.0, mA13mA23mA33, 0.0, 0.0, A(0, 2), 0.0, 0.0, A(1, 2), 0.0, 0.0, A(2, 2), 0.0,
                    0.0, 0.0, mA11mA21mA31, 0.0, 0.0, A(0, 0), 0.0, 0.0, A(1, 0), 0.0, 0.0, A(2, 0),
                    0.0, 0.0, mA12mA22mA32, 0.0, 0.0, A(0, 1), 0.0, 0.0, A(1, 1), 0.0, 0.0, A(2, 1),
                    0.0, 0.0, mA13mA23mA33, 0.0, 0.0, A(0, 2), 0.0, 0.0, A(1, 2), 0.0, 0.0, A(2, 2);
            }
        }
#ifdef USE_TBB
        );
#endif
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        globalLinSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        globalLinSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        globalLinSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        
        timer_temp3.stop();
    }
    template<int dim>
    ADMMTimeStepper<dim>::~ADMMTimeStepper(void)
    {
        delete globalLinSysSolver;
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::precompute(void)
    {
        timer_temp3.start(0);
        
        Base::computeEnergyVal(Base::result, true, Base::lastEnergyVal);
        
        // construct and prefactorize the linear system
        std::vector<Eigen::Triplet<double>> triplet;
        triplet.reserve(Base::result.F.rows() * dim * (dim + 1));
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            const Eigen::Matrix<int, 1, dim + 1>& triVInd = Base::result.F.row(triI);
            for(int localVI = 0; localVI < dim + 1; localVI++) {
                //TODO: simplify!
                triplet.emplace_back(triI * dim, triVInd[localVI],
                                     D_array[triI](0, localVI * dim));
                triplet.emplace_back(triI * dim + 1, triVInd[localVI],
                                     D_array[triI](1, localVI * dim));
                if(dim == 3) {
                    triplet.emplace_back(triI * dim + 2, triVInd[localVI],
                                         D_array[triI](2, localVI * dim));
                }
            }
        }
        D.resize(Base::result.F.rows() * dim, Base::result.V.rows());
        D.setZero();
        D.setFromTriplets(triplet.begin(), triplet.end());
        
        triplet.resize(0);
        triplet.reserve(Base::result.F.rows() * dim);
        for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
            int rowIStart = elemI * dim;
            for(int rowI = 0; rowI < dim; ++rowI) {
                triplet.emplace_back(rowIStart + rowI, rowIStart + rowI, 0.0);
            }
        }
        W.resize(Base::result.F.rows() * dim, Base::result.F.rows() * dim);
        W.setZero();
        W.setFromTriplets(triplet.begin(), triplet.end());
        
        W_elemPtr.resize(0);
        W_elemPtr.reserve(W.nonZeros());
        for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
            int rowIStart = elemI * dim;
            for(int rowI = 0; rowI < dim; ++rowI) {
                W_elemPtr.emplace_back(&W.coeffRef(rowIStart + rowI, rowIStart + rowI));
            }
        }
        
        Eigen::SparseMatrix<double> coefMtr = D.transpose() * W * D;
        offset_fixVerts.resize(0);
        offset_fixVerts.resize(Base::result.V.rows() * dim);
        for (int k = 0; k < coefMtr.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(coefMtr, k); it; ++it)
            {
                bool fixed_rowV = (Base::result.fixedVert.find(it.row()) !=
                                   Base::result.fixedVert.end());
                bool fixed_colV = (Base::result.fixedVert.find(it.col()) !=
                                   Base::result.fixedVert.end());
                if(fixed_rowV || fixed_colV) {
                    if(!fixed_rowV) {
                        offset_fixVerts[it.row()][it.col()] = 0.0;
                    }
                }
            }
        }
        coefMtr.makeCompressed();
        
        globalLinSysSolver->set_type(1, 2);
        globalLinSysSolver->set_pattern(coefMtr);
        globalLinSysSolver->analyze_pattern();
        
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
        timer_temp3.stop();
        
        initWeights();
        initGlobalLinSysSolver();
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::updatePrecondMtrAndFactorize(void)
    {
        precompute();
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::getFaceFieldForVis(Eigen::VectorXd& field)
    {
        field = Eigen::VectorXd::LinSpaced(Base::result.F.rows(), 0, Base::result.F.rows() - 1);
    }
    
    template<int dim>
    bool ADMMTimeStepper<dim>::fullyImplicit(void)
    {
        timer_temp3.start(1);
        // initialize x with xHat
        Base::initX(Base::animConfig.warmStart);
        
        Base::computeGradient(Base::result, true, Base::gradient);
        Base::computeEnergyVal(Base::result, false, Base::lastEnergyVal);
        std::cout << "After initX: E = " << Base::lastEnergyVal <<
            ", ||g||^2 = " << Base::gradient.squaredNorm() << std::endl;
        Base::file_iterStats << Base::globalIterNum << " " << Base::lastEnergyVal << " " << Base::gradient.squaredNorm() << std::endl;
        
        // initialize M_mult_xHat
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < Base::result.V.rows(); vI++)
#endif
        {
            if(Base::result.fixedVert.find(vI) == Base::result.fixedVert.end()) {
                M_mult_xHat.segment<dim>(vI * dim) = (Base::result.massMatrix.coeffRef(vI, vI) *
                                                  (Base::resultV_n.row(vI).transpose() +
                                                   Base::dt * Base::velocity.segment(vI * dim, dim) +
                                                   Base::dtSq * Base::gravity));
            }
            else {
                M_mult_xHat.segment<dim>(vI * dim) = (Base::result.massMatrix.coeffRef(vI, vI) *
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
        timer_temp3.stop();
        
        // ADMM iterations
        int ADMMIterAmt = Base::animConfig.maxIter_APD;
        int ADMMIterI = 0;
        for(; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            Base::file_iterStats << Base::globalIterNum << " ";
            
#ifdef SVSPACE_FSTEP
            zuUpdate_SV();
#else
            zuUpdate();
#endif
            xUpdate();
//            xUpdate();
//            zuUpdate();
//            checkRes();
            
            Base::computeGradient(Base::result, true, Base::gradient);
            Base::computeEnergyVal(Base::result, false, Base::lastEnergyVal);
            double sqn_g = Base::gradient.squaredNorm();
            std::cout << "Step" << Base::globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            Base::file_iterStats << Base::lastEnergyVal << " " << sqn_g << std::endl;
            if(sqn_g < Base::targetGRes) {
                break;
            }
        }
        
        if(ADMMIterI >= ADMMIterAmt) {
            logFile << "ADMM PD exceeds iteration cap" << std::endl;
//            exit(0);
        }
        
        Base::innerIterAmt += ((ADMMIterI == ADMMIterAmt) ?
                               ADMMIterAmt : (ADMMIterI + 1));
        
#ifndef OVERBYAPD
        initWeights();
        initGlobalLinSysSolver();
#endif
        
        return (ADMMIterI == ADMMIterAmt);
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::zuUpdate(void)
    {
        timer_temp3.start(5);
        
        int localMaxIter = 100; // fail-safe
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
#endif
        {
            Eigen::RowVectorXd zi = z.row(triI);
            Eigen::Matrix<double, dim * dim, 1> g;
            int j = 0;
            for( ; j < localMaxIter; j++) {
                computeGradient_zUpdate(triI, zi, g);
//                std::cout << "  " << triI << "-" << j << " ||g_local||^2 = "
//                    << g.squaredNorm() << std::endl;
//                if(g.squaredNorm() < localTol) {
//                    break;
//                }
                
                Eigen::Matrix<double, dim * dim, dim * dim> P;
                computeHessianProxy_zUpdate(triI, zi, P);
                //NOTE: for z being the deformation gradient,
                // no need to consider position constraints
                
                // solve for search direction
                Eigen::Matrix<double, dim * dim, 1> p = P.ldlt().solve(-g);
                
                // line search init
                double alpha = 1.0;
                
                // Armijo's rule:
//                const double m = p.dot(g);
//                const double c1m = 1.0e-4 * m;
                const double c1m = 0.0;
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
                
                if(std::abs((E0 - E) / E0) < 1e-3 * alpha) {
                    break;
                }
            }
            
            if(j >= localMaxIter) {
                logFile << "!!! maxIter reached for local solve!" << std::endl;
            }
            
            dz.row(triI) = zi - z.row(triI);
            z.row(triI) = zi;
            
            du.row(triI) = D_mult_x.row(triI) - z.row(triI);
            u.row(triI) += du.row(triI);
        }
#ifdef USE_TBB
        );
#endif
        
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMTimeStepper<dim>::zuUpdate_SV(void)
    {
        timer_temp3.start(5);
        
        // solve in singular value space
        int localMaxIter = 100; // fail-safe
        Eigen::VectorXd localIterCount(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
#endif
        {
            // find SV space
            Eigen::RowVectorXd Dx_plus_u = D_mult_x.row(triI) + u.row(triI);
            Eigen::Matrix<double, dim, dim> Dx_plus_u_mtr;
            Dx_plus_u_mtr.row(0) = Dx_plus_u.segment<dim>(0);
            Dx_plus_u_mtr.row(1) = Dx_plus_u.segment<dim>(dim);
            if(dim == 3) {
                Dx_plus_u_mtr.row(2) = Dx_plus_u.segment<dim>(dim * 2);
            }
            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Dx_plus_u_mtr,
                                                             Eigen::ComputeFullU |
                                                             Eigen::ComputeFullV);
            
            Eigen::Matrix<double, dim, 1> sigma_triI = svd.singularValues(); // initial guess
            Eigen::Matrix<double, dim, 1> g;
            int j = 0;
            for( ; j < localMaxIter; j++) {
                computeGradient_zUpdate_SV(triI, sigma_triI, svd.singularValues(), g);
                //                std::cout << "  " << triI << "-" << j << " ||g_local||^2 = "
                //                    << g.squaredNorm() << std::endl;
                //                if(g.squaredNorm() < localTol) {
                //                    break;
                //                }
                
                Eigen::Matrix<double, dim, dim> P;
                computeHessianProxy_zUpdate_SV(triI, sigma_triI, svd.singularValues(), P);
                //NOTE: for z being the deformation gradient,
                // no need to consider position constraints
                
                // solve for search direction
                Eigen::Matrix<double, dim, 1> p = P.ldlt().solve(-g);
                
                // line search init
                double alpha = 1.0;
                
                const double c1m = 0.0;
                const Eigen::Matrix<double, dim, 1> sigma0_triI = sigma_triI;
                double E0;
                computeEnergyVal_zUpdate_SV(triI, sigma0_triI, svd.singularValues(), E0);
                sigma_triI = sigma0_triI + alpha * p;
                double E;
                computeEnergyVal_zUpdate_SV(triI, sigma_triI, svd.singularValues(), E);
                while(E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    sigma_triI = sigma0_triI + alpha * p;
                    computeEnergyVal_zUpdate_SV(triI, sigma_triI, svd.singularValues(), E);
                }
                
                if(std::abs((E0 - E) / E0) < 1e-3 * alpha) {
                    break;
                }
            }
            
            if(j >= localMaxIter) {
                logFile << "!!! maxIter reached for local solve!" << std::endl;
            }
            localIterCount[triI] = j + 1;
            
            // recover F
            Eigen::Matrix<double, dim, dim> F_triI = svd.matrixU();
            F_triI.col(0) *= sigma_triI[0];
            F_triI.col(1) *= sigma_triI[1];
            if(dim == 3) {
                F_triI.col(2) *= sigma_triI[2];
            }
            F_triI *= svd.matrixV().transpose();
            
            Eigen::RowVectorXd zi;
            zi.resize(dim * dim);
            zi.segment<dim>(0) = F_triI.row(0);
            zi.segment<dim>(dim) = F_triI.row(1);
            if(dim == 3) {
                zi.segment<dim>(dim * 2) = F_triI.row(2);
            }
            
            dz.row(triI) = zi - z.row(triI);
            z.row(triI) = zi;
            
            du.row(triI) = D_mult_x.row(triI) - z.row(triI);
            u.row(triI) += du.row(triI);
        }
#ifdef USE_TBB
        );
#endif
        
        timer_temp3.stop();
        
//        std::cout << "avgLocalIterCount = " << localIterCount.mean() << std::endl;
    }
    template<int dim>
    void ADMMTimeStepper<dim>::checkRes(void)
    {
//        Eigen::VectorXd s = Eigen::VectorXd::Zero(Base::result.V.rows() * dim);
//        double r_part_max = 0.0, rmax = 0.0;
//        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
//            double r0I = D_mult_x.row(triI).norm();
//            if(r_part_max < r0I) {
//                r_part_max = r0I;
//            }
//            double r1I = z.row(triI).norm();
//            if(r_part_max < r1I) {
//                r_part_max = r1I;
//            }
//            
//            Eigen::RowVectorXd diff = D_mult_x.row(triI) - z.row(triI);
//            double rI = diff.norm();
//            if(rmax < rI) {
//                rmax = rI;
//            }
//            
//            const Eigen::Matrix<int, 1, dim + 1>& triVInd = Base::result.F.row(triI);
//            
//            const Eigen::VectorXd& s_triI = (D_array[triI].transpose() *
//                                             (GW[triI] * diff.transpose()));
//            s.segment<dim>(triVInd[0] * dim) += s_triI.segment<dim>(0);
//            s.segment<dim>(triVInd[1] * dim) += s_triI.segment<dim>(dim);
//            s.segment<dim>(triVInd[2] * dim) += s_triI.segment<dim>(dim * 2);
//            if(dim == 3) {
//                s.segment<dim>(triVInd[3] * dim) += s_triI.segment<dim>(dim * 3);
//            }
//        }
//        
//        double smax = 0.0, s_part_max = 0.0;
//        for(int vI = 0; vI < Base::result.V.rows(); vI++) {
//            if(Base::result.fixedVert.find(vI) != Base::result.fixedVert.end()) {
//                continue;
//            }
//            
//            double sI = s.segment<dim>(vI * dim).norm();
//            if(smax < sI) {
//                smax = sI;
//            }
//            
//            double s0I = M_mult_xHat.segment<dim>(vI * dim).norm();
//            if(s_part_max < s0I) {
//                s_part_max = s0I;
//            }
//            double s1I = (Base::result.massMatrix.coeffRef(vI, vI) *
//                          Base::result.V.row(vI).transpose()).norm();
//            if(s_part_max < s1I) {
//                s_part_max = s1I;
//            }
//        }
//        
//        double norm_s = smax / s_part_max, norm_r = rmax / r_part_max;
//        std::cout << "||s||_relInf = " << norm_s << " " << s_part_max <<
//            ", ||r||_relInf = " << norm_r << " " << r_part_max << ", ";
//        Base::file_iterStats << norm_s << " " << norm_r << " ";
//        
//        //TEST varying weights
//        double mu = 2.0;
//        if((norm_r > mu * norm_s) || (norm_s > mu * norm_r)) {
//            double multiplier = std::sqrt(norm_r / norm_s);
//            std::cout << "multiplyW " << multiplier << " ";
//            updateWeights(multiplier);
//            initGlobalLinSysSolver();
//        }
        
        double duNorm = 0.0, dzNorm = 0.0;
        for(int triI = 0; triI < Base::result.F.rows(); triI++) {
            duNorm += GW[triI].norm() * du.row(triI).transpose().squaredNorm();
            dzNorm += GW[triI].norm() * dz.row(triI).transpose().squaredNorm();
        }
        std::cout << "||du|| = " << duNorm << ", ||dz|| = " << dzNorm << ", ";
    }
    template<int dim>
    void ADMMTimeStepper<dim>::xUpdate(void)
    {
        timer_temp3.start(6);
        
        // compute rhs
        std::vector<Eigen::Matrix<double, dim * (dim + 1), 1>> rhs_right_tri(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); ++triI)
#endif
        {
            rhs_right_tri[triI] = D_array[triI].transpose() * (GW[triI](0, 0) * (z.row(triI) - u.row(triI)).transpose()); //TODO: can still be simplified by not multiplying 0 in D
        }
#ifdef USE_TBB
        );
#endif
        
        rhs_xUpdate = M_mult_xHat;
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < Base::result.V.rows(); vI++)
#endif
        {
            int _dimVI = vI * dim;
            for(const auto FLocI : Base::result.vFLoc[vI]) {
                rhs_xUpdate.segment<dim>(_dimVI) +=
                    rhs_right_tri[FLocI.first].segment(FLocI.second * dim, dim);
            }
        }
#ifdef USE_TBB
        );
#endif
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)offset_fixVerts.size(), 1, [&](int rowI)
#else
        for(int rowI = 0; rowI < offset_fixVerts.size(); rowI++)
#endif
        {
            for(const auto& entryI : offset_fixVerts[rowI]) {
                int vI = entryI.first;
                rhs_xUpdate.segment<dim>(rowI * dim) -= entryI.second *
                                                        Base::result.V.row(vI).transpose();
            }
        }
#ifdef USE_TBB
        );
#endif
        for(const auto& fVI : Base::result.fixedVert) {
            rhs_xUpdate.segment<dim>(fVI * dim) = Base::result.V.row(fVI).transpose();
        }
        
        Base::dimSeparatedSolve(globalLinSysSolver, rhs_xUpdate, Base::result.V);
        
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
        
        timer_temp3.stop();
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::compute_Di_mult_xi(int triI)
    {
        assert(triI < Base::result.F.rows());
        
        const Eigen::Matrix<int, 1, dim + 1>& triVInd = Base::result.F.row(triI);
        
        Eigen::Matrix<double, dim, dim> triMtr;
        triMtr.col(0) = (Base::result.V.row(triVInd[1]) -
                         Base::result.V.row(triVInd[0])).transpose();
        triMtr.col(1) = (Base::result.V.row(triVInd[2]) -
                         Base::result.V.row(triVInd[0])).transpose();
        if(dim == 3) {
            triMtr.col(2) = (Base::result.V.row(triVInd[3]) -
                             Base::result.V.row(triVInd[0])).transpose();
        }
        Eigen::Matrix<double, dim, dim> F;
        F = triMtr * Base::result.restTriInv[triI];
        
        D_mult_x.block<1, dim>(triI, 0) = F.row(0);
        D_mult_x.block<1, dim>(triI, dim) = F.row(1);
        if(dim == 3) {
            D_mult_x.block<1, dim>(triI, dim * 2) = F.row(2);
        }
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::initWeights(void)
    {
        timer_temp3.start(3);
        
        int W_elemPtrI = 0;
        for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
#ifdef OVERBYAPD
            // ADMM PD [Overby et al. 2017] weights
            double bulkModulus;
            Base::energyTerms[0]->getBulkModulus(Base::result.u[elemI],
                                                 Base::result.lambda[elemI],
                                                 bulkModulus);
            GW[elemI].setZero();
            GW[elemI].diagonal().setConstant(Base::dtSq * bulkModulus *
                                             Base::result.triArea[elemI]);
#else
            Base::F[elemI].row(0) = z.row(elemI).segment(0, dim);
            Base::F[elemI].row(1) = z.row(elemI).segment(dim, dim);
            if(dim == 3) {
                Base::F[elemI].row(2) = z.row(elemI).segment(dim * 2, dim);
            }
            
            Base::svd[elemI].compute(Base::F[elemI], Eigen::ComputeFullU | Eigen::ComputeFullV);
            
            Eigen::Matrix<double, dim * dim, dim * dim> wdP_div_dF;
            Base::energyTerms[0]->compute_dP_div_dF(Base::svd[elemI],
                                                    Base::result.u[elemI],
                                                    Base::result.lambda[elemI],
                                                    wdP_div_dF,
                                                    Base::dtSq * Base::result.triArea[elemI],
                                                    false);
            GW[elemI].setZero();
//            GW[elemI].diagonal() = wdP_div_dF.diagonal();
//            GW[elemI] = wdP_div_dF;
//            GW[elemI].diagonal().setConstant(wdP_div_dF.diagonal().maxCoeff());
//            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, dim * dim, dim * dim>> eigenSolver(wdP_div_dF);
//            GW[elemI].diagonal().setConstant(eigenSolver.eigenvalues()[dim * dim - 1]);
            GW[elemI].diagonal().setConstant(wdP_div_dF.norm());
#endif
            
            for(int rowI = 0; rowI < dim; ++rowI) {
                *W_elemPtr[W_elemPtrI] = GW[elemI](rowI, rowI);
                ++W_elemPtrI;
            }
        }
        
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMTimeStepper<dim>::updateWeights(double multiplier)
    {
        timer_temp3.start(3);
        
        int W_elemPtrI = 0;
        for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
            GW[elemI] *= multiplier;
            for(int rowI = 0; rowI < dim; ++rowI) {
                *W_elemPtr[W_elemPtrI] = GW[elemI](rowI, rowI);
                ++W_elemPtrI;
            }
        }
        
        // update scaled dual variable
        for(int elemI = 0; elemI < u.rows(); ++elemI) {
            u.row(elemI) /= multiplier;
        }
        
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMTimeStepper<dim>::initGlobalLinSysSolver(void)
    {
        timer_temp3.start(0);
        
        //TODO: directly compute the entries for coefMtr
        Eigen::SparseMatrix<double> coefMtr = D.transpose() * W * D;
        for(int vI = 0; vI < Base::result.V.rows(); ++vI) {
            double massI = Base::result.massMatrix.coeffRef(vI, vI);
            coefMtr.coeffRef(vI, vI) += massI;
        }
        for (int k = 0; k < coefMtr.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(coefMtr, k); it; ++it)
            {
                bool fixed_rowV = (Base::result.fixedVert.find(it.row()) !=
                                   Base::result.fixedVert.end());
                bool fixed_colV = (Base::result.fixedVert.find(it.col()) !=
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
            coefMtr.coeffRef(fVI, fVI) = 1.0;
        }
        coefMtr.makeCompressed();
        
        globalLinSysSolver->update_a(coefMtr);
        globalLinSysSolver->factorize(); //TODO: error check
        
        timer_temp3.stop();
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::computeEnergyVal_zUpdate(int triI,
                                                        const Eigen::RowVectorXd& zi,
                                                        double& Ei) const
    {
        Base::energyTerms[0]->computeEnergyValBySVD_F(Base::result, triI, zi, Ei);
        Ei *= Base::dtSq;
        Eigen::RowVectorXd vec = D_mult_x.row(triI) - zi + u.row(triI);
        Ei += (vec * GW[triI] * vec.transpose())[0] / 2.0;
    }
    template<int dim>
    void ADMMTimeStepper<dim>::computeGradient_zUpdate(int triI,
                                                       const Eigen::RowVectorXd& zi,
                                                       Eigen::Matrix<double, dim * dim, 1>& g) const
    {
//        Base::energyTerms[0]->computeGradientBySVD_F(Base::result, triI, zi, g);
        Base::energyTerms[0]->computeGradientByPK_F(Base::result, triI, zi, g);
        g *= Base::dtSq;
        g -= GW[triI] * (D_mult_x.row(triI) - zi + u.row(triI)).transpose();
    }
    template<int dim>
    void ADMMTimeStepper<dim>::computeHessianProxy_zUpdate(int triI,
                                                           const Eigen::RowVectorXd& zi,
                                                           Eigen::Matrix<double, dim * dim, dim * dim>& P) const
    {
//        Base::energyTerms[0]->computeHessianBySVD_F(Base::result, triI, zi, P);
        Base::energyTerms[0]->computeHessianByPK_F(Base::result, triI, zi, P);
        P *= Base::dtSq;
        P += GW[triI];
    }
    
    template<int dim>
    void ADMMTimeStepper<dim>::computeEnergyVal_zUpdate_SV(int triI,
                                                           const Eigen::Matrix<double, dim, 1>& sigma_triI,
                                                           const Eigen::Matrix<double, dim, 1>& sigma_Dx_plus_u,
                                                           double& Ei) const
    {
        Base::energyTerms[0]->compute_E(sigma_triI,
                                        Base::result.u[triI],
                                        Base::result.lambda[triI],
                                        Ei);
        Ei *= Base::result.triArea[triI] * Base::dtSq;
        Eigen::Matrix<double, dim, 1> vec = sigma_Dx_plus_u - sigma_triI;
        Ei += vec.squaredNorm() * GW[triI](0, 0) / 2.0;
    }
    template<int dim>
    void ADMMTimeStepper<dim>::computeGradient_zUpdate_SV(int triI,
                                                          const Eigen::Matrix<double, dim, 1>& sigma_triI,
                                                          const Eigen::Matrix<double, dim, 1>& sigma_Dx_plus_u,
                                                          Eigen::Matrix<double, dim, 1>& g) const
    {
        Base::energyTerms[0]->compute_dE_div_dsigma(sigma_triI,
                                                    Base::result.u[triI],
                                                    Base::result.lambda[triI],
                                                    g);
        g *= Base::result.triArea[triI] * Base::dtSq;
        g -= GW[triI](0, 0) * (sigma_Dx_plus_u - sigma_triI);
    }
    template<int dim>
    void ADMMTimeStepper<dim>::computeHessianProxy_zUpdate_SV(int triI,
                                                              const Eigen::Matrix<double, dim, 1>& sigma_triI,
                                                              const Eigen::Matrix<double, dim, 1>& sigma_Dx_plus_u,
                                                              Eigen::Matrix<double, dim, dim>& P) const
    {
        Base::energyTerms[0]->compute_d2E_div_dsigma2(sigma_triI,
                                                      Base::result.u[triI],
                                                      Base::result.lambda[triI],
                                                      P);
        IglUtils::makePD(P);
        P *= Base::result.triArea[triI] * Base::dtSq;
        P.diagonal().array() += GW[triI](0, 0);
    }
    
    template class ADMMTimeStepper<DIM>;
    
}
