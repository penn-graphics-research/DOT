//
//  ADMMDDTimeStepper.cpp
//  FracCuts
//
//  Created by Minchen Li on 7/17/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "ADMMDDTimeStepper.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <iostream>

namespace FracCuts {
    
    ADMMDDTimeStepper::ADMMDDTimeStepper(const TriangleSoup& p_data0,
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
        //TODO: try different partition
        //TODO: output per timestep iteration count, check time count, write report
//        1.) Use the current ADMM DD on all 4 of your stress-test examples and compare it against Newton and ADMM PD as you increase resolution (use a high poisson and Youngs) ; 2.) Try the same experiment but now with shared elements instead of shared vertices; 3) enable visualization of the inner iterations of your ADMM method per a single timestep to see the changes between the proximal step and the averaging step.
        // divide domain
        const int partitionAmt = 4;
        assert(result.F.rows() % partitionAmt == 0);
        mesh_subdomain.resize(partitionAmt);
        
        elemList_subdomain.resize(mesh_subdomain.size());
        globalVIToLocal_subdomain.resize(mesh_subdomain.size());
        xHat_subdomain.resize(mesh_subdomain.size());
        int subdomainTriAmt = result.F.rows() / mesh_subdomain.size();
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            elemList_subdomain[subdomainI] = Eigen::VectorXi::LinSpaced(subdomainTriAmt,
                                                                        subdomainTriAmt * subdomainI,
                                                                        subdomainTriAmt * (subdomainI + 1) - 1);
            result.constructSubmesh(elemList_subdomain[subdomainI], mesh_subdomain[subdomainI],
                                    globalVIToLocal_subdomain[subdomainI]);
            
            xHat_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].V.rows(), 2);
        }
#ifdef USE_TBB
        );
#endif
        
        // find overlapping vertices
        dualDim = 0;
        globalVIToDual_subdomain.resize(mesh_subdomain.size());
        u_subdomain.resize(mesh_subdomain.size());
        du_subdomain.resize(mesh_subdomain.size());
        dz_subdomain.resize(mesh_subdomain.size());
        weights_subdomain.resize(mesh_subdomain.size());
        weightSum.resize(result.V.rows());
        weightSum.setZero();
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            int sharedVertexAmt = 0;
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                for(int subdomainJ = 0; subdomainJ < mesh_subdomain.size(); subdomainJ++) {
                    if(subdomainJ != subdomainI) {
                        auto finder = globalVIToLocal_subdomain[subdomainJ].find(mapperI.first);
                        if(finder != globalVIToLocal_subdomain[subdomainJ].end()) {
                            globalVIToDual_subdomain[subdomainI][mapperI.first] = sharedVertexAmt;
                            
                            weights_subdomain[subdomainI].conservativeResize(sharedVertexAmt + 1);
                            double weight = 0.01; //TODO: initialize per-element weight
                            weights_subdomain[subdomainI][sharedVertexAmt] = weight;
                            weightSum[mapperI.first] += weight;
                            
                            sharedVertexAmt++;
                            break;
                        }
                    }
                }
            }
            dualDim += sharedVertexAmt;
            u_subdomain[subdomainI].resize(sharedVertexAmt, 2);
            du_subdomain[subdomainI].resize(sharedVertexAmt, 2);
            dz_subdomain[subdomainI].resize(sharedVertexAmt, 2);
        }
        dualDim *= 2; // in 2D
        
        sharedVerts.resize(0);
        for(int vI = 0; vI < weightSum.size(); vI++) {
            if(weightSum[vI] > 0.0) {
                sharedVerts.conservativeResize(sharedVerts.size() + 1);
                sharedVerts.tail(1) << vI;
            }
        }
        
        linSysSolver_subdomain.resize(mesh_subdomain.size());
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
#ifdef LINSYSSOLVER_USE_CHOLMOD
            linSysSolver_subdomain[subdomainI] = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
            linSysSolver_subdomain[subdomainI] = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
            linSysSolver_subdomain[subdomainI] = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        }
    }
    
    ADMMDDTimeStepper::~ADMMDDTimeStepper(void)
    {
        for(int subdomainI = 0; subdomainI < linSysSolver_subdomain.size(); subdomainI++) {
            delete linSysSolver_subdomain[subdomainI];
        }
    }
    
    void ADMMDDTimeStepper::precompute(void)
    {
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            Eigen::VectorXd V;
            Eigen::VectorXi I, J;
            computeHessianProxy_subdomain(subdomainI, V, I, J);
            
            linSysSolver_subdomain[subdomainI]->set_type(1, 2);
            linSysSolver_subdomain[subdomainI]->set_pattern(I, J, V, mesh_subdomain[subdomainI].vNeighbor,
                                                            mesh_subdomain[subdomainI].fixedVert);
            linSysSolver_subdomain[subdomainI]->analyze_pattern();
        }
#ifdef USE_TBB
        );
#endif
    }
    
    void ADMMDDTimeStepper::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field.resize(result.F.rows());
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(int elemII = 0; elemII < elemList_subdomain[subdomainI].size(); elemII++) {
                field[elemList_subdomain[subdomainI][elemII]] = subdomainI;
            }
        }
    }
    void ADMMDDTimeStepper::getSharedVerts(Eigen::VectorXi& sharedVerts) const
    {
        sharedVerts = this->sharedVerts;
    }
    
    bool ADMMDDTimeStepper::fullyImplicit(void)
    {
        // TODO: compare different initial guess, add comments
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < result.V.rows(); vI++)
#endif
        {
            if(result.fixedVert.find(vI) == result.fixedVert.end()) {
                result.V.row(vI) += (dt * velocity.segment(vI * 2, 2) + dtSq * gravity).transpose();
            }
        }
#ifdef USE_TBB
        );
#endif
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            u_subdomain[subdomainI].setZero();
            
//            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
//                mesh_subdomain[subdomainI].V.row(globalVIToLocal_subdomain[subdomainI][dualMapperI.first]) = result.V.row(dualMapperIapperI.first);
//            }
            
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(mesh_subdomain[subdomainI].fixedVert.find(mapperI.second) ==
                   mesh_subdomain[subdomainI].fixedVert.end())
                {
                    xHat_subdomain[subdomainI].row(mapperI.second) = resultV_n.row(mapperI.first) + dt * velocity.segment(mapperI.first * 2, 2).transpose() + dtSq * gravity.transpose();
                }
                else {
                    xHat_subdomain[subdomainI].row(mapperI.second) = resultV_n.row(mapperI.first);
                }
            }
            mesh_subdomain[subdomainI].V = xHat_subdomain[subdomainI];
        }
#ifdef USE_TBB
        );
#endif
        
        // ADMM iterations
        int ADMMIterAmt = 2000;
        for(int ADMMIterI = 0; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            file_iterStats << globalIterNum << " ";
            
            subdomainSolve();
            checkRes();
            boundaryConsensusSolve();
            
            computeGradient(result, scaffold, gradient);
            double sqn_g = gradient.squaredNorm();
            std::cout << "Step" << globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            file_iterStats << sqn_g << std::endl;
            if(sqn_g < targetGRes * 10.0) { //10~100!!!
                break;
            }
            
            if(ADMMIterI == ADMMIterAmt - 1) {
                return true;
            }
        }
        
        return false;
    }
    
    void ADMMDDTimeStepper::subdomainSolve(void) // local solve
    {
        int localMaxIter = __INT_MAX__;
        double localTol = targetGRes / mesh_subdomain.size();
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            // primal
            Eigen::MatrixXd V0_ADMM = mesh_subdomain[subdomainI].V;
            for(int j = 0; j < localMaxIter; j++) {
                Eigen::VectorXd g;
                computeGradient_subdomain(subdomainI, g);
//                std::cout << "  " << triI << "-" << j << " ||g_local||^2 = "
//                    << g.squaredNorm() << std::endl;
                if(g.squaredNorm() < localTol) {
                    break;
                }
                
                Eigen::VectorXd V;
                Eigen::VectorXi I, J;
                computeHessianProxy_subdomain(subdomainI, V, I, J);
                
                // solve for search direction
                linSysSolver_subdomain[subdomainI]->update_a(I, J, V);
                linSysSolver_subdomain[subdomainI]->factorize();
                Eigen::VectorXd p, rhs = -g;
                linSysSolver_subdomain[subdomainI]->solve(rhs, p);
                
                // line search init
                double alpha = 1.0;
                energyTerms[0]->initStepSize(mesh_subdomain[subdomainI], p, alpha);
                alpha *= 0.99;
                
                // Armijo's rule:
                const double m = p.dot(g);
                const double c1m = 1.0e-4 * m;
                Eigen::MatrixXd V0 = mesh_subdomain[subdomainI].V;
                double E0;
                computeEnergyVal_subdomain(subdomainI, E0);
                for(int vI = 0; vI < V0.rows(); vI++) {
                    mesh_subdomain[subdomainI].V.row(vI) = V0.row(vI) + alpha * p.segment(vI * 2, 2).transpose();
                }
                double E;
                computeEnergyVal_subdomain(subdomainI, E);
                while(E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    for(int vI = 0; vI < V0.rows(); vI++) {
                        mesh_subdomain[subdomainI].V.row(vI) = V0.row(vI) + alpha * p.segment(vI * 2, 2).transpose();
                    }
                    computeEnergyVal_subdomain(subdomainI, E);
                }
            }
            
            // dual
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                dz_subdomain[subdomainI].row(dualMapperI.second) = mesh_subdomain[subdomainI].V.row(localI) - V0_ADMM.row(localI);
                du_subdomain[subdomainI].row(dualMapperI.second) =
                    mesh_subdomain[subdomainI].V.row(localI) - result.V.row(dualMapperI.first);
                u_subdomain[subdomainI].row(dualMapperI.second) += du_subdomain[subdomainI].row(dualMapperI.second);
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    void ADMMDDTimeStepper::checkRes(void)
    {
        double sqn_r = 0.0, sqn_s = 0.0;;
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            sqn_r += du_subdomain[subdomainI].squaredNorm();
            
            Eigen::MatrixXd sI = Eigen::MatrixXd::Zero(mesh_subdomain[subdomainI].V.rows(), 2);
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                sI.row(localI) = weights_subdomain[subdomainI][dualMapperI.second] * dz_subdomain[subdomainI].row(dualMapperI.second);
            }
            sqn_s += sI.squaredNorm();
        }
        std::cout << "||s||^2 = " << sqn_s << ", ||r||^2 = " << sqn_r << ", ";
        file_iterStats << sqn_s << " " << sqn_r << " ";
    }
    void ADMMDDTimeStepper::boundaryConsensusSolve(void) // global solve
    {
        result.V.setZero();
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(weightSum[mapperI.first] > 0.0) {
                    int dualI = globalVIToDual_subdomain[subdomainI][mapperI.first];
                    result.V.row(mapperI.first) += weights_subdomain[subdomainI][dualI] *
                        (mesh_subdomain[subdomainI].V.row(mapperI.second) +
                         u_subdomain[subdomainI].row(dualI));
                }
                else {
                    result.V.row(mapperI.first) = mesh_subdomain[subdomainI].V.row(mapperI.second);
                }
            }
        }
        for(int vI = 0; vI < result.V.rows(); vI++) {
            if(weightSum[vI] > 0.0) {
                result.V.row(vI) /= weightSum[vI];
            }
        }
    }
    
    // subdomain energy computation
    void ADMMDDTimeStepper::computeEnergyVal_subdomain(int subdomainI, double& Ei) const
    {
        // incremental potential:
        energyTerms[0]->computeEnergyValBySVD(mesh_subdomain[subdomainI], Ei);
        Ei *= dtSq;
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            Ei += (mesh_subdomain[subdomainI].V.row(vI) - xHat_subdomain[subdomainI].row(vI)).squaredNorm() * massI / 2.0;
        }
        
        // augmented Lagrangian:
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            Ei += weights_subdomain[subdomainI][dualMapperI.second] / 2.0 *
                (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                 result.V.row(dualMapperI.first) + u_subdomain[subdomainI].row(dualMapperI.second)).squaredNorm();
        }
    }
    void ADMMDDTimeStepper::computeGradient_subdomain(int subdomainI, Eigen::VectorXd& g) const
    {
        // incremental potential:
        energyTerms[0]->computeGradientBySVD(mesh_subdomain[subdomainI], g);
        g *= dtSq;
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            g.segment(vI * 2, 2) += massI * (mesh_subdomain[subdomainI].V.row(vI) - xHat_subdomain[subdomainI].row(vI)).transpose();
        }
        
        // augmented Lagrangian:
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            g.segment(localVIFinder->second * 2, 2) += weights_subdomain[subdomainI][dualMapperI.second] *
                (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                 result.V.row(dualMapperI.first) + u_subdomain[subdomainI].row(dualMapperI.second)).transpose();
        }
    }
    void ADMMDDTimeStepper::computeHessianProxy_subdomain(int subdomainI, Eigen::VectorXd& V,
                                                          Eigen::VectorXi& I, Eigen::VectorXi& J) const
    {
        // incremental potential:
        I.resize(0);
        J.resize(0);
        V.resize(0);
        energyTerms[0]->computeHessianBySVD(mesh_subdomain[subdomainI], &V, &I, &J);
        V *= dtSq;
        int curTripletSize = static_cast<int>(I.size());
        I.conservativeResize(I.size() + mesh_subdomain[subdomainI].V.rows() * 2);
        J.conservativeResize(J.size() + mesh_subdomain[subdomainI].V.rows() * 2);
        V.conservativeResize(V.size() + mesh_subdomain[subdomainI].V.rows() * 2);
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            I[curTripletSize + vI * 2] = vI * 2;
            J[curTripletSize + vI * 2] = vI * 2;
            V[curTripletSize + vI * 2] = massI;
            I[curTripletSize + vI * 2 + 1] = vI * 2 + 1;
            J[curTripletSize + vI * 2 + 1] = vI * 2 + 1;
            V[curTripletSize + vI * 2 + 1] = massI;
        }
        
        // augmented Lagrangian:
        curTripletSize = static_cast<int>(I.size());
        I.conservativeResize(I.size() + globalVIToDual_subdomain[subdomainI].size() * 2);
        J.conservativeResize(J.size() + globalVIToDual_subdomain[subdomainI].size() * 2);
        V.conservativeResize(V.size() + globalVIToDual_subdomain[subdomainI].size() * 2);
        int dualI = 0;
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            I[curTripletSize + dualI * 2] = localVIFinder->second * 2;
            J[curTripletSize + dualI * 2] = localVIFinder->second * 2;
            V[curTripletSize + dualI * 2] = weights_subdomain[subdomainI][dualMapperI.second];
            I[curTripletSize + dualI * 2 + 1] = localVIFinder->second * 2 + 1;
            J[curTripletSize + dualI * 2 + 1] = localVIFinder->second * 2 + 1;
            V[curTripletSize + dualI * 2 + 1] = weights_subdomain[subdomainI][dualMapperI.second];
            dualI++;
        }
    }
    
}
