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

#include <igl/writeOBJ.h>

#include <sys/stat.h>

#include <iostream>

extern std::string outputFolderPath;

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
        weightSum.resize(result.V.rows(), 2);
        weightSum.setZero();
        isSharedVert.resize(0);
        isSharedVert.resize(result.V.rows(), false);
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            int sharedVertexAmt = 0;
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                for(int subdomainJ = 0; subdomainJ < mesh_subdomain.size(); subdomainJ++) {
                    if(subdomainJ != subdomainI) {
                        auto finder = globalVIToLocal_subdomain[subdomainJ].find(mapperI.first);
                        if(finder != globalVIToLocal_subdomain[subdomainJ].end()) {
                            globalVIToDual_subdomain[subdomainI][mapperI.first] = sharedVertexAmt++;
                            isSharedVert[mapperI.first] = true;
                            break;
                        }
                    }
                }
            }
            dualDim += sharedVertexAmt;
            u_subdomain[subdomainI].resize(sharedVertexAmt, 2);
            du_subdomain[subdomainI].resize(sharedVertexAmt, 2);
            dz_subdomain[subdomainI].resize(sharedVertexAmt, 2);
            weights_subdomain[subdomainI].resize(sharedVertexAmt, 2);
            weights_subdomain[subdomainI].setZero();
        }
        dualDim *= 2; // in 2D
        
        sharedVerts.resize(0);
        for(int vI = 0; vI < weightSum.rows(); vI++) {
            if(isSharedVert[vI]) {
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
        
        // for weights computation
        Eigen::VectorXi I, J;
        Eigen::VectorXd V;
        computePrecondMtr(result, scaffold, I, J, V);
        linSysSolver->set_type(1, 2);
        linSysSolver->set_pattern(I, J, V, result.vNeighbor, result.fixedVert);
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
    
    void ADMMDDTimeStepper::writeMeshToFile(const std::string& filePath_pre) const
    {
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            Eigen::MatrixXd V(mesh_subdomain[subdomainI].V.rows(), 3);
            V << mesh_subdomain[subdomainI].V, Eigen::VectorXd::Zero(V.rows());
            igl::writeOBJ(filePath_pre + "_subdomain" + std::to_string(subdomainI) + ".obj",
                          V, mesh_subdomain[subdomainI].F);
        }
        Eigen::MatrixXd V(result.V.rows(), 3);
        V << result.V, Eigen::VectorXd::Zero(V.rows());
        igl::writeOBJ(filePath_pre + ".obj", V, result.F);
    }
    
    bool ADMMDDTimeStepper::fullyImplicit(void)
    {
        initWeights();
        
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
            
            // precompute xHat and update local vertices
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                // a more general way that also valid for other initialization:
//                if(mesh_subdomain[subdomainI].fixedVert.find(mapperI.second) ==
//                   mesh_subdomain[subdomainI].fixedVert.end())
//                {
//                    xHat_subdomain[subdomainI].row(mapperI.second) = resultV_n.row(mapperI.first) + dt * velocity.segment(mapperI.first * 2, 2).transpose() + dtSq * gravity.transpose();
//                }
//                else {
//                    // scripted
//                    xHat_subdomain[subdomainI].row(mapperI.second) = result.V.row(mapperI.first);
//                }
                mesh_subdomain[subdomainI].V.row(mapperI.second) = result.V.row(mapperI.first);
            }
            // a more convenient way when using xHat as initial guess
            xHat_subdomain[subdomainI] = mesh_subdomain[subdomainI].V;
        }
#ifdef USE_TBB
        );
#endif
        int outputTimestepAmt = 3;
        std::string curOutputFolderPath;
        if(globalIterNum < outputTimestepAmt) {
            curOutputFolderPath = outputFolderPath + "timestep" + std::to_string(globalIterNum);
            mkdir(curOutputFolderPath.c_str(), 0777);
            curOutputFolderPath += '/';
        }
        
        // ADMM iterations
        int ADMMIterAmt = 200, ADMMIterI = 0;
        for(; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            file_iterStats << globalIterNum << " ";
            
            subdomainSolve();
            checkRes();
            boundaryConsensusSolve();
            
            computeGradient(result, scaffold, gradient);
            double sqn_g = gradient.squaredNorm();
            std::cout << "Step" << globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            file_iterStats << sqn_g << std::endl;
            if(sqn_g < targetGRes * 10.0) {
                break;
            }
            
            if(globalIterNum < outputTimestepAmt) {
                std::string filePath_pre = curOutputFolderPath + std::to_string(ADMMIterI);
                writeMeshToFile(filePath_pre);
            }
        }
        innerIterAmt += ADMMIterI;
        
        return (ADMMIterI == ADMMIterAmt);
    }
    
    void ADMMDDTimeStepper::initWeights(void)
    {
        //TODO: rest shape Hessian v.s. per time step Hessian?
        //TODO: write into report: spd will decrease the speed
        //TODO: initial guess of primal and dual variables?
        
        Eigen::VectorXi I, J;
        Eigen::VectorXd V;
        computePrecondMtr(result, scaffold, I, J, V);
        linSysSolver->update_a(I, J, V);
        
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                for(int dimI = 0; dimI < 2; dimI++) {
                        double offset = (linSysSolver->coeffMtr(dualMapperI.first * 2 + dimI,
                                                               dualMapperI.first * 2 + dimI) -
                                         linSysSolver_subdomain[subdomainI]->coeffMtr(localI * 2 + dimI,
                                                                                      localI * 2 + dimI));
                        weights_subdomain[subdomainI](dualMapperI.second, dimI) += offset;
                        weightSum(dualMapperI.first, dimI) += offset;
                }
            }
        }
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
                sI.row(localI) = weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct(dz_subdomain[subdomainI].row(dualMapperI.second));
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
                if(isSharedVert[mapperI.first]) {
                    int dualI = globalVIToDual_subdomain[subdomainI][mapperI.first];
                    result.V.row(mapperI.first) +=
                        weights_subdomain[subdomainI].row(dualI).cwiseProduct                        (mesh_subdomain[subdomainI].V.row(mapperI.second) + u_subdomain[subdomainI].row(dualI));
                }
                else {
                    result.V.row(mapperI.first) = mesh_subdomain[subdomainI].V.row(mapperI.second);
                }
            }
        }
        for(int vI = 0; vI < result.V.rows(); vI++) {
            if(isSharedVert[vI]) {
                result.V(vI, 0) /= weightSum(vI, 0);
                result.V(vI, 1) /= weightSum(vI, 1);
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
            Eigen::RowVector2d vec = (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                                      result.V.row(dualMapperI.first) +
                                      u_subdomain[subdomainI].row(dualMapperI.second));
            Ei += weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct(vec).dot(vec) / 2.0;
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
            Eigen::RowVector2d vec = (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                                      result.V.row(dualMapperI.first) +
                                      u_subdomain[subdomainI].row(dualMapperI.second));
            g.segment(localVIFinder->second * 2, 2) +=
                weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct(vec).transpose();
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
            V[curTripletSize + dualI * 2] = weights_subdomain[subdomainI](dualMapperI.second, 0);
            I[curTripletSize + dualI * 2 + 1] = localVIFinder->second * 2 + 1;
            J[curTripletSize + dualI * 2 + 1] = localVIFinder->second * 2 + 1;
            V[curTripletSize + dualI * 2 + 1] = weights_subdomain[subdomainI](dualMapperI.second, 1);
            dualI++;
        }
    }
    
}
