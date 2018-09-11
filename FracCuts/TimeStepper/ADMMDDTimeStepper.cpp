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

#ifdef USE_METIS
#include "METIS.hpp"
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
        mesh_subdomain.resize(partitionAmt);
        
        elemList_subdomain.resize(mesh_subdomain.size());
        globalVIToLocal_subdomain.resize(mesh_subdomain.size());
        globalTriIToLocal_subdomain.resize(mesh_subdomain.size());
        xHat_subdomain.resize(mesh_subdomain.size());
        svd_subdomain.resize(mesh_subdomain.size());
        
#ifdef USE_METIS
        METIS partitions(result);
        partitions.partMesh(partitionAmt);
#endif
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
#ifdef USE_METIS
            partitions.getElementList(subdomainI, elemList_subdomain[subdomainI]);
#else
            // partition according to face index
            int subdomainTriAmt = result.F.rows() / mesh_subdomain.size();
            int triI_begin = subdomainTriAmt * subdomainI;
            int triI_end = subdomainTriAmt * (subdomainI + 1) - 1;
            if(subdomainI + 1 == mesh_subdomain.size()) {
                triI_end = result.F.rows() - 1;
            }
            elemList_subdomain[subdomainI] = Eigen::VectorXi::LinSpaced(triI_end - triI_begin + 1,
                                                                        triI_begin,
                                                                        triI_end);
//            // grid test only:
//            int partitionWidth = std::sqrt(partitionAmt);
//            assert(partitionWidth * partitionWidth == partitionAmt);
//            int gridWidth = sqrt(result.F.rows() / 2);
//            assert(gridWidth * gridWidth == result.F.rows() / 2);
//            assert(gridWidth % partitionWidth == 0);
//            int innerWidth = gridWidth / partitionWidth;
//            int rowI = subdomainI / partitionWidth;
//            int colI = subdomainI % partitionWidth;
//            assert(result.F.rows() % partitionWidth == 0);
//            int triI_head = rowI * (result.F.rows() / partitionWidth) + colI * innerWidth * 2;
//            for(int innerRowI = 0; innerRowI < innerWidth; innerRowI++) {
//                for(int innerColI = 0; innerColI < innerWidth * 2; innerColI++) {
//                    int oldSize = elemList_subdomain[subdomainI].size();
//                    elemList_subdomain[subdomainI].conservativeResize(oldSize + 1);
//                    elemList_subdomain[subdomainI][oldSize] =
//                        triI_head + innerRowI * gridWidth * 2 + innerColI;
//                }
//            }
#endif
            result.constructSubmesh(elemList_subdomain[subdomainI], mesh_subdomain[subdomainI],
                                    globalVIToLocal_subdomain[subdomainI],
                                    globalTriIToLocal_subdomain[subdomainI]);
            
            xHat_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].V.rows(), 2);
            svd_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
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
        
        // find shared elements
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < result.F.rows(); triI++)
#endif
        {
            std::vector<std::map<int, int>::iterator> mapper(mesh_subdomain.size());
            double weight = 0.0;
            for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
                mapper[subdomainI] = globalTriIToLocal_subdomain[subdomainI].find(triI);
                if(mapper[subdomainI] != globalTriIToLocal_subdomain[subdomainI].end()) {
                    weight += 1.0;
                }
            }
            if(weight > 1.0) {
                weight = 1.0 / weight;
                for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
                    if(mapper[subdomainI] != globalTriIToLocal_subdomain[subdomainI].end()) {
                        mesh_subdomain[subdomainI].triWeight[mapper[subdomainI]->second] = weight;
                    }
                }
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
            mesh_subdomain[subdomainI].computeMassMatrix();
        }
#ifdef USE_TBB
        );
#endif
        
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
            computeHessianProxy_subdomain(subdomainI, true, V, I, J);
            
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
        computePrecondMtr(result, scaffold, true, I, J, V);
        linSysSolver->set_type(1, 2);
        linSysSolver->set_pattern(I, J, V, result.vNeighbor, result.fixedVert);
        
        initWeights();
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
        initPrimal(1);
//        writeMeshToFile(outputFolderPath + "init");
        initDual();
        
        int outputTimestepAmt = 100;
        std::string curOutputFolderPath;
        if(globalIterNum == outputTimestepAmt) {
            curOutputFolderPath = outputFolderPath + "timestep" + std::to_string(globalIterNum);
            mkdir(curOutputFolderPath.c_str(), 0777);
            curOutputFolderPath += '/';
        }
        
        // ADMM iterations
        //TODO: adaptive tolerances
        int ADMMIterAmt = __INT_MAX__, ADMMIterI = 0;
        for(; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            file_iterStats << globalIterNum << " ";
            
            subdomainSolve();
            checkRes();
            boundaryConsensusSolve();
            
            computeGradient(result, scaffold, true, gradient);
            double sqn_g = gradient.squaredNorm();
            std::cout << "Step" << globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            file_iterStats << sqn_g << std::endl;
            if(sqn_g < targetGRes) {
                break;
            }
            
            if((globalIterNum == outputTimestepAmt) && (ADMMIterI < 100)) {
                std::string filePath_pre = curOutputFolderPath + std::to_string(ADMMIterI);
                writeMeshToFile(filePath_pre);
            }
        }
        innerIterAmt += ADMMIterI + 1;
        
        initWeights();
        
        return (ADMMIterI == ADMMIterAmt);
    }
    
    void ADMMDDTimeStepper::initPrimal(int option)
    {
        // global:
        initX(option);
        
        // local:
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            // precompute xHat and update local primal
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(mesh_subdomain[subdomainI].fixedVert.find(mapperI.second) ==
                   mesh_subdomain[subdomainI].fixedVert.end())
                {
                    xHat_subdomain[subdomainI].row(mapperI.second) = resultV_n.row(mapperI.first) + dt * velocity.segment(mapperI.first * 2, 2).transpose() + dtSq * gravity.transpose();
                }
                else {
                    // scripted
                    xHat_subdomain[subdomainI].row(mapperI.second) = result.V.row(mapperI.first);
                }
                mesh_subdomain[subdomainI].V.row(mapperI.second) = result.V.row(mapperI.first);
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    void ADMMDDTimeStepper::initDual(void)
    {
        Eigen::VectorXd g;
        computeGradient(result, scaffold, true, g); //TODO: only need to compute for shared vertices
        file_iterStats << globalIterNum << " 0 0 " << g.squaredNorm() << std::endl;
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            u_subdomain[subdomainI].setZero();
            
            Eigen::VectorXd g_subdomain;
            computeGradient_subdomain(subdomainI, true, g_subdomain); //TODO: only need to compute for shared vertices
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                if(result.fixedVert.find(dualMapperI.first) == result.fixedVert.end()) {
                    int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                    u_subdomain[subdomainI].row(dualMapperI.second) = (g.segment(dualMapperI.first * 2, 2) - g_subdomain.segment(localI * 2, 2)).transpose();
                    u_subdomain[subdomainI](dualMapperI.second, 0) /= weights_subdomain[subdomainI](dualMapperI.second, 0);
                    u_subdomain[subdomainI](dualMapperI.second, 1) /= weights_subdomain[subdomainI](dualMapperI.second, 1);
                }
            }
        }
#ifdef USE_TBB
        );
#endif
                              
    }
    void ADMMDDTimeStepper::initWeights(void)
    {
        Eigen::VectorXi I, J;
        Eigen::VectorXd V;
        computePrecondMtr(result, scaffold, true, I, J, V);
        linSysSolver->update_a(I, J, V); //TODO: only need to compute for shared vertices
        
        double multiplier = 1.0;
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
//            Eigen::VectorXd V;
//            Eigen::VectorXi I, J;
//            computeHessianProxy_subdomain(subdomainI, V, I, J);
//            linSysSolver_subdomain[subdomainI]->update_a(I, J, V);
            
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                for(int dimI = 0; dimI < 2; dimI++) {
                    double offset = (linSysSolver->coeffMtr(dualMapperI.first * 2 + dimI,
                                                           dualMapperI.first * 2 + dimI) -
                                     linSysSolver_subdomain[subdomainI]->coeffMtr(localI * 2 + dimI,
                                                                                  localI * 2 + dimI));
                    weightSum(dualMapperI.first, dimI) -= weights_subdomain[subdomainI](dualMapperI.second, dimI);
                    weights_subdomain[subdomainI](dualMapperI.second, dimI) += offset;
                    weights_subdomain[subdomainI](dualMapperI.second, dimI) *= multiplier;
                    weightSum(dualMapperI.first, dimI) += weights_subdomain[subdomainI](dualMapperI.second, dimI);
                }
            }
        }
    }
    
    void ADMMDDTimeStepper::subdomainSolve(void) // local solve
    {
        int localMaxIter = __INT_MAX__;
        double localTol = targetGRes / mesh_subdomain.size() / 100.0; //TODO: needs to be more adaptive to global tol
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
                computeGradient_subdomain(subdomainI, false, g); // i = 0 also no redoSVD because of dual init
//                std::cout << "  " << subdomainI << "-" << j << " ||g_local||^2 = "
//                    << g.squaredNorm() << std::endl;
                if(g.squaredNorm() < localTol) {
                    break;
                }
                
                Eigen::VectorXd V;
                Eigen::VectorXi I, J;
                computeHessianProxy_subdomain(subdomainI, false, V, I, J);
                
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
                computeEnergyVal_subdomain(subdomainI, false, E0);
                for(int vI = 0; vI < V0.rows(); vI++) {
                    mesh_subdomain[subdomainI].V.row(vI) = V0.row(vI) + alpha * p.segment(vI * 2, 2).transpose();
                }
                double E;
                computeEnergyVal_subdomain(subdomainI, true, E);
                while(E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    for(int vI = 0; vI < V0.rows(); vI++) {
                        mesh_subdomain[subdomainI].V.row(vI) = V0.row(vI) + alpha * p.segment(vI * 2, 2).transpose();
                    }
                    computeEnergyVal_subdomain(subdomainI, true, E);
                }
//                std::cout << "stepsize = " << alpha << std::endl;
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
    void ADMMDDTimeStepper::computeEnergyVal_subdomain(int subdomainI, bool redoSVD, double& Ei)
    {
        // incremental potential:
        energyTerms[0]->computeEnergyValBySVD(mesh_subdomain[subdomainI], redoSVD,
                                              svd_subdomain[subdomainI], Ei);
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
    void ADMMDDTimeStepper::computeGradient_subdomain(int subdomainI,
                                                      bool redoSVD,
                                                      Eigen::VectorXd& g)
    {
        // incremental potential:
        energyTerms[0]->computeGradientByPK(mesh_subdomain[subdomainI], redoSVD,
                                            svd_subdomain[subdomainI], g);
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
    void ADMMDDTimeStepper::computeHessianProxy_subdomain(int subdomainI,
                                                          bool redoSVD,
                                                          Eigen::VectorXd& V,
                                                          Eigen::VectorXi& I,
                                                          Eigen::VectorXi& J)
    {
        // incremental potential:
        I.resize(0);
        J.resize(0);
        V.resize(0);
        energyTerms[0]->computeHessianByPK(mesh_subdomain[subdomainI], redoSVD,
                                           svd_subdomain[subdomainI], dtSq, &V, &I, &J);
        int curTripletSize = static_cast<int>(I.size());
        I.conservativeResize(I.size() + mesh_subdomain[subdomainI].V.rows() * 2);
        J.conservativeResize(J.size() + mesh_subdomain[subdomainI].V.rows() * 2);
        V.conservativeResize(V.size() + mesh_subdomain[subdomainI].V.rows() * 2);
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            int ind0 = vI * 2;
            int ind1 = ind0 + 1;
            I[curTripletSize + ind0] = ind0;
            J[curTripletSize + ind0] = ind0;
            V[curTripletSize + ind0] = massI;
            I[curTripletSize + ind1] = ind1;
            J[curTripletSize + ind1] = ind1;
            V[curTripletSize + ind1] = massI;
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
            int _2dualI = dualI * 2;
            int _2dualIp1 = _2dualI + 1;
            int _2localVI = localVIFinder->second * 2;
            int _2localVIp1 = _2localVI + 1;
            I[curTripletSize + _2dualI] = _2localVI;
            J[curTripletSize + _2dualI] = _2localVI;
            V[curTripletSize + _2dualI] = weights_subdomain[subdomainI](dualMapperI.second, 0);
            I[curTripletSize + _2dualIp1] = _2localVIp1;
            J[curTripletSize + _2dualIp1] = _2localVIp1;
            V[curTripletSize + _2dualIp1] = weights_subdomain[subdomainI](dualMapperI.second, 1);
            dualI++;
        }
    }
    
}
