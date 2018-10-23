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

#include "IglUtils.hpp"
#include "Timer.hpp"

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
extern std::ofstream logFile;
extern Timer timer_temp3;

namespace FracCuts {
    
    template<int dim>
    ADMMDDTimeStepper<dim>::ADMMDDTimeStepper(const TriangleSoup<dim>& p_data0,
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
        timer_temp3.start(0);
        // divide domain
        const int partitionAmt = animConfig.partitionAmt;
        mesh_subdomain.resize(partitionAmt);
        
        elemList_subdomain.resize(mesh_subdomain.size());
        globalVIToLocal_subdomain.resize(mesh_subdomain.size());
        localVIToGlobal_subdomain.resize(mesh_subdomain.size());
        globalTriIToLocal_subdomain.resize(mesh_subdomain.size());
        xHat_subdomain.resize(mesh_subdomain.size());
        svd_subdomain.resize(mesh_subdomain.size());
        F_subdomain.resize(mesh_subdomain.size());
        
#ifdef USE_METIS
        METIS<dim> partitions(Base::result);
#if(USE_METIS == 1)
        partitions.partMesh(partitionAmt);
#elif(USE_METIS == 2)
        partitions.partMesh_slice(Base::result, partitionAmt, 1);
#endif
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
            int subdomainTriAmt = Base::result.F.rows() / mesh_subdomain.size();
            int triI_begin = subdomainTriAmt * subdomainI;
            int triI_end = subdomainTriAmt * (subdomainI + 1) - 1;
            if(subdomainI + 1 == mesh_subdomain.size()) {
                triI_end = Base::result.F.rows() - 1;
            }
            elemList_subdomain[subdomainI] = Eigen::VectorXi::LinSpaced(triI_end - triI_begin + 1,
                                                                        triI_begin,
                                                                        triI_end);
//            // 2D grid test only:
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
            Base::result.constructSubmesh(elemList_subdomain[subdomainI], mesh_subdomain[subdomainI],
                                          globalVIToLocal_subdomain[subdomainI],
                                          globalTriIToLocal_subdomain[subdomainI]);
            
            localVIToGlobal_subdomain[subdomainI].resize(globalVIToLocal_subdomain[subdomainI].size());
            for(const auto& g2lI : globalVIToLocal_subdomain[subdomainI]) {
                localVIToGlobal_subdomain[subdomainI][g2lI.second] = g2lI.first;
            }
            
            xHat_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].V.rows(), dim);
            svd_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
            F_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
        }
#ifdef USE_TBB
        );
#endif
        
        // find overlapping vertices
        dualDim = 0;
        globalVIToDual_subdomain.resize(mesh_subdomain.size());
        dualIndIToLocal_subdomain.resize(mesh_subdomain.size());
        dualIndIToShared_subdomain.resize(mesh_subdomain.size());
        u_subdomain.resize(mesh_subdomain.size());
        du_subdomain.resize(mesh_subdomain.size());
        dz_subdomain.resize(mesh_subdomain.size());
        weights_subdomain.resize(mesh_subdomain.size());
        weightMtr_subdomain.resize(mesh_subdomain.size());
        weightMtrFixed_subdomain.resize(mesh_subdomain.size());
        weightSum.resize(Base::result.V.rows(), dim);
        weightSum.setZero();
        isSharedVert.resize(0);
        isSharedVert.resize(Base::result.V.rows(), false);
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            int sharedVertexAmt = 0;
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                for(int subdomainJ = 0; subdomainJ < mesh_subdomain.size(); subdomainJ++) {
                    if(subdomainJ != subdomainI) {
                        auto finder = globalVIToLocal_subdomain[subdomainJ].find(mapperI.first);
                        if(finder != globalVIToLocal_subdomain[subdomainJ].end()) {
                            dualIndIToLocal_subdomain[subdomainI].emplace_back(mapperI.second * dim);
                            dualIndIToLocal_subdomain[subdomainI].emplace_back(mapperI.second * dim + 1);
                            dualIndIToShared_subdomain[subdomainI].emplace_back(mapperI.first * dim);
                            dualIndIToShared_subdomain[subdomainI].emplace_back(mapperI.first * dim + 1);
                            if(dim == 3) {
                                dualIndIToLocal_subdomain[subdomainI].emplace_back(mapperI.second * dim + 2);
                                dualIndIToShared_subdomain[subdomainI].emplace_back(mapperI.first * dim + 2);
                            }
                            globalVIToDual_subdomain[subdomainI][mapperI.first] = sharedVertexAmt++;
                            isSharedVert[mapperI.first] = true;
                            break;
                        }
                    }
                }
            }
            dualDim += sharedVertexAmt;
            u_subdomain[subdomainI].resize(sharedVertexAmt, dim);
            du_subdomain[subdomainI].resize(sharedVertexAmt, dim);
            dz_subdomain[subdomainI].resize(sharedVertexAmt, dim);
            weights_subdomain[subdomainI].resize(sharedVertexAmt, dim);
            weights_subdomain[subdomainI].setZero();
            logFile << "subdomain " << subdomainI << ": " << sharedVertexAmt <<
                " shared vertices out of " << mesh_subdomain[subdomainI].V.rows() << std::endl;
        }
        dualDim *= dim;
        
        sharedVerts.resize(0);
        for(int vI = 0; vI < weightSum.rows(); vI++) {
            if(isSharedVert[vI]) {
                globalVIToShared[vI] = sharedVerts.size();
                sharedVerts.conservativeResize(sharedVerts.size() + 1);
                sharedVerts.tail(1) << vI;
            }
        }
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(auto& dualIndMapperI : dualIndIToShared_subdomain[subdomainI]) {
                auto finder = globalVIToShared.find(dualIndMapperI / dim);
                assert(finder != globalVIToShared.end());
                dualIndMapperI = finder->second * dim + dualIndMapperI % dim;
            }
        }
        
        // find shared elements
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); triI++)
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
        timer_temp3.stop();
    }
    
    template<int dim>
    ADMMDDTimeStepper<dim>::~ADMMDDTimeStepper(void)
    {
        for(int subdomainI = 0; subdomainI < linSysSolver_subdomain.size(); subdomainI++) {
            delete linSysSolver_subdomain[subdomainI];
        }
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::precompute(void)
    {
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            linSysSolver_subdomain[subdomainI]->set_type(1, 2);
            linSysSolver_subdomain[subdomainI]->set_pattern(mesh_subdomain[subdomainI].vNeighbor,
                                                            mesh_subdomain[subdomainI].fixedVert);
            linSysSolver_subdomain[subdomainI]->analyze_pattern();
        }
#ifdef USE_TBB
        );
#endif
        
        // for weights computation
        Base::linSysSolver->set_type(1, 2);
        Base::linSysSolver->set_pattern(Base::result.vNeighbor, Base::result.fixedVert);
        
        initWeights();
        initConsensusSolver();
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field.resize(Base::result.F.rows());
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(int elemII = 0; elemII < elemList_subdomain[subdomainI].size(); elemII++) {
                field[elemList_subdomain[subdomainI][elemII]] = subdomainI;
            }
        }
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::getSharedVerts(Eigen::VectorXi& sharedVerts) const
    {
        sharedVerts = this->sharedVerts;
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::writeMeshToFile(const std::string& filePath_pre) const
    {
        if(dim == 2) {
            for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
                Eigen::MatrixXd V(mesh_subdomain[subdomainI].V.rows(), 3);
                V.leftCols(2) = mesh_subdomain[subdomainI].V;
                V.col(2).setZero();
                igl::writeOBJ(filePath_pre + "_subdomain" + std::to_string(subdomainI) + ".obj",
                              V, mesh_subdomain[subdomainI].F);
            }
            Eigen::MatrixXd V(Base::result.V.rows(), 3);
            V.leftCols(2) = Base::result.V;
            V.col(2).setZero();
            igl::writeOBJ(filePath_pre + ".obj", V, Base::result.F);
        }
        else {
            for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
                IglUtils::saveTetMesh(filePath_pre + "_subdomain" +
                                      std::to_string(subdomainI) + ".msh",
                                      mesh_subdomain[subdomainI].V,
                                      mesh_subdomain[subdomainI].F);
            }
            IglUtils::saveTetMesh(filePath_pre + ".msh", Base::result.V, Base::result.F);
        }
    }
    
    template<int dim>
    bool ADMMDDTimeStepper<dim>::fullyImplicit(void)
    {
        initPrimal(1);
//        writeMeshToFile(outputFolderPath + "init");
        initDual();
        
        int outputTimestepAmt = 99;
        std::string curOutputFolderPath;
        if(Base::globalIterNum == outputTimestepAmt) {
            curOutputFolderPath = outputFolderPath + "timestep" + std::to_string(Base::globalIterNum);
            mkdir(curOutputFolderPath.c_str(), 0777);
            curOutputFolderPath += '/';
        }
        
        // ADMM iterations
        //TODO: adaptive tolerances
        int ADMMIterAmt = __INT_MAX__, ADMMIterI = 0;
        for(; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            Base::file_iterStats << Base::globalIterNum << " ";
            
            subdomainSolve();
//            checkRes();
            boundaryConsensusSolve();
            
            Base::computeGradient(Base::result, Base::scaffold, true, Base::gradient);
            double sqn_g = Base::gradient.squaredNorm();
            std::cout << "Step" << Base::globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            Base::file_iterStats << sqn_g << std::endl;
            if(sqn_g < Base::targetGRes) {
                break;
            }
            
            if((Base::globalIterNum == outputTimestepAmt) && (ADMMIterI < 100)) {
                std::string filePath_pre = curOutputFolderPath + std::to_string(ADMMIterI);
                writeMeshToFile(filePath_pre);
            }
        }
        Base::innerIterAmt += ADMMIterI + 1;
        
        initWeights();
        initConsensusSolver();
        
        return (ADMMIterI == ADMMIterAmt);
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::initPrimal(int option)
    {
        timer_temp3.start(1);
        // global:
        Base::initX(option);
        
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
                    xHat_subdomain[subdomainI].row(mapperI.second) = Base::resultV_n.row(mapperI.first) + Base::dt * Base::velocity.segment(mapperI.first * dim, dim).transpose() + Base::dtSq * Base::gravity.transpose();
                }
                else {
                    // scripted
                    xHat_subdomain[subdomainI].row(mapperI.second) = Base::result.V.row(mapperI.first);
                }
                mesh_subdomain[subdomainI].V.row(mapperI.second) = Base::result.V.row(mapperI.first);
            }
        }
#ifdef USE_TBB
        );
#endif
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::initDual(void)
    {
        timer_temp3.start(2);
        Eigen::VectorXd g;
        Base::computeGradient(Base::result, Base::scaffold, true, g); //TODO: only need to compute for shared vertices
        Base::file_iterStats << Base::globalIterNum << " 0 0 " << g.squaredNorm() << std::endl;
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            u_subdomain[subdomainI].setZero();
            
            Eigen::VectorXd g_subdomain;
            computeGradient_subdomain(subdomainI, true, g_subdomain); //TODO: only need to compute for shared vertices
#ifndef USE_GW
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                if(Base::result.fixedVert.find(dualMapperI.first) == Base::result.fixedVert.end()) {
                    int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                    u_subdomain[subdomainI].row(dualMapperI.second) = (g.segment<dim>(dualMapperI.first * dim) - g_subdomain.segment<dim>(localI * dim)).transpose();
                    u_subdomain[subdomainI](dualMapperI.second, 0) /= weights_subdomain[subdomainI](dualMapperI.second, 0);
                    u_subdomain[subdomainI](dualMapperI.second, 1) /= weights_subdomain[subdomainI](dualMapperI.second, 1);
                    if(dim == 3) {
                        u_subdomain[subdomainI](dualMapperI.second, 2) /= weights_subdomain[subdomainI](dualMapperI.second, 2);
                    }
                }
            }
#else
            Eigen::VectorXd rhs(globalVIToDual_subdomain[subdomainI].size() * dim);
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                rhs.segment<dim>(dualMapperI.second * dim) = g.segment<dim>(dualMapperI.first * dim) - g_subdomain.segment<dim>(localI * dim);
            }
            //TODO: consider using sparse matrix
            Eigen::MatrixXd coefMtr(rhs.size(), rhs.size());
            coefMtr.setZero();
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                coefMtr(entryI.first.first, entryI.first.second) = entryI.second;
            }
            for(const auto& fVI : Base::result.fixedVert) {
                const auto dualFinder = globalVIToDual_subdomain[subdomainI].find(fVI);
                if(dualFinder != globalVIToDual_subdomain[subdomainI].end()) {
                    coefMtr.block<dim, dim>(dualFinder->second * dim, dualFinder->second * dim).setIdentity();
                }
            }
            Eigen::VectorXd dualVec = coefMtr.ldlt().solve(rhs);
            for(int uI = 0; uI < u_subdomain[subdomainI].rows(); uI++) {
                u_subdomain[subdomainI].row(uI) = dualVec.segment<dim>(uI * dim).transpose();
            }
//            std::cout << u_subdomain[subdomainI] << std::endl; //DEBUG
#endif
        }
#ifdef USE_TBB
        );
#endif
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::initWeights(void)
    {
        timer_temp3.start(3);
        Base::computePrecondMtr(Base::result, Base::scaffold, true, Base::linSysSolver); //TODO: only need to compute for shared vertices
        
        double multiplier = 1.0;
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                mesh_subdomain[subdomainI].V.row(mapperI.second) = Base::result.V.row(mapperI.first);
            }
            computeHessianProxy_subdomain(subdomainI, true);
            
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
#ifndef USE_GW
                for(int dimI = 0; dimI < dim; dimI++) {
                    double offset = (Base::linSysSolver->coeffMtr(dualMapperI.first * dim + dimI,
                                                           dualMapperI.first * dim + dimI) -
                                     linSysSolver_subdomain[subdomainI]->coeffMtr(localI * dim + dimI,
                                                                                  localI * dim + dimI));
                    weightSum(dualMapperI.first, dimI) -= weights_subdomain[subdomainI](dualMapperI.second, dimI);
                    weights_subdomain[subdomainI](dualMapperI.second, dimI) += offset;
                    weights_subdomain[subdomainI](dualMapperI.second, dimI) *= multiplier;
                    weightSum(dualMapperI.first, dimI) += weights_subdomain[subdomainI](dualMapperI.second, dimI);
                }
#else
                if(mesh_subdomain[subdomainI].fixedVert.find(localI) !=
                   mesh_subdomain[subdomainI].fixedVert.end())
                {
                    // fixed vertex
                    continue;
                }
                for(int rowI = 0; rowI < dim; rowI++) {
                    for(int colI = 0; colI < dim; colI++) {
                        double offset = (Base::linSysSolver->coeffMtr(dualMapperI.first * dim + rowI,
                                                                      dualMapperI.first * dim + colI) -
                                         linSysSolver_subdomain[subdomainI]->coeffMtr(localI * dim + rowI,
                                                                                      localI * dim + colI));
                        // fill with dual index
                        weightMtr_subdomain[subdomainI][std::pair<int, int>(dualMapperI.second * dim + rowI, dualMapperI.second * dim + colI)] += offset;
                    }
                }
                for(const auto& nbVI_local : mesh_subdomain[subdomainI].vNeighbor[localI]) {
                    const auto& nbVI_global = localVIToGlobal_subdomain[subdomainI][nbVI_local];
                    auto finder = globalVIToDual_subdomain[subdomainI].find(nbVI_global);
                    if(finder != globalVIToDual_subdomain[subdomainI].end()) {
                        bool fixed = (mesh_subdomain[subdomainI].fixedVert.find(nbVI_local) !=
                                      mesh_subdomain[subdomainI].fixedVert.end());
                        for(int rowI = 0; rowI < dim; rowI++) {
                            for(int colI = 0; colI < dim; colI++) {
                                double offset = (Base::linSysSolver->coeffMtr(dualMapperI.first * dim + rowI,
                                                                        nbVI_global * dim + colI) -
                                                 linSysSolver_subdomain[subdomainI]->coeffMtr(localI * dim + rowI,
                                                                                              nbVI_local * dim + colI));
                                // fill with dual index
                                if(fixed) {
                                    weightMtrFixed_subdomain[subdomainI][std::pair<int, int>(dualMapperI.second * dim + rowI, finder->second * dim + colI)] += offset;
                                }
                                else {
                                    weightMtr_subdomain[subdomainI][std::pair<int, int>(dualMapperI.second * dim + rowI, finder->second * dim + colI)] += offset;
                                }
                            }
                        }
                    }
                }
#endif
            }
            //DEBUG
//            IglUtils::writeSparseMatrixToFile("/Users/mincli/Desktop/OptCuts_dynamic/output/WM" +
//                                              std::to_string(subdomainI),
//                                              weightMtr_subdomain[subdomainI], true);
        }
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::initConsensusSolver(void)
    {
        timer_temp3.start(4);
#ifdef USE_GW
        //TODO: consider using sparse matrix
        consensusMtr.conservativeResize(sharedVerts.size() * dim, sharedVerts.size() * dim);
        consensusMtr.setZero();
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                consensusMtr(dualIndIToShared_subdomain[subdomainI][entryI.first.first],
                        dualIndIToShared_subdomain[subdomainI][entryI.first.second]) += entryI.second;
            }
        }
        for(const auto& fVI : Base::result.fixedVert) {
            auto finder = globalVIToShared.find(fVI);
            if(finder != globalVIToShared.end()) {
                consensusMtr.block<dim, dim>(finder->second * dim, finder->second * dim).setIdentity();
            }
        }
        consensusSolver = consensusMtr.ldlt();
#endif
        timer_temp3.stop();
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::subdomainSolve(void) // local solve
    {
        timer_temp3.start(5);
        int localMaxIter = __INT_MAX__;
        double localTol = Base::targetGRes / mesh_subdomain.size() / 100.0; //TODO: needs to be more adaptive to global tol
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
                
                computeHessianProxy_subdomain(subdomainI, false);
                
                // solve for search direction
                linSysSolver_subdomain[subdomainI]->factorize();
                Eigen::VectorXd p, rhs = -g;
                linSysSolver_subdomain[subdomainI]->solve(rhs, p);
                
                // line search init
                double alpha = 1.0;
                Base::energyTerms[0]->filterStepSize(mesh_subdomain[subdomainI], p, alpha);
                
                // Armijo's rule:
//                const double m = p.dot(g);
//                const double c1m = 1.0e-4 * m;
                const double c1m = 0.0;
                Eigen::MatrixXd V0 = mesh_subdomain[subdomainI].V;
                double E0;
                computeEnergyVal_subdomain(subdomainI, false, E0);
                for(int vI = 0; vI < V0.rows(); vI++) {
                    mesh_subdomain[subdomainI].V.row(vI) = V0.row(vI) + alpha * p.segment<dim>(vI * dim).transpose();
                }
                double E;
                computeEnergyVal_subdomain(subdomainI, true, E);
                while(E > E0 + alpha * c1m) {
                    alpha /= 2.0;
                    for(int vI = 0; vI < V0.rows(); vI++) {
                        mesh_subdomain[subdomainI].V.row(vI) = V0.row(vI) + alpha * p.segment<dim>(vI * dim).transpose();
                    }
                    computeEnergyVal_subdomain(subdomainI, true, E);
                }
//                std::cout << "stepsize = " << alpha << std::endl;
            }
            
            // dual
            //TODO-GW?
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                dz_subdomain[subdomainI].row(dualMapperI.second) = mesh_subdomain[subdomainI].V.row(localI) - V0_ADMM.row(localI);
                du_subdomain[subdomainI].row(dualMapperI.second) =
                    0.5 * (mesh_subdomain[subdomainI].V.row(localI) - Base::result.V.row(dualMapperI.first));
                u_subdomain[subdomainI].row(dualMapperI.second) += du_subdomain[subdomainI].row(dualMapperI.second);
            }
        }
#ifdef USE_TBB
        );
#endif
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::checkRes(void)
    {
        double sqn_r = 0.0, sqn_s = 0.0;;
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            sqn_r += du_subdomain[subdomainI].squaredNorm();
            
            Eigen::MatrixXd sI = Eigen::MatrixXd::Zero(mesh_subdomain[subdomainI].V.rows(), dim);
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                sI.row(localI) = weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct(dz_subdomain[subdomainI].row(dualMapperI.second));
            }
            sqn_s += sI.squaredNorm();
        }
        std::cout << "||s||^2 = " << sqn_s << ", ||r||^2 = " << sqn_r << ", ";
        Base::file_iterStats << sqn_s << " " << sqn_r << " ";
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::boundaryConsensusSolve(void) // global solve
    {
        timer_temp3.start(6);
#ifndef USE_GW
        Base::result.V.setZero();
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(isSharedVert[mapperI.first]) {
                    int dualI = globalVIToDual_subdomain[subdomainI][mapperI.first];
                    Base::result.V.row(mapperI.first) +=
                        weights_subdomain[subdomainI].row(dualI).cwiseProduct                        (mesh_subdomain[subdomainI].V.row(mapperI.second) + u_subdomain[subdomainI].row(dualI));
                }
                else {
                    Base::result.V.row(mapperI.first) = mesh_subdomain[subdomainI].V.row(mapperI.second);
                }
            }
        }
        for(int vI = 0; vI < Base::result.V.rows(); vI++) {
            if(isSharedVert[vI]) {
                Base::result.V(vI, 0) /= weightSum(vI, 0);
                Base::result.V(vI, 1) /= weightSum(vI, 1);
                if(dim == 3) {
                    Base::result.V(vI, 2) /= weightSum(vI, 2);
                }
            }
        }
#else
        Eigen::VectorXd rhs(sharedVerts.size() * dim);
        rhs.setZero();
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            Eigen::VectorXd augVec(globalVIToDual_subdomain[subdomainI].size() * dim);
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
                assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
                augVec.segment<dim>(dualMapperI.second * dim) =
                    (mesh_subdomain[subdomainI].V.row(localVIFinder->second) +
                     u_subdomain[subdomainI].row(dualMapperI.second)).transpose();
            }
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                rhs[dualIndIToShared_subdomain[subdomainI][entryI.first.first]] +=
                    entryI.second * augVec[entryI.first.second];
            }
        }
        
        for(const auto& fVI : Base::result.fixedVert) {
            auto finder = globalVIToShared.find(fVI);
            if(finder != globalVIToShared.end()) {
                rhs.segment<dim>(finder->second * dim) = Base::result.V.row(fVI).transpose();
            }
        }
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& entryI : weightMtrFixed_subdomain[subdomainI]) {
                rhs[dualIndIToShared_subdomain[subdomainI][entryI.first.first]] -=
                    entryI.second * rhs[dualIndIToShared_subdomain[subdomainI][entryI.first.second]];
            }
        }
        
        Eigen::VectorXd solvedSharedVerts = consensusSolver.solve(rhs);
        for(int svI = 0; svI < sharedVerts.size(); svI++) {
            Base::result.V.row(sharedVerts[svI]) = solvedSharedVerts.segment<dim>(svI * dim).transpose();
        }
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(!isSharedVert[mapperI.first]) {
                    Base::result.V.row(mapperI.first) = mesh_subdomain[subdomainI].V.row(mapperI.second);
                }
            }
        }
#endif
        timer_temp3.stop();
    }
    
    // subdomain energy computation
    template<int dim>
    void ADMMDDTimeStepper<dim>::computeEnergyVal_subdomain(int subdomainI, bool redoSVD, double& Ei)
    {
        // incremental potential:
        Base::energyTerms[0]->computeEnergyVal(mesh_subdomain[subdomainI], redoSVD,
                                               svd_subdomain[subdomainI],
                                               F_subdomain[subdomainI], Base::dtSq, Ei);
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            Ei += (mesh_subdomain[subdomainI].V.row(vI) - xHat_subdomain[subdomainI].row(vI)).squaredNorm() * massI / 2.0;
        }
        
        // augmented Lagrangian:
#ifndef USE_GW
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            Eigen::Matrix<double, 1, dim> vec = (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                                      Base::result.V.row(dualMapperI.first) +
                                      u_subdomain[subdomainI].row(dualMapperI.second));
            Ei += weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct(vec).dot(vec) / 2.0;
        }
#else
        // general weighting matrix
        Eigen::VectorXd augVec(globalVIToDual_subdomain[subdomainI].size() * dim);
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            augVec.segment<dim>(dualMapperI.second * dim) =
                (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                 Base::result.V.row(dualMapperI.first) +
                 u_subdomain[subdomainI].row(dualMapperI.second)).transpose();
        }
        Eigen::VectorXd g(globalVIToDual_subdomain[subdomainI].size() * dim);
        g.setZero();
        for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
            g[entryI.first.first] += entryI.second * augVec[entryI.first.second];
        }
        Ei += g.dot(augVec) / 2.0;
#endif
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::computeGradient_subdomain(int subdomainI,
                                                      bool redoSVD,
                                                      Eigen::VectorXd& g)
    {
        // incremental potential:
        Base::energyTerms[0]->computeGradient(mesh_subdomain[subdomainI], redoSVD,
                                              svd_subdomain[subdomainI],
                                              F_subdomain[subdomainI], Base::dtSq, g);
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            g.segment<dim>(vI * dim) += massI * (mesh_subdomain[subdomainI].V.row(vI) - xHat_subdomain[subdomainI].row(vI)).transpose();
        }
        
        // augmented Lagrangian:
#ifndef USE_GW
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            Eigen::Matrix<double, 1, dim> vec = (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                                      Base::result.V.row(dualMapperI.first) +
                                      u_subdomain[subdomainI].row(dualMapperI.second));
            g.segment<dim>(localVIFinder->second * dim) +=
                weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct(vec).transpose();
        }
#else
        Eigen::VectorXd augVec(globalVIToDual_subdomain[subdomainI].size() * dim);
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            augVec.segment<dim>(dualMapperI.second * dim) =
                (mesh_subdomain[subdomainI].V.row(localVIFinder->second) -
                 Base::result.V.row(dualMapperI.first) +
                 u_subdomain[subdomainI].row(dualMapperI.second)).transpose();
        }
        for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
            g[dualIndIToLocal_subdomain[subdomainI][entryI.first.first]] +=
                entryI.second * augVec[entryI.first.second];
        }
#endif
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::computeHessianProxy_subdomain(int subdomainI,
                                                          bool redoSVD)
    {
        // incremental potential:
        linSysSolver_subdomain[subdomainI]->setZero();
        Base::energyTerms[0]->computeHessian(mesh_subdomain[subdomainI], redoSVD,
                                             svd_subdomain[subdomainI],
                                             F_subdomain[subdomainI], Base::dtSq,
                                             linSysSolver_subdomain[subdomainI]);
        
        for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
            double massI = mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            int ind0 = vI * dim;
            int ind1 = ind0 + 1;
            linSysSolver_subdomain[subdomainI]->addCoeff(ind0, ind0, massI);
            linSysSolver_subdomain[subdomainI]->addCoeff(ind1, ind1, massI);
            if(dim == 3) {
                int ind2 = ind0 + 2;
                linSysSolver_subdomain[subdomainI]->addCoeff(ind2, ind2, massI);
            }
        }
        
        // augmented Lagrangian:
#ifndef USE_GW
        for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
            auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
            assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
            int _2localVI = localVIFinder->second * dim;
            int _2localVIp1 = _2localVI + 1;
            linSysSolver_subdomain[subdomainI]->addCoeff(_2localVI, _2localVI,
                                                         weights_subdomain[subdomainI](dualMapperI.second, 0));
            linSysSolver_subdomain[subdomainI]->addCoeff(_2localVIp1, _2localVIp1,
                                                         weights_subdomain[subdomainI](dualMapperI.second, 1));
            if(dim == 3) {
                int _2localVIp2 = _2localVI + 2;
                linSysSolver_subdomain[subdomainI]->addCoeff(_2localVIp2, _2localVIp2,
                                                             weights_subdomain[subdomainI](dualMapperI.second, 2));
            }
        }
#else
        for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
            linSysSolver_subdomain[subdomainI]->addCoeff(dualIndIToLocal_subdomain[subdomainI][entryI.first.first],
                                     dualIndIToLocal_subdomain[subdomainI][entryI.first.second],
                                     entryI.second);
        }
#endif
    }
    
    template class ADMMDDTimeStepper<DIM>;
    
}
