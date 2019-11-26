//
//  ADMMDDTimeStepper.cpp
//  DOT
//
//  Created by Minchen Li on 7/17/18.
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
extern Timer timer_step, timer_temp3;
extern Eigen::MatrixXi SF;
extern std::vector<int> sTri2Tet;

namespace DOT {
    
    template<int dim>
    ADMMDDTimeStepper<dim>::ADMMDDTimeStepper(const Mesh<dim>& p_data0,
                                         const std::vector<Energy<dim>*>& p_energyTerms,
                                         const std::vector<double>& p_energyParams,
                                         bool p_mute,
                                         const Config& animConfig) :
        Base(p_data0, p_energyTerms, p_energyParams, p_mute, animConfig)
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
        U_sbd.resize(mesh_subdomain.size());
        U_sbd.resize(mesh_subdomain.size());
        V_sbd.resize(mesh_subdomain.size());
        Sigma_sbd.resize(mesh_subdomain.size());
        tol_subdomain.resize(mesh_subdomain.size());
        localIterCount_sbd.resize(mesh_subdomain.size(), 0);
        
        timer_sbd.resize(mesh_subdomain.size());
        for(int sbdI = 0; sbdI < mesh_subdomain.size(); ++sbdI) {
            timer_sbd[sbdI].new_activity("compG");
            timer_sbd[sbdI].new_activity("compH");
            timer_sbd[sbdI].new_activity("solveLinSys");
            timer_sbd[sbdI].new_activity("lineSearch");
            timer_sbd[sbdI].new_activity("compH_elasticity");
            timer_sbd[sbdI].new_activity("compH_mass");
            timer_sbd[sbdI].new_activity("compH_augLag");
        }
        timer_sum.new_activity("compG");
        timer_sum.new_activity("compH");
        timer_sum.new_activity("solveLinSys");
        timer_sum.new_activity("lineSearch");
        timer_sum.new_activity("compH_elasticity");
        timer_sum.new_activity("compH_mass");
        timer_sum.new_activity("compH_augLag");
        
#ifdef USE_METIS
        METIS<dim> partitions(Base::result);
        
#if(USE_METIS == 1)
        partitions.partMesh(partitionAmt);
        
#elif(USE_METIS == 2)
        partitions.partMesh_slice(Base::result, partitionAmt, 1);
        
#elif(USE_METIS == 3)
        assert(DIM == 3);
        std::cout << "computing node graph distance to surface" << std::endl;
        
        std::vector<int> shortestDistToSurface(Base::result.V.rows(), -1);
        for(int sfI = 0; sfI < SF.rows(); ++sfI) {
            shortestDistToSurface[SF(sfI, 0)] = 0;
            shortestDistToSurface[SF(sfI, 1)] = 0;
            shortestDistToSurface[SF(sfI, 2)] = 0;
        }
        Base::result.computeShortestDistToSurface(shortestDistToSurface);
        
        std::vector<idx_t> elementWeights(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int fI)
#else
        for(int fI = 0; fI < Base::result.F.rows(); ++fI)
#endif
        {
            const Eigen::Matrix<int, 1, dim + 1>& elemVInds = Base::result.F.row(fI);
            elementWeights[fI] = (shortestDistToSurface[elemVInds[0]] +
                                  shortestDistToSurface[elemVInds[1]] +
                                  shortestDistToSurface[elemVInds[2]]);
            if(dim == 3) {
                elementWeights[fI] += shortestDistToSurface[elemVInds[3]];
            }
        }
#ifdef USE_TBB
        );
#endif
        partitions.partMesh(partitionAmt, elementWeights.data());
        
#elif(USE_METIS == 4)
        std::vector<idx_t> elementWeights(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int fI)
#else
        for(int fI = 0; fI < Base::result.F.rows(); ++fI)
#endif
        {
            const Eigen::Matrix<int, 1, dim + 1>& elemVInds = Base::result.F.row(fI);
            elementWeights[fI] = (Base::result.vNeighbor[elemVInds[0]].size() + 1 +
                                  Base::result.vNeighbor[elemVInds[1]].size() + 1 +
                                  Base::result.vNeighbor[elemVInds[2]].size() + 1);
            if(dim == 3) {
                elementWeights[fI] += Base::result.vNeighbor[elemVInds[3]].size() + 1;
            }
        }
#ifdef USE_TBB
        );
#endif
        partitions.partMesh(partitionAmt, elementWeights.data());
        
#endif // USE_METIS ==
        
#endif // USE_METIS
        
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
            Base::result.constructSubmesh(elemList_subdomain[subdomainI],
                                          mesh_subdomain[subdomainI],
                                          globalVIToLocal_subdomain[subdomainI],
                                          globalTriIToLocal_subdomain[subdomainI]);
            
            localVIToGlobal_subdomain[subdomainI].resize(globalVIToLocal_subdomain[subdomainI].size());
            for(const auto& g2lI : globalVIToLocal_subdomain[subdomainI]) {
                localVIToGlobal_subdomain[subdomainI][g2lI.second] = g2lI.first;
            }
            
            xHat_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].V.rows(), dim);
            svd_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
            F_subdomain[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
            U_sbd[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
            V_sbd[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
            Sigma_sbd[subdomainI].resize(mesh_subdomain[subdomainI].F.rows());
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
        y_subdomain.resize(mesh_subdomain.size());
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
            y_subdomain[subdomainI].resize(sharedVertexAmt, dim);
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
        
        // find interface elements
        interElemGlobalI_sbd.resize(0);
        interElemGlobalI_sbd.resize(mesh_subdomain.size());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            std::vector<bool> isInterV(Base::result.V.rows(), false);
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                isInterV[dualMapperI.first] = true;
            }
            
            for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
                const Eigen::Matrix<int, 1, dim + 1>& elemVInd = Base::result.F.row(elemI);
                for(int cI = 0; cI < dim + 1; ++cI) {
                    if(isInterV[elemVInd[cI]]) {
                        interElemGlobalI_sbd[subdomainI].emplace_back(elemI);
                        break;
                    }
                }
            }
        }
#ifdef USE_TBB
        );
#endif
        
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
        
        if(useDense) {
            hessian_sbd.resize(mesh_subdomain.size());
            denseSolver_sbd.resize(mesh_subdomain.size());
        }
        else {
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
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        consensusSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        consensusSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        consensusSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        
        timer_temp3.stop();
        
        // output surface label
        std::vector<int> tetLabel(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            for(int elemII = 0; elemII < elemList_subdomain[subdomainI].size(); ++elemII) {
                tetLabel[elemList_subdomain[subdomainI][elemII]] = subdomainI;
            }
        }
#ifdef USE_TBB
        );
#endif
        FILE *out = fopen((outputFolderPath + "label.obj").c_str(), "w");
        assert(out);
        for(int sfI = 0; sfI < SF.rows(); ++sfI) {
            fprintf(out, "v %d 0 0\n", tetLabel[sTri2Tet[sfI]]);
        }
        fclose(out);
        
        // output wire frame
        std::vector<bool> isSurfNode(Base::result.V.rows(), false);
        for(int tI = 0; tI < SF.rows(); ++tI) {
            isSurfNode[SF(tI, 0)] = true;
            isSurfNode[SF(tI, 1)] = true;
            isSurfNode[SF(tI, 2)] = true;
        }
        
        std::vector<int> tetIndToSurf(Base::result.V.rows(), -1);
        std::vector<int> surfIndToTet(Base::result.V.rows(), -1);
        int sVI = 0;
        for(int vI = 0; vI < isSurfNode.size(); ++vI) {
            if(isSurfNode[vI]) {
                tetIndToSurf[vI] = sVI;
                surfIndToTet[sVI] = vI;
                ++sVI;
            }
        }
        
        Eigen::MatrixXd V_surf(sVI, 3);
        for(int vI = 0; vI < sVI; ++vI) {
            V_surf.row(vI) = Base::result.V.row(surfIndToTet[vI]);
        }
        Eigen::MatrixXi F_surf(SF.rows(), 3);
        for(int tI = 0; tI < SF.rows(); ++tI) {
            F_surf(tI, 0) = tetIndToSurf[SF(tI, 0)];
            F_surf(tI, 1) = tetIndToSurf[SF(tI, 1)];
            F_surf(tI, 2) = tetIndToSurf[SF(tI, 2)];
        }
        
        out = fopen((outputFolderPath + "wire.poly").c_str(), "w");
        assert(out);
        fprintf(out, "POINTS\n");
        for(int vI = 0; vI < V_surf.rows(); ++vI) {
            fprintf(out, "%d: %le %le %le\n", vI + 1, V_surf(vI, 0),
                    V_surf(vI, 1), V_surf(vI, 2));
        }
        fprintf(out, "POLYS\n");
        for(int fI = 0; fI < F_surf.rows(); ++fI) {
            int indStart = fI * 3;
            fprintf(out, "%d: %d %d\n", indStart + 1, F_surf(fI, 0) + 1, F_surf(fI, 1) + 1);
            fprintf(out, "%d: %d %d\n", indStart + 2, F_surf(fI, 1) + 1, F_surf(fI, 2) + 1);
            fprintf(out, "%d: %d %d\n", indStart + 3, F_surf(fI, 2) + 1, F_surf(fI, 0) + 1);
        }
        fprintf(out, "END\n");
        fclose(out);
    }
    
    template<int dim>
    ADMMDDTimeStepper<dim>::~ADMMDDTimeStepper(void)
    {
        if(!useDense) {
            for(int subdomainI = 0; subdomainI < linSysSolver_subdomain.size(); subdomainI++) {
                delete linSysSolver_subdomain[subdomainI];
            }
        }
        delete consensusSolver;
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::precompute(void)
    {
        timer_temp3.start(0);
        
        Base::computeEnergyVal(Base::result, true, Base::lastEnergyVal);
        
        if(useDense) {
            for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
                hessian_sbd[subdomainI].resize(mesh_subdomain[subdomainI].V.rows() * dim,
                                               mesh_subdomain[subdomainI].V.rows() * dim);
            }
        }
        else {
            timer_step.start(2);
#ifdef USE_TBB
            tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
            for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
            {
                // for enhancing local Hessian with missing information at interfaces
                std::vector<std::set<int>> vNeighborExt = mesh_subdomain[subdomainI].vNeighbor;
                for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                    int localVI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                    
                    for(const auto& nbVI_g : Base::result.vNeighbor[dualMapperI.first]) {
                        const auto localFinder = globalVIToLocal_subdomain[subdomainI].find(nbVI_g);
                        if(localFinder != globalVIToLocal_subdomain[subdomainI].end()) {
                            vNeighborExt[localVI].insert(localFinder->second);
                        }
                    }
                }
                
                linSysSolver_subdomain[subdomainI]->set_type(1, 2);
                linSysSolver_subdomain[subdomainI]->set_pattern(vNeighborExt,
                                                                mesh_subdomain[subdomainI].fixedVert);
                
                linSysSolver_subdomain[subdomainI]->analyze_pattern();
            }
#ifdef USE_TBB
            );
#endif
            timer_step.stop();
        }
        
//        timer_step.start(1);
        // for weights computation
        Base::linSysSolver->set_type(1, 2);
        Base::linSysSolver->set_pattern(Base::result.vNeighbor, Base::result.fixedVert);
//        timer_step.stop();
        
        timer_temp3.stop();
        
        m_isUpdateElemHessian.resize(0);
        m_isUpdateElemHessian.resize(Base::result.F.rows(), true);
        initWeights_fast(false);
        initConsensusSolver();
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::getFaceFieldForVis(Eigen::VectorXd& field)
    {
        field.resize(Base::result.F.rows());
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(int elemII = 0; elemII < elemList_subdomain[subdomainI].size(); elemII++) {
                field[elemList_subdomain[subdomainI][elemII]] = subdomainI;
            }
        }
        
        // visualize error distribution
//        Base::computeGradient(Base::result, true, Base::gradient);
//        Eigen::VectorXd gNorm(Base::gradient.size());
//        for(int vI = 0; vI < Base::result.V.rows(); ++vI) {
//            gNorm[vI] = Base::gradient.segment(vI * dim, dim).norm();
//        }
//        field.resize(Base::result.F.rows());
//        for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
//            const Eigen::Matrix<int, 1, dim + 1>& elemVInd = Base::result.F.row(elemI);
//            field[elemI] = (gNorm[elemVInd[0]] + gNorm[elemVInd[1]] + gNorm[elemVInd[2]]) / 3.0;
//        }
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
    const Eigen::MatrixXd& ADMMDDTimeStepper<dim>::getDenseMatrix(int sbdI) const
    {
        assert(useDense);
        assert(sbdI < hessian_sbd.size());

        return hessian_sbd[sbdI];
    }

    template<int dim>
    void ADMMDDTimeStepper<dim>::getGradient(int sbdI, Eigen::VectorXd& gradient)
    {
        computeGradient_subdomain(sbdI, false, gradient);
    }

    template<int dim>
    LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* ADMMDDTimeStepper<dim>::getSparseSolver(int sbdI)
    {
        assert(!useDense);
        assert(sbdI < linSysSolver_subdomain.size());

        return linSysSolver_subdomain[sbdI];
    }

    template<int dim>
    bool ADMMDDTimeStepper<dim>::fullyImplicit(void)
    {
#ifdef USE_SIMD
        std::cout << "ADMMDD must run without USE_SIMD macro!" << std::endl;
        std::cout << "please comment out #define USE_SIMD in src/Utils/Types.hpp," << std::endl;
        std::cout << "then rebuild and try again:)" << std::endl;
        logFile << "ADMMDD must run without USE_SIMD macro!" << std::endl;
        logFile << "please comment out #define USE_SIMD in src/Utils/Types.hpp," << std::endl;
        logFile << "then rebuild and try again:)" << std::endl;
        exit(0);
#endif

        initPrimal(Base::animConfig.warmStart);
        resultVk = Base::result.V;
//        writeMeshToFile(outputFolderPath + "init");
        initDual();
        for(int sbdI = 0; sbdI < mesh_subdomain.size(); ++sbdI) {
            tol_subdomain[sbdI] = __DBL_MAX__;
        }

        Base::computeGradient(Base::result, true, Base::gradient);
        Base::computeEnergyVal(Base::result, false, Base::lastEnergyVal);
        std::cout << "After initX: E = " << Base::lastEnergyVal <<
            ", ||g||^2 = " << Base::gradient.squaredNorm() << std::endl;
        Base::file_iterStats << Base::globalIterNum << " " << Base::lastEnergyVal << " " <<
        Base::gradient.squaredNorm() << std::endl;
        
        int outputTimestepAmt = -1;
        std::string curOutputFolderPath;
        if(Base::globalIterNum == outputTimestepAmt) {
            curOutputFolderPath = outputFolderPath + "timestep" + std::to_string(Base::globalIterNum);
            mkdir(curOutputFolderPath.c_str(), 0777);
            curOutputFolderPath += '/';
        }
        
        // ADMM iterations
        int ADMMIterAmt = 1000, ADMMIterI = 0;
        for(; ADMMIterI < ADMMIterAmt; ADMMIterI++) {
            Base::file_iterStats << Base::globalIterNum << " ";
            
            subdomainSolve(1, 1, (ADMMIterI % 20 == 0), true); //TODO: can make this Hessian factorization update adaptive as for DVS
            boundaryConsensusSolve(1.8);
            dualSolve(1.0, 1.8);

            resultVk = Base::result.V;
            
            checkRes();
            
            Base::computeGradient(Base::result, true, Base::gradient);
            Eigen::VectorXd grad_KKT;
            sqn_g = Base::gradient.squaredNorm();
            std::cout << "Step" << Base::globalIterNum << "-" << ADMMIterI <<
                " ||gradient||^2 = " << sqn_g << std::endl;
            Base::computeEnergyVal(Base::result, false, Base::lastEnergyVal);
            Base::file_iterStats << Base::lastEnergyVal << " " << sqn_g << std::endl;
            if(sqn_g < Base::targetGRes) {
//                // sanityCheck
//                Eigen::VectorXd sbdGradSum(Base::gradient.size());
//                sbdGradSum.setZero();
//                for(int sbdI = 0; sbdI < mesh_subdomain.size(); ++sbdI) {
//                    Eigen::VectorXd gradI;
//                    computeGradient_subdomain(sbdI, false, gradI);
//                    for(int localVI = 0; localVI < localVIToGlobal_subdomain[sbdI].size();
//                        ++localVI)
//                    {
//                        sbdGradSum.segment<dim>(localVIToGlobal_subdomain[sbdI][localVI]
//                                                * dim) += gradI.segment<dim>(localVI * dim);
//                    }
//                }
//                std::cout << "grad diff l_inf_rel = " <<
//                    (sbdGradSum - Base::gradient).cwiseAbs().maxCoeff() /
//                    Base::gradient.cwiseAbs().maxCoeff() << std::endl;
                double err_b, err_in;
                computeError(Base::gradient, err_in, err_b);
                break;
            }
            
            if((Base::globalIterNum == outputTimestepAmt) && (ADMMIterI < 100)) {
                std::string filePath_pre = curOutputFolderPath + std::to_string(ADMMIterI);
                writeMeshToFile(filePath_pre);
            }
            
//            if(ADMMIterI % 2 == 0) {
//                initWeights_fast();
//                initDual();
//            }
        }
        Base::innerIterAmt += ADMMIterI + 1;
        
        logFile << "avglocalIterCount =";
        for(int sbdI = 0; sbdI < localIterCount_sbd.size(); ++sbdI) {
            logFile << " " << localIterCount_sbd[sbdI] / float(ADMMIterI + 1);
            localIterCount_sbd[sbdI] = 0;
        }
        logFile << std::endl;
        
//        for(int sbdI = 0; sbdI < timer_sbd.size(); ++sbdI) {
//            timer_sum += timer_sbd[sbdI];
//        }
//        timer_sum.print(logFile);
        
        initWeights_fast(false);
        updateConsensusSolver();
        
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
        Base::computeGradient(Base::result, true, g); //TODO: only need to compute for shared vertices
        sqn_g = g.squaredNorm();
        Base::file_iterStats << Base::globalIterNum << " 0 0 " << sqn_g << std::endl;
        
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
    void ADMMDDTimeStepper<dim>::initWeights(bool overwriteSbdH)
    {
        timer_temp3.start(3);
        Base::computePrecondMtr(Base::result, true, Base::linSysSolver); //TODO: only need to compute for shared vertices
        
        double multiplier = 1.0;
        //TODO: parallelize
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                mesh_subdomain[subdomainI].V.row(mapperI.second) = Base::result.V.row(mapperI.first);
            }
            computeHessianProxy_subdomain(subdomainI, true, true);
            
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
#ifndef USE_GW
                for(int dimI = 0; dimI < dim; dimI++) {
                    int sbdRowI = localI * dim + dimI;
                    const double& sbdEntry = (useDense ?
                                              hessian_sbd[subdomainI](sbdRowI, sbdRowI) :
                                              linSysSolver_subdomain[subdomainI]->coeffMtr(sbdRowI, sbdRowI));
                    const double& globalEntry = Base::linSysSolver->coeffMtr(dualMapperI.first * dim + dimI,
                                                                             dualMapperI.first * dim + dimI);
                    double offset = globalEntry- sbdEntry;
                    weightSum(dualMapperI.first, dimI) -= weights_subdomain[subdomainI](dualMapperI.second, dimI);
                    weights_subdomain[subdomainI](dualMapperI.second, dimI) += offset;
                    weights_subdomain[subdomainI](dualMapperI.second, dimI) *= multiplier;
                    weightSum(dualMapperI.first, dimI) += weights_subdomain[subdomainI](dualMapperI.second, dimI);
                    
                    if(overwriteSbdH) {
                        if(useDense) {
                            hessian_sbd[subdomainI](sbdRowI, sbdRowI) = globalEntry;
                        }
                        else {
                            linSysSolver_subdomain[subdomainI]->setCoeff(sbdRowI, sbdRowI,
                                                                         globalEntry);
                        }
                    }
                }
#else
                //TODO: overwriteSbdH
                if(mesh_subdomain[subdomainI].fixedVert.find(localI) !=
                   mesh_subdomain[subdomainI].fixedVert.end())
                {
                    // fixed vertex
                    continue;
                }
                for(int rowI = 0; rowI < dim; rowI++) {
                    for(int colI = 0; colI < dim; colI++) {
                        int sbdRowI = localI * dim + rowI;
                        int sbdColI = localI * dim + colI;
                        const double& sbdEntry = (useDense ?
                                                  hessian_sbd[subdomainI](sbdRowI, sbdColI) :
                                                  linSysSolver_subdomain[subdomainI]->coeffMtr(sbdRowI, sbdColI));
                        double offset = (Base::linSysSolver->coeffMtr(dualMapperI.first * dim + rowI,
                                                                      dualMapperI.first * dim + colI)
                                         - sbdEntry);
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
                                int sbdRowI = localI * dim + rowI;
                                int sbdColI = nbVI_local * dim + colI;
                                const double& sbdEntry = (useDense ?
                                                          hessian_sbd[subdomainI](sbdRowI, sbdColI) :
                                                          linSysSolver_subdomain[subdomainI]->coeffMtr(sbdRowI, sbdColI));
                                double offset = (Base::linSysSolver->coeffMtr(dualMapperI.first * dim + rowI, nbVI_global * dim + colI) -
                                                 sbdEntry);
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
//            IglUtils::writeSparseMatrixToFile("/Users/mincli/Desktop/DOT/output/WM" +
//                                              std::to_string(subdomainI),
//                                              weightMtr_subdomain[subdomainI], true);
        }
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::initWeights_fast(bool overwriteSbdH)
    {
        timer_temp3.start(3);
        
        std::vector<Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)>> elemHessians;
        std::vector<Eigen::Matrix<int, 1, dim + 1>> vInds;
        Base::energyTerms[0]->computeElemHessianByPK(Base::result, false,
                                                     Base::svd, Base::F,
                                                     Base::energyParams[0] * Base::dtSq,
                                                     m_isUpdateElemHessian,
                                                     elemHessians, vInds, true);

        // missing Hessian information
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int sbdI)
#else
        for(int sbdI = 0; sbdI < mesh_subdomain.size(); ++sbdI)
#endif
        {
            
#ifdef USE_GW
            //TODO: only allocate storage in the first iteration
            weightMtr_subdomain[sbdI].clear();
//            weightMtrFixed_subdomain[sbdI].clear();
#endif
            
            for(const auto& dualMapperI : globalVIToDual_subdomain[sbdI]) {
                if(Base::result.isFixedVert[dualMapperI.first]) {
#ifndef USE_GW
                    weights_subdomain[sbdI].row(dualMapperI.second).setZero();
#endif
                }
                else {
                    int localVI = globalVIToLocal_subdomain[sbdI][dualMapperI.first];
                    
                    // add missing mass
                    double massDif = (Base::result.massMatrix.coeff(dualMapperI.first,
                                                                    dualMapperI.first) -
                                      mesh_subdomain[sbdI].massMatrix.coeff(localVI,
                                                                            localVI));
#ifdef USE_GW
                    int startInd = dualMapperI.second * dim;
                    int startIndp1 = startInd + 1;
                    int startIndp2 = startInd + 2;
                    weightMtr_subdomain[sbdI][std::pair<int, int>(startInd, startInd)] += massDif;
                    weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp1, startIndp1)] += massDif;
                    if(dim == 3) {
                        weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp2, startIndp2)] += massDif;
                    }
#else
                    weights_subdomain[sbdI].row(dualMapperI.second).setConstant(massDif);
#endif
                    
                    // check whether any elements are missing
                    for(const auto& FLocI_g : Base::result.vFLoc[dualMapperI.first]) {
                        const auto elemFinder = globalTriIToLocal_subdomain[sbdI].find(FLocI_g.first);
                        if(elemFinder == globalTriIToLocal_subdomain[sbdI].end()) {
                            // for missing element, add the missing Hessian

                            const Eigen::Matrix<double, dim, dim>& hessianI = elemHessians[FLocI_g.first].block(FLocI_g.second * dim, FLocI_g.second * dim, dim, dim);
                            
#ifdef USE_GW
                            weightMtr_subdomain[sbdI][std::pair<int, int>(startInd, startInd)] += hessianI(0, 0);
                            weightMtr_subdomain[sbdI][std::pair<int, int>(startInd, startIndp1)] += hessianI(0, 1);
                            weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp1, startInd)] += hessianI(1, 0);
                            weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp1, startIndp1)] += hessianI(1, 1);
                            if(dim == 3) {
                                weightMtr_subdomain[sbdI][std::pair<int, int>(startInd, startIndp2)] += hessianI(0, 2);
                                weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp1, startIndp2)] += hessianI(1, 2);
                                
                                weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp2, startInd)] += hessianI(2, 0);
                                weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp2, startIndp1)] += hessianI(2, 1);
                                weightMtr_subdomain[sbdI][std::pair<int, int>(startIndp2, startIndp2)] += hessianI(2, 2);
                            }
                            
                            // check for off-diagonal blocks
                            const Eigen::Matrix<int, 1, dim + 1>& elemVInds_g = Base::result.F.row(FLocI_g.first);
                            for(int vI_elem = 0; vI_elem < dim + 1; ++vI_elem) {
                                if(vI_elem == FLocI_g.second) {
                                    continue;
                                }
                                
                                std::map<std::pair<int, int>, double>* ptr_W_sbd =
                                    &weightMtr_subdomain[sbdI];
                                if(Base::result.isFixedVert[elemVInds_g[vI_elem]]) {
//                                    ptr_W_sbd = &weightMtrFixed_subdomain[sbdI];
                                    continue;
                                }
                                
                                const auto dualNFinder = globalVIToDual_subdomain[sbdI].find(elemVInds_g[vI_elem]);
                                if(dualNFinder != globalVIToDual_subdomain[sbdI].end()) {
                                    int startInd_col = dualNFinder->second * dim;
                                    int startIndp1_col = startInd_col + 1;
                                    int startIndp2_col = startInd_col + 2;
                                    
                                    const Eigen::Matrix<double, dim, dim>& hessianJ = elemHessians[FLocI_g.first].block(FLocI_g.second * dim, vI_elem * dim, dim, dim);
                                    
                                    (*ptr_W_sbd)[std::pair<int, int>(startInd, startInd_col)] += hessianJ(0, 0);
                                    (*ptr_W_sbd)[std::pair<int, int>(startInd, startIndp1_col)] += hessianJ(0, 1);
                                    (*ptr_W_sbd)[std::pair<int, int>(startIndp1, startInd_col)] += hessianJ(1, 0);
                                    (*ptr_W_sbd)[std::pair<int, int>(startIndp1, startIndp1_col)] += hessianJ(1, 1);
                                    if(dim == 3) {
                                        (*ptr_W_sbd)[std::pair<int, int>(startInd, startIndp2_col)] += hessianJ(0, 2);
                                        (*ptr_W_sbd)[std::pair<int, int>(startIndp1, startIndp2_col)] += hessianJ(1, 2);
                                        
                                        (*ptr_W_sbd)[std::pair<int, int>(startIndp2, startInd_col)] += hessianJ(2, 0);
                                        (*ptr_W_sbd)[std::pair<int, int>(startIndp2, startIndp1_col)] += hessianJ(2, 1);
                                        (*ptr_W_sbd)[std::pair<int, int>(startIndp2, startIndp2_col)] += hessianJ(2, 2);
                                    }
                                }
                            }
#else
                            weights_subdomain[sbdI](dualMapperI.second, 0) += hessianI(0, 0);
                            weights_subdomain[sbdI](dualMapperI.second, 1) += hessianI(1, 1);
                            if(dim == 3) {
                                weights_subdomain[sbdI](dualMapperI.second, 2) += hessianI(2, 2);
                            }
#endif
                        }
                    }
                }
            }
        }
#ifdef USE_TBB
        );
#endif
        
#ifndef USE_GW
        weightSum.setZero();
        for(int sbdI = 0; sbdI < mesh_subdomain.size(); ++sbdI) {
            for(const auto& dualMapperI : globalVIToDual_subdomain[sbdI]) {
                weightSum.row(dualMapperI.first) +=
                    weights_subdomain[sbdI].row(dualMapperI.second);
            }
        }
#endif
        
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::initConsensusSolver(void)
    {
        timer_temp3.start(4);
#ifdef USE_GW
        // construct matrix
        std::vector<Eigen::Triplet<double>> triplet;
        triplet.reserve(sharedVerts.size() * (1 + dim * (dim - 1)) * dim * dim);
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); ++subdomainI) {
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                triplet.emplace_back(dualIndIToShared_subdomain[subdomainI][entryI.first.first],
                                     dualIndIToShared_subdomain[subdomainI][entryI.first.second],
                                     entryI.second);
            }
        }
        for(const auto& fVI : Base::result.fixedVert) {
            auto finder = globalVIToShared.find(fVI);
            if(finder != globalVIToShared.end()) {
                int startInd = finder->second * dim;
                triplet.emplace_back(startInd, startInd, 1.0);
                triplet.emplace_back(startInd + 1, startInd + 1, 1.0);
                triplet.emplace_back(startInd + 2, startInd + 2, 1.0);
            }
        }
        consensusMtr.conservativeResize(sharedVerts.size() * dim, sharedVerts.size() * dim);
        consensusMtr.setFromTriplets(triplet.begin(), triplet.end());
        
        // factorize
        consensusSolver->set_pattern(consensusMtr);
        consensusSolver->analyze_pattern();
        consensusSolver->factorize();
        
        // record matrix entry locations for fast access
        CM_elemPtr.resize(0);
        CM_elemPtr.reserve(triplet.size());
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); ++subdomainI) {
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                CM_elemPtr.emplace_back(&consensusMtr.coeffRef(dualIndIToShared_subdomain[subdomainI][entryI.first.first], dualIndIToShared_subdomain[subdomainI][entryI.first.second]));
            }
        }
        for(const auto& fVI : Base::result.fixedVert) {
            auto finder = globalVIToShared.find(fVI);
            if(finder != globalVIToShared.end()) {
                int startInd = finder->second * dim;
                CM_elemPtr.emplace_back(&consensusMtr.coeffRef(startInd, startInd));
                CM_elemPtr.emplace_back(&consensusMtr.coeffRef(startInd + 1, startInd + 1));
                CM_elemPtr.emplace_back(&consensusMtr.coeffRef(startInd + 2, startInd + 2));
            }
        }
#endif
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::updateConsensusSolver(void)
    {
        timer_temp3.start(4);
#ifdef USE_GW
        std::memset(consensusMtr.valuePtr(), 0, consensusMtr.nonZeros() * sizeof(double));
        int elemPtrI = 0;
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                *CM_elemPtr[elemPtrI++] += entryI.second;
            }
        }
        while(elemPtrI < CM_elemPtr.size()) {
            *CM_elemPtr[elemPtrI++] = 1.0;
        }
        consensusSolver->update_a(consensusMtr);
        consensusSolver->factorize();
#endif
        timer_temp3.stop();
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::subdomainSolve(int localMaxIter,
                                                int localMinIter,
                                                bool updateH,
                                                bool linesearch) // local solve
    {
        timer_temp3.start(5);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            double localTol = ((localMaxIter <= 1) ? 0.0 :
                               Base::computeCharNormSq(mesh_subdomain[subdomainI],
                                                       Base::energyTerms[0],
                                                       Base::relGL2Tol)); // exact tol

            if(Base::animConfig.inexactSolve) {
                double ratio = sqn_g / Base::targetGRes / 4.0;
                if(ratio > 1.0) {
                    localTol *= ratio;
                }
                
                if(localTol > tol_subdomain[subdomainI]) {
                    localTol = tol_subdomain[subdomainI];
                }
                else {
                    tol_subdomain[subdomainI] = localTol;
                }
            }
            
            // primal
            int j = 0;
            for(; j < localMaxIter; j++) {
//                std::string filePath_pre = (outputFolderPath + std::to_string(subdomainI) +
//                                            "_" + std::to_string(j));
//                writeMeshToFile(filePath_pre);
                
                timer_sbd[subdomainI].start(0);
                Eigen::VectorXd g;
                computeGradient_subdomain(subdomainI, false, g); // i = 0 also no redoSVD because of dual init
                timer_sbd[subdomainI].stop();
                double sqn_g_local;
                sqn_g_local = g.squaredNorm();
//                if(subdomainI == 0) {
//                    std::cout << "  " << subdomainI << "-" << j << " ||g_local||^2 = "
//                        << sqn_g_local << std::endl;
//                }
                
                if((sqn_g_local < localTol) && (j >= localMinIter)) {
                    break;
                }
                
                if(updateH) {
                    timer_sbd[subdomainI].start(1);
                    computeHessianProxy_subdomain(subdomainI, false, true);
                    timer_sbd[subdomainI].stop();
                }
                
                // solve for search direction
                Eigen::VectorXd p;
                timer_sbd[subdomainI].start(2);
                if(useDense) {
                    if(updateH) {
                        denseSolver_sbd[subdomainI].compute(hessian_sbd[subdomainI]);
                    }
                    p = denseSolver_sbd[subdomainI].solve(-g);
                }
                else {
                    Eigen::VectorXd rhs = -g;
                    if(updateH) {
                        linSysSolver_subdomain[subdomainI]->factorize();
                    }
                    linSysSolver_subdomain[subdomainI]->solve(rhs, p);
                }
                timer_sbd[subdomainI].stop();
                
                if(linesearch) {
                    timer_sbd[subdomainI].start(3);
                    // line search init
                    double alpha = 1.0;
                    
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
    //                if(subdomainI == 0) {
    //                    std::cout << "stepsize = " << alpha << std::endl;
    //                }
                    timer_sbd[subdomainI].stop();
                }
                else {
                    for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); ++vI) {
                        mesh_subdomain[subdomainI].V.row(vI) += p.segment<dim>(vI * dim).transpose();
                    }
                    double E;
                    computeEnergyVal_subdomain(subdomainI, true, E);
                }
            }
            
            localIterCount_sbd[subdomainI] += j;
            if(j >= localMaxIter) {
                logFile << "!!! maxIter reached for subdomain " << subdomainI << std::endl;
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
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); ++subdomainI) {
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                Eigen::Matrix<double, 1, dim> diff = (mesh_subdomain[subdomainI].V.row(localI) -
                                                      Base::result.V.row(dualMapperI.first));
                sqn_r += weights_subdomain[subdomainI].row(dualMapperI.second).
                    cwiseProduct(diff).dot(diff);
                
                sqn_s += weights_subdomain[subdomainI].row(dualMapperI.second).cwiseProduct
                    (dz_subdomain[subdomainI].row(dualMapperI.second)).squaredNorm();
                
                //TODO: GW
            }
        }
        std::cout << "||s|| = " << std::sqrt(sqn_s) << ", ||r|| = " << std::sqrt(sqn_r) << ", ";
        Base::file_iterStats << std::sqrt(sqn_s) << " " << std::sqrt(sqn_r) << " ";
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::boundaryConsensusSolve(double relaxParam) // global solve
    {
        timer_temp3.start(6);
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); ++subdomainI) {
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                dz_subdomain[subdomainI].row(dualMapperI.second) =
                    Base::result.V.row(dualMapperI.first);
            }
        }
#ifndef USE_GW
        Base::result.V.setZero();
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(isSharedVert[mapperI.first] &&
                   (!Base::result.isFixedVert[mapperI.first]))
                {
                    int dualI = globalVIToDual_subdomain[subdomainI][mapperI.first];
                    Base::result.V.row(mapperI.first) +=
                        weights_subdomain[subdomainI].row(dualI).cwiseProduct
                            (relaxParam * mesh_subdomain[subdomainI].V.row(mapperI.second) +
                             (1.0 - relaxParam) * resultVk.row(mapperI.first) +
                             u_subdomain[subdomainI].row(dualI));
                }
                else {
                    Base::result.V.row(mapperI.first) = mesh_subdomain[subdomainI].V.row(mapperI.second);
                }
            }
        }
        for(int vI = 0; vI < Base::result.V.rows(); vI++) {
            if(isSharedVert[vI] &&
               (!Base::result.isFixedVert[vI]))
            {
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
                    (relaxParam * mesh_subdomain[subdomainI].V.row(localVIFinder->second) +
                     (1.0 - relaxParam) * resultVk.row(localVIFinder->first) +
                     u_subdomain[subdomainI].row(dualMapperI.second) -
                     Base::result.V.row(localVIFinder->first)).transpose();
                // changed to solve for dz, which is simpler
            }
            for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                rhs[dualIndIToShared_subdomain[subdomainI][entryI.first.first]] +=
                    entryI.second * augVec[entryI.first.second];
            }
        }
        
        for(const auto& fVI : Base::result.fixedVert) {
            auto finder = globalVIToShared.find(fVI);
            if(finder != globalVIToShared.end()) {
                rhs.segment<dim>(finder->second * dim).setZero();
                // for fixed nodes, project dz to 0
            }
        }
        
        consensusSolver->solve(rhs, solvedSharedVerts);
        for(int svI = 0; svI < sharedVerts.size(); svI++) {
            Base::result.V.row(sharedVerts[svI]) += solvedSharedVerts.segment<dim>(svI * dim).transpose();
            // += dz
        }
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++) {
            for(const auto& mapperI : globalVIToLocal_subdomain[subdomainI]) {
                if(!isSharedVert[mapperI.first]) {
                    Base::result.V.row(mapperI.first) = mesh_subdomain[subdomainI].V.row(mapperI.second);
                }
            }
        }
#endif
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); ++subdomainI) {
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                dz_subdomain[subdomainI].row(dualMapperI.second) -=
                    Base::result.V.row(dualMapperI.first);
            }
        }
        
        timer_temp3.stop();
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::dualSolve(double stepSize, double relaxParam)
    {
        timer_temp3.start(6);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                int localI = globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                du_subdomain[subdomainI].row(dualMapperI.second) =
                    stepSize * (relaxParam * mesh_subdomain[subdomainI].V.row(localI) +
                                (1.0 - relaxParam) * resultVk.row(dualMapperI.first)
                                - Base::result.V.row(dualMapperI.first));
                u_subdomain[subdomainI].row(dualMapperI.second) += du_subdomain[subdomainI].row(dualMapperI.second);
            }
        }
#ifdef USE_TBB
        );
#endif
        timer_temp3.stop();
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::uToY(void)
    {
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            //TODO: GW
            for(int dualI = 0; dualI < u_subdomain[subdomainI].rows(); ++dualI) {
                y_subdomain[subdomainI](dualI, 0) = std::sqrt(weights_subdomain[subdomainI](dualI, 0)) * u_subdomain[subdomainI](dualI, 0);
                y_subdomain[subdomainI](dualI, 1) = std::sqrt(weights_subdomain[subdomainI](dualI, 1)) * u_subdomain[subdomainI](dualI, 1);
                if(dim == 3) {
                    y_subdomain[subdomainI](dualI, 2) = std::sqrt(weights_subdomain[subdomainI](dualI, 2)) * u_subdomain[subdomainI](dualI, 2);
                }
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    template<int dim>
    void ADMMDDTimeStepper<dim>::yToU(void)
    {
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < mesh_subdomain.size(); subdomainI++)
#endif
        {
            //TODO: GW
            for(int dualI = 0; dualI < u_subdomain[subdomainI].rows(); ++dualI) {
                u_subdomain[subdomainI](dualI, 0) = y_subdomain[subdomainI](dualI, 0) / std::sqrt(weights_subdomain[subdomainI](dualI, 0));
                u_subdomain[subdomainI](dualI, 1) = y_subdomain[subdomainI](dualI, 1) / std::sqrt(weights_subdomain[subdomainI](dualI, 1));
                if(dim == 3) {
                    u_subdomain[subdomainI](dualI, 2) = y_subdomain[subdomainI](dualI, 2) / std::sqrt(weights_subdomain[subdomainI](dualI, 2));
                }
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::computeError(const Eigen::VectorXd& residual,
                                              double& err_in, double& err_b) const
    {
        assert(residual.size() == dim * isSharedVert.size());
        
        err_in = 0.0;
        err_b = 0.0;
        double maxErr_in = 0.0, maxErr_b = 0.0;
        double minErr_in = __DBL_MAX__, minErr_b = __DBL_MAX__;
        for(int vI = 0; vI < isSharedVert.size(); ++vI) {
            double errI = residual.segment<dim>(vI * dim).norm();
            if(isSharedVert[vI]) {
                err_b += errI;
                if(maxErr_b < errI) {
                    maxErr_b = errI;
                }
                if(minErr_b > errI) {
                    minErr_b = errI;
                }
            }
            else {
                err_in += errI;
                if(maxErr_in < errI) {
                    maxErr_in = errI;
                }
                if(minErr_in > errI) {
                    minErr_in = errI;
                }
            }
        }
        err_in /= (isSharedVert.size() - sharedVerts.size());
        err_b /= sharedVerts.size();
        
        logFile << err_b << " " << minErr_b << " " << maxErr_b << " " <<
            err_in << " " << minErr_in << " " << maxErr_in << std::endl;
    }
    
    // subdomain energy computation
    template<int dim>
    void ADMMDDTimeStepper<dim>::computeEnergyVal_subdomain(int subdomainI, bool redoSVD, double& Ei)
    {
        // incremental potential:
        Base::energyTerms[0]->computeEnergyVal(mesh_subdomain[subdomainI], redoSVD,
                                               svd_subdomain[subdomainI], F_subdomain[subdomainI],
                                               U_sbd[subdomainI], V_sbd[subdomainI], Sigma_sbd[subdomainI],
                                               Base::dtSq, Ei);
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
                                              svd_subdomain[subdomainI], F_subdomain[subdomainI],
                                              U_sbd[subdomainI], V_sbd[subdomainI], Sigma_sbd[subdomainI],
                                              Base::dtSq, g);
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
                                                               bool redoSVD,
                                                               bool augmentLag)
    {
        if(useDense) {
            // incremental potential:
            hessian_sbd[subdomainI].setZero();
            Base::energyTerms[0]->computeHessian(mesh_subdomain[subdomainI], redoSVD,
                                                 svd_subdomain[subdomainI],
                                                 F_subdomain[subdomainI], Base::dtSq,
                                                 hessian_sbd[subdomainI]);
            
            for(int vI = 0; vI < mesh_subdomain[subdomainI].V.rows(); vI++) {
                hessian_sbd[subdomainI].diagonal().segment(vI * dim, dim).array() +=
                    mesh_subdomain[subdomainI].massMatrix.coeff(vI, vI);
            }
            
            if(augmentLag) {
            // augmented Lagrangian:
#ifndef USE_GW
                for(const auto& dualMapperI : globalVIToDual_subdomain[subdomainI]) {
                    auto localVIFinder = globalVIToLocal_subdomain[subdomainI].find(dualMapperI.first);
                    assert(localVIFinder != globalVIToLocal_subdomain[subdomainI].end());
                    hessian_sbd[subdomainI].diagonal().segment(localVIFinder->second * dim, dim) +=
                        weights_subdomain[subdomainI].row(dualMapperI.second).transpose();
                }
#else
                for(const auto& entryI : weightMtr_subdomain[subdomainI]) {
                    hessian_sbd[subdomainI](dualIndIToLocal_subdomain[subdomainI][entryI.first.first],
                                            dualIndIToLocal_subdomain[subdomainI][entryI.first.second]) +=
                        entryI.second;
                }
#endif
            }
        }
        else {
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
            
            if(augmentLag) {
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
        }
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::extract(const Eigen::VectorXd& src_g,
                                         int sbdI,
                                         Eigen::VectorXd& dst_l) const
    {
        assert(src_g.size() == Base::result.V.rows() * dim);
        
        dst_l.conservativeResize(mesh_subdomain[sbdI].V.rows() * dim);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain[sbdI].V.rows(), 1, [&](int localVI)
#else
        for(int localVI = 0; localVI < mesh_subdomain[sbdI].V.rows(); ++localVI)
#endif
        {
            dst_l.segment<dim>(localVI * dim) =
                src_g.segment<dim>(localVIToGlobal_subdomain[sbdI][localVI] * dim);
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    void ADMMDDTimeStepper<dim>::fill(int sbdI, const Eigen::VectorXd& src_l,
                                      Eigen::VectorXd& dst_g) const
    {
        assert(src_l.size() == mesh_subdomain[sbdI].V.rows() * dim);
        
        dst_g.conservativeResize(Base::result.V.rows() * dim);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh_subdomain[sbdI].V.rows(), 1, [&](int localVI)
#else
        for(int localVI = 0; localVI < mesh_subdomain[sbdI].V.rows(); ++localVI)
#endif
        {
             dst_g.segment<dim>(localVIToGlobal_subdomain[sbdI][localVI] * dim) =
                src_l.segment<dim>(localVI * dim);
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template class ADMMDDTimeStepper<DIM>;
    
}
