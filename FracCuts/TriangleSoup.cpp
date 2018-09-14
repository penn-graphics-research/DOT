//
//  TriangleSoup.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "TriangleSoup.hpp"
#include "IglUtils.hpp"
#include "SymStretchEnergy.hpp"
#include "Optimizer.hpp"
#include "Timer.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/writeOBJ.h>
#include <igl/list_to_matrix.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <fstream>

extern std::ofstream logFile;
extern Timer timer_step;

extern std::vector<std::pair<double, double>> energyChanges_bSplit, energyChanges_iSplit, energyChanges_merge;
extern std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
extern std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit, newVertPoses_merge;

namespace FracCuts {
    
    TriangleSoup::TriangleSoup(void)
    {
        initSeamLen = 0.0;
    }
    
    TriangleSoup::TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                               const Eigen::MatrixXd& UV_mesh, const Eigen::MatrixXi& FUV_mesh,
                               bool separateTri, double p_initSeamLen, double p_areaThres_AM)
    {
        initSeamLen = p_initSeamLen;
        areaThres_AM = p_areaThres_AM;
        
        bool multiComp = false; //TODO: detect whether the mesh is multi-component
        if(separateTri)
        {
            // duplicate vertices and edges, use new face vertex indices,
            // construct cohesive edge pairs,
            // compute triangle matrix to save rest shapes
            V_rest.resize(F_mesh.rows() * F_mesh.cols(), 3);
            V.resize(F_mesh.rows() * F_mesh.cols(), 2);
            F.resize(F_mesh.rows(), F_mesh.cols());
            std::map<std::pair<int, int>, Eigen::Vector3i> edge2DupInd;
            int cohEAmt = 0;
            for(int triI = 0; triI < F_mesh.rows(); triI++)
            {
                int vDupIndStart = triI * 3;
                
                if(UV_mesh.rows() == V_mesh.rows()) {
                    // bijective map without seams, usually Tutte
                    V.row(vDupIndStart) = UV_mesh.row(F_mesh.row(triI)[0]);
                    V.row(vDupIndStart + 1) = UV_mesh.row(F_mesh.row(triI)[1]);
                    V.row(vDupIndStart + 2) = UV_mesh.row(F_mesh.row(triI)[2]);
                }
                
//                // perturb for testing separation energy
//                V.row(vDupIndStart + 1) = V.row(vDupIndStart) + 0.5 * (V.row(vDupIndStart + 1) - V.row(vDupIndStart));
//                V.row(vDupIndStart + 2) = V.row(vDupIndStart) + 0.5 * (V.row(vDupIndStart + 2) - V.row(vDupIndStart));
                
                V_rest.row(vDupIndStart) = V_mesh.row(F_mesh.row(triI)[0]);
                V_rest.row(vDupIndStart + 1) = V_mesh.row(F_mesh.row(triI)[1]);
                V_rest.row(vDupIndStart + 2) = V_mesh.row(F_mesh.row(triI)[2]);
                
                F(triI, 0) = vDupIndStart;
                F(triI, 1) = vDupIndStart + 1;
                F(triI, 2) = vDupIndStart + 2;
                
                for(int vI = 0; vI < 3; vI++)
                {
                    int vsI = F_mesh.row(triI)[vI], veI = F_mesh.row(triI)[(vI + 1) % 3];
                    auto cohEFinder = edge2DupInd.find(std::pair<int, int>(veI, vsI));
                    if(cohEFinder == edge2DupInd.end()) {
                        cohEAmt++;
                        edge2DupInd[std::pair<int, int>(vsI, veI)] = Eigen::Vector3i(cohEAmt, F(triI, vI), F(triI, (vI + 1) % 3));
                    }
                    else {
                        edge2DupInd[std::pair<int, int>(vsI, veI)] = Eigen::Vector3i(-cohEFinder->second[0], F(triI, vI), F(triI, (vI + 1) % 3));
                    }
                }
            }
            
            cohE.resize(cohEAmt, 4);
            cohE.setConstant(-1);
            for(const auto& cohPI : edge2DupInd) {
                if(cohPI.second[0] > 0) {
                    cohE.row(cohPI.second[0] - 1)[0] = cohPI.second[1];
                    cohE.row(cohPI.second[0] - 1)[1] = cohPI.second[2];
                }
                else {
                    cohE.row(-cohPI.second[0] - 1)[2] = cohPI.second[2];
                    cohE.row(-cohPI.second[0] - 1)[3] = cohPI.second[1];
                }
            }
//            std::cout << cohE << std::endl;
            
            if(UV_mesh.rows() == 0) {
                // no input UV
                initRigidUV();
            }
            else if(UV_mesh.rows() != V_mesh.rows()) {
                // input UV with seams
                assert(0 && "TODO: separate each triangle in UV space according to FUV!");
            }
        }
        else {
            // deal with mesh
            if(UV_mesh.rows() == V_mesh.rows()) {
                // same vertex and uv index
                V_rest = V_mesh;
                V = UV_mesh;
                F = F_mesh;
            }
            else if(UV_mesh.rows() != 0) {
                // different vertex and uv index, split 3D surface according to UV and merge back while saving into files
                assert(F_mesh.rows() == FUV_mesh.rows());
                // UV map contains seams
                // Split triangles along the seams on the surface (construct cohesive edges there)
                // to construct a bijective map
                std::set<std::pair<int, int>> HE_UV;
                std::map<std::pair<int, int>, std::pair<int, int>> HE;
                for(int triI = 0; triI < FUV_mesh.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd_UV = FUV_mesh.row(triI);
                    HE_UV.insert(std::pair<int, int>(triVInd_UV[0], triVInd_UV[1]));
                    HE_UV.insert(std::pair<int, int>(triVInd_UV[1], triVInd_UV[2]));
                    HE_UV.insert(std::pair<int, int>(triVInd_UV[2], triVInd_UV[0]));
                    const Eigen::RowVector3i& triVInd = F_mesh.row(triI);
                    HE[std::pair<int, int>(triVInd[0], triVInd[1])] = std::pair<int, int>(triI, 0);
                    HE[std::pair<int, int>(triVInd[1], triVInd[2])] = std::pair<int, int>(triI, 1);
                    HE[std::pair<int, int>(triVInd[2], triVInd[0])] = std::pair<int, int>(triI, 2);
                }
                std::vector<std::vector<int>> cohEdges;
                for(int triI = 0; triI < FUV_mesh.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd_UV = FUV_mesh.row(triI);
                    const Eigen::RowVector3i& triVInd = F_mesh.row(triI);
                    for(int eI = 0; eI < 3; eI++) {
                        int vI = eI, vI_post = (eI + 1) % 3;
                        if(HE_UV.find(std::pair<int, int>(triVInd_UV[vI_post], triVInd_UV[vI])) == HE_UV.end()) {
                            // boundary edge in UV space
                            const auto finder = HE.find(std::pair<int, int>(triVInd[vI_post], triVInd[vI]));
                            if(finder != HE.end()) {
                                // non-boundary edge on the surface
                                // construct cohesive edge pair
                                cohEdges.resize(cohEdges.size() + 1);
                                cohEdges.back().emplace_back(triVInd_UV[vI]);
                                cohEdges.back().emplace_back(triVInd_UV[vI_post]);
                                cohEdges.back().emplace_back(FUV_mesh(finder->second.first, (finder->second.second + 1) % 3));
                                cohEdges.back().emplace_back(FUV_mesh(finder->second.first, finder->second.second));
                                HE.erase(std::pair<int, int>(triVInd[vI], triVInd[vI_post])); // prevent from inserting again
                            }
                        }
                    }
                }
                bool makeCoh = true;
                if(makeCoh) {
                    igl::list_to_matrix(cohEdges, cohE);
                }
                
                V_rest.resize(UV_mesh.rows(), 3);
                V = UV_mesh;
                F = FUV_mesh;
                std::vector<bool> updated(UV_mesh.rows(), false);
                for(int triI = 0; triI < F_mesh.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd = F_mesh.row(triI);
                    const Eigen::RowVector3i& triVInd_UV = FUV_mesh.row(triI);
                    for(int vI = 0; vI < 3; vI++) {
                        if(!updated[triVInd_UV[vI]]) {
                            V_rest.row(triVInd_UV[vI]) = V_mesh.row(triVInd[vI]);
                            updated[triVInd_UV[vI]] = true;
                        }
                    }
                }
                
                if(!makeCoh) {
                    for(const auto& cohI : cohEdges) {
                        initSeamLen += (V_rest.row(cohI[0]) - V_rest.row(cohI[1])).norm();
                    }
                }
                else {
                    initSeams = cohE;
                }
            }
            else {
                assert(V_mesh.rows() > 0);
                assert(F_mesh.rows() > 0);
                V_rest = V_mesh;
                F = F_mesh;
                V = Eigen::MatrixXd::Zero(V_rest.rows(), 2);
                std::cout << "No UV provided, initialized to all 0" << std::endl;
            }
        }
        
        triWeight = Eigen::VectorXd::Ones(F.rows());
        computeFeatures(false, true);
        
        vertWeight = Eigen::VectorXd::Ones(V.rows());
        //TEST: compute gaussian curvature for regional seam placement
//        vertWeight = Eigen::VectorXd::Ones(V.rows()) * 2.0 * M_PI;
//        for(int triI = 0; triI < F.rows(); triI++) {
//            const Eigen::RowVector3i& triVInd = F.row(triI);
//            const Eigen::RowVector3d v[3] = {
//                V_rest.row(triVInd[0]),
//                V_rest.row(triVInd[1]),
//                V_rest.row(triVInd[2])
//            };
//            for(int vI = 0; vI < 3; vI++) {
//                int vI_post = (vI + 1) % 3;
//                int vI_pre = (vI + 2) % 3;
//                const Eigen::RowVector3d e0 = v[vI_pre] - v[vI];
//                const Eigen::RowVector3d e1 = v[vI_post] - v[vI];
//                vertWeight[triVInd[vI]] -= std::acos(std::max(-1.0, std::min(1.0, e0.dot(e1) / e0.norm() / e1.norm())));
//            }
//        }
//        const double maxRatio = 4.0;
//        for(int vI = 0; vI < V.rows(); vI++) {
//            if(isBoundaryVert(vI)) {
//                vertWeight[vI] -= M_PI;
//            }
//            if(vertWeight[vI] < 0) {
//                vertWeight[vI] = -vertWeight[vI];
//            }
//            vertWeight[vI] = maxRatio - (maxRatio - 1.0) * vertWeight[vI] / (2.0 * M_PI);
//        }
    }
    
    void initCylinder(double r1_x, double r1_y, double r2_x, double r2_y, double height, int circle_res, int height_resolution,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXd * uv_coords_per_face = NULL,
        Eigen::MatrixXi * uv_coords_face_ids = NULL)
    {
        int nvertices = circle_res * (height_resolution+1);
        int nfaces = 2*circle_res * height_resolution;
        
        V.resize(nvertices, 3);
        if(uv_coords_per_face) {
            uv_coords_per_face->resize(nvertices, 2);
        }
        F.resize(nfaces, 3);
        for (int j=0; j<height_resolution+1; j++) {
            for (int i=0; i<circle_res; i++)
            {
                double t = (double)j / (double)height_resolution;
                double h = height * t;
                double theta = i * 2*M_PI / circle_res;
                double r_x = r1_x * t + r2_x * (1-t);
                double r_y = r1_y * t + r2_y * (1-t);
                V.row(j*circle_res+i) = Eigen::Vector3d(r_x*cos(theta), height-h, r_y*sin(theta));
                if(uv_coords_per_face) {
                    uv_coords_per_face->row(j*circle_res+i) = Eigen::Vector2d(r_x*cos(theta), r_y*sin(theta));
                }
                
                if (j<height_resolution)
                {
                    int vl0 = j*circle_res+i;
                    int vl1 = j*circle_res+(i+1)%circle_res;
                    int vu0 = (j+1)*circle_res+i;
                    int vu1 = (j+1)*circle_res+(i+1)%circle_res;
                    F.row(2*(j*circle_res+i)+0) = Eigen::Vector3i(vl0, vl1, vu1);
                    F.row(2*(j*circle_res+i)+1) = Eigen::Vector3i(vu0, vl0, vu1);
                }
            }
        }
    }
    
    TriangleSoup::TriangleSoup(Primitive primitive, double size, int elemAmt, bool separateTri)
    {
        assert(size > 0.0);
        assert(elemAmt > 0);
        
        switch(primitive)
        {
            case P_GRID: {
                double spacing = size / std::sqrt(elemAmt / 2.0);
                assert(size >= spacing);
                int gridSize = static_cast<int>(size / spacing) + 1;
                spacing = size / (gridSize - 1);
                V_rest.resize(gridSize * gridSize, 3);
                V.resize(gridSize * gridSize, 2);
                borderVerts_primitive.resize(2);
                for(int rowI = 0; rowI < gridSize; rowI++)
                {
                    for(int colI = 0; colI < gridSize; colI++)
                    {
                        int vI = rowI * gridSize + colI;
                        V_rest.row(vI) = Eigen::Vector3d(spacing * colI, spacing * rowI, 0.0);
                        V.row(vI) = spacing * Eigen::Vector2d(colI, rowI);
                        
                        if(colI == 0) {
                            borderVerts_primitive[0].emplace_back(vI);
                        }
                        else if(colI == gridSize - 1) {
                            borderVerts_primitive[1].emplace_back(vI);
                        }
                    }
                }
                
                F.resize((gridSize - 1) * (gridSize - 1) * 2, 3);
                for(int rowI = 0; rowI < gridSize - 1; rowI++)
                {
                    for(int colI = 0; colI < gridSize - 1; colI++)
                    {
                        int squareI = rowI * (gridSize - 1) + colI;
                        F.row(squareI * 2) = Eigen::Vector3i(
                            rowI * gridSize + colI, (rowI + 1) * gridSize + colI + 1, (rowI + 1) * gridSize + colI);
                        F.row(squareI * 2 + 1) = Eigen::Vector3i(
                            rowI * gridSize + colI, rowI * gridSize + colI + 1, (rowI + 1) * gridSize + colI + 1);
                    }
                }
                break;
            }
                
            case P_SQUARE: {
                double spacing = size / std::sqrt(elemAmt / 2.0);
                assert(size >= spacing);
                int gridSize = static_cast<int>(size / spacing) + 1;
                spacing = size / (gridSize - 1);
                
                Eigen::MatrixXd UV_bnds(gridSize * 4 - 4, 2);
                borderVerts_primitive.resize(2);
                int vI = 0;
                for(int rowI = 0; rowI < gridSize; rowI++)
                {
                    for(int colI = 0; colI < gridSize; colI++)
                    {
                        if((colI > 0) && (colI < gridSize - 1) &&
                           (rowI > 0) && (rowI < gridSize - 1))
                        {
                            continue;
                        }
                        
                        UV_bnds.row(vI) = spacing * Eigen::Vector2d(colI, rowI);
                        
                        if(colI == 0) {
                            borderVerts_primitive[0].emplace_back(vI);
                        }
                        else if(colI == gridSize - 1) {
                            borderVerts_primitive[1].emplace_back(vI);
                        }
                        
                        vI++;
                    }
                }
                
                Eigen::MatrixXi E(gridSize * 4 - 4, 2);
                for(int colI = 0; colI + 1 < gridSize; colI++) {
                    E.row(colI) << colI, colI + 1;
                }
                for(int rowI = 0; rowI + 1 < gridSize; rowI++) {
                    E.row(gridSize - 1 + rowI) <<
                        borderVerts_primitive[1][rowI],
                        borderVerts_primitive[1][rowI + 1];
                }
                for(int colI = 0; colI + 1 < gridSize; colI++) {
                    E.row(gridSize * 2 - 2 + colI) << UV_bnds.rows() - 1 - colI, UV_bnds.rows() - 2 - colI;
                }
                for(int rowI = 0; rowI + 1 < gridSize; rowI++) {
                    E.row(gridSize * 3 - 3 + rowI) <<
                        borderVerts_primitive[0][gridSize - 1 - rowI],
                        borderVerts_primitive[0][gridSize - 2 - rowI];
                }
                
                std::string flag("qQa" + std::to_string(size * size / elemAmt * 1.1));
                // "q" for high quality mesh generation
                // "Q" for quiet mode (no output)
                // "a" for area upper bound
                
                igl::triangle::triangulate(UV_bnds, E, Eigen::MatrixXd(), flag, V, F);
                
                V_rest.resize(V.rows(), 3);
                V_rest.leftCols(2) = V;
                V_rest.rightCols(1).setZero();
                
                break;
            }
                
            case P_SPIKES: {
                Eigen::MatrixXd UV_bnds_temp(7, 2);
                UV_bnds_temp <<
                    0.0, 0.0,
                    1.0, 0.0,
                    0.8, 0.7,
                    1.0, 1.0,
                    0.7, 0.9,
                    0.0, 1.0,
                    0.25, 0.4;
                UV_bnds_temp *= size;
                
                double spacing = std::sqrt(size * size * 0.725 / elemAmt * 4.0 / std::sqrt(3.0));
                Eigen::MatrixXd inBetween1, inBetween2, inBetween5, inBetween6;
                IglUtils::sampleSegment(UV_bnds_temp.row(1), UV_bnds_temp.row(2), spacing, inBetween1);
                IglUtils::sampleSegment(UV_bnds_temp.row(2), UV_bnds_temp.row(3), spacing, inBetween2);
                IglUtils::sampleSegment(UV_bnds_temp.row(5), UV_bnds_temp.row(6), spacing, inBetween5);
                IglUtils::sampleSegment(UV_bnds_temp.row(6), UV_bnds_temp.row(0), spacing, inBetween6);
                
                Eigen::MatrixXd UV_bnds(UV_bnds_temp.rows() + inBetween1.rows() + inBetween2.rows() +
                                        inBetween5.rows() + inBetween6.rows(), 2);
                UV_bnds <<
                    UV_bnds_temp.row(0),
                    UV_bnds_temp.row(1),
                    inBetween1,
                    UV_bnds_temp.row(2),
                    inBetween2,
                    UV_bnds_temp.row(3),
                    UV_bnds_temp.row(4),
                    UV_bnds_temp.row(5),
                    inBetween5,
                    UV_bnds_temp.row(6),
                    inBetween6;
                
                borderVerts_primitive.resize(2);
                
                borderVerts_primitive[0].emplace_back(0);
                for(int i = 0; i < inBetween6.rows(); i++) {
                    borderVerts_primitive[0].emplace_back(6 + inBetween1.rows() + inBetween2.rows() +
                                                          inBetween5.rows() + inBetween6.rows() - i);
                }
                borderVerts_primitive[0].emplace_back(6 + inBetween1.rows() + inBetween2.rows() +
                                                      inBetween5.rows());
                for(int i = 0; i < inBetween5.rows(); i++) {
                    borderVerts_primitive[0].emplace_back(6 + inBetween1.rows() + inBetween2.rows() +
                                                          inBetween5.rows() - 1 - i);
                }
                borderVerts_primitive[0].emplace_back(5 + inBetween1.rows() + inBetween2.rows());
                
                borderVerts_primitive[1].emplace_back(1);
                for(int i = 0; i < inBetween1.rows(); i++) {
                    borderVerts_primitive[1].emplace_back(1 + 1 + i);
                }
                borderVerts_primitive[1].emplace_back(2 + inBetween1.rows());
                for(int i = 0; i < inBetween2.rows(); i++) {
                    borderVerts_primitive[1].emplace_back(2 + inBetween1.rows() + 1 + i);
                }
                borderVerts_primitive[1].emplace_back(3 + inBetween1.rows() + inBetween2.rows());
                
                Eigen::MatrixXi E(UV_bnds.rows(), 2);
                for(int i = 0; i < E.rows(); i++) {
                    E.row(i) << i, i + 1;
                }
                E(E.rows() - 1, 1) = 0;
                
                std::string flag("qQa" + std::to_string(size * size * 0.725 / elemAmt * 1.1));
                // "q" for high quality mesh generation
                // "Q" for quiet mode (no output)
                // "a" for area upper bound
                
                igl::triangle::triangulate(UV_bnds, E, Eigen::MatrixXd(), flag, V, F);
                
                V_rest.resize(V.rows(), 3);
                V_rest.leftCols(2) = V;
                V_rest.rightCols(1).setZero();
                break;
            }
                
            case P_SHARKEY: {
                V.resize(406,2);
                V <<581,1904 , 580,1896 , 610,1872 , 658,1847 , 597,1837 , 577,1823 , 592,1732 , 598,1728 , 606,1735 , 610,1717 , 631,1723 , 617,1705 , 626,1680 , 637,1674 , 632,1667 , 609,1674 , 597,1504 , 589,1499 , 549,1520 , 504,1562 , 492,1564 , 491,1557 , 548,1508 , 607,1485 , 597,1401 , 454,1353 , 228,1301 , 220,1301 , 192,1336 , 175,1337 , 157,1306 , 138,1322 , 124,1322 , 111,1285 , 121,1066 , 154,1065 , 188,1038 , 323,1037 , 471,1061 , 591,1093 , 582,1051 , 509,980 , 473,954 , 436,946 , 453,914 , 398,887 , 358,877 , 312,879 , 311,867 , 300,869 , 271,890 , 253,931 , 246,930 , 239,886 , 232,883 , 202,939 , 200,966 , 185,963 , 177,972 , 176,1010 , 160,1010 , 151,999 , 163,949 , 180,935 , 213,868 , 207,831 , 220,794 , 234,770 , 264,761 , 269,745 , 263,737 , 284,696 , 310,698 , 315,707 , 313,725 , 284,779 , 305,798 , 315,801 , 319,789 , 356,806 , 476,889 , 499,872 , 495,799 , 505,749 , 524,707 , 482,682 , 454,654 , 420,599 , 413,604 , 420,618 , 398,631 , 388,589 , 330,648 , 267,673 , 219,683 , 149,683 , 122,678 , 119,669 , 172,641 , 224,598 , 317,548 , 356,505 , 384,438 , 393,433 , 402,436 , 413,461 , 427,546 , 473,579 , 450,599 , 472,639 , 503,670 , 541,690 , 603,662 , 589,605 , 590,533 , 542,500 , 466,422 , 428,356 , 417,287 , 438,286 , 466,304 , 607,430 , 644,436 , 679,457 , 679,397 , 602,372 , 556,324 , 541,283 , 539,223 , 556,150 , 580,116 , 638,78 , 711,67 , 766,80 , 822,128 , 849,195 , 853,263 , 830,322 , 794,364 , 758,386 , 701,400 , 701,650 , 803,661 , 811,686 , 826,691 , 854,683 , 956,620 , 983,617 , 993,623 , 992,636 , 961,658 , 960,707 , 893,719 , 913,774 , 857,795 , 889,856 , 837,868 , 936,906 , 958,932 , 960,974 , 944,982 , 945,954 , 929,934 , 896,931 , 872,941 , 843,962 , 808,1003 , 795,1066 , 903,1091 , 1002,1129 , 1024,1131 , 1023,1141 , 1061,1144 , 1087,1098 , 1112,1028 , 1162,1000 , 1183,999 , 1238,1012 , 1255,1028 , 1302,1047 , 1293,1058 , 1300,1070 , 1295,1094 , 1306,1103 , 1324,1101 , 1338,1125 , 1366,1120 , 1384,1131 , 1393,1122 , 1396,1094 , 1402,1094 , 1440,1113 , 1467,1107 , 1506,1124 , 1520,1134 , 1521,1153 , 1502,1198 , 1481,1214 , 1434,1320 , 1420,1313 , 1412,1330 , 1397,1321 , 1401,1340 , 1390,1344 , 1372,1329 , 1353,1338 , 1353,1354 , 1379,1364 , 1377,1386 , 1365,1392 , 1369,1406 , 1353,1416 , 1353,1428 , 1385,1441 , 1345,1522 , 1349,1545 , 1334,1576 , 1314,1604 , 1249,1605 , 1168,1561 , 970,1474 , 936,1450 , 938,1422 , 1007,1282 , 999,1269 , 1002,1233 , 988,1215 , 976,1224 , 952,1185 , 889,1132 , 833,1104 , 793,1099 , 793,1120 , 807,1149 , 795,1158 , 803,1317 , 799,1377 , 793,1398 , 785,1397 , 783,1453 , 772,1489 , 836,1506 , 872,1528 , 883,1548 , 876,1550 , 859,1532 , 825,1514 , 789,1505 , 771,1636 , 779,1708 , 805,1820 , 706,1839 , 677,650 , 679,473 , 619,522 , 617,538 , 623,634 , 632,647 , 666,650 , 669.37,1484.16 , 256.722,1205.87 , 710.262,878.775 , 718.791,241.321 , 577.5,1496.5 , 701,525 , 643.5,1871.5 , 1399,1108.48 , 246.841,858.039 , 252.926,906.516 , 224,1270.76 , 406.174,587.043 , 612.25,1887.75 , 660.961,1651.6 , 595.195,1896.62 , 286.678,725.492 , 299.888,833.205 , 678,561.5 , 671.5,691.582 , 677.362,618.002 , 519.5,1532.5 , 643.888,1699.71 , 569,1509.5 , 1422.52,1160.52 , 1307.64,1422 , 584.5,1777.5 , 799.599,745.203 , 701,462.5 , 1266.48,1140.08 , 1037.77,1254.11 , 804,1497.5 , 1459.94,1261.49 , 1292.52,1346 , 1446.97,1290.75 , 700.754,319.635 , 713.616,1109.5 , 1176.28,1078.88 , 1328.48,1261.68 , 596.763,762.763 , 604.494,1610.16 , 748.024,692.372 , 727.895,1342.13 , 1035.75,1192.31 , 720.017,1561.26 , 1301.91,1541.34 , 708.569,1750.71 , 1389.02,1260.99 , 312.104,1274.49 , 643.231,1780.52 , 115.161,1193.87 , 139.482,1130.91 , 642.891,185.864 , 1127.83,1384.35 , 1146.82,1219.61 , 679,427 , 701,431.25 , 701,415.625 , 701,446.875 , 685.185,412 , 685.426,442 , 678.5,517.25 , 620,586 , 701,587.5 , 677.681,589.751 , 701,556.25 , 701,618.75 , 637.57,1438.77 , 712.014,1420.46 , 716.467,1016.95 , 764.818,1026.93 , 746.172,1072.53 , 382.507,1161.01 , 717.531,1176.74 , 341,1327 , 284.5,1314 , 315.953,1306.58 , 408.442,1255.68 , 661.491,1369.44 , 566.071,1256.13 , 666.133,1286.34 , 599.629,1324.7 , 683.678,1329 , 327.459,1205.44 , 291.585,1122.11 , 378.918,1212.88 , 799,1237.5 , 651.037,1187.49 , 525.5,1377 , 510.035,1304.57 , 452.344,1294.45 , 400.904,1325.2 , 550.527,1333.79 , 615.073,1274.65 , 681.204,1233.46 , 616.319,942.686 , 752.497,946.107 , 684.119,955.917 , 644.661,1015.92 , 658.982,904.398 , 740.246,1231.28 , 761.813,1190.3 , 764.897,1279.07 , 762.206,1319.87 , 716.106,1290.73 , 487.465,1183.15 , 505.436,1242.61 , 397,1049 , 425.148,1109.59 , 351.43,1095.85 , 649.495,1122.15 , 572.012,988.241 , 569.564,1173.55 , 604.163,1216.38 , 521.415,1117.61 , 629.722,1062.74 , 435.268,1205.59 , 697.732,1062.69 , 561.25,1389 , 590.012,1362.57 , 561.42,1360.32 , 612.142,1031.34 , 625.439,1097.16 , 404.421,1188.59 , 726.02,1047.57 , 677.57,1095.63 , 680.385,1028.88 , 545.5,1015.5 , 602,1443 , 397.5,1340 , 472.446,1220.13 , 469.896,1259.32 , 444.884,1156.04 , 473.026,1116.67 , 412.412,1142.42 , 496.652,1146.34 , 440.179,1240.45 , 564.625,1129.11 , 531,1077 , 557.926,1096.53 , 495.943,1087.96 , 561,1085 , 574.463,1094.76 , 544.463,1086.76 , 603.01,1068.46 , 608.786,1146.69 , 641.561,1155.03 , 586.5,1072;
                
                F.resize(515,3);
                F <<99,94,98 , 100,92,99 , 92,100,91 , 93,99,92 , 96,97,95 , 93,94,99 , 95,97,98 , 95,98,94 , 298,302,272 , 102,103,104 , 104,105,102 , 102,105,101 , 299,261,277 , 91,101,106 , 10,307,8 , 88,90,91 , 270,88,91 , 91,100,101 , 106,101,105 , 311,312,291 , 274,74,75 , 274,69,70 , 66,75,267 , 66,67,75 , 68,75,67 , 69,75,68 , 274,75,69 , 72,73,274 , 297,82,83 , 274,73,74 , 277,261,297 , 91,106,270 , 275,76,77 , 190,191,266 , 78,79,77 , 70,71,274 , 52,53,268 , 77,79,275 , 280,272,304 , 275,75,76 , 342,35,36 , 48,275,79 , 90,88,89 , 117,118,119 , 127,310,293 , 117,120,116 , 120,117,119 , 116,121,115 , 128,129,310 , 293,126,127 , 310,127,128 , 121,116,120 , 115,254,114 , 115,121,254 , 254,255,114 , 254,121,122 , 255,320,114 , 87,106,108 , 88,270,87 , 185,296,287 , 107,108,106 , 108,86,87 , 2,273,1 , 253,264,319 , 109,85,86 , 110,85,109 , 108,109,86 , 308,30,33 , 248,272,302 , 240,302,259 , 156,285,154 , 112,113,256 , 256,257,112 , 113,320,256 , 82,297,81 , 50,267,49 , 85,110,84 , 84,111,297 , 267,65,66 , 111,84,110 , 277,297,112 , 204,305,201 , 83,84,297 , 112,297,111 , 291,296,205 , 4,307,3 , 249,304,272 , 129,130,310 , 3,251,265 , 55,63,64 , 55,64,54 , 64,267,54 , 65,267,64 , 143,285,299 , 62,57,58 , 62,63,57 , 250,304,249 , 55,56,57 , 63,55,57 , 125,293,124 , 267,53,54 , 308,34,309 , 46,47,48 , 79,46,48 , 14,15,298 , 260,308,309 , 52,268,51 , 283,303,219 , 218,219,303 , 268,50,51 , 79,45,46 , 50,53,267 , 48,49,275 , 261,355,357 , 308,260,269 , 61,59,60 , 35,309,34 , 32,33,31 , 30,31,33 , 269,30,308 , 58,59,61 , 37,342,36 , 342,309,35 , 29,30,28 , 269,26,27 , 28,30,27 , 269,306,333 , 269,260,306 , 367,37,365 , 61,62,58 , 199,201,305 , 42,43,44 , 185,287,183 , 42,44,80 , 45,79,80 , 44,45,80 , 42,80,81 , 388,374,363 , 306,335,332 , 263,17,281 , 41,42,81 , 314,316,318 , 257,277,112 , 41,369,385 , 336,377,339 , 355,356,353 , 23,17,263 , 386,24,325 , 16,23,259 , 263,281,22 , 16,17,23 , 16,259,302 , 16,302,298 , 274,71,72 , 297,357,81 , 279,22,18 , 20,21,19 , 279,19,21 , 271,273,2 , 5,284,4 , 284,6,8 , 12,280,11 , 131,310,130 , 280,12,13 , 4,284,307 , 284,8,307 , 10,8,9 , 8,6,7 , 307,251,3 , 2,3,265 , 232,331,294 , 22,281,18 , 257,258,277 , 126,293,125 , 310,132,133 , 124,293,140 , 314,313,315 , 258,252,277 , 252,324,141 , 123,253,122 , 286,264,253 , 283,311,291 , 254,122,253 , 262,133,134 , 262,310,133 , 134,135,262 , 262,135,136 , 197,305,282 , 293,138,139 , 293,137,138 , 262,136,137 , 139,140,293 , 141,142,299 , 354,165,166 , 141,299,277 , 252,278,324 , 277,252,141 , 329,232,294 , 106,87,270 , 231,167,230 , 382,328,329 , 232,329,231 , 27,30,269 , 299,142,143 , 402,379,373 , 152,285,144 , 344,360,358 , 152,154,285 , 296,291,312 , 342,330,341 , 156,261,285 , 167,168,230 , 156,165,354 , 156,164,165 , 163,164,156 , 247,240,289 , 272,14,298 , 156,154,155 , 2,265,271 , 145,152,144 , 143,144,285 , 220,311,219 , 49,267,275 , 328,166,167 , 362,358,360 , 295,176,177 , 158,162,157 , 312,301,172 , 145,146,150 , 157,163,156 , 286,253,123 , 162,163,157 , 160,161,159 , 191,188,266 , 161,162,158 , 75,275,267 , 150,152,145 , 153,154,152 , 146,149,150 , 149,146,147 , 148,149,147 , 312,287,296 , 26,269,333 , 321,322,276 , 288,225,301 , 158,159,161 , 150,151,152 , 175,176,295 , 295,287,312 , 182,295,178 , 180,178,179 , 177,178,295 , 174,175,295 , 311,283,219 , 312,288,301 , 295,182,287 , 182,180,181 , 180,182,178 , 173,174,295 , 304,307,10 , 185,183,184 , 189,190,266 , 193,282,192 , 189,266,188 , 182,183,287 , 230,168,229 , 53,50,268 , 238,239,326 , 197,290,305 , 172,173,312 , 234,232,233 , 262,293,310 , 237,238,236 , 300,236,238 , 282,191,192 , 259,23,325 , 352,358,362 , 227,228,226 , 226,171,301 , 228,171,226 , 228,229,169 , 206,291,205 , 299,285,261 , 187,305,296 , 288,224,225 , 226,301,225 , 262,137,293 , 221,222,220 , 169,171,228 , 234,331,232 , 229,168,169 , 222,223,311 , 13,14,272 , 240,247,302 , 302,247,248 , 240,259,326 , 239,240,326 , 272,280,13 , 307,304,251 , 304,250,251 , 10,280,304 , 280,10,11 , 0,1,273 , 248,249,272 , 247,289,246 , 246,241,245 , 289,241,246 , 242,243,244 , 242,244,245 , 245,241,242 , 220,222,311 , 361,236,300 , 19,279,18 , 209,283,206 , 204,296,305 , 187,185,186 , 187,296,185 , 282,305,187 , 288,223,224 , 203,204,201 , 312,173,295 , 292,305,290 , 205,296,204 , 191,282,188 , 170,171,169 , 196,193,195 , 197,282,196 , 194,195,193 , 200,201,199 , 292,198,199 , 305,292,199 , 187,188,282 , 196,282,193 , 215,216,303 , 288,311,223 , 214,303,283 , 211,212,283 , 215,303,214 , 217,218,303 , 217,303,216 , 206,283,291 , 207,208,209 , 214,283,212 , 209,210,211 , 283,209,211 , 212,213,214 , 207,209,206 , 201,202,203 , 301,171,172 , 311,288,312 , 132,310,131 , 318,316,123 , 317,315,313 , 313,124,317 , 315,317,140 , 286,123,316 , 140,317,124 , 123,313,318 , 314,318,313 , 264,276,319 , 114,320,113 , 276,323,321 , 324,322,321 , 276,264,323 , 322,324,278 , 325,336,326 , 326,300,238 , 259,325,326 , 325,24,336 , 383,373,375 , 354,327,355 , 329,328,167 , 354,328,327 , 231,329,167 , 329,294,375 , 403,371,370 , 306,260,341 , 300,326,336 , 344,235,360 , 341,343,306 , 306,334,333 , 306,332,334 , 332,333,334 , 347,364,337 , 335,348,349 , 336,24,377 , 300,336,340 , 388,389,394 , 371,403,345 , 338,340,339 , 331,352,345 , 347,346,25 , 351,339,337 , 339,340,336 , 359,358,331 , 342,341,260 , 343,341,330 , 342,260,309 , 398,391,38 , 330,392,381 , 306,343,335 , 362,340,338 , 359,344,358 , 352,351,371 , 364,370,337 , 348,347,25 , 378,339,377 , 350,337,339 , 349,348,25 , 394,335,374 , 387,332,349 , 335,349,332 , 346,347,350 , 337,350,347 , 351,337,371 , 338,339,351 , 351,352,338 , 404,331,345 , 353,369,81 , 261,354,355 , 354,261,156 , 328,354,166 , 355,327,384 , 355,353,357 , 375,384,382 , 373,379,356 , 81,357,353 , 297,261,357 , 352,331,358 , 331,234,359 , 344,359,234 , 235,361,360 , 300,340,362 , 236,361,235 , 362,361,300 , 362,338,352 , 362,360,361 , 390,366,391 , 393,395,370 , 389,347,348 , 370,364,363 , 365,38,366 , 37,367,342 , 366,392,367 , 366,367,365 , 330,342,367 , 380,402,373 , 331,368,294 , 81,369,41 , 356,369,353 , 370,363,393 , 368,403,380 , 371,345,352 , 337,370,371 , 390,391,393 , 400,395,397 , 384,375,373 , 380,373,383 , 335,343,374 , 330,381,343 , 383,375,294 , 384,373,356 , 350,378,346 , 376,378,377 , 24,376,377 , 346,378,376 , 339,378,350 , 40,369,379 , 369,356,379 , 395,403,370 , 402,380,39 , 381,392,390 , 374,343,381 , 328,382,327 , 329,375,382 , 294,368,383 , 383,368,380 , 355,384,356 , 382,384,327 , 369,40,385 , 325,23,386 , 349,25,387 , 363,364,388 , 389,388,364 , 347,389,364 , 394,348,335 , 363,374,390 , 381,390,374 , 38,391,366 , 393,391,372 , 367,392,330 , 366,390,392 , 393,372,395 , 390,393,363 , 388,394,374 , 348,394,389 , 401,372,396 , 403,39,380 , 398,372,391 , 372,397,395 , 399,400,397 , 38,396,398 , 372,398,396 , 401,396,399 , 39,395,400 , 399,39,400 , 397,372,401 , 399,397,401 , 405,40,402 , 379,402,40 , 403,395,39 , 404,403,368 , 331,404,368 , 345,403,404 , 402,39,405;
                //TODO: save into file
                
                // remeshing for different resolution
                Eigen::VectorXi bnd;
                igl::boundary_loop(F, bnd);
                Eigen::MatrixXi E(bnd.size(), 2);
                Eigen::MatrixXd UV_bnds(bnd.size(), 2);
                for(int i = 0; i < bnd.size(); i++) {
                    E.row(i) << i, i + 1;
                    UV_bnds.row(i) = V.row(bnd[i]);
                }
                E(bnd.size() - 1, 1) = 0;
                
                double area = 0.0;
                for(int triI = 0; triI < F.rows(); triI++) {
                    const Eigen::RowVector2d v01 = V.row(F(triI, 1)) - V.row(F(triI, 0));
                    const Eigen::RowVector2d v02 = V.row(F(triI, 2)) - V.row(F(triI, 0));
                    area += v01[0] * v02[1] - v01[1] * v02[0];
                }
                area /= 2.0;
                std::string flag("qQa" + std::to_string(area / elemAmt * 1.1));
                // "q" for high quality mesh generation
                // "Q" for quiet mode (no output)
                // "a" for area upper bound
                
                igl::triangle::triangulate(UV_bnds, E, Eigen::MatrixXd(), flag, V, F);
                
                // resize to match size
                Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
                Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
                Eigen::RowVectorXd range = topRight - bottomLeft;
                double scale = size / range[0];
                V *= scale;
                bottomLeft *= scale;
                topRight *= scale;
                range *= scale;
                
                V_rest.resize(V.rows(), 3);
                V_rest.leftCols(2) = V;
                V_rest.rightCols(1).setZero();
                
                borderVerts_primitive.resize(2);
                for(int vI = 0; vI < V.rows(); vI++) {
                    if(V(vI, 0) < bottomLeft[0] + range[0] / 100.0) {
                        borderVerts_primitive[0].emplace_back(vI);
                    }
                    else if(V(vI, 0) > topRight[0] - range[0] / 100.0) {
                        borderVerts_primitive[1].emplace_back(vI);
                    }
                }
                
                break;
            }
                
            case P_CYLINDER: {
                initCylinder(0.5, 0.5, 1.0, 1.0, 1.0, 20, 20, V_rest, F, &V);
                break;
            }
                
            default:
                assert(0 && "no such primitive to construct!");
                break;
        }
        
        if(separateTri) {
            *this = TriangleSoup(V_rest, F, V);
        }
        else {
            triWeight = Eigen::VectorXd::Ones(F.rows());
            computeFeatures(false, true);
        }
        initSeamLen = 0.0;
        
        vertWeight = Eigen::VectorXd::Ones(V.rows());
    }
    
    void TriangleSoup::computeLaplacianMtr(void)
    {
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V_rest, F, L);
        LaplacianMtr.resize(L.rows(), L.cols());
        LaplacianMtr.setZero();
        LaplacianMtr.reserve(L.nonZeros());
        for (int k = 0; k < L.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            {
                if((fixedVert.find(static_cast<int>(it.row())) == fixedVert.end()) &&
                   (fixedVert.find(static_cast<int>(it.col())) == fixedVert.end()))
                {
                    LaplacianMtr.insert(it.row(), it.col()) = -it.value();
                }
            }
        }
        for(const auto fixedVI : fixedVert) {
            LaplacianMtr.insert(fixedVI, fixedVI) = 1.0;
        }
        LaplacianMtr.makeCompressed();
//        //        Eigen::SparseMatrix<double> M;
//        //        massmatrix(data.V_rest, data.F, igl::MASSMATRIX_TYPE_DEFAULT, M);
//                    LaplacianMtr.insert(it.row() * 2, it.col() * 2) = -it.value();// * M.coeffRef(it.row(), it.row());
//                    LaplacianMtr.insert(it.row() * 2 + 1, it.col() * 2 + 1) = -it.value();// * M.coeffRef(it.row(), it.row());
    }
    
    void TriangleSoup::computeMassMatrix(const igl::MassMatrixType type)
    {
        const Eigen::MatrixXd & V = V_rest;
        Eigen::SparseMatrix<double>& M = massMatrix;
        
        typedef double Scalar;
        
        using namespace Eigen;
        using namespace std;
        using namespace igl;
        
        const int n = V.rows();
        const int m = F.rows();
        const int simplex_size = F.cols();
        
        MassMatrixType eff_type = type;
        // Use voronoi of for triangles by default, otherwise barycentric
        if(type == MASSMATRIX_TYPE_DEFAULT)
        {
            eff_type = (simplex_size == 3?MASSMATRIX_TYPE_VORONOI:MASSMATRIX_TYPE_BARYCENTRIC);
        }
        
        // Not yet supported
        assert(type!=MASSMATRIX_TYPE_FULL);
        
        Matrix<int,Dynamic,1> MI;
        Matrix<int,Dynamic,1> MJ;
        Matrix<Scalar,Dynamic,1> MV;
        if(simplex_size == 3)
        {
            // Triangles
            // edge lengths numbered same as opposite vertices
            Matrix<Scalar,Dynamic,3> l(m,3);
            // loop over faces
            for(int i = 0;i<m;i++)
            {
                l(i,0) = (V.row(F(i,1))-V.row(F(i,2))).norm();
                l(i,1) = (V.row(F(i,2))-V.row(F(i,0))).norm();
                l(i,2) = (V.row(F(i,0))-V.row(F(i,1))).norm();
            }
            Matrix<Scalar,Dynamic,1> dblA;
            doublearea(l,0.,dblA);
            
            //!!! weighted area
            for(int i = 0; i < m; i++) {
                dblA[i] *= triWeight[i];
            }
            
            switch(eff_type)
            {
                case MASSMATRIX_TYPE_BARYCENTRIC:
                    // diagonal entries for each face corner
                    MI.resize(m*3,1); MJ.resize(m*3,1); MV.resize(m*3,1);
                    MI.block(0*m,0,m,1) = F.col(0);
                    MI.block(1*m,0,m,1) = F.col(1);
                    MI.block(2*m,0,m,1) = F.col(2);
                    MJ = MI;
                    repmat(dblA,3,1,MV);
                    MV.array() /= 6.0;
                    break;
                case MASSMATRIX_TYPE_VORONOI:
                {
                    // diagonal entries for each face corner
                    // http://www.alecjacobson.com/weblog/?p=874
                    MI.resize(m*3,1); MJ.resize(m*3,1); MV.resize(m*3,1);
                    MI.block(0*m,0,m,1) = F.col(0);
                    MI.block(1*m,0,m,1) = F.col(1);
                    MI.block(2*m,0,m,1) = F.col(2);
                    MJ = MI;
                    
                    // Holy shit this needs to be cleaned up and optimized
                    Matrix<Scalar,Dynamic,3> cosines(m,3);
                    cosines.col(0) =
                    (l.col(2).array().pow(2)+l.col(1).array().pow(2)-l.col(0).array().pow(2))/(l.col(1).array()*l.col(2).array()*2.0);
                    cosines.col(1) =
                    (l.col(0).array().pow(2)+l.col(2).array().pow(2)-l.col(1).array().pow(2))/(l.col(2).array()*l.col(0).array()*2.0);
                    cosines.col(2) =
                    (l.col(1).array().pow(2)+l.col(0).array().pow(2)-l.col(2).array().pow(2))/(l.col(0).array()*l.col(1).array()*2.0);
                    Matrix<Scalar,Dynamic,3> barycentric = cosines.array() * l.array();
                    normalize_row_sums(barycentric,barycentric);
                    Matrix<Scalar,Dynamic,3> partial = barycentric;
                    partial.col(0).array() *= dblA.array() * 0.5;
                    partial.col(1).array() *= dblA.array() * 0.5;
                    partial.col(2).array() *= dblA.array() * 0.5;
                    Matrix<Scalar,Dynamic,3> quads(partial.rows(),partial.cols());
                    quads.col(0) = (partial.col(1)+partial.col(2))*0.5;
                    quads.col(1) = (partial.col(2)+partial.col(0))*0.5;
                    quads.col(2) = (partial.col(0)+partial.col(1))*0.5;
                    
                    quads.col(0) = (cosines.col(0).array()<0).select( 0.25*dblA,quads.col(0));
                    quads.col(1) = (cosines.col(0).array()<0).select(0.125*dblA,quads.col(1));
                    quads.col(2) = (cosines.col(0).array()<0).select(0.125*dblA,quads.col(2));
                    
                    quads.col(0) = (cosines.col(1).array()<0).select(0.125*dblA,quads.col(0));
                    quads.col(1) = (cosines.col(1).array()<0).select(0.25*dblA,quads.col(1));
                    quads.col(2) = (cosines.col(1).array()<0).select(0.125*dblA,quads.col(2));
                    
                    quads.col(0) = (cosines.col(2).array()<0).select(0.125*dblA,quads.col(0));
                    quads.col(1) = (cosines.col(2).array()<0).select(0.125*dblA,quads.col(1));
                    quads.col(2) = (cosines.col(2).array()<0).select( 0.25*dblA,quads.col(2));
                    
                    MV.block(0*m,0,m,1) = quads.col(0);
                    MV.block(1*m,0,m,1) = quads.col(1);
                    MV.block(2*m,0,m,1) = quads.col(2);
                    
                    break;
                }
                case MASSMATRIX_TYPE_FULL:
                    assert(false && "Implementation incomplete");
                    break;
                default:
                    assert(false && "Unknown Mass matrix eff_type");
            }
            
        }else if(simplex_size == 4)
        {
            assert(V.cols() == 3);
            assert(eff_type == MASSMATRIX_TYPE_BARYCENTRIC);
            MI.resize(m*4,1); MJ.resize(m*4,1); MV.resize(m*4,1);
            MI.block(0*m,0,m,1) = F.col(0);
            MI.block(1*m,0,m,1) = F.col(1);
            MI.block(2*m,0,m,1) = F.col(2);
            MI.block(3*m,0,m,1) = F.col(3);
            MJ = MI;
            // loop over tets
            for(int i = 0;i<m;i++)
            {
                // http://en.wikipedia.org/wiki/Tetrahedron#Volume
                Matrix<Scalar,3,1> v0m3,v1m3,v2m3;
                v0m3.head(V.cols()) = V.row(F(i,0)) - V.row(F(i,3));
                v1m3.head(V.cols()) = V.row(F(i,1)) - V.row(F(i,3));
                v2m3.head(V.cols()) = V.row(F(i,2)) - V.row(F(i,3));
                Scalar v = fabs(v0m3.dot(v1m3.cross(v2m3)))/6.0;
                MV(i+0*m) = v/4.0;
                MV(i+1*m) = v/4.0;
                MV(i+2*m) = v/4.0;
                MV(i+3*m) = v/4.0;
            }
        }else
        {
            // Unsupported simplex size
            assert(false && "Unsupported simplex size");
        }
        sparse(MI,MJ,MV,n,n,M);
        
        massMatrix *= density;
    }
    
    void TriangleSoup::computeFeatures(bool multiComp, bool resetFixedV)
    {
        //TODO: if the mesh is multi-component, then fix more vertices
        if(resetFixedV) {
            fixedVert.clear();
            fixedVert.insert(0);
            isFixedVert.resize(0);
            isFixedVert.resize(V.rows(), false);
            isFixedVert[0] = true;
        }
        
        boundaryEdge.resize(cohE.rows());
        edgeLen.resize(cohE.rows());
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(cohE.row(cohI).minCoeff() >= 0) {
                boundaryEdge[cohI] = 0;
            }
            else {
                boundaryEdge[cohI] = 1;
            }
            edgeLen[cohI] = (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
        }
        
//        igl::cotmatrix_entries(V_rest, F, cotVals);
        
        restTriInv.resize(F.rows());
        triNormal.resize(F.rows(), 3);
        triArea.resize(F.rows());
        surfaceArea = 0.0;
        triAreaSq.resize(F.rows());
        e0SqLen.resize(F.rows());
        e1SqLen.resize(F.rows());
        e0dote1.resize(F.rows());
        e0SqLen_div_dbAreaSq.resize(F.rows());
        e1SqLen_div_dbAreaSq.resize(F.rows());
        e0dote1_div_dbAreaSq.resize(F.rows());
        std::vector<Eigen::RowVector3d> vertNormals(V_rest.rows(), Eigen::Vector3d::Zero());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector3d& P1 = V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = V_rest.row(triVInd[2]);
            
            Eigen::Vector3d x0_3D[3] = { P1, P2, P3 };
            Eigen::Vector2d x0[3];
            IglUtils::mapTriangleTo2D(x0_3D, x0);
            Eigen::Matrix2d X0;
            X0.col(0) = x0[1] - x0[0];
            X0.col(1) = x0[2] - x0[0];
            restTriInv[triI] = X0.inverse();
            //TODO: support areaThres_AM
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            const Eigen::RowVector3d normalVec = P2m1.cross(P3m1);
            
            triNormal.row(triI) = normalVec.normalized();
            triArea[triI] = 0.5 * normalVec.norm();
            if(triArea[triI] < areaThres_AM) {
                // air mesh triangle degeneracy prevention
                triArea[triI] = areaThres_AM;
                surfaceArea += areaThres_AM;
                triAreaSq[triI] = areaThres_AM * areaThres_AM;
                e0SqLen[triI] = e1SqLen[triI] = 4.0 / std::sqrt(3.0) * areaThres_AM;
                e0dote1[triI] = e0SqLen[triI] / 2.0;
                
                e0SqLen_div_dbAreaSq[triI] = e1SqLen_div_dbAreaSq[triI] = 2.0 / std::sqrt(3.0) / areaThres_AM;
                e0dote1_div_dbAreaSq[triI] = e0SqLen_div_dbAreaSq[triI] / 2.0;
            }
            else {
                surfaceArea += triArea[triI];
                triAreaSq[triI] = triArea[triI] * triArea[triI];
                e0SqLen[triI] = P2m1.squaredNorm();
                e1SqLen[triI] = P3m1.squaredNorm();
                e0dote1[triI] = P2m1.dot(P3m1);
                
                e0SqLen_div_dbAreaSq[triI] = e0SqLen[triI] / 2. / triAreaSq[triI];
                e1SqLen_div_dbAreaSq[triI] = e1SqLen[triI] / 2. / triAreaSq[triI];
                e0dote1_div_dbAreaSq[triI] = e0dote1[triI] / 2. / triAreaSq[triI];
            }
            vertNormals[triVInd[0]] += normalVec;
            vertNormals[triVInd[1]] += normalVec;
            vertNormals[triVInd[2]] += normalVec;
        }
        avgEdgeLen = igl::avg_edge_length(V_rest, F);
        virtualRadius = std::sqrt(surfaceArea / M_PI);
        for(auto& vNI : vertNormals) {
            vNI.normalize();
        }
        
        computeLaplacianMtr();
#ifndef STATIC_SOLVE
        computeMassMatrix(igl::MASSMATRIX_TYPE_VORONOI);
#endif STATIC_SOLVE
        
//        //!! for edge count minimization of separation energy
//        for(int cohI = 0; cohI < cohE.rows(); cohI++)
//        {
//            edgeLen[cohI] = avgEdgeLen;
//        }
        
        bbox.block(0, 0, 1, 3) = V_rest.row(0);
        bbox.block(1, 0, 1, 3) = V_rest.row(0);
        for(int vI = 1; vI < V_rest.rows(); vI++) {
            const Eigen::RowVector3d& v = V_rest.row(vI);
            for(int dimI = 0; dimI < 3; dimI++) {
                if(v[dimI] < bbox(0, dimI)) {
                    bbox(0, dimI) = v[dimI];
                }
                if(v[dimI] > bbox(1, dimI)) {
                    bbox(1, dimI) = v[dimI];
                }
            }
        }
        
        edge2Tri.clear();
        vNeighbor.resize(0);
        vNeighbor.resize(V_rest.rows());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            for(int vI = 0; vI < 3; vI++) {
                int vI_post = (vI + 1) % 3;
                edge2Tri[std::pair<int, int>(triVInd[vI], triVInd[vI_post])] = triI;
                vNeighbor[triVInd[vI]].insert(triVInd[vI_post]);
                vNeighbor[triVInd[vI_post]].insert(triVInd[vI]);
            }
        }
        cohEIndex.clear();
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            const Eigen::RowVector4i& cohEI = cohE.row(cohI);
            if(cohEI.minCoeff() >= 0) {
                cohEIndex[std::pair<int, int>(cohEI[0], cohEI[1])] = cohI;
                cohEIndex[std::pair<int, int>(cohEI[3], cohEI[2])] = -cohI - 1;
            }
        }
        
        validSplit.resize(V_rest.rows());
        for(int vI = 0; vI < V_rest.rows(); vI++) {
            validSplit[vI].clear();
            if(isBoundaryVert(vI)) {
                continue;
            }
            
            std::vector<int> nbVs(vNeighbor[vI].begin(), vNeighbor[vI].end());
            std::vector<Eigen::RowVector3d> projectedEdge(nbVs.size());
            for(int nbI = 0; nbI < nbVs.size(); nbI++) {
                const Eigen::RowVector3d edge = V_rest.row(nbVs[nbI]) - V_rest.row(vI);
                projectedEdge[nbI] = (edge - edge.dot(vertNormals[vI]) * vertNormals[vI]).normalized();
            }
            for(int nbI = 0; nbI + 1 < nbVs.size(); nbI++) {
                for(int nbJ = nbI + 1; nbJ < nbVs.size(); nbJ++) {
                    if(projectedEdge[nbI].dot(projectedEdge[nbJ]) <= 0.0) {
                        validSplit[vI].insert(std::pair<int, int>(nbVs[nbI], nbVs[nbJ]));
                        validSplit[vI].insert(std::pair<int, int>(nbVs[nbJ], nbVs[nbI]));
                    }
                }
            }
        }
        
        // init fracture tail record
        fracTail.clear();
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            if(cohE(cohI, 0) == cohE(cohI, 2)) {
                fracTail.insert(cohE(cohI, 0));
            }
            else if(cohE(cohI, 1) == cohE(cohI, 3)) {
                fracTail.insert(cohE(cohI, 1));
            }
        }
        // tails of initial seams doesn't count as fracture tails for propagation
        for(int initSeamI = 0; initSeamI < initSeams.rows(); initSeamI++) {
            if(initSeams(initSeamI, 0) == initSeams(initSeamI, 2)) {
                fracTail.erase(initSeams(initSeamI, 0));
            }
            else if(initSeams(initSeamI, 1) == initSeams(initSeamI, 3)) {
                fracTail.erase(initSeams(initSeamI, 1));
            }
        }
    }
    
    void TriangleSoup::updateFeatures(void)
    {
        // assumption: triangle shapes stay the same
        
        const int nCE = static_cast<int>(boundaryEdge.size());
        boundaryEdge.conservativeResize(cohE.rows());
        edgeLen.conservativeResize(cohE.rows());
        for(int cohI = nCE; cohI < cohE.rows(); cohI++)
        {
            if(cohE.row(cohI).minCoeff() >= 0) {
                boundaryEdge[cohI] = 0;
            }
            else {
                boundaryEdge[cohI] = 1;
            }
            edgeLen[cohI] = (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
        }
        
        computeLaplacianMtr();
#ifndef STATIC_SOLVE
        computeMassMatrix(igl::MASSMATRIX_TYPE_VORONOI);
#endif
    }
    
    void TriangleSoup::resetFixedVert(const std::set<int>& p_fixedVert)
    {
        isFixedVert.resize(0);
        isFixedVert.resize(V.rows(), false);
        for(const auto& vI : p_fixedVert) {
            assert(vI < V.rows());
            isFixedVert[vI] = true;
        }
        
        fixedVert = p_fixedVert;
        computeLaplacianMtr();
    }
    void TriangleSoup::addFixedVert(int vI)
    {
        assert(vI < V.rows());
        fixedVert.insert(vI);
        isFixedVert[vI] = true;
    }
    void TriangleSoup::addFixedVert(const std::vector<int>& p_fixedVert)
    {
        for(const auto& vI : p_fixedVert) {
            assert(vI < V.rows());
            isFixedVert[vI] = true;
        }
        
        fixedVert.insert(p_fixedVert.begin(), p_fixedVert.end());
        computeLaplacianMtr();
    }
    
    bool TriangleSoup::separateTriangle(const Eigen::VectorXd& measure, double thres)
    {
        assert(measure.size() == F.rows());
        
        // separate triangles when necessary
        bool changed = false;
        for(int triI = 0; triI < F.rows(); triI++) {
            if(measure[triI] <= thres) {
                continue;
            }
            
            const Eigen::RowVector3i triVInd = F.row(triI);
            Eigen::Vector3i needSeparate = Eigen::Vector3i::Zero();
            std::map<std::pair<int, int>, int>::iterator edgeFinder[3];
            for(int eI = 0; eI < 3; eI++) {
                if(edge2Tri.find(std::pair<int, int>(triVInd[(eI + 1) % 3], triVInd[eI])) != edge2Tri.end()) {
                    needSeparate[eI] = 1;
                    edgeFinder[eI] = edge2Tri.find(std::pair<int, int>(triVInd[eI], triVInd[(eI + 1) % 3]));
                }
            }
            if(needSeparate.sum() == 0) {
                continue;
            }
            
            changed = true;
            if(needSeparate.sum() == 1) {
                // duplicate the edge
                for(int eI = 0; eI < 3; eI++) {
                    if(needSeparate[eI]) {
                        const int vI = triVInd[eI], vI_post = triVInd[(eI + 1) % 3];
                        const int nV = static_cast<int>(V_rest.rows());
                        V_rest.conservativeResize(nV + 2, 3);
                        V_rest.row(nV) = V_rest.row(vI);
                        V_rest.row(nV + 1) = V_rest.row(vI_post);
                        V.conservativeResize(nV + 2, 2);
                        V.row(nV) = V.row(vI);
                        V.row(nV + 1) = V.row(vI_post);
                        
                        F(triI, eI) = nV;
                        F(triI, (eI + 1) % 3) = nV + 1;
                        
                        const int nCE = static_cast<int>(cohE.rows());
                        cohE.conservativeResize(nCE + 1, 4);
                        cohE.row(nCE) << nV, nV + 1, vI, vI_post;
                        cohEIndex[std::pair<int, int>(nV, nV + 1)] = nCE;
                        cohEIndex[std::pair<int, int>(vI_post, vI)] = -nCE - 1;
                        
                        edge2Tri.erase(edgeFinder[eI]);
                        edge2Tri[std::pair<int, int>(nV, nV + 1)] = triI;
                        
                        const int vI_pre = triVInd[(eI + 2) % 3];
                        auto finder0 = cohEIndex.find(std::pair<int, int>(vI_post, vI_pre));
                        if(finder0 != cohEIndex.end()) {
                            if(finder0->second >= 0) {
                                cohE(finder0->second, 0) = nV + 1;
                            }
                            else {
                                cohE(-finder0->second - 1, 3) = nV + 1;
                            }
                            cohEIndex[std::pair<int, int>(nV + 1, vI_pre)] = finder0->second;
                            cohEIndex.erase(finder0);
                        }
                        auto finder1 = cohEIndex.find(std::pair<int, int>(vI_pre, vI));
                        if(finder1 != cohEIndex.end()) {
                            if(finder1->second >= 0) {
                                cohE(finder1->second, 1) = nV;
                            }
                            else {
                                cohE(-finder1->second - 1, 2) = nV;
                            }
                            cohEIndex[std::pair<int, int>(vI_pre, nV)] = finder1->second;
                            cohEIndex.erase(finder1);
                        }
                        
                        break;
                    }
                }
            }
            else if(needSeparate.sum() > 1) {
                std::vector<std::vector<int>> tri_toSep;
                std::vector<std::pair<int, int>> boundaryEdge;
                std::vector<int> vI_toSplit, vI_toSplit_post;
                std::vector<bool> needSplit;
                for(int eI = 0; eI < 3; eI++) {
                    if(needSeparate[eI] && needSeparate[(eI + 2) % 3]) {
                        vI_toSplit.emplace_back(triVInd[eI]);
                        vI_toSplit_post.emplace_back(triVInd[(eI + 1) % 3]);
                        tri_toSep.resize(tri_toSep.size() + 1);
                        boundaryEdge.resize(boundaryEdge.size() + 1);
                        needSplit.push_back(isBoundaryVert(vI_toSplit.back(), vI_toSplit_post.back(),
                                                              tri_toSep.back(), boundaryEdge.back()));
                    }
                }
                
                // duplicate all vertices
                const int vI0 = triVInd[0], vI1 = triVInd[1], vI2 = triVInd[2];
                const int nV = static_cast<int>(V_rest.rows());
                V_rest.conservativeResize(nV + 3, 3);
                V_rest.row(nV) = V_rest.row(vI0);
                V_rest.row(nV + 1) = V_rest.row(vI1);
                V_rest.row(nV + 2) = V_rest.row(vI2);
                V.conservativeResize(nV + 3, 2);
                V.row(nV) = V.row(vI0);
                V.row(nV + 1) = V.row(vI1);
                V.row(nV + 2) = V.row(vI2);
                
                F.row(triI) << nV, nV + 1, nV + 2;
                
                // construct cohesive edges:
                for(int eI = 0; eI < 3; eI++) {
                    if(needSeparate[eI]) {
                        const int nCE = static_cast<int>(cohE.rows());
                        cohE.conservativeResize(nCE + 1, 4);
                        const int vI = eI, vI_post = (eI + 1) % 3;
                        cohE.row(nCE) << nV + vI, nV + vI_post, triVInd[vI], triVInd[vI_post];
                        cohEIndex[std::pair<int, int>(nV + vI, nV + vI_post)] = nCE;
                        cohEIndex[std::pair<int, int>(triVInd[vI_post], triVInd[vI])] = -nCE - 1;
                        
                        edge2Tri.erase(edgeFinder[eI]);
                        edge2Tri[std::pair<int, int>(nV + vI, nV + vI_post)] = triI;
                    }
                    else {
                        int vI = eI, vI_post = (eI + 1) % 3;
                        auto finder = cohEIndex.find(std::pair<int, int>(triVInd[vI], triVInd[vI_post]));
                        if(finder != cohEIndex.end()) {
                            if(finder->second >= 0) {
                                cohE(finder->second, 0) = nV + vI;
                                cohE(finder->second, 1) = nV + vI_post;
                            }
                            else {
                                cohE(-finder->second - 1, 3) = nV + vI;
                                cohE(-finder->second - 1, 2) = nV + vI_post;
                            }
                            cohEIndex[std::pair<int, int>(nV + vI, nV + vI_post)] = finder->second;
                            cohEIndex.erase(finder);
                        }
                    }
                }
                
                for(int sI = 0; sI < needSplit.size(); sI++) {
                    if(!needSplit[sI]) {
                        continue;
                    }
                    
                    assert(!tri_toSep.empty());
                    const int nV = static_cast<int>(V_rest.rows());
                    V_rest.conservativeResize(nV + 1, 3);
                    V_rest.row(nV) = V_rest.row(vI_toSplit[sI]);
                    V.conservativeResize(nV + 1, 2);
                    V.row(nV) = V.row(vI_toSplit[sI]);
                    for(const auto triToSepI : tri_toSep[sI]) {
                        int i = 0;
                        for(; i < 3; i++) {
                            if(F(triToSepI, i) == vI_toSplit[sI]) {
                                F(triToSepI, i) = nV;
                                int vI_post = F(triToSepI, (i + 1) % 3);
                                int vI_pre = F(triToSepI, (i + 2) % 3);
                                edge2Tri[std::pair<int, int>(nV, vI_post)] = triToSepI;
                                edge2Tri[std::pair<int, int>(vI_pre, nV)] = triToSepI;
                                edge2Tri.erase(std::pair<int, int>(vI_toSplit[sI], vI_post));
                                edge2Tri.erase(std::pair<int, int>(vI_pre, vI_toSplit[sI]));
                                break;
                            }
                        }
                        assert(i < 3);
                    }
                    auto finder = cohEIndex.find(std::pair<int, int>(vI_toSplit_post[sI], vI_toSplit[sI]));
                    assert(finder != cohEIndex.end());
                    if(finder->second >= 0) {
                        cohE(finder->second, 1) = nV;
                    }
                    else {
                        cohE(-finder->second - 1, 2) = nV;
                    }
                    cohEIndex[std::pair<int, int>(vI_toSplit_post[sI], nV)] = finder->second;
                    cohEIndex.erase(finder);
                    
                    finder = cohEIndex.find(boundaryEdge[sI]);
                    if(finder != cohEIndex.end()) {
                        if(finder->second >= 0) {
                            cohE(finder->second, 0) = nV;
                        }
                        else {
                            cohE(-finder->second - 1, 3) = nV;
                        }
                        cohEIndex[std::pair<int, int>(nV, boundaryEdge[sI].second)] = finder->second;
                        cohEIndex.erase(finder);
                    }
                }
            }
        }
        
        if(changed) {
            updateFeatures();
        }
        
        return changed;
    }
    
    bool TriangleSoup::splitVertex(const Eigen::VectorXd& measure, double thres)
    {
        assert(measure.rows() == V.rows());
        
        bool modified = false;
        for(int vI = 0; vI < measure.size(); vI++) {
            if(measure[vI] > thres) {
                if(isBoundaryVert(vI)) {
                    // right now only on boundary vertices
                    int vI_interior = -1;
                    for(const auto& vI_neighbor : vNeighbor[vI]) {
                        if((edge2Tri.find(std::pair<int, int>(vI, vI_neighbor)) != edge2Tri.end()) &&
                           (edge2Tri.find(std::pair<int, int>(vI_neighbor, vI)) != edge2Tri.end()))
                        {
                            vI_interior = vI_neighbor;
                            break;
                        }
                    }
                    if(vI_interior >= 0) {
//                        splitEdgeOnBoundary(std::pair<int, int>(vI, vI_interior), edge2Tri, vNeighbor, cohEIndex);
                        modified = true;
                    }
                }
            }
        }
        
        if(modified) {
            updateFeatures();
        }
        
        return modified;
    }
    
    void TriangleSoup::querySplit(double lambda_t, bool propagate, bool splitInterior,
                                  double& EwDec_max, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max,
                                  std::pair<double, double>& energyChanges_max) const
    {
        timer_step.start(6 + splitInterior);
        
        const double filterExp_b = 0.8, filterMult_b = 1.0; //TODO: better use ratio
        const double filterExp_in = 0.6; // smaller than 0.5 is not recommanded
        
        std::vector<int> bestCandVerts;
        if(!propagate) {
            SymStretchEnergy SD;
            //            double energyVal;
            //            SD.computeEnergyVal(*this, energyVal);
            Eigen::VectorXd divGradPerVert;
            SD.computeDivGradPerVert(*this, divGradPerVert);
            //            Eigen::VectorXd maxUnweightedEnergyValPerVert;
            //            SD.getMaxUnweightedEnergyValPerVert(*this, maxUnweightedEnergyValPerVert);
            
            std::map<double, int> sortedCandVerts_b, sortedCandVerts_in;
            if(splitInterior) {
                for(int vI = 0; vI < V_rest.rows(); vI++) {
                    if(vNeighbor[vI].size() <= 2) {
                        // this vertex is impossible to be splitted further
                        continue;
                    }
                    
                    if(!isBoundaryVert(vI)) {
                        bool connectToBound = false;
                        for(const auto& nbVI : vNeighbor[vI]) {
                            if(isBoundaryVert(nbVI)) {
                                connectToBound = true;
                                break;
                            }
                        }
                        if(!connectToBound) {
                            // don't split vertices connected to boundary here
                        //                        if(maxUnweightedEnergyValPerVert[vI] > energyVal) {
                            sortedCandVerts_in[-divGradPerVert[vI] / vertWeight[vI]] = vI;
                        //                        }
                        }
                    }
                }
            }
            else {
                for(int vI = 0; vI < V_rest.rows(); vI++) {
                    if(vNeighbor[vI].size() <= 2) {
                        // this vertex is impossible to be splitted further
                        continue;
                    }
                    
                    if(isBoundaryVert(vI)) {
                        //                        if(maxUnweightedEnergyValPerVert[vI] > energyVal) {
                        sortedCandVerts_b[-divGradPerVert[vI] / vertWeight[vI]] = vI;
                        //                        }
                    }
                }
            }
            
            if(!splitInterior)
            {
                assert(!sortedCandVerts_b.empty());
                int bestCandAmt_b = static_cast<int>(std::pow(sortedCandVerts_b.size(), filterExp_b) * filterMult_b);
                if(bestCandAmt_b < 2) {
                    bestCandAmt_b = 2;
                }
                bestCandVerts.reserve(bestCandAmt_b);
                for(const auto& candI : sortedCandVerts_b) {
                    bestCandVerts.emplace_back(candI.second);
                    if(bestCandVerts.size() >= bestCandAmt_b) {
                        break;
                    }
                }
            }
            else
            {
                assert(!sortedCandVerts_in.empty());
                int bestCandAmt_in = static_cast<int>(std::pow(sortedCandVerts_in.size(), filterExp_in));
                if(bestCandAmt_in < 2) {
                    bestCandAmt_in = 2;
                }
                bestCandVerts.reserve(bestCandVerts.size() + bestCandAmt_in);
                for(const auto& candI : sortedCandVerts_in) {
                    bestCandVerts.emplace_back(candI.second);
                    if(bestCandVerts.size() >= bestCandAmt_in) {
                        break;
                    }
                }
            }
        }
        else {
            // see whether fracture could be propagated from each fracture tail
//#define PROPAGATE_MULTIPLE_TAIL 1
#ifdef PROPAGATE_MULTIPLE_TAIL
            if(fracTail.empty()) {
#else
            if(curFracTail < 0) {
#endif
                if(curInteriorFracTails.first < 0) {
                    EwDec_max = -__DBL_MAX__;
                    path_max.resize(0);
                    newVertPos_max.resize(0, 2);
                    timer_step.stop();
                    return;
                }
                else {
                    assert(curInteriorFracTails.second >= 0);
                    splitInterior = false;
                    bestCandVerts.emplace_back(curInteriorFracTails.first);
                    bestCandVerts.emplace_back(curInteriorFracTails.second);
                }
            }
            else {
                splitInterior = false;
#ifdef PROPAGATE_MULTIPLE_TAIL
                bestCandVerts.insert(bestCandVerts.end(), fracTail.begin(), fracTail.end());
#else
                bestCandVerts.emplace_back(curFracTail);
#endif
            }
        }
        
        assert(!bestCandVerts.empty()); //TODO: extra filter might cause this!!!
        
        // evaluate local energy decrease
        std::cout << "evaluate vertex splits, " << bestCandVerts.size() << " candidate verts" << std::endl;
        // run in parallel:
        static std::vector<double> EwDecs;
        EwDecs.resize(bestCandVerts.size());
        static std::vector<std::vector<int>> paths_p;
        static std::vector<Eigen::MatrixXd> newVertPoses_p;
        static std::vector<std::pair<double, double>> energyChanges_p;
        int operationType = -1;
        // query boundary splits
        if(!splitInterior) {
            if(propagate) {
                paths_p.resize(0);
                paths_p.resize(bestCandVerts.size());
                newVertPoses_p.resize(bestCandVerts.size());
                energyChanges_p.resize(bestCandVerts.size());
#ifdef USE_TBB
                tbb::parallel_for(0, (int)bestCandVerts.size(), 1, [&](int candI)
#else
                for(int candI = 0; candI < bestCandVerts.size(); candI++)
#endif
                {
                    EwDecs[candI] = computeLocalEwDec(bestCandVerts[candI], lambda_t, paths_p[candI], newVertPoses_p[candI],
                                                      energyChanges_p[candI]);
                }
#ifdef USE_TBB
                );
#endif
            }
            else {
                operationType = 0;
                paths_bSplit.resize(0);
                paths_bSplit.resize(bestCandVerts.size());
                newVertPoses_bSplit.resize(bestCandVerts.size());
                energyChanges_bSplit.resize(bestCandVerts.size());
#ifdef USE_TBB
                tbb::parallel_for(0, (int)bestCandVerts.size(), 1, [&](int candI)
#else
                for(int candI = 0; candI < bestCandVerts.size(); candI++)
#endif
                {
                    EwDecs[candI] = computeLocalEwDec(bestCandVerts[candI], lambda_t, paths_bSplit[candI], newVertPoses_bSplit[candI],
                                                      energyChanges_bSplit[candI]);
                }
#ifdef USE_TBB
                );
#endif
            }
        }
        else {
            assert(!propagate);
            operationType = 1;
            // query interior splits
            paths_iSplit.resize(0);
            paths_iSplit.resize(bestCandVerts.size());
            newVertPoses_iSplit.resize(bestCandVerts.size());
            energyChanges_iSplit.resize(bestCandVerts.size());
#ifdef USE_TBB
            tbb::parallel_for(0, (int)bestCandVerts.size(), 1, [&](int candI)
#else
            for(int candI = 0; candI < bestCandVerts.size(); candI++)
#endif
            {
                EwDecs[candI] = computeLocalEwDec(bestCandVerts[candI], lambda_t, paths_iSplit[candI], newVertPoses_iSplit[candI],
                                                        energyChanges_iSplit[candI]);
                if(EwDecs[candI] != -__DBL_MAX__) {
                    EwDecs[candI] *= 0.5;
                }
            }
#ifdef USE_TBB
            );
#endif
        }
            
        int candI_max = 0;
        for(int candI = 1; candI < bestCandVerts.size(); candI++) {
            if(EwDecs[candI] > EwDecs[candI_max]) {
                candI_max = candI;
            }
        }
            
        EwDec_max = EwDecs[candI_max];
        switch(operationType) {
            case -1:
                path_max = paths_p[candI_max];
                newVertPos_max = newVertPoses_p[candI_max];
                energyChanges_max = energyChanges_p[candI_max];
                break;
                
            case 0:
                path_max = paths_bSplit[candI_max];
                newVertPos_max = newVertPoses_bSplit[candI_max];
                energyChanges_max = energyChanges_bSplit[candI_max];
                break;
                
            case 1:
                path_max = paths_iSplit[candI_max];
                newVertPos_max = newVertPoses_iSplit[candI_max];
                energyChanges_max = energyChanges_iSplit[candI_max];
                break;
                
            default:
                assert(0);
                break;
        }
//        std::cout << path_max[0] << " " << path_max[1] << std::endl;
//        std::cout << newVertPoses[candI_max] << std::endl;
            
        timer_step.stop();
    }
    
    bool TriangleSoup::splitEdge(double lambda_t, double thres, bool propagate, bool splitInterior)
    {
        double EwDec_max;
        std::vector<int> path_max;
        Eigen::MatrixXd newVertPos_max;
        std::pair<double, double> energyChanges_max;
        querySplit(lambda_t, propagate, splitInterior,
                   EwDec_max, path_max, newVertPos_max,
                   energyChanges_max);
        
        std::cout << "E_dec threshold = " << thres << std::endl;
        if(EwDec_max > thres) {
            if(!splitInterior) {
                // boundary split
                std::cout << "boundary split E_dec = " << EwDec_max << std::endl;
                splitEdgeOnBoundary(std::pair<int, int>(path_max[0], path_max[1]),
                                    newVertPos_max);
                //TODO: process fractail here!
                updateFeatures();
            }
            else {
                // interior split
                assert(!propagate);
                std::cout << "interior split E_dec = " << EwDec_max << std::endl;
                cutPath(path_max, true, 1, newVertPos_max);
                logFile << "interior edge splitted" << std::endl;
                fracTail.insert(path_max[0]);
                fracTail.insert(path_max[2]);
                curInteriorFracTails.first = path_max[0];
                curInteriorFracTails.second = path_max[2];
                curFracTail = -1;
            }
            return true;
        }
        else {
            std::cout << "max E_dec = " << EwDec_max << " < thres" << std::endl;
            return false;
        }
    }
    
    void TriangleSoup::queryMerge(double lambda, bool propagate,
                                  double& localEwDec_max, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max,
                                  std::pair<double, double>& energyChanges_max)
    {
        //TODO: local index updates in mergeBoundaryEdge()
        //TODO: parallelize the query
        timer_step.start(8);
        
        std::cout << "evaluate edge merge, " << cohE.rows() << " cohesive edge pairs." << std::endl;
        localEwDec_max = -__DBL_MAX__;
        if(!propagate) {
            paths_merge.resize(0);
            newVertPoses_merge.resize(0);
            energyChanges_merge.resize(0);
        }
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            int forkVI = 0;
            if(cohE(cohI, 0) == cohE(cohI, 2)) {
                forkVI = 0;
            }
            else if(cohE(cohI, 1) == cohE(cohI, 3)) {
                forkVI = 1;
            }
            else {
                //!!! only consider "zipper bottom" edge pairs for now
                continue;
            }
            
            if(propagate) {
                if(cohE(cohI, forkVI) != curFracTail) {
                    continue;
                }
            }
            
            // find incident triangles for inversion check and local energy decrease evaluation
            std::vector<int> triangles;
            int firstVertIncTriAmt = 0;
            for(int mergeVI = 1; mergeVI <= 3; mergeVI += 2) {
                for(const auto& nbVI : vNeighbor[cohE(cohI, mergeVI - forkVI)]) {
                    auto finder = edge2Tri.find(std::pair<int, int>(cohE(cohI, mergeVI - forkVI), nbVI));
                    if(finder != edge2Tri.end()) {
                        triangles.emplace_back(finder->second);
                    }
                }
                if(mergeVI == 1) {
                    firstVertIncTriAmt = static_cast<int>(triangles.size());
                    assert(firstVertIncTriAmt >= 1);
                }
            }
            
            Eigen::RowVector2d mergedPos = (V.row(cohE(cohI, 1 - forkVI)) + V.row(cohE(cohI, 3 - forkVI))) / 2.0;
            const Eigen::RowVector2d backup0 = V.row(cohE(cohI, 1 - forkVI));
            const Eigen::RowVector2d backup1 = V.row(cohE(cohI, 3 - forkVI));
            V.row(cohE(cohI, 1 - forkVI)) = mergedPos;
            V.row(cohE(cohI, 3 - forkVI)) = mergedPos;
            if(checkInversion(true, triangles)) {
                V.row(cohE(cohI, 1 - forkVI)) = backup0;
                V.row(cohE(cohI, 3 - forkVI)) = backup1;
            }
            else {
                // project mergedPos to feasible set via Relaxation method for linear inequalities
                
                // find inequality constraints by opposite edge in incident triangles
                Eigen::MatrixXd inequalityConsMtr;
                Eigen::VectorXd inequalityConsVec;
                for(int triII = 0; triII < triangles.size(); triII++) {
                    int triI = triangles[triII];
                    int vI_toMerge = ((triII < firstVertIncTriAmt) ? cohE(cohI, 1 - forkVI) : cohE(cohI, 3 - forkVI));
                    for(int i = 0; i < 3; i++) {
                        if(F(triI, i) == vI_toMerge) {
                            const Eigen::RowVector2d& v1 = V.row(F(triI, (i + 1) % 3));
                            const Eigen::RowVector2d& v2 = V.row(F(triI, (i + 2) % 3));
                            Eigen::RowVector2d coef(v2[1] - v1[1], v1[0] - v2[0]);
                            inequalityConsMtr.conservativeResize(inequalityConsMtr.rows() + 1, 2);
                            inequalityConsMtr.row(inequalityConsMtr.rows() - 1) = coef / coef.norm();
                            inequalityConsVec.conservativeResize(inequalityConsVec.size() + 1);
                            inequalityConsVec[inequalityConsVec.size() - 1] = (v1[0] * v2[1] - v1[1] * v2[0]) / coef.norm();
                            break;
                        }
                    }
                }
                assert(inequalityConsMtr.rows() == triangles.size());
                assert(inequalityConsVec.size() == triangles.size());
                
                // Relaxation method for linear inequalities
//                logFile << "Relaxation method for linear inequalities" << std::endl; //DEBUG
                int maxIter = 70;
                const double eps_IC = 1.0e-6 * avgEdgeLen;
                for(int iterI = 0; iterI < maxIter; iterI++) {
                    double maxRes = -__DBL_MAX__;
                    for(int consI = 0; consI < inequalityConsMtr.rows(); consI++) {
                        double res = inequalityConsMtr.row(consI) * mergedPos.transpose() - inequalityConsVec[consI];
                        if(res > -eps_IC) {
                            // project
                            mergedPos -= (res + eps_IC) * inequalityConsMtr.row(consI).transpose();
                        }
                        if(res > maxRes) {
                            maxRes = res;
                        }
                    }
                    
//                    logFile << maxRes << std::endl; //DEBUG
                    if(maxRes < 0.0) {
                        // converged (non-inversion satisfied)
                        //NOTE: although this maxRes is 1 iteration behind, it is OK for a convergence check
                        break;
                    }
                }
                
                V.row(cohE(cohI, 1 - forkVI)) = mergedPos;
                V.row(cohE(cohI, 3 - forkVI)) = mergedPos;
                bool noInversion = checkInversion(true, triangles);
                V.row(cohE(cohI, 1 - forkVI)) = backup0;
                V.row(cohE(cohI, 3 - forkVI)) = backup1;
                if(!noInversion) {
                    // because propagation is not at E_SD stationary, so it's possible to have no feasible region
                    continue;
                }
            }
            
            // optimize local distortion
            std::vector<int> path;
            if(forkVI) {
                path.emplace_back(cohE(cohI, 0));
                path.emplace_back(cohE(cohI, 1));
                path.emplace_back(cohE(cohI, 2));
            }
            else {
                path.emplace_back(cohE(cohI, 3));
                path.emplace_back(cohE(cohI, 2));
                path.emplace_back(cohE(cohI, 1));
            }
            Eigen::MatrixXd newVertPos;
            std::pair<double, double> energyChanges;
            double localEwDec = computeLocalEwDec(0, lambda, path, newVertPos, energyChanges, triangles, mergedPos);
            if(!propagate) {
                paths_merge.emplace_back(path);
                newVertPoses_merge.emplace_back(newVertPos);
                energyChanges_merge.emplace_back(energyChanges);
            }
            
            if(localEwDec > localEwDec_max) {
                localEwDec_max = localEwDec;
                newVertPos_max = newVertPos;
                path_max = path;
                energyChanges_max = energyChanges;
            }
        }
        
        timer_step.stop();
    }
    
    bool TriangleSoup::mergeEdge(double lambda, double EDecThres, bool propagate)
    {
        double localEwDec_max;
        std::vector<int> path_max;
        Eigen::MatrixXd newVertPos_max;
        std::pair<double, double> energyChanges_max;
        queryMerge(lambda, propagate, localEwDec_max, path_max, newVertPos_max,
                   energyChanges_max);
        
        std::cout << "E_dec threshold = " << EDecThres << std::endl;
        if(localEwDec_max > EDecThres) {
            std::cout << "merge edge E_dec = " << localEwDec_max << std::endl;
            mergeBoundaryEdges(std::pair<int, int>(path_max[0], path_max[1]),
                               std::pair<int, int>(path_max[1], path_max[2]), newVertPos_max.row(0));
            logFile << "edge merged" << std::endl;

            computeFeatures(); //TODO: only update locally
            return true;
        }
        else {
            std::cout << "max E_dec = " << localEwDec_max << " < thres" << std::endl;
            return false;
        }
    }
    
    bool TriangleSoup::splitOrMerge(double lambda_t, double EDecThres, bool propagate, bool splitInterior,
                                    bool& isMerge)
    {
        assert((!propagate) && "propagation is supported separately for split and merge!");
        
        double EwDec_max;
        std::vector<int> path_max;
        Eigen::MatrixXd newVertPos_max;
        isMerge = false;
        std::pair<double, double> energyChanes_split, energyChanes_merge;
        if(splitInterior) {
            querySplit(lambda_t, propagate, splitInterior,
                       EwDec_max, path_max, newVertPos_max,
                       energyChanes_split);
        }
        else {
            double EwDec_max_split, EwDec_max_merge;
            std::vector<int> path_max_split, path_max_merge;
            Eigen::MatrixXd newVertPos_max_split, newVertPos_max_merge;
            querySplit(lambda_t, propagate, splitInterior,
                       EwDec_max_split, path_max_split, newVertPos_max_split,
                       energyChanes_split);
            queryMerge(lambda_t, propagate, EwDec_max_merge, path_max_merge, newVertPos_max_merge,
                       energyChanes_merge);
            
            if(EwDec_max_merge > EwDec_max_split) {
                isMerge = true;
                EwDec_max = EwDec_max_merge;
                path_max = path_max_merge;
                newVertPos_max = newVertPos_max_merge;
            }
            else {
                EwDec_max = EwDec_max_split;
                path_max = path_max_split;
                newVertPos_max = newVertPos_max_split;
            }
        }
        
        if(EwDec_max > EDecThres) {
            if(isMerge) {
                std::cout << "merge edge E_dec = " << EwDec_max << std::endl;
                mergeBoundaryEdges(std::pair<int, int>(path_max[0], path_max[1]),
                                   std::pair<int, int>(path_max[1], path_max[2]), newVertPos_max.row(0));
                logFile << "edge merged" << std::endl;
                computeFeatures(); //TODO: only update locally
            }
            else {
                if(!splitInterior) {
                    // boundary split
                    std::cout << "boundary split E_dec = " << EwDec_max << std::endl;
                    splitEdgeOnBoundary(std::pair<int, int>(path_max[0], path_max[1]),
                                        newVertPos_max);
                    logFile << "boundary edge splitted" << std::endl;
                    //TODO: process fractail here!
                    updateFeatures();
                }
                else {
                    // interior split
                    std::cout << "Interior split E_dec = " << EwDec_max << std::endl;
                    cutPath(path_max, true, 1, newVertPos_max);
                    logFile << "interior edge splitted" << std::endl;
                    fracTail.insert(path_max[0]);
                    fracTail.insert(path_max[2]);
                    curInteriorFracTails.first = path_max[0];
                    curInteriorFracTails.second = path_max[2];
                    curFracTail = -1;
                }
            }
            return true;
        }
        else {
            std::cout << "max E_dec = " << EwDec_max << " < thres" << std::endl;
            return false;
        }
    }
    
    void TriangleSoup::onePointCut(int vI)
    {
        assert((vI >= 0) && (vI < V_rest.rows()));
        std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());
        assert(path.size() >= 3);
        path[1] = vI;
        path.resize(3);
        
        bool makeCoh = true;
        if(!makeCoh) {
            for(int pI = 0; pI + 1 < path.size(); pI++) {
                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
            }
        }
        
        cutPath(path, makeCoh);
        
        if(makeCoh) {
            initSeams = cohE;
        }
    }
    
    void TriangleSoup::highCurvOnePointCut(void)
    {
        std::vector<double> gaussianCurv(V.rows(), 2.0 * M_PI);
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            const Eigen::RowVector3d v[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            for(int vI = 0; vI < 3; vI++) {
                int vI_post = (vI + 1) % 3;
                int vI_pre = (vI + 2) % 3;
                const Eigen::RowVector3d e0 = v[vI_pre] - v[vI];
                const Eigen::RowVector3d e1 = v[vI_post] - v[vI];
                gaussianCurv[triVInd[vI]] -= std::acos(std::max(-1.0, std::min(1.0, e0.dot(e1) / e0.norm() / e1.norm())));
            }
        }
        
        for(auto& gcI : gaussianCurv) {
            if(gcI < 0) {
                gcI = -gcI;
            }
        }
        int vI_maxGC = 0;
        for(int vI = 1; vI < gaussianCurv.size(); vI++) {
            if(gaussianCurv[vI] > gaussianCurv[vI_maxGC]) {
                vI_maxGC = vI;
            }
        }
        
        assert(vNeighbor[vI_maxGC].size() >= 3);
        int vJ_maxGC = *vNeighbor[vI_maxGC].begin();
        for(const auto& vINb : vNeighbor[vI_maxGC]) {
            if(gaussianCurv[vINb] > gaussianCurv[vJ_maxGC]) {
                vJ_maxGC = vINb;
            }
        }
        
        //!!! use smoothest cut as initial?
        assert(vNeighbor[vJ_maxGC].size() >= 3);
        int vK_maxGC = -1;
        double gc_vK = -__DBL_MAX__;
        for(const auto& vJNb : vNeighbor[vJ_maxGC]) {
            if((gaussianCurv[vJNb] > gc_vK) &&
               (vJNb != vI_maxGC))
            {
                vK_maxGC = vJNb;
                gc_vK = gaussianCurv[vJNb];
            }
        }
            
        std::vector<int> path(3);
        path[0] = vI_maxGC;
        path[1] = vJ_maxGC;
        path[2] = vK_maxGC;
        
        bool makeCoh = true;
        if(!makeCoh) {
            for(int pI = 0; pI + 1 < path.size(); pI++) {
                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
            }
        }
        
        cutPath(path, makeCoh);
        
        if(makeCoh) {
            initSeams = cohE;
        }
    }
    
    // A utility function to find the vertex with minimum distance value, from
    // the set of vertices not yet included in shortest path tree
    int minDistance(const std::vector<double>& dist, const std::vector<bool>& sptSet)
    {
        // Initialize min value
        double min = __DBL_MAX__;
        int min_index = -1;
        
        for (int v = 0; v < dist.size(); v++) {
            if ((!sptSet[v]) && (dist[v] <= min)) {
                min = dist[v], min_index = v;
            }
        }
        
        return min_index;
    }
    
    // Funtion that implements Dijkstra's single source shortest path algorithm
    // for a graph represented using adjacency matrix representation
    void dijkstra(const std::vector<std::map<int, double>>& graph, int src, std::vector<double>& dist, std::vector<int>& parent)
    {
        int nV = static_cast<int>(graph.size());
        
        dist.resize(0);
        dist.resize(nV, __DBL_MAX__);
        // The output array.  dist[i] will hold the shortest
        // distance from src to i
        
        std::vector<bool> sptSet(nV, false); // sptSet[i] will true if vertex i is included in shortest
        // path tree or shortest distance from src to i is finalized
        
        parent.resize(0);
        parent.resize(nV, -1);
        
        // Distance of source vertex from itself is always 0
        dist[src] = 0.0;
        
        // Find shortest path for all vertices
        for (int count = 0; count + 1 < nV; count++)
        {
            // Pick the minimum distance vertex from the set of vertices not
            // yet processed. u is always equal to src in first iteration.
            int u = minDistance(dist, sptSet);
            
            // Mark the picked vertex as processed
            sptSet[u] = true;
            
            for(const auto v : graph[u]) {
                // Update dist[v] only if is not in sptSet, there is an edge from
                // u to v, and total weight of path from src to  v through u is
                // smaller than current value of dist[v]
                if ((!sptSet[v.first]) && (dist[u] != __DBL_MAX__)
                    && (dist[u] + v.second < dist[v.first]))
                {
                    dist[v.first] = dist[u] + v.second;
                    parent[v.first] = u;
                }
            }
        }
    }
    
    int getFarthestPoint(const std::vector<std::map<int, double>>& graph, int src)
    {
        int nV = static_cast<int>(graph.size());
        std::vector<double> dist;
        std::vector<int> parent;
        dijkstra(graph, src, dist, parent);
        
        double maxDist = 0.0;
        int vI_maxDist = -1;
        for(int vI = 0; vI < nV; vI++) {
            if(dist[vI] > maxDist) {
                maxDist = dist[vI];
                vI_maxDist = vI;
            }
        }
        assert(vI_maxDist >= 0);
        return vI_maxDist;
    }
    
    void getFarthestPointPath(const std::vector<std::map<int, double>>& graph, int src, std::vector<int>& path)
    {
        int nV = static_cast<int>(graph.size());
        std::vector<double> dist;
        std::vector<int> parent;
        dijkstra(graph, src, dist, parent);
        
        double maxDist = 0.0;
        int vI_maxDist = -1;
        for(int vI = 0; vI < nV; vI++) {
            if(dist[vI] > maxDist) {
                maxDist = dist[vI];
                vI_maxDist = vI;
            }
        }
        assert(vI_maxDist >= 0);
        path.resize(0);
        while(vI_maxDist >= 0) {
            path.emplace_back(vI_maxDist);
            vI_maxDist = parent[vI_maxDist];
        }
        std::reverse(path.begin(), path.end());
    }
    
    void TriangleSoup::farthestPointCut(void)
    {
        assert(vNeighbor.size() == V_rest.rows());
        
        std::vector<std::map<int, double>> graph(vNeighbor.size());
        for(int vI = 0; vI < vNeighbor.size(); vI++) {
            for(const auto nbI : vNeighbor[vI]) {
                if(nbI > vI) {
                    graph[nbI][vI] = graph[vI][nbI] = (V_rest.row(vI) - V_rest.row(nbI)).norm();
                }
            }
        }
        
        std::vector<int> path;
        getFarthestPointPath(graph, getFarthestPoint(graph, 0), path);
        
        bool makeCoh = true;
        if(!makeCoh) {
            for(int pI = 0; pI + 1 < path.size(); pI++) {
                initSeamLen += (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
            }
        }
        
        cutPath(path, makeCoh);
//        save("/Users/mincli/Desktop/meshes/test_triSoup.obj");
//        saveAsMesh("/Users/mincli/Desktop/meshes/test_mesh.obj");
        
        if(makeCoh) {
            initSeams = cohE;
        }
    }
    
    void TriangleSoup::geomImgCut(TriangleSoup& data_findExtrema)
    {
        // compute UV map for find extremal point (interior)
        data_findExtrema = *this;
        const int mapType = 0; // 0: SD, 1: harmonic (uniform), 2: harmonic (cotangent), 3: harmonic (MVC)
        if(mapType) {
            Eigen::VectorXi bnd;
            igl::boundary_loop(this->F, bnd); // Find the open boundary
            assert(bnd.size());
            //TODO: ensure it doesn't have multiple boundaries? or multi-components?
            
            // Map the boundary to a circle, preserving edge proportions
            Eigen::MatrixXd bnd_uv;
            //            igl::map_vertices_to_circle(V, bnd, bnd_uv);
            FracCuts::IglUtils::map_vertices_to_circle(this->V_rest, bnd, bnd_uv);
            
            Eigen::MatrixXd UV_Tutte;
            
            switch (mapType) {
                case 1: {
                    // Harmonic map with uniform weights
                    Eigen::SparseMatrix<double> A, M;
                    FracCuts::IglUtils::computeUniformLaplacian(this->F, A);
                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                    break;
                }
                    
                case 2: {
                    // Harmonic parametrization
                    igl::harmonic(V, F, bnd, bnd_uv, 1, UV_Tutte);
                    break;
                }
                    
                case 3: {
                    // Shape Preserving Mesh Parameterization
                    // (Harmonic map with MVC weights)
                    Eigen::SparseMatrix<double> A;
                    FracCuts::IglUtils::computeMVCMtr(this->V_rest, this->F, A);
                    FracCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV_Tutte);
                    break;
                }
                    
                default:
                    assert(0 && "Unknown map specified for finding Geometry Image cuts");
                    break;
            }
            
            
            data_findExtrema.V = UV_Tutte;
        }
        
        // pick the vertex with largest L2 stretch
        int vI_extremal = -1;
        Eigen::VectorXd L2stretchPerElem, vertScores;
        data_findExtrema.computeL2StretchPerElem(L2stretchPerElem);
        vertScores.resize(V_rest.rows());
        vertScores.setZero();
        for(int triI = 0; triI < F.rows(); triI++) {
            for(int i = 0; i < 3; i++) {
                if(vertScores[F(triI, i)] < L2stretchPerElem[triI]) {
                    vertScores[F(triI, i)] = L2stretchPerElem[triI];
                }
            }
        }
        double extremal = 0.0;
        for(int vI = 0; vI < vertScores.size(); vI++) {
            if(!isBoundaryVert(vI)) {
                if(extremal < vertScores[vI]) {
                    extremal = vertScores[vI];
                    vI_extremal = vI;
                }
            }
        }
        assert(vI_extremal >= 0);
        
        // construct mesh graph
        assert(vNeighbor.size() == V_rest.rows());
        std::vector<std::map<int, double>> graph(vNeighbor.size());
        for(int vI = 0; vI < vNeighbor.size(); vI++) {
            for(const auto nbI : vNeighbor[vI]) {
                if(nbI > vI) {
                    graph[nbI][vI] = graph[vI][nbI] = (V_rest.row(vI) - V_rest.row(nbI)).norm();
                }
            }
        }
        
        // find closest point on boundary
        int nV = static_cast<int>(graph.size());
        std::vector<double> dist;
        std::vector<int> parent;
        dijkstra(graph, vI_extremal, dist, parent);
        double minDistToBound = __DBL_MAX__;
        int vI_minDistToBound = -1;
        for(int vI = 0; vI < nV; vI++) {
            if(isBoundaryVert(vI)) {
                if(dist[vI] < minDistToBound) {
                    minDistToBound = dist[vI];
                    vI_minDistToBound = vI;
                }
            }
        }
        assert((vI_minDistToBound >= 0) && "No boundary on the mesh!");
        
        // find shortest path to closest point on boundary
        std::vector<int> path;
        while(vI_minDistToBound >= 0) {
            path.emplace_back(vI_minDistToBound);
            vI_minDistToBound = parent[vI_minDistToBound];
        }
        std::reverse(path.begin(), path.end());
        
        cutPath(path, true);
//        save("/Users/mincli/Desktop/meshes/test_triSoup.obj");
//        saveAsMesh("/Users/mincli/Desktop/meshes/test_mesh.obj");
    }
    
    void TriangleSoup::cutPath(std::vector<int> path, bool makeCoh, int changePos,
                               const Eigen::MatrixXd& newVertPos, bool allowCutThrough)
    {
        assert(path.size() >= 2);
        if(changePos) {
            assert((changePos == 1) && "right now only support change 1"); //!!! still only allow 1?
            assert(newVertPos.cols() == 2);
            assert(changePos * 2 == newVertPos.rows());
        }
        
        for(int pI = 1; pI + 1 < path.size(); pI++) {
            assert(!isBoundaryVert(path[pI]) &&
                   "Boundary vertices detected on split path, please split multiple times!");
        }
        
        bool isFromBound = isBoundaryVert(path[0]);
        bool isToBound = isBoundaryVert(path.back());
        if(allowCutThrough && (isFromBound || isToBound)) {
            bool cutThrough = false;
            if(isFromBound && isToBound) {
                cutThrough = true;
            }
            else if(isToBound) {
                std::reverse(path.begin(), path.end());
                // always start cut from boundary
            }
            
            for(int vI = 0; vI + 1 < path.size(); vI++) {
                //TODO: enable change pos!
                int vInd_s = path[vI];
                int vInd_e = path[vI + 1];
                assert(edge2Tri.find(std::pair<int, int>(vInd_s, vInd_e)) != edge2Tri.end());
                assert(edge2Tri.find(std::pair<int, int>(vInd_e, vInd_s)) != edge2Tri.end());
                Eigen::MatrixXd newVertPos;
                if(cutThrough) {
                    newVertPos.resize(4, 2);
                    newVertPos << V.row(vInd_s), V.row(vInd_s), V.row(vInd_e), V.row(vInd_e);
                }
                else {
                    newVertPos.resize(2, 2);
                    newVertPos << V.row(vInd_s), V.row(vInd_s);
                }
                splitEdgeOnBoundary(std::pair<int, int>(vInd_s, vInd_e), newVertPos, true); //!!! make coh?
                updateFeatures();
            }
        }
        else {
            // path is interior
            assert(path.size() >= 3);
            
            std::vector<int> tri_left;
            int vI = path[1];
            int vI_new = path[0];
            while(1) {
                auto finder = edge2Tri.find(std::pair<int, int>(vI, vI_new));
                assert(finder != edge2Tri.end());
                tri_left.emplace_back(finder->second);
                const Eigen::RowVector3i& triVInd = F.row(finder->second);
                for(int i = 0; i < 3; i++) {
                    if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                        vI_new = triVInd[i];
                        break;
                    }
                }
                
                if(vI_new == path[2]) {
                    break;
                }
                if(vI_new == path[0]) {
                    assert(0 && "not a valid path!");
                }
            }
            
            int nV = static_cast<int>(V_rest.rows());
            V_rest.conservativeResize(nV + 1, 3);
            V_rest.row(nV) = V_rest.row(path[1]);
            vertWeight.conservativeResize(nV + 1);
            vertWeight[nV] = vertWeight[path[1]];
            V.conservativeResize(nV + 1, 2);
            if(changePos) {
                V.row(nV) = newVertPos.block(0, 0, 1, 2);
                V.row(path[1]) = newVertPos.block(1, 0, 1, 2);
            }
            else {
                V.row(nV) = V.row(path[1]);
            }
            for(const auto triI : tri_left) {
                for(int vI = 0; vI < 3; vI++) {
                    if(F(triI, vI) == path[1]) {
                        F(triI, vI) = nV;
                        break;
                    }
                }
            }
            if(makeCoh) {
                int nCoh = static_cast<int>(cohE.rows());
                cohE.conservativeResize(nCoh + 2, 4);
                cohE.row(nCoh) << nV, path[0], path[1], path[0];
                cohE.row(nCoh + 1) << path[2], nV, path[2], path[1];
            }
            
            computeFeatures(); //TODO: only update locally
            
            for(int vI = 2; vI + 1 < path.size(); vI++) {
                //TODO: enable change pos!
                int vInd_s = path[vI];
                int vInd_e = path[vI + 1];
                assert(edge2Tri.find(std::pair<int, int>(vInd_s, vInd_e)) != edge2Tri.end());
                assert(edge2Tri.find(std::pair<int, int>(vInd_e, vInd_s)) != edge2Tri.end());
                Eigen::Matrix2d newVertPos;
                newVertPos << V.row(vInd_s), V.row(vInd_s);
                splitEdgeOnBoundary(std::pair<int, int>(vInd_s, vInd_e), newVertPos, true, allowCutThrough); //!!! make coh?
                updateFeatures();
            }
        }
    }
    
    void TriangleSoup::computeSeamScore(Eigen::VectorXd& seamScore) const
    {
        seamScore.resize(cohE.rows());
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(boundaryEdge[cohI]) {
                seamScore[cohI] = -1.0;
            }
            else {
                seamScore[cohI] = std::max((V.row(cohE(cohI, 0)) - V.row(cohE(cohI, 2))).norm(),
                    (V.row(cohE(cohI, 1)) - V.row(cohE(cohI, 3))).norm()) / avgEdgeLen;
            }
        }
    }
    void TriangleSoup::computeBoundaryLen(double& boundaryLen) const
    {
        boundaryLen = 0.0;
        for(const auto& e : edge2Tri) {
            if(edge2Tri.find(std::pair<int, int>(e.first.second, e.first.first)) == edge2Tri.end()) {
                boundaryLen += (V_rest.row(e.first.second) - V_rest.row(e.first.first)).norm();
            }
        }
    }
    void TriangleSoup::computeSeamSparsity(double& sparsity, bool triSoup) const
    {
        const double thres = 1.0e-2;
        sparsity = 0.0;
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(!boundaryEdge[cohI]) {
                if((!triSoup) ||
                   ((V.row(cohE(cohI, 0)) - V.row(cohE(cohI, 2))).norm() / avgEdgeLen > thres) ||
                   ((V.row(cohE(cohI, 1)) - V.row(cohE(cohI, 3))).norm() / avgEdgeLen > thres))
                {
                    sparsity += edgeLen[cohI];
                }
            }
        }
        sparsity += initSeamLen;
    }
    void TriangleSoup::computeL2StretchPerElem(Eigen::VectorXd& L2StretchPerElem) const
    {
        L2StretchPerElem.resize(F.rows());
        for(int triI = 0; triI < F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = F.row(triI);
            const Eigen::Vector3d x_3D[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            const Eigen::Vector2d uv[3] = {
                V.row(triVInd[0]),
                V.row(triVInd[1]),
                V.row(triVInd[2])
            };
            Eigen::Matrix2d dg;
            IglUtils::computeDeformationGradient(x_3D, uv, dg);
            
            const double a = Eigen::Vector2d(dg.block(0, 0, 2, 1)).squaredNorm();
            const double c = Eigen::Vector2d(dg.block(0, 1, 2, 1)).squaredNorm();
            const double t0 = a + c;
            
            L2StretchPerElem[triI] = std::sqrt(t0 / 2.0);
        }
    }
    void TriangleSoup::computeStandardStretch(double& stretch_l2, double& stretch_inf, double& stretch_shear, double& compress_inf) const
    {
        stretch_l2 = 0.0;
        stretch_inf = -__DBL_MAX__;
        stretch_shear = 0.0;
        compress_inf = __DBL_MAX__;
        for(int triI = 0; triI < F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = F.row(triI);
            const Eigen::Vector3d x_3D[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            const Eigen::Vector2d uv[3] = {
                V.row(triVInd[0]),
                V.row(triVInd[1]),
                V.row(triVInd[2])
            };
            Eigen::Matrix2d dg;
            IglUtils::computeDeformationGradient(x_3D, uv, dg);
            
            const double a = Eigen::Vector2d(dg.block(0, 0, 2, 1)).squaredNorm();
            const double b = Eigen::Vector2d(dg.block(0, 0, 2, 1)).dot(Eigen::Vector2d(dg.block(0, 1, 2, 1)));
            const double c = Eigen::Vector2d(dg.block(0, 1, 2, 1)).squaredNorm();
            const double t0 = a + c;
            const double t1 = std::sqrt((a - c) * (a - c) + 4. * b * b);
            const double tau = std::sqrt((t0 + t1) / 2.);
            const double gamma = std::sqrt((t0 - t1) / 2.);
            
            stretch_l2 += t0 / 2.0 * triArea[triI];
            
            if(tau > stretch_inf) {
                stretch_inf = tau;
            }
            
            stretch_shear += b * b / a / c * triArea[triI];
            
            if(gamma < compress_inf) {
                compress_inf = gamma;
            }
        }
        stretch_l2 /= surfaceArea;
        stretch_l2 = std::sqrt(stretch_l2);
        stretch_shear /= surfaceArea;
        stretch_shear = std::sqrt(stretch_shear);
        
        double surfaceArea_UV = 0.0;
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector2d& U1 = V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            surfaceArea_UV += 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
        }
        
        // area scaling:
        const double scaleFactor = std::sqrt(surfaceArea_UV / surfaceArea);
        stretch_l2 *= scaleFactor;
        stretch_inf *= scaleFactor;
        compress_inf *= scaleFactor; // not meaningful now...
        // stretch_shear won't be affected by area scaling
    }
    void TriangleSoup::outputStandardStretch(std::ofstream& file) const
    {
        double stretch_l2, stretch_inf, stretch_shear, compress_inf;
        computeStandardStretch(stretch_l2, stretch_inf, stretch_shear, compress_inf);
        file << stretch_l2 << " " << stretch_inf << " " << stretch_shear << " " << compress_inf << std::endl;
    }
    void TriangleSoup::computeAbsGaussianCurv(double& absGaussianCurv) const
    {
        //!!! it's easy to optimize this way actually...
        std::vector<double> weights(V.rows(), 0.0);
        std::vector<double> gaussianCurv(V.rows(), 2.0 * M_PI);
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            const Eigen::RowVector3d v[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            for(int vI = 0; vI < 3; vI++) {
                int vI_post = (vI + 1) % 3;
                int vI_pre = (vI + 2) % 3;
                const Eigen::RowVector3d e0 = v[vI_pre] - v[vI];
                const Eigen::RowVector3d e1 = v[vI_post] - v[vI];
                gaussianCurv[triVInd[vI]] -= std::acos(std::max(-1.0, std::min(1.0, e0.dot(e1) / e0.norm() / e1.norm())));
                weights[triVInd[vI]] += triArea[triI];
            }
        }
        
        absGaussianCurv = 0.0;
        for(int vI = 0; vI < V.rows(); vI++) {
            if(!isBoundaryVert(vI)) {
                absGaussianCurv += std::abs(gaussianCurv[vI]) * weights[vI];
            }
        }
        absGaussianCurv /= surfaceArea * 3.0;
    }

    void TriangleSoup::initRigidUV(void)
    {
        V.resize(V_rest.rows(), 2);
        for(int triI = 0; triI < F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector3d x_3D[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x[3];
            IglUtils::mapTriangleTo2D(x_3D, x);
            
            V.row(triVInd[0]) = x[0];
            V.row(triVInd[1]) = x[1];
            V.row(triVInd[2]) = x[2];
        }
    }
    
    bool TriangleSoup::checkInversion(int triI, bool mute) const
    {
        assert(triI < F.rows());
        
        const double eps = 0.0;//1.0e-20 * avgEdgeLen * avgEdgeLen;

        const Eigen::Vector3i& triVInd = F.row(triI);
        
        const Eigen::Vector2d e_u[2] = {
            V.row(triVInd[1]) - V.row(triVInd[0]),
            V.row(triVInd[2]) - V.row(triVInd[0])
        };
        
        const double dbArea = e_u[0][0] * e_u[1][1] - e_u[0][1] * e_u[1][0];
        if(dbArea < eps) {
            if(!mute) {
                std::cout << "***Element inversion detected: " << dbArea << " < " << eps << std::endl;
                std::cout << "mesh triangle count: " << F.rows() << std::endl;
                logFile << "***Element inversion detected: " << dbArea << " < " << eps << std::endl;
            }
            return false;
        }
        else {
            return true;
        }
    }
    bool TriangleSoup::checkInversion(bool mute, const std::vector<int>& triangles) const
    {
        if(triangles.empty()) {
            for(int triI = 0; triI < F.rows(); triI++)
            {
                if(!checkInversion(triI, mute)) {
                    return false;
                }
            }
        }
        else {
            for(const auto& triI : triangles)
            {
                if(!checkInversion(triI, mute)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    void TriangleSoup::save(const std::string& filePath, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                            const Eigen::MatrixXd UV, const Eigen::MatrixXi& FUV) const
    {
        std::ofstream out;
        out.open(filePath);
        assert(out.is_open());
        
        for(int vI = 0; vI < V.rows(); vI++) {
            const Eigen::RowVector3d& v = V.row(vI);
            out << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
        }
        
        for(int vI = 0; vI < UV.rows(); vI++) {
            const Eigen::RowVector2d& uv = UV.row(vI);
            out << "vt " << uv[0] << " " << uv[1] << std::endl;
        }
        
        if(FUV.rows() == F.rows()) {
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::RowVector3i& tri = F.row(triI);
                const Eigen::RowVector3i& tri_UV = FUV.row(triI);
                out << "f " << tri[0] + 1 << "/" << tri_UV[0] + 1 <<
                " " << tri[1] + 1 << "/" << tri_UV[1] + 1 <<
                " " << tri[2] + 1 << "/" << tri_UV[2] + 1 << std::endl;
            }
        }
        else {
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::RowVector3i& tri = F.row(triI);
                out << "f " << tri[0] + 1 << "/" << tri[0] + 1 <<
                " " << tri[1] + 1 << "/" << tri[1] + 1 <<
                " " << tri[2] + 1 << "/" << tri[2] + 1 << std::endl;
            }
        }
        
        out.close();
    }
    
    void TriangleSoup::save(const std::string& filePath) const
    {
        save(filePath, V_rest, F, V);
    }
    
    void TriangleSoup::saveAsMesh(const std::string& filePath, bool scaleUV) const
    {
        const double thres = 1.0e-2;
        std::vector<int> dupVI2GroupI(V.rows());
        std::vector<std::set<int>> meshVGroup(V.rows());
        std::vector<int> dupVI2GroupI_3D(V_rest.rows());
        std::vector<std::set<int>> meshVGroup_3D(V_rest.rows());
        for(int dupI = 0; dupI < V.rows(); dupI++) {
            dupVI2GroupI[dupI] = dupI;
            meshVGroup[dupI].insert(dupI);
            dupVI2GroupI_3D[dupI] = dupI;
            meshVGroup_3D[dupI].insert(dupI);
        }
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            if(boundaryEdge[cohI]) {
                continue;
            }
            
            for(int pI = 0; pI < 2; pI++) {
                int groupI0_3D = dupVI2GroupI_3D[cohE(cohI, 0 + pI)];
                int groupI2_3D = dupVI2GroupI_3D[cohE(cohI, 2 + pI)];
                if(groupI0_3D != groupI2_3D) {
                    for(const auto& vI : meshVGroup_3D[groupI2_3D]) {
                        dupVI2GroupI_3D[vI] = groupI0_3D;
                        meshVGroup_3D[groupI0_3D].insert(vI);
                    }
                    meshVGroup_3D[groupI2_3D].clear();
                }
           
                // this is only for gluing the triangle soup
//                if((V.row(cohE(cohI, 0 + pI)) - V.row(cohE(cohI, 2 + pI))).norm() / avgEdgeLen > thres) {
                    continue;
//                }
                
                int groupI0 = dupVI2GroupI[cohE(cohI, 0 + pI)];
                int groupI2 = dupVI2GroupI[cohE(cohI, 2 + pI)];
                if(groupI0 != groupI2) {
                    for(const auto& vI : meshVGroup[groupI2]) {
                        dupVI2GroupI[vI] = groupI0;
                        meshVGroup[groupI0].insert(vI);
                    }
                    meshVGroup[groupI2].clear();
                }
            }
        }
        
        int meshVAmt = 0;
        for(int gI = 0; gI < meshVGroup.size(); gI++) {
            if(!meshVGroup[gI].empty()) {
                meshVAmt++;
            }
        }
        std::vector<int> groupI2meshVI(meshVGroup.size(), -1);
        Eigen::MatrixXd UV_mesh;
        UV_mesh.resize(meshVAmt, 2);
        
        int meshVAmt_3D = 0;
        for(int gI = 0; gI < meshVGroup_3D.size(); gI++) {
            if(!meshVGroup_3D[gI].empty()) {
                meshVAmt_3D++;
            }
        }
        std::vector<int> groupI2meshVI_3D(meshVGroup_3D.size(), -1);
        Eigen::MatrixXd V_mesh;
        V_mesh.resize(meshVAmt_3D, 3);
        
        int nextVI = 0;
        for(int gI = 0; gI < meshVGroup.size(); gI++) {
            if(meshVGroup[gI].empty()) {
                continue;
            }
            
            groupI2meshVI[gI] = nextVI;
            UV_mesh.row(nextVI) = Eigen::RowVector2d::Zero();
            for(const auto dupVI : meshVGroup[gI]) {
                UV_mesh.row(nextVI) += V.row(dupVI);
            }
            UV_mesh.row(nextVI) /= meshVGroup[gI].size();
            nextVI++;
        }
        int nextVI_3D = 0;
        for(int gI = 0; gI < meshVGroup_3D.size(); gI++) {
            if(meshVGroup_3D[gI].empty()) {
                continue;
            }
            
            groupI2meshVI_3D[gI] = nextVI_3D;
            V_mesh.row(nextVI_3D) = Eigen::RowVector3d::Zero();
            for(const auto dupVI : meshVGroup_3D[gI]) {
                V_mesh.row(nextVI_3D) += V_rest.row(dupVI);
            }
            V_mesh.row(nextVI_3D) /= meshVGroup_3D[gI].size();
            nextVI_3D++;
        }
        
        Eigen::MatrixXi F_mesh, FUV_mesh;
        F_mesh.resize(F.rows(), 3);
        FUV_mesh.resize(F.rows(), 3);
        for(int triI = 0; triI < F.rows(); triI++) {
            for(int vI = 0; vI < 3; vI++) {
                int groupI = dupVI2GroupI[F(triI, vI)];
                assert(groupI >= 0);
                int meshVI = groupI2meshVI[groupI];
                assert(meshVI >= 0);
                FUV_mesh(triI, vI) = meshVI;
                
                int groupI_3D = dupVI2GroupI_3D[F(triI, vI)];
                assert(groupI_3D >= 0);
                int meshVI_3D = groupI2meshVI_3D[groupI_3D];
                assert(meshVI_3D >= 0);
                F_mesh(triI, vI) = meshVI_3D;
            }
        }
        
        if(scaleUV) {
            const Eigen::VectorXd& u = UV_mesh.col(0);
            const Eigen::VectorXd& v = UV_mesh.col(1);
            const double uMin = u.minCoeff();
            const double vMin = v.minCoeff();
            const double uScale = u.maxCoeff() - uMin;
            const double vScale = v.maxCoeff() - vMin;
            const double scale = std::max(uScale, vScale);
            for(int uvI = 0; uvI < UV_mesh.rows(); uvI++) {
                UV_mesh(uvI, 0) = (UV_mesh(uvI, 0) - uMin) / scale;
                UV_mesh(uvI, 1) = (UV_mesh(uvI, 1) - vMin) / scale;
            }
        }
        
        save(filePath, V_mesh, F_mesh, UV_mesh, FUV_mesh);
    }
        
    void TriangleSoup::constructSubmesh(const Eigen::VectorXi& triangles,
                                        TriangleSoup& submesh,
                                        std::map<int, int>& globalVIToLocal,
                                        std::map<int, int>& globalTriIToLocal) const
    {
        Eigen::MatrixXi F_sub;
        F_sub.resize(triangles.size(), 3);
        Eigen::MatrixXd V_rest_sub, V_sub;
        globalVIToLocal.clear();
        for(int localTriI = 0; localTriI < triangles.size(); localTriI++) {
            int triI = triangles[localTriI];
            for(int vI = 0; vI < 3; vI++) {
                int globalVI = F(triI, vI);
                auto localVIFinder = globalVIToLocal.find(globalVI);
                if(localVIFinder == globalVIToLocal.end()) {
                    int localVI = static_cast<int>(V_rest_sub.rows());
                    V_rest_sub.conservativeResize(localVI + 1, 3);
                    V_rest_sub.row(localVI) = V_rest.row(globalVI);
                    V_sub.conservativeResize(localVI + 1, 2);
                    V_sub.row(localVI) = V.row(globalVI);
                    F_sub(localTriI, vI) = localVI;
                    globalVIToLocal[globalVI] = localVI;
                }
                else {
                    F_sub(localTriI, vI) = localVIFinder->second;
                }
            }
            globalTriIToLocal[triI] = localTriI;
        }
        submesh = TriangleSoup(V_rest_sub, F_sub, V_sub, Eigen::MatrixXi(), false);
        
        std::set<int> fixedVert_sub;
        for(const auto& fixedVI : fixedVert) {
            auto finder = globalVIToLocal.find(fixedVI);
            if(finder != globalVIToLocal.end()) {
                fixedVert_sub.insert(finder->second);
            }
        }
        submesh.resetFixedVert(fixedVert_sub);
    }
    
    bool TriangleSoup::findBoundaryEdge(int vI, const std::pair<int, int>& startEdge,
                                        std::pair<int, int>& boundaryEdge)
    {
        auto finder = edge2Tri.find(startEdge);
        assert(finder != edge2Tri.end());
        bool proceed = (startEdge.first != vI);
        const int vI_neighbor = (proceed ? startEdge.first : startEdge.second);
        int vI_new = vI_neighbor;
        while(1) {
            const Eigen::RowVector3i& triVInd = F.row(finder->second);
            for(int i = 0; i < 3; i++) {
                if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                    vI_new = triVInd[i];
                    break;
                }
            }
            
            if(vI_new == vI_neighbor) {
                return false;
            }
            
            finder = edge2Tri.find(proceed ? std::pair<int, int>(vI_new, vI) : std::pair<int, int>(vI, vI_new));
            if(finder == edge2Tri.end()) {
                boundaryEdge.first = (proceed ? vI : vI_new);
                boundaryEdge.second = (proceed ? vI_new : vI);
                return true;
            }
        }
    }
    
    bool TriangleSoup::insideTri(int triI, const Eigen::RowVector2d& pos) const
    {
        const Eigen::RowVector3i& triVInd = F.row(triI);
        const Eigen::RowVector2d e01 = V.row(triVInd[1]) - V.row(triVInd[0]);
        const Eigen::RowVector2d e02 = V.row(triVInd[2]) - V.row(triVInd[0]);
        const Eigen::RowVector2d e0p = pos - V.row(triVInd[0]);
        
        // represent ep0 using e01 and e02
        Eigen::Matrix2d A;
        A << e01.transpose(), e02.transpose();
        Eigen::Vector2d coef = A.colPivHouseholderQr().solve(e0p.transpose());
        
        // check inside
        if((coef[0] >= 0.0) && (coef[1] >= 0.0) && (coef[0] + coef[1] <= 1.0)) {
            return true;
        }
        else {
            return false;
        }
    }
        
    bool TriangleSoup::insideUVRegion(const std::vector<int>& triangles, const Eigen::RowVector2d& pos) const
    {
        for(const auto& triI : triangles) {
            if(insideTri(triI, pos)) {
                return true;
            }
        }
        return false;
    }

    bool TriangleSoup::isBoundaryVert(int vI, int vI_neighbor,
                                      std::vector<int>& tri_toSep, std::pair<int, int>& boundaryEdge, bool toBound) const
    {
//        const auto inputEdgeTri = edge2Tri.find(toBound ? std::pair<int, int>(vI, vI_neighbor) :
//                                                std::pair<int, int>(vI_neighbor, vI));
//        assert(inputEdgeTri != edge2Tri.end());
        
        tri_toSep.resize(0);
        auto finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_neighbor, vI):
                                    std::pair<int, int>(vI, vI_neighbor));
        if(finder == edge2Tri.end()) {
            boundaryEdge.first = (toBound ? vI : vI_neighbor);
            boundaryEdge.second = (toBound ? vI_neighbor : vI);
            return true;
        }
        
        int vI_new = vI_neighbor;
        do {
            tri_toSep.emplace_back(finder->second);
            const Eigen::RowVector3i& triVInd = F.row(finder->second);
            for(int i = 0; i < 3; i++) {
                if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                    vI_new = triVInd[i];
                    break;
                }
            }
            
            if(vI_new == vI_neighbor) {
                return false;
            }
            
            finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_new, vI) :
                                   std::pair<int, int>(vI, vI_new));
            if(finder == edge2Tri.end()) {
                boundaryEdge.first = (toBound ? vI : vI_new);
                boundaryEdge.second = (toBound ? vI_new : vI);
                return true;
            }
        } while(1);
    }
    
    bool TriangleSoup::isBoundaryVert(int vI) const
    {
        assert(vNeighbor.size() == V.rows());
        assert(vI < vNeighbor.size());
        
        for(const auto vI_neighbor : vNeighbor[vI]) {
            if((edge2Tri.find(std::pair<int, int>(vI, vI_neighbor)) == edge2Tri.end()) ||
                (edge2Tri.find(std::pair<int, int>(vI_neighbor, vI)) == edge2Tri.end()))
            {
                return true;
            }
        }
        
        return false;
    }
        
    void TriangleSoup::compute2DInwardNormal(int vI, Eigen::RowVector2d& normal) const
    {
        std::vector<int> incTris[2];
        std::pair<int, int> boundaryEdge[2];
        if(!isBoundaryVert(vI, *vNeighbor[vI].begin(), incTris[0], boundaryEdge[0], 0)) {
            return;
        }
        isBoundaryVert(vI, *vNeighbor[vI].begin(), incTris[1], boundaryEdge[1], 1);
        assert(!(incTris[0].empty() && incTris[1].empty()));
        
        Eigen::RowVector2d boundaryEdgeDir[2] = {
            (V.row(boundaryEdge[0].first) - V.row(boundaryEdge[0].second)).normalized(),
            (V.row(boundaryEdge[1].second) - V.row(boundaryEdge[1].first)).normalized(),
        };
        normal = (boundaryEdgeDir[0] + boundaryEdgeDir[1]).normalized();
        if(boundaryEdgeDir[1][0] * normal[1] - boundaryEdgeDir[1][1] * normal[0] < 0.0) {
            normal *= -1.0;
        }
    }
    
    double TriangleSoup::computeLocalEwDec(int vI, double lambda_t, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max,
                                           std::pair<double, double>& energyChanges_max,
                                           const std::vector<int>& incTris, const Eigen::RowVector2d& initMergedPos) const
    {
        if(!path_max.empty()) {
            // merge query
            assert(path_max.size() >= 3);
            for(const auto& pI : path_max) {
                assert(isBoundaryVert(pI));
            }
            
            if(path_max.size() == 3) {
                // zipper merge
                double seDec = (V_rest.row(path_max[0]) - V_rest.row(path_max[1])).norm() /
                    virtualRadius * (vertWeight[path_max[0]] + vertWeight[path_max[1]]) / 2.0;
                // closing up splitted diamond
                bool closeup = false;
                for(const auto& nbVI : vNeighbor[path_max[0]]) {
                    if(nbVI != path_max[1]) {
                        if(isBoundaryVert(nbVI)) {
                            if(vNeighbor[path_max[2]].find(nbVI) != vNeighbor[path_max[2]].end()) {
                                seDec += (V_rest.row(path_max[0]) - V_rest.row(nbVI)).norm() / virtualRadius * (vertWeight[path_max[0]] + vertWeight[nbVI]) / 2.0;
                                closeup = true;
                                break;
                            }
                        }
                    }
                }
                
                assert(incTris.size() >= 2);
                std::set<int> freeVert;
                freeVert.insert(path_max[0]);
                freeVert.insert(path_max[2]);
                std::map<int, int> mergeVert;
                mergeVert[path_max[0]] = path_max[2];
                mergeVert[path_max[2]] = path_max[0];
                std::map<int, Eigen::RowVector2d> newVertPos;
                const double SDInc = -computeLocalEDec(path_max, incTris, freeVert, newVertPos, mergeVert, initMergedPos, closeup);
                energyChanges_max.first = SDInc;
                energyChanges_max.second = -seDec;
                if(SDInc == __DBL_MAX__) {
                    return -__DBL_MAX__;
                }
                else {
                    auto finder = newVertPos.find(path_max[0]);
                    assert(finder != newVertPos.end());
                    newVertPos_max.resize(1, 2);
                    newVertPos_max.row(0) = finder->second;
                    
                    return lambda_t * seDec - (1.0 - lambda_t) * SDInc;
                }
            }
            else {
                assert(0 && "currently not considering \"interior\" merge!");
            }
        }
        
        // split:
        std::vector<int> umbrella;
        std::pair<int, int> boundaryEdge;
        if(isBoundaryVert(vI, *(vNeighbor[vI].begin()), umbrella, boundaryEdge, false)) {
            // boundary split
            double maxEwDec = -__DBL_MAX__;
            energyChanges_max.first = __DBL_MAX__;
            energyChanges_max.second = __DBL_MAX__;
            path_max.resize(2);
            for(const auto& nbVI : vNeighbor[vI]) {
                const std::pair<int, int> edge(vI, nbVI);
                if((edge2Tri.find(edge) != edge2Tri.end()) &&
                   (edge2Tri.find(std::pair<int, int>(nbVI, vI)) != edge2Tri.end()))
                {
                    // interior edge
                    
                    Eigen::MatrixXd newVertPosI;
                    const double SDDec = computeLocalEDec(edge, newVertPosI);
                    
//                    // test overlap locally (not necessary if global bijectivity is enforced)
//                    const Eigen::RowVector2d e = V.row(vI) - V.row(nbVI);
//                    const Eigen::RowVector2d a = newVertPosI.row(0) - V.row(nbVI);
//                    const Eigen::RowVector2d b = newVertPosI.row(1) - V.row(nbVI);
//                    if(IglUtils::computeRotAngle(a, e) + IglUtils::computeRotAngle(e, b) > 0.0) {
//                    std::cout << vI << "-" << nbVI << ": " << IglUtils::computeRotAngle(a, e) + IglUtils::computeRotAngle(e, b) << std::endl;
//                    assert(IglUtils::computeRotAngle(a, e) + IglUtils::computeRotAngle(e, b) > 0.0); //TODO: cut through may violate
                        const double seInc = (V_rest.row(vI) - V_rest.row(nbVI)).norm() /
                            virtualRadius * (vertWeight[vI] + vertWeight[nbVI]) / 2.0;
                        const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                        if(curEwDec > maxEwDec) {
                            maxEwDec = curEwDec;
                            path_max[0] = vI;
                            path_max[1] = nbVI;
                            newVertPos_max = newVertPosI;
                            energyChanges_max.first = -SDDec;
                            energyChanges_max.second = seInc;
                        }
//                    }
                    //TODO: cut through check?
                }
            }
            return maxEwDec;
        }
        else {
            // interior split
            for(const auto& nbVI : vNeighbor[vI]) {
                if(isBoundaryVert(nbVI)) {
                    energyChanges_max.first = __DBL_MAX__;
                    energyChanges_max.second = __DBL_MAX__;
                    assert(0 && "should have prevented this case outside");
                    return -__DBL_MAX__; // don't split vertices connected to boundary here
                }
            }
            
//            std::cout << "umbrella obtained, size " << umbrella.size() << std::endl;
            if(umbrella.size() > 10) {
                std::cout << "large degree vert, " << umbrella.size() << " incident tris" << std::endl;
                logFile << "large degree vert, " << umbrella.size() << " incident tris" << std::endl;
            }
            
            path_max.resize(3);
            double EwDec_max = -__DBL_MAX__;
            energyChanges_max.first = __DBL_MAX__;
            energyChanges_max.second = __DBL_MAX__;
            std::set<int> freeVert;
            freeVert.insert(vI);
            std::map<int, Eigen::RowVector2d> newVertPosMap;
            std::vector<int> path(3);
            path[1] = vI;
            for(int startI = 0; startI + 1 < umbrella.size(); startI++) {
                for(int i = 0; i < 3; i++) {
                    if(F(umbrella[startI], i) == vI) {
                        path[0] = F(umbrella[startI], (i + 1) % 3);
                        break;
                    }
                }
                
                for(int endI = startI + 1; endI < umbrella.size(); endI++) {
                    for(int i = 0; i < 3; i++) {
                        if(F(umbrella[endI], i) == vI) {
                            path[2] = F(umbrella[endI], (i + 1) % 3);
                            break;
                        }
                    }
                    
//                    // don't make sharp turn splits
//                    if(validSplit[vI].find(std::pair<int, int>(path[0], path[2])) == validSplit[vI].end()) {
//                        continue;
//                    }
                    
                    double SDDec = 0.0;
                    Eigen::MatrixXd newVertPos;
                    
                    SDDec += computeLocalEDec_in(umbrella, freeVert, path, newVertPos);
                    //TODO: share local mesh before split, also for boundary splits
                    
//                    if(scaffold) {
//                        // test overlap
//                        const double eps_ang = 1.0e-3;
//                        const Eigen::RowVector2d p0p1 = V.row(path[1]) - V.row(path[0]);
//                        const Eigen::RowVector2d p0nv0 = newVertPos.block(0, 0, 1, 2) - V.row(path[0]);
//                        const Eigen::RowVector2d p0nv1 = newVertPos.block(1, 0, 1, 2) - V.row(path[0]);
//                        const double ang_nv0p0p1 = IglUtils::computeRotAngle(p0nv0, p0p1);
//                        const double ang_p1p0nv1 = IglUtils::computeRotAngle(p0p1, p0nv1);
//                        if(ang_nv0p0p1 + ang_p1p0nv1 <= eps_ang) {
//                            continue;
//                        }
//                        const Eigen::RowVector2d p2p1 = V.row(path[1]) - V.row(path[2]);
//                        const Eigen::RowVector2d p2nv0 = newVertPos.block(0, 0, 1, 2) - V.row(path[2]);
//                        const Eigen::RowVector2d p2nv1 = newVertPos.block(1, 0, 1, 2) - V.row(path[2]);
//                        const double ang_nv1p2p1 = IglUtils::computeRotAngle(p2nv1, p2p1);
//                        const double ang_p1p2nv0 = IglUtils::computeRotAngle(p2p1, p2nv0);
//                        if(ang_nv1p2p1 + ang_p1p2nv0 <= eps_ang) {
//                            continue;
//                        }
//                        // test on corners of non-moving vertices is unnecessary
//                        double ang_p0p1p2 = IglUtils::computeRotAngle(-p0p1, -p2p1);
//                        if(ang_p0p1p2 < 0.0) {
//                            ang_p0p1p2 += 2.0 * M_PI;
//                        }
////                        assert(ang_p1p0nv1 + ang_nv1p2p1 < ang_p0p1p2 + eps_ang);
//                        if(ang_p1p0nv1 + ang_nv1p2p1 >= ang_p0p1p2) {
//                            continue;
//                        }
////                        assert(ang_p1p2nv0 + ang_nv0p0p1 < 2.0 * M_PI - ang_p0p1p2 + eps_ang);
//                        if(ang_p1p2nv0 + ang_nv0p0p1 >= 2.0 * M_PI - ang_p0p1p2) {
//                            continue;
//                        }
//                    }
                    
                    const double seInc = ((V_rest.row(path[0]) - V_rest.row(path[1])).norm() * (vertWeight[path[0]] + vertWeight[path[1]]) +
                                          (V_rest.row(path[1]) - V_rest.row(path[2])).norm() * (vertWeight[path[1]] + vertWeight[path[2]])) / virtualRadius / 2.0;
                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                    if(EwDec > EwDec_max) {
                        EwDec_max = EwDec;
                        newVertPos_max = newVertPos;
                        path_max = path;
                        energyChanges_max.first = -SDDec;
                        energyChanges_max.second = seInc;
                    }
                }
            }
            return EwDec_max;
        }
    }
        
    double TriangleSoup::computeLocalEDec_in(const std::vector<int>& triangles, const std::set<int>& freeVert,
                                          const std::vector<int>& path, Eigen::MatrixXd& newVertPos, int maxIter) const
    {
        assert(triangles.size() && freeVert.size());
        
        // construct local mesh
        Eigen::MatrixXi localF;
        localF.resize(triangles.size(), 3);
        Eigen::MatrixXd localV_rest, localV;
        std::set<int> fixedVert;
        std::map<int, int> globalVI2local;
        int localTriI = 0;
        for(const auto triI : triangles) {
            for(int vI = 0; vI < 3; vI++) {
                int globalVI = F(triI, vI);
                auto localVIFinder = globalVI2local.find(globalVI);
                if(localVIFinder == globalVI2local.end()) {
                    int localVI = static_cast<int>(localV_rest.rows());
                    if(freeVert.find(globalVI) == freeVert.end()) {
                        fixedVert.insert(localVI);
                    }
                    localV_rest.conservativeResize(localVI + 1, 3);
                    localV_rest.row(localVI) = V_rest.row(globalVI);
                    localV.conservativeResize(localVI + 1, 2);
                    localV.row(localVI) = V.row(globalVI);
                    localF(localTriI, vI) = localVI;
                    globalVI2local[globalVI] = localVI;
                }
                else {
                    localF(localTriI, vI) = localVIFinder->second;
                }
            }
            localTriI++;
        }
        TriangleSoup localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
        localMesh.resetFixedVert(fixedVert);
        
        SymStretchEnergy SD;
        double initE = 0.0;
        for(const auto& triI : triangles) {
            double energyValI;
            SD.getEnergyValByElemID(*this, triI, energyValI);
            initE += energyValI;
        }
        initE *= surfaceArea / localMesh.surfaceArea;
        
        // convert split path global index to local index
        std::vector<int> path_local;
        path_local.reserve(path.size());
        for(const auto& pvI : path) {
            const auto finder = globalVI2local.find(pvI);
            assert(finder != globalVI2local.end());
            path_local.emplace_back(finder->second);
        }
        // split
        localMesh.cutPath(path_local, true, 0, Eigen::MatrixXd(), false);
        
        bool isBijective = !!scaffold;
        
        // construct air mesh
        Eigen::MatrixXd UV_bnds;
        Eigen::MatrixXi E;
        Eigen::VectorXi bnd;
        if(isBijective) {
            // separate vertex
            //TODO: write into a function
            Eigen::RowVector2d splittedV[2] = {
                localMesh.V.row(path_local[1]),
                localMesh.V.row(localMesh.V.rows() - 1)
            };
            Eigen::RowVector2d sepDir_oneV[2];
            localMesh.compute2DInwardNormal(path_local[1], sepDir_oneV[0]);
            localMesh.compute2DInwardNormal(localMesh.V.rows() - 1, sepDir_oneV[1]);
            Eigen::VectorXd sepDir[2] = {
                Eigen::VectorXd::Zero(localMesh.V.rows() * 2),
                Eigen::VectorXd::Zero(localMesh.V.rows() * 2)
            };
            sepDir[0].block(path_local[1] * 2, 0, 2, 1) = sepDir_oneV[0].transpose();
            sepDir[1].block((localMesh.V.rows() - 1) * 2, 0, 2, 1) = sepDir_oneV[1].transpose();
            const double eps_sep = (V.row(path[1]) - V.row(path[0])).squaredNorm() * 1.0e-4;
            double curSqDist = (splittedV[0]-splittedV[1]).squaredNorm();
            while(curSqDist < eps_sep) {
                for(int i = 0; i < 2; i++) {
                    double stepSize_sep = 1.0;
                    SD.initStepSize(localMesh, sepDir[i], stepSize_sep);
                    splittedV[i] += 0.1 * stepSize_sep * sepDir_oneV[i];
                }
                localMesh.V.row(path_local[1]) = splittedV[0];
                localMesh.V.row(localMesh.V.rows() - 1) = splittedV[1];
                
                double lastSqDist = curSqDist;
                curSqDist = (splittedV[0]-splittedV[1]).squaredNorm();
                if(std::abs(curSqDist - lastSqDist) / lastSqDist < 1.0e-3) {
                    break;
                }
//                    //TODO: may update search dir, and accelerate
//                    localMesh.compute2DInwardNormal(splitPath_local[0], sepDir_oneV[0]);
//                    localMesh.compute2DInwardNormal(localMesh.V.rows() - 1, sepDir_oneV[1]);
//                    sepDir[0].block(splitPath_local[0] * 2, 0, 2, 1) = sepDir_oneV[0].transpose();
//                    sepDir[1].bottomRows(2) = sepDir_oneV[1].transpose();
            }
            
            // establish air mesh information
            UV_bnds.resize(4, 2);
            E.resize(4, 2);
            E << 0, 1, 1, 2, 2, 3, 3, 0;
            bnd.resize(4);
            bnd << path_local[2], path_local[1], path_local[0], static_cast<int>(localMesh.V.rows()) - 1;
        }
        
        // conduct optimization on local mesh
        std::vector<FracCuts::Energy*> energyTerms(1, &SD);
        std::vector<double> energyParams(1, 1.0);
        Optimizer optimizer(localMesh, energyTerms, energyParams, 0, true, isBijective, UV_bnds, E, bnd);
        optimizer.precompute();
        //        optimizer.result.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_optimized.obj");
        //        optimizer.scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test0_AM.obj");
        optimizer.setRelGL2Tol(1.0e-4);
        optimizer.solve(maxIter); //do not output, the other part
        //            std::cout << "local opt " << optimizer.getIterNum() << " iters" << std::endl;
        double curE;
        optimizer.computeEnergyVal(optimizer.getResult(), optimizer.getScaffold(), true, curE, true);
        const double eDec = (initE - curE) * localMesh.surfaceArea / surfaceArea; //!!! this should be written in a more general way, cause this way it only works for E_SD
        
        // get new vertex positions
        newVertPos.resize(2, 2);
        newVertPos.row(0) = optimizer.getResult().V.bottomRows(1);
        newVertPos.row(1) = optimizer.getResult().V.row(path_local[1]);
        
        return eDec;
    }
    
    double TriangleSoup::computeLocalEDec(const std::vector<int>& path, const std::vector<int>& triangles,
                                          const std::set<int>& freeVert, std::map<int, Eigen::RowVector2d>& newVertPos,
                                          const std::map<int, int>& mergeVert, const Eigen::RowVector2d& initMergedPos,
                                          bool closeup, int maxIter) const
    {
        assert(triangles.size() && freeVert.size());
        assert(!mergeVert.empty());
        
        bool isBijective = ((!!scaffold) && (!closeup));
        
        // construct local mesh
        Eigen::MatrixXi localF;
        localF.resize(triangles.size(), 3);
        Eigen::MatrixXd localV_rest, localV;
        std::set<int> fixedVert;
        std::map<int, int> globalVI2local;
        int localTriI = 0;
        for(const auto triI : triangles) {
            for(int vI = 0; vI < 3; vI++) {
                int globalVI = F(triI, vI);
                auto mergeFinder = mergeVert.find(globalVI);
                if(mergeFinder == mergeVert.end()) {
                    // normal vertices
                    auto localVIFinder = globalVI2local.find(globalVI);
                    if(localVIFinder == globalVI2local.end()) {
                        int localVI = static_cast<int>(localV_rest.rows());
                        if(freeVert.find(globalVI) == freeVert.end()) {
                            fixedVert.insert(localVI);
                        }
                        localV_rest.conservativeResize(localVI + 1, 3);
                        localV_rest.row(localVI) = V_rest.row(globalVI);
                        localV.conservativeResize(localVI + 1, 2);
                        localV.row(localVI) = V.row(globalVI);
                        localF(localTriI, vI) = localVI;
                        globalVI2local[globalVI] = localVI;
                    }
                    else {
                        localF(localTriI, vI) = localVIFinder->second;
                    }
                }
                else {
                    // one of the vertices to be merged
                    auto localVIFinder = globalVI2local.find(globalVI);
                    auto localVIFinder_mergePair = globalVI2local.find(mergeFinder->second);
                    bool selfAdded = (localVIFinder != globalVI2local.end());
                    bool mergePairAdded = (localVIFinder_mergePair != globalVI2local.end());
                    if(selfAdded) {
                        assert(mergePairAdded);
                        localF(localTriI, vI) = localVIFinder->second;
                    }
                    else {
                        assert(!mergePairAdded);
                        int localVI = static_cast<int>(localV_rest.rows());
                        if(freeVert.find(globalVI) == freeVert.end()) {
                            fixedVert.insert(localVI);
                        }
                        localV_rest.conservativeResize(localVI + 1, 3);
                        localV_rest.row(localVI) = V_rest.row(globalVI);
                        localV.conservativeResize(localVI + 1, 2);
                        localV.row(localVI) = initMergedPos;
                        localF(localTriI, vI) = localVI;
                        globalVI2local[globalVI] = localVI;
                        
                        globalVI2local[mergeFinder->second] = localVI;
                    }
                }
            }
            localTriI++;
        }
        TriangleSoup localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
        localMesh.resetFixedVert(fixedVert);
//        localMesh.save("/Users/mincli/Desktop/meshes/test.obj");
//        save("/Users/mincli/Desktop/meshes/test_full.obj");
        
        SymStretchEnergy SD;
        double initE = 0.0;
        for(const auto& triI : triangles) {
            double energyValI;
            SD.getEnergyValByElemID(*this, triI, energyValI);
            initE += energyValI;
        }
        initE *= surfaceArea / localMesh.surfaceArea;
        
        // construct air mesh
        Eigen::MatrixXd UV_bnds;
        Eigen::MatrixXi E;
        Eigen::VectorXi bnd;
        if(isBijective) {
            if(!scaffold->getCornerAirLoop(path, initMergedPos, UV_bnds, E, bnd)) {
                // if initPos causes the composite loop to self-intersect, or the loop is totally inverted
                // (potentially violating bijectivity), abandon this query
                return -__DBL_MAX__;
            }
            
            for(int bndI = 0; bndI < bnd.size(); bndI++) {
                const auto finder = globalVI2local.find(bnd[bndI]);
                assert(finder != globalVI2local.end());
                bnd[bndI] = finder->second;
            }
        }
        
        // conduct optimization on local mesh
        std::vector<FracCuts::Energy*> energyTerms(1, &SD);
        std::vector<double> energyParams(1, 1.0);
        Optimizer optimizer(localMesh, energyTerms, energyParams, 0, true, isBijective, UV_bnds, E, bnd);
        optimizer.precompute();
        //        optimizer.result.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_optimized.obj");
        //        optimizer.scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test0_AM.obj");
        optimizer.setRelGL2Tol(1.0e-4);
        optimizer.solve(maxIter); //do not output, the other part
        //            std::cout << "local opt " << optimizer.getIterNum() << " iters" << std::endl;
        double curE;
        optimizer.computeEnergyVal(optimizer.getResult(), optimizer.getScaffold(), true, curE, true);
        const double eDec = (initE - curE) * localMesh.surfaceArea / surfaceArea; //!!! this should be written in a more general way, cause this way it only works for E_SD
        
        // get new vertex positions
        newVertPos.clear();
        for(const auto& vI_free : freeVert) {
            newVertPos[vI_free] = optimizer.getResult().V.row(globalVI2local[vI_free]);
        }
        
        return eDec;
    }
        
    double TriangleSoup::computeLocalEDec(const std::vector<int>& triangles, const std::set<int>& freeVert,
                                          const std::vector<int>& splitPath, Eigen::MatrixXd& newVertPos,
                                          int maxIter) const
    {
        assert(triangles.size() && freeVert.size());
        
        // construct local mesh
        Eigen::MatrixXi localF;
        localF.resize(triangles.size(), 3);
        Eigen::MatrixXd localV_rest, localV;
        std::set<int> fixedVert;
        std::map<int, int> globalVI2local;
        int localTriI = 0;
        for(const auto triI : triangles) {
            for(int vI = 0; vI < 3; vI++) {
                int globalVI = F(triI, vI);
                auto localVIFinder = globalVI2local.find(globalVI);
                if(localVIFinder == globalVI2local.end()) {
                    int localVI = static_cast<int>(localV_rest.rows());
                    if(freeVert.find(globalVI) == freeVert.end()) {
                        fixedVert.insert(localVI);
                    }
                    localV_rest.conservativeResize(localVI + 1, 3);
                    localV_rest.row(localVI) = V_rest.row(globalVI);
                    localV.conservativeResize(localVI + 1, 2);
                    localV.row(localVI) = V.row(globalVI);
                    localF(localTriI, vI) = localVI;
                    globalVI2local[globalVI] = localVI;
                }
                else {
                    localF(localTriI, vI) = localVIFinder->second;
                }
            }
            localTriI++;
        }
        TriangleSoup localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
        localMesh.resetFixedVert(fixedVert);
        
        // compute initial symmetric Dirichlet Energy value
        SymStretchEnergy SD;
        double initE = 0.0;
        for(const auto& triI : triangles) {
            double energyValI;
            SD.getEnergyValByElemID(*this, triI, energyValI);
            initE += energyValI;
        }
        initE *= surfaceArea / localMesh.surfaceArea;
        
        // split edge
        Eigen::MatrixXd UV_bnds;
        Eigen::MatrixXi E;
        Eigen::VectorXi bnd;
        bool cutThrough = false;
        switch(splitPath.size()) {
            case 0: // nothing to split
                assert(0 && "currently we don't use this function without splitting!");
                break;
                
            case 2: {// boundary split
                assert(freeVert.find(splitPath[0]) != freeVert.end());
                if(freeVert.find(splitPath[1]) != freeVert.end()) {
                    cutThrough = true;
                }
                
                // convert splitPath global index to local index
                std::vector<int> splitPath_local;
                splitPath_local.reserve(splitPath.size());
                for(const auto& pvI : splitPath) {
                    const auto finder = globalVI2local.find(pvI);
                    assert(finder != globalVI2local.end());
                    splitPath_local.emplace_back(finder->second);
                }
                
                // split
                localMesh.splitEdgeOnBoundary(std::pair<int, int>(splitPath_local[0], splitPath_local[1]),
                                              Eigen::Matrix2d(), false, cutThrough);
//                localMesh.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_afterSplit.obj");
//                std::cout << splitPath[0] << "-" << splitPath[1] << std::endl;
                
                if(scaffold) {
                    // separate the splitted vertices to leave room for airmesh
                    Eigen::RowVector2d splittedV[2] = {
                        localMesh.V.row(splitPath_local[0]),
                        localMesh.V.row(localMesh.V.rows() - 1 - cutThrough)
                    };
                    Eigen::RowVector2d sepDir_oneV[2];
                    localMesh.compute2DInwardNormal(splitPath_local[0], sepDir_oneV[0]);
                    localMesh.compute2DInwardNormal(localMesh.V.rows() - 1 - cutThrough, sepDir_oneV[1]);
                    Eigen::VectorXd sepDir[2] = {
                        Eigen::VectorXd::Zero(localMesh.V.rows() * 2),
                        Eigen::VectorXd::Zero(localMesh.V.rows() * 2)
                    };
                    sepDir[0].block(splitPath_local[0] * 2, 0, 2, 1) = sepDir_oneV[0].transpose();
                    sepDir[1].block((localMesh.V.rows() - 1 - cutThrough) * 2, 0, 2, 1) = sepDir_oneV[1].transpose();
                    const double eps_sep = (V.row(splitPath[1]) - V.row(splitPath[0])).squaredNorm() * 1.0e-4;
                    double curSqDist = (splittedV[0]-splittedV[1]).squaredNorm();
                    while(curSqDist < eps_sep) {
                        for(int i = 0; i < 2; i++) {
                            double stepSize_sep = 1.0;
                            SD.initStepSize(localMesh, sepDir[i], stepSize_sep);
                            splittedV[i] += 0.1 * stepSize_sep * sepDir_oneV[i];
                        }
                        localMesh.V.row(splitPath_local[0]) = splittedV[0];
                        localMesh.V.row(localMesh.V.rows() - 1 - cutThrough) = splittedV[1];
                        
                        double lastSqDist = curSqDist;
                        curSqDist = (splittedV[0]-splittedV[1]).squaredNorm();
                        if(std::abs(curSqDist - lastSqDist) / lastSqDist < 1.0e-3) {
                            break;
                        }
    //                    //TODO: may update search dir, and accelerate
    //                    localMesh.compute2DInwardNormal(splitPath_local[0], sepDir_oneV[0]);
    //                    localMesh.compute2DInwardNormal(localMesh.V.rows() - 1, sepDir_oneV[1]);
    //                    sepDir[0].block(splitPath_local[0] * 2, 0, 2, 1) = sepDir_oneV[0].transpose();
    //                    sepDir[1].bottomRows(2) = sepDir_oneV[1].transpose();
                    }
                    assert(localMesh.checkInversion());
                    
                    if(cutThrough) {
                        splittedV[0] = localMesh.V.row(splitPath_local[1]);
                        splittedV[1] = localMesh.V.bottomRows(1);
                        localMesh.compute2DInwardNormal(splitPath_local[1], sepDir_oneV[0]);
                        localMesh.compute2DInwardNormal(localMesh.V.rows() - 1, sepDir_oneV[1]);
                        sepDir[0] = sepDir[1] = Eigen::VectorXd::Zero(localMesh.V.rows() * 2);
                        sepDir[0].block(splitPath_local[1] * 2, 0, 2, 1) = sepDir_oneV[0].transpose();
                        sepDir[1].bottomRows(2) = sepDir_oneV[1].transpose();
                        double curSqDist = (splittedV[0]-splittedV[1]).squaredNorm();
                        while(curSqDist < eps_sep) {
                            for(int i = 0; i < 2; i++) {
                                double stepSize_sep = 1.0;
                                SD.initStepSize(localMesh, sepDir[i], stepSize_sep);
                                splittedV[i] += 0.1 * stepSize_sep * sepDir_oneV[i];
                            }
                            localMesh.V.row(splitPath_local[1]) = splittedV[0];
                            localMesh.V.bottomRows(1) = splittedV[1];
                            
                            double lastSqDist = curSqDist;
                            curSqDist = (splittedV[0]-splittedV[1]).squaredNorm();
                            if(std::abs(curSqDist - lastSqDist) / lastSqDist < 1.0e-3) {
                                break;
                            }
                            //                    //TODO: may update search dir, and accelerate
                        }
                        assert(localMesh.checkInversion());
                    }
    //                std::cout << std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) << std::endl;
//                    localMesh.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_separated.obj");
                
                    // prepare local air mesh boundary
                    Eigen::MatrixXd UV_temp;
                    Eigen::VectorXi bnd_temp;
                    std::set<int> loop_AMVI;
                    scaffold->get1RingAirLoop(splitPath[0], UV_temp, E, bnd_temp, loop_AMVI);
                    int loopVAmt_beforeSplit = E.rows();
                    if(!cutThrough) {
                        E.bottomRows(1) << loopVAmt_beforeSplit - 1, loopVAmt_beforeSplit;
                        E.conservativeResize(loopVAmt_beforeSplit + 2, 2);
                        E.bottomRows(2) << loopVAmt_beforeSplit, loopVAmt_beforeSplit + 1,
                            loopVAmt_beforeSplit + 1, 0;
                        
                        UV_bnds.resize(loopVAmt_beforeSplit + 2, 2);
                        UV_bnds.bottomRows(loopVAmt_beforeSplit - 3) = UV_temp.bottomRows(loopVAmt_beforeSplit - 3);
                        //NOTE: former vertices will be filled with mesh coordinates while constructing the local air mesh
                        
                        bnd.resize(bnd_temp.size() + 2);
                        bnd[0] = bnd_temp[0];
                        bnd[1] = localMesh.V.rows() - 1 - cutThrough;
                        bnd[2] = splitPath[1];
                        bnd.bottomRows(2) = bnd_temp.bottomRows(2);
                        for(int bndI = 0; bndI < bnd.size(); bndI++) {
                            if(bndI != 1) {
                                const auto finder = globalVI2local.find(bnd[bndI]);
                                assert(finder != globalVI2local.end());
                                bnd[bndI] = finder->second;
                            }
                        }
                    }
                    else {
                        Eigen::MatrixXd UV_temp1;
                        Eigen::VectorXi bnd_temp1;
                        Eigen::MatrixXi E1;
                        std::set<int> loop1_AMVI;
                        scaffold->get1RingAirLoop(splitPath[1], UV_temp1, E1, bnd_temp1, loop1_AMVI);
                        // avoid generating air mesh with duplicated vertices
                        //NOTE: this also avoid forming tiny charts
                        for(const auto& i : loop1_AMVI) {
                            if(loop_AMVI.find(i) != loop_AMVI.end()) {
                                return -__DBL_MAX__;
                            }
                        }
                        int loopVAmt1_beforeSplit = E1.rows();
                        
                        UV_bnds.resize(loopVAmt_beforeSplit + loopVAmt1_beforeSplit + 2, 2);
                        UV_bnds.bottomRows(UV_bnds.rows() - 8) << UV_temp1.bottomRows(loopVAmt1_beforeSplit - 3),
                            UV_temp.bottomRows(loopVAmt_beforeSplit - 3);
                        //NOTE: former vertices will be filled with mesh coordinates while constructing the local air mesh
                        
                        bnd.resize(8);
                        bnd[0] = bnd_temp[0];
                        bnd[1] = localMesh.V.rows() - 2;
                        bnd[2] = splitPath[1];
                        bnd[3] = bnd_temp1[2];
                        bnd[4] = bnd_temp1[0];
                        bnd[5] = localMesh.V.rows() - 1;
                        bnd[6] = splitPath[0];
                        bnd[7] = bnd_temp[2];
                        for(int bndI = 0; bndI < bnd.size(); bndI++) {
                            if((bndI != 1) && (bndI != 5)) {
                                const auto finder = globalVI2local.find(bnd[bndI]);
                                assert(finder != globalVI2local.end());
                                bnd[bndI] = finder->second;
                            }
                        }
                        
                        E.resize(UV_bnds.rows(), 2);
                        E.row(0) << 0, 1; E.row(1) << 1, 2; E.row(2) << 2, 3;
                        if(loopVAmt1_beforeSplit - 3 == 0) {
                            E.row(3) << 3, 4;
                        }
                        else {
                            E.row(3) << 3, 8;
                            for(int i = 0; i < loopVAmt1_beforeSplit - 3; i++) {
                                E.row(4 + i) << 8 + i, 9 + i;
                            }
                            E(loopVAmt1_beforeSplit, 1) = 4;
                        }
                        E.row(loopVAmt1_beforeSplit + 1) << 4, 5;
                        E.row(loopVAmt1_beforeSplit + 2) << 5, 6;
                        E.row(loopVAmt1_beforeSplit + 3) << 6, 7;
                        if(loopVAmt_beforeSplit - 3 == 0) {
                            E.row(loopVAmt1_beforeSplit + 4) << 7, 0;
                        }
                        else {
                            E.row(loopVAmt1_beforeSplit + 4) << 7, loopVAmt1_beforeSplit + 5;
                            for(int i = 0; i < loopVAmt_beforeSplit - 3; i++) {
                                E.row(loopVAmt1_beforeSplit + 5 + i) << loopVAmt1_beforeSplit + 5 + i, loopVAmt1_beforeSplit + 6 + i;
                            }
                            E(loopVAmt1_beforeSplit + loopVAmt_beforeSplit + 1, 1) = 0;
                        }
                    }
                }
                
                break;
            }
                
            case 3: // interior split
                //TODO later
                break;
                
            default:
                assert(0 && "invalid split path!");
                break;
        }
        
        // conduct optimization on local mesh
        std::vector<FracCuts::Energy*> energyTerms(1, &SD);
        std::vector<double> energyParams(1, 1.0);
        Optimizer optimizer(localMesh, energyTerms, energyParams, 0, true, !!scaffold, UV_bnds, E, bnd);
        optimizer.precompute();
//        optimizer.scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_separated_AM.obj");
        optimizer.setRelGL2Tol(1.0e-4);
        optimizer.solve(maxIter);
        //            std::cout << "local opt " << optimizer.getIterNum() << " iters" << std::endl;
//        optimizer.result.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_optimized.obj");
//        optimizer.scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test" + std::to_string(splitPath[0]) + "-" + std::to_string(splitPath[1]) + "_optimized_AM.obj");
        
        double curE;
        optimizer.computeEnergyVal(optimizer.getResult(), optimizer.getScaffold(), true, curE, true);
        const double eDec = (initE - curE) * localMesh.surfaceArea / surfaceArea; //!!! this should be written in a more general way, cause this way it only works for E_SD
        
        // get new vertex positions
        newVertPos.resize(2, 2);
        newVertPos << optimizer.getResult().V.row(globalVI2local[splitPath[0]]),
            optimizer.getResult().V.row(localMesh.V.rows() - 1 - cutThrough);
        if(cutThrough) {
            newVertPos.conservativeResize(4, 2);
            newVertPos.row(2) = optimizer.getResult().V.bottomRows(1);
            newVertPos.row(3) = optimizer.getResult().V.row(globalVI2local[splitPath[1]]);
        }
        
        return eDec;
    }
    
    double TriangleSoup::computeLocalEDec(const std::pair<int, int>& edge, Eigen::MatrixXd& newVertPos) const
    {
        assert(vNeighbor.size() == V.rows());
        auto edgeTriIndFinder = edge2Tri.find(edge);
        auto edgeTriIndFinder_dual = edge2Tri.find(std::pair<int, int>(edge.second, edge.first));
        assert(edgeTriIndFinder != edge2Tri.end());
        assert(edgeTriIndFinder_dual != edge2Tri.end());
        
        int vI_boundary = edge.first, vI_interior = edge.second;
        bool cutThrough = false;
        if(isBoundaryVert(edge.first)) {
            if(isBoundaryVert(edge.second)) {
                cutThrough = true;
            }
        }
        else {
            assert(isBoundaryVert(edge.second) && "Input edge must attach mesh boundary!");
            
            vI_boundary = edge.second;
            vI_interior = edge.first;
        }
        
        if(cutThrough) {
            newVertPos.resize(4, 2);
        }
        else {
            newVertPos.resize(2, 2);
        }
        
//        if(cutThrough) {
//            //!!! old mechanism with no bijectivity enforcement on local stencil
//            
//            double eDec = 0.0;
//            for(int toBound = 0; toBound < 2; toBound++) {
//                std::set<int> freeVertGID;
//                freeVertGID.insert(vI_boundary);
//                if(cutThrough) {
//                    freeVertGID.insert(vI_interior);
//                }
//                
//                std::vector<int> tri_toSep;
//                std::pair<int, int> boundaryEdge;
//                isBoundaryVert(vI_boundary, vI_interior, tri_toSep, boundaryEdge, toBound);
//                assert(!tri_toSep.empty());
//                if(cutThrough) {
//                    std::vector<int> tri_interior;
//                    std::pair<int, int> boundaryEdge_interior;
//                    isBoundaryVert(vI_interior, vI_boundary, tri_interior, boundaryEdge_interior, !toBound);
//                    for(const auto& triI : tri_interior) {
//                        bool newTri = true;
//                        for(const auto& triI_b : tri_toSep) {
//                            if(triI_b == triI) {
//                                newTri = false;
//                                break;
//                            }
//                        }
//                        if(newTri) {
//                            tri_toSep.emplace_back(triI);
//                        }
//                    }
//                }
//                
//                std::map<int, Eigen::RowVector2d> newVertPosMap;
//                eDec += computeLocalEDec(tri_toSep, freeVertGID, newVertPosMap);
//                newVertPos.block(toBound, 0, 1, 2) = newVertPosMap[vI_boundary];
//                if(cutThrough) {
//                    newVertPos.row(2 + toBound) = newVertPosMap[vI_interior];
//                }
//            }
//            return eDec;
//        }
//        else {
            std::set<int> freeVertGID;
            freeVertGID.insert(vI_boundary);
            if(cutThrough) {
                freeVertGID.insert(vI_interior);
            }
            
            std::vector<int> tri_toSep, tri_toSep1;
            std::pair<int, int> boundaryEdge;
            isBoundaryVert(vI_boundary, vI_interior, tri_toSep, boundaryEdge, 0);
            assert(!tri_toSep.empty());
            isBoundaryVert(vI_boundary, vI_interior, tri_toSep1, boundaryEdge, 1);
            assert(!tri_toSep1.empty());
            tri_toSep.insert(tri_toSep.end(), tri_toSep1.begin(), tri_toSep1.end());
            if(cutThrough) {
                for(int clockwise = 0; clockwise < 2; clockwise++) {
                    std::vector<int> tri_interior;
                    std::pair<int, int> boundaryEdge_interior;
                    isBoundaryVert(vI_interior, vI_boundary, tri_interior, boundaryEdge_interior, clockwise);
                    for(const auto& triI : tri_interior) {
                        bool newTri = true;
                        for(const auto& triI_b : tri_toSep) {
                            if(triI_b == triI) {
                                newTri = false;
                                break;
                            }
                        }
                        if(newTri) {
                            tri_toSep.emplace_back(triI);
                        }
                    }
                }
            }
        
            std::vector<int> splitPath(2);
            splitPath[0] = vI_boundary;
            splitPath[1] = vI_interior;
            return computeLocalEDec(tri_toSep, freeVertGID, splitPath, newVertPos);
//        }
    }
    
    void TriangleSoup::splitEdgeOnBoundary(const std::pair<int, int>& edge,
                                           const Eigen::MatrixXd& newVertPos,
                                           bool changeVertPos, bool allowCutThrough)
    {
        assert(vNeighbor.size() == V.rows());
        auto edgeTriIndFinder = edge2Tri.find(edge);
        auto edgeTriIndFinder_dual = edge2Tri.find(std::pair<int, int>(edge.second, edge.first));
        assert(edgeTriIndFinder != edge2Tri.end());
        assert(edgeTriIndFinder_dual != edge2Tri.end());
        
        bool duplicateBoth = false;
        int vI_boundary = edge.first, vI_interior = edge.second;
        if(isBoundaryVert(edge.first)) {
            if(allowCutThrough && isBoundaryVert(edge.second)) {
                if(changeVertPos) {
                    assert(newVertPos.rows() == 4);
                }
                duplicateBoth = true;
            }
        }
        else {
            assert(isBoundaryVert(edge.second) && "Input edge must attach mesh boundary!");
            
            vI_boundary = edge.second;
            vI_interior = edge.first;
        }
        
        fracTail.erase(vI_boundary);
        if(!duplicateBoth) {
            fracTail.insert(vI_interior);
            curFracTail = vI_interior;
        }
        else {
            curFracTail = -1;
        }
        curInteriorFracTails.first = curInteriorFracTails.second = -1;
        
        // duplicate vI_boundary
        std::vector<int> tri_toSep[2];
        std::pair<int, int> boundaryEdge[2];
        for(int toBound = 0; toBound < 2; toBound++) { //!!! why?
            isBoundaryVert(vI_boundary, vI_interior, tri_toSep[1], boundaryEdge[1], toBound);
            assert(!tri_toSep[1].empty());
        }
        if(duplicateBoth) {
            isBoundaryVert(vI_interior, vI_boundary, tri_toSep[0], boundaryEdge[0], true);
            assert(!tri_toSep[0].empty());
        }
        
        int nV = static_cast<int>(V_rest.rows());
        V_rest.conservativeResize(nV + 1, 3);
        V_rest.row(nV) = V_rest.row(vI_boundary);
        vertWeight.conservativeResize(nV + 1);
        vertWeight[nV] = vertWeight[vI_boundary];
        V.conservativeResize(nV + 1, 2);
        if(changeVertPos) {
            V.row(nV) = newVertPos.block(1, 0, 1, 2);
            V.row(vI_boundary) = newVertPos.block(0, 0, 1, 2);
        }
        else {
            V.row(nV) = V.row(vI_boundary);
        }
        
        for(const auto triI : tri_toSep[1]) {
            for(int vI = 0; vI < 3; vI++) {
                if(F(triI, vI) == vI_boundary) {
                    // update triangle vertInd, edge2Tri and vNeighbor
                    int vI_post = F(triI, (vI + 1) % 3);
                    int vI_pre = F(triI, (vI + 2) % 3);
                    
                    F(triI, vI) = nV;
                    
                    edge2Tri.erase(std::pair<int, int>(vI_boundary, vI_post));
                    edge2Tri[std::pair<int, int>(nV, vI_post)] = triI;
                    edge2Tri.erase(std::pair<int, int>(vI_pre, vI_boundary));
                    edge2Tri[std::pair<int, int>(vI_pre, nV)] = triI;
                    
                    vNeighbor[vI_pre].erase(vI_boundary);
                    vNeighbor[vI_pre].insert(nV);
                    vNeighbor[vI_post].erase(vI_boundary);
                    vNeighbor[vI_post].insert(nV);
                    vNeighbor[vI_boundary].erase(vI_pre);
                    vNeighbor[vI_boundary].erase(vI_post);
                    vNeighbor.resize(nV + 1);
                    vNeighbor[nV].insert(vI_pre);
                    vNeighbor[nV].insert(vI_post);
                    
                    break;
                }
            }
        }
        vNeighbor[vI_boundary].insert(vI_interior);
        vNeighbor[vI_interior].insert(vI_boundary);
        
        // add cohesive edge pair and update cohEIndex
        const int nCE = static_cast<int>(cohE.rows());
        cohE.conservativeResize(nCE + 1, 4);
        cohE.row(nCE) << vI_interior, nV, vI_interior, vI_boundary; //!! is it a problem?
        cohEIndex[std::pair<int, int>(vI_interior, nV)] = nCE;
        cohEIndex[std::pair<int, int>(vI_boundary, vI_interior)] = -nCE - 1;
        auto CEIfinder = cohEIndex.find(boundaryEdge[1]);
        if(CEIfinder != cohEIndex.end()) {
            if(CEIfinder->second >= 0) {
                cohE(CEIfinder->second, 0) = nV;
            }
            else {
                cohE(-CEIfinder->second - 1, 3) = nV;
            }
            cohEIndex[std::pair<int, int>(nV, boundaryEdge[1].second)] = CEIfinder->second;
            cohEIndex.erase(CEIfinder);
        }
        
        if(duplicateBoth) {
            int nV = static_cast<int>(V_rest.rows());
//            subOptimizerInfo[1].first.insert(nV);
            V_rest.conservativeResize(nV + 1, 3);
            V_rest.row(nV) = V_rest.row(vI_interior);
            vertWeight.conservativeResize(nV + 1);
            vertWeight[nV] = vertWeight[vI_interior];
            V.conservativeResize(nV + 1, 2);
            if(changeVertPos) {
                V.row(nV) = newVertPos.block(2, 0, 1, 2);
                V.row(vI_interior) = newVertPos.block(3, 0, 1, 2);
            }
            else {
                V.row(nV) = V.row(vI_interior);
            }
            
            for(const auto triI : tri_toSep[0]) {
                for(int vI = 0; vI < 3; vI++) {
                    if(F(triI, vI) == vI_interior) {
                        // update triangle vertInd, edge2Tri and vNeighbor
                        int vI_post = F(triI, (vI + 1) % 3);
                        int vI_pre = F(triI, (vI + 2) % 3);
                        
                        F(triI, vI) = nV;
                        
                        edge2Tri.erase(std::pair<int, int>(vI_interior, vI_post));
                        edge2Tri[std::pair<int, int>(nV, vI_post)] = triI;
                        edge2Tri.erase(std::pair<int, int>(vI_pre, vI_interior));
                        edge2Tri[std::pair<int, int>(vI_pre, nV)] = triI;

                        vNeighbor[vI_pre].erase(vI_interior);
                        vNeighbor[vI_pre].insert(nV);
                        vNeighbor[vI_post].erase(vI_interior);
                        vNeighbor[vI_post].insert(nV);
                        vNeighbor[vI_interior].erase(vI_pre);
                        vNeighbor[vI_interior].erase(vI_post);
                        vNeighbor.resize(nV + 1);
                        vNeighbor[nV].insert(vI_pre);
                        vNeighbor[nV].insert(vI_post);
                        
                        break;
                    }
                }
            }
            
            // update cohesive edge pair and update cohEIndex
            cohE(nCE, 2) = nV;
            cohEIndex.erase(std::pair<int, int>(vI_boundary, vI_interior));
            cohEIndex[std::pair<int, int>(vI_boundary, nV)] = -nCE - 1;
            auto CEIfinder = cohEIndex.find(boundaryEdge[0]);
            if(CEIfinder != cohEIndex.end()) {
                if(CEIfinder->second >= 0) {
                    cohE(CEIfinder->second, 0) = nV;
                }
                else {
                    cohE(-CEIfinder->second - 1, 3) = nV;
                }
                cohEIndex[std::pair<int, int>(nV, boundaryEdge[0].second)] = CEIfinder->second;
                cohEIndex.erase(CEIfinder);
            }
        }
    }
    
    void TriangleSoup::mergeBoundaryEdges(const std::pair<int, int>& edge0, const std::pair<int, int>& edge1,
                                          const Eigen::RowVectorXd& mergedPos)
    {
        assert(edge0.second == edge1.first);
        assert(edge2Tri.find(std::pair<int, int>(edge0.second, edge0.first)) == edge2Tri.end());
        assert(edge2Tri.find(std::pair<int, int>(edge1.second, edge1.first)) == edge2Tri.end());
        assert(vNeighbor.size() == V.rows());
        
        fracTail.erase(edge0.second);
        fracTail.insert(edge0.first);
        curFracTail = edge0.first;
        
        V.row(edge0.first) = mergedPos;
        int vBackI = static_cast<int>(V.rows()) - 1;
        if(edge1.second < vBackI) {
            V_rest.row(edge1.second) = V_rest.row(vBackI);
            vertWeight[edge1.second] = vertWeight[vBackI];
            V.row(edge1.second) = V.row(vBackI);
            
            auto finder = fracTail.find(vBackI);
            if(finder != fracTail.end()) {
                fracTail.erase(finder);
                fracTail.insert(edge1.second);
            }
        }
        else {
            assert(edge1.second == vBackI);
        }
        V_rest.conservativeResize(vBackI, 3);
        vertWeight.conservativeResize(vBackI);
        V.conservativeResize(vBackI, 2);
        
//        for(const auto& nbI : vNeighbor[edge1.second]) {
//            std::pair<int, int> edgeToFind[2] = {
//                std::pair<int, int>(edge1.second, nbI),
//                std::pair<int, int>(nbI, edge1.second)
//            };
//            for(int eI = 0; eI < 2; eI++) {
//                auto edgeTri = edge2Tri.find(edgeToFind[eI]);
//                if(edgeTri != edge2Tri.end()) {
//                    for(int vI = 0; vI < 3; vI++) {
//                        if(F(edgeTri->second, vI) == edge1.second) {
//                            F(edgeTri->second, vI) = edge0.first;
//                            break;
//                        }
//                    }
//                }
//            }
//        }
        
        for(int triI = 0; triI < F.rows(); triI++) {
            for(int vI = 0; vI < 3; vI++) {
                if(F(triI, vI) == edge1.second) {
                    F(triI, vI) = edge0.first;
                    break;
                }
            }
        }

        if(edge1.second < vBackI) {
            for(int triI = 0; triI < F.rows(); triI++) {
                for(int vI = 0; vI < 3; vI++) {
                    if(F(triI, vI) == vBackI) {
                        F(triI, vI) = edge1.second;
                    }
                }
            }
//            // not valid because vNeighbor is not updated
//            for(const auto& nbI : vNeighbor[vBackI]) {
//                std::pair<int, int> edgeToFind[2] = {
//                    std::pair<int, int>(vBackI, nbI),
//                    std::pair<int, int>(nbI, vBackI)
//                };
//                for(int eI = 0; eI < 2; eI++) {
//                    auto edgeTri = edge2Tri.find(edgeToFind[eI]);
//                    if(edgeTri != edge2Tri.end()) {
//                        for(int vI = 0; vI < 3; vI++) {
//                            if(F(edgeTri->second, vI) == vBackI) {
//                                F(edgeTri->second, vI) = edge1.second;
//                                break;
//                            }
//                        }
//                    }
//                }
//            }
        }
        
        auto cohEFinder = cohEIndex.find(edge0);
        assert(cohEFinder != cohEIndex.end());
        int cohEBackI = static_cast<int>(cohE.rows()) - 1;
        if(cohEFinder->second >= 0) {
            if(cohEFinder->second < cohEBackI) {
                cohE.row(cohEFinder->second) = cohE.row(cohEBackI);
            }
            else {
                assert(cohEFinder->second == cohEBackI);
            }
        }
        else {
            if(-cohEFinder->second - 1 < cohEBackI) {
                cohE.row(-cohEFinder->second - 1) = cohE.row(cohEBackI);
            }
            else {
                assert(-cohEFinder->second - 1 == cohEBackI);
            }
        }
        cohE.conservativeResize(cohEBackI, 4);
        
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            for(int pI = 0; pI < 4; pI++) {
                if(cohE(cohI, pI) == edge1.second) {
                    cohE(cohI, pI) = edge0.first;
                }
            }
        }
        if(edge1.second < vBackI) {
            for(int cohI = 0; cohI < cohE.rows(); cohI++) {
                for(int pI = 0; pI < 4; pI++) {
                    if(cohE(cohI, pI) == vBackI) {
                        cohE(cohI, pI) = edge1.second;
                    }
                }
            }
        }
        
        // closeup just interior splitted diamond
        //TODO: do it faster by knowing the edge in advance and locate using cohIndex
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            if((cohE(cohI, 0) == cohE(cohI, 2)) &&
                (cohE(cohI, 1) == cohE(cohI, 3)))
           {
               fracTail.erase(cohE(cohI, 0));
               fracTail.erase(cohE(cohI, 1));
               curFracTail = -1;
               
               if(cohI < cohE.rows() - 1) {
                   cohE.row(cohI) = cohE.row(cohE.rows() - 1);
               }
               cohE.conservativeResize(cohE.rows() - 1, 4);
               break;
           }
        }
        
        //TODO: locally update edge2Tri, vNeighbor, cohEIndex
    }
    
}
