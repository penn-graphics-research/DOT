//
//  MeshProcessing.hpp
//  FracCuts
//
//  Created by Minchen Li on 1/31/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef MeshProcessing_hpp
#define MeshProcessing_hpp

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/euler_characteristic.h>
#include <igl/per_vertex_normals.h>

#include <cstdio>

extern std::string outputFolderPath;

namespace FracCuts {
    class MeshProcessing
    {
    public:
        static void run(int argc, char *argv[])
        {
            if(argc > 2) {
                Eigen::MatrixXd V, UV, N;
                Eigen::MatrixXi F, FUV, FN;
                std::string meshPath = std::string(argv[2]);
                std::string meshFileName = meshPath.substr(meshPath.find_last_of('/') + 1);
                std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
                const std::string suffix = meshFileName.substr(meshFileName.find_last_of('.'));
                if(suffix == ".off") {
                    igl::readOFF(meshPath, V, F);
                }
                else if(suffix == ".obj") {
                    igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                }
                else {
                    std::cout << "unkown mesh file format!" << std::endl;
                    return;
                }
                
                if(argc > 3) {
                    int procMode = 0;
                    procMode = std::stoi(argv[3]);
                    switch(procMode) {
                        case 0: {
                            // invert normal of a mesh
                            for(int triI = 0; triI < F.rows(); triI++) {
                                const Eigen::RowVector3i temp = F.row(triI);
                                F(triI, 1) = temp[2];
                                F(triI, 2) = temp[1];
                            }
                            igl::writeOBJ(outputFolderPath + meshName + "_processed.obj", V, F);
                            break;
                        }
                            
                        case 2: {
                            // save texture as mesh
                            if(UV.rows() == 0) {
                                // no input UV
                                std::cout << "compute harmonic UV map" << std::endl;
                                Eigen::VectorXi bnd;
                                igl::boundary_loop(F, bnd); // Find the open boundary
                                if(bnd.size()) {
                                    std::cout << "disk-topology surface" << std::endl;
                                    FUV.resize(0, 3);
                                    
                                    //TODO: what if it has multiple boundaries? or multi-components?
                                    // Map the boundary to a circle, preserving edge proportions
                                    Eigen::MatrixXd bnd_uv;
                                    //            igl::map_vertices_to_circle(V, bnd, bnd_uv);
                                    FracCuts::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);
                                    
                                    //            // Harmonic parametrization
                                    //            igl::harmonic(V, F, bnd, bnd_uv, 1, UV);
                                    
                                    // Harmonic map with uniform weights
                                    Eigen::SparseMatrix<double> A, M;
                                    FracCuts::IglUtils::computeUniformLaplacian(F, A);
                                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV);
                                    //            FracCuts::IglUtils::computeMVCMtr(V, F, A);
                                    //            FracCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV);
                                }
                                else {
                                    // closed surface
                                    std::cout << "closed surface" << std::endl;
                                    if(igl::euler_characteristic(V, F) != 2) {
                                        std::cout << "Input surface genus > 0 or has multiple connected components!" << std::endl;
                                        exit(-1);
                                    }
                                    
                                    FracCuts::TriangleSoup<DIM> *temp = new FracCuts::TriangleSoup<DIM>(V, F, Eigen::MatrixXd());
                                    //            temp->farthestPointCut(); // open up a boundary for Tutte embedding
                                    //                temp->highCurvOnePointCut();
                                    temp->onePointCut();
                                    FUV = temp->F;
                                    
                                    igl::boundary_loop(temp->F, bnd);
                                    assert(bnd.size());
                                    Eigen::MatrixXd bnd_uv;
                                    FracCuts::IglUtils::map_vertices_to_circle(temp->V_rest, bnd, bnd_uv);
                                    Eigen::SparseMatrix<double> A, M;
                                    FracCuts::IglUtils::computeUniformLaplacian(temp->F, A);
                                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV);
                                    
                                    delete temp;
                                }
                            }
                            else {
                                std::cout << "use input UV" << std::endl;
                            }
                            
                            Eigen::MatrixXd V_uv;
                            V_uv.resize(UV.rows(), 3);
                            V_uv << UV, Eigen::VectorXd::Zero(UV.rows(), 1);
                            if(FUV.rows() == 0) {
                                assert(F.rows() > 0);
                                std::cout << "output with F" << std::endl;
                                igl::writeOBJ(outputFolderPath + meshName + "_UV.obj", V_uv, F);
                            }
                            else {
                                std::cout << "output with FUV" << std::endl;
                                igl::writeOBJ(outputFolderPath + meshName + "_UV.obj", V_uv, FUV);
                            }
                            
                            std::cout << "texture saved as mesh into " << outputFolderPath << meshName << "_UV.obj" << std::endl;
                            
                            break;
                        }
                            
                        default:
                            std::cout << "No procMode " << procMode << std::endl;
                            break;
                    }
                }
                else {
                    std::cout << "Please enter procMode!" << std::endl;
                }
            }
            else {
                std::cout << "Please enter mesh file path!" << std::endl;
            }
        }
    };
}

#endif /* MeshProcessing_hpp */
