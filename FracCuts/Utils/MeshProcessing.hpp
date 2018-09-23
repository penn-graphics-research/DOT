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
#include <igl/opengl/glfw/Viewer.h>

#include <cstdio>


extern std::string outputFolderPath;

extern igl::opengl::glfw::Viewer viewer;


namespace FracCuts {
    class MeshProcessing
    {
    public:
        static void run(int argc, char *argv[])
        {
            if(argc > 2) {
                Eigen::MatrixXd V, UV, N, TV;
                Eigen::MatrixXi F, FUV, FN, TT;
                
                const std::string meshPath = std::string(argv[2]);
                const std::string meshFolderPath = meshPath.substr(0, meshPath.find_last_of('/') + 1);
                const std::string meshFileName = meshPath.substr(meshPath.find_last_of('/') + 1);
                const std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
                const std::string suffix = meshFileName.substr(meshFileName.find_last_of('.'));
                
                if(suffix == ".off") {
                    igl::readOFF(meshPath, V, F);
                }
                else if(suffix == ".obj") {
                    igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                }
                else if(suffix == ".msh") {
                    IglUtils::readTetMesh(meshPath, TV, TT);
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
                            
                        case 3: { // tetrahedralize the input surface mesh
                            // Tetrahedralized interior
                            Eigen::MatrixXd TV;
                            Eigen::MatrixXi TT;
                            Eigen::MatrixXi TF;
                            
                            using namespace Eigen;
                            using namespace std;
                            
                            // Tetrahedralize the interior
                            igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);
                            
                            // Compute barycenters
                            Eigen::MatrixXd B;
                            igl::barycenter(TV,TT,B);
                            
                            unsigned char key = '5';
                            if (key >= '1' && key <= '9')
                            {
                                double t = double((key - '1')+1) / 9.0;
                                
                                VectorXd v = B.col(2).array() - B.col(2).minCoeff();
                                v /= v.col(0).maxCoeff();
                                
                                vector<int> s;
                                
                                for (unsigned i=0; i<v.size();++i)
                                    if (v(i) < t)
                                        s.push_back(i);
                                
                                MatrixXd V_temp(s.size()*4,3);
                                MatrixXi F_temp(s.size()*4,3);
                                
                                for (unsigned i=0; i<s.size();++i)
                                {
                                    V_temp.row(i*4+0) = TV.row(TT(s[i],0));
                                    V_temp.row(i*4+1) = TV.row(TT(s[i],1));
                                    V_temp.row(i*4+2) = TV.row(TT(s[i],2));
                                    V_temp.row(i*4+3) = TV.row(TT(s[i],3));
                                    F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
                                    F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
                                    F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
                                    F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
                                }
                                
                                viewer.data().clear();
                                viewer.data().set_mesh(V_temp,F_temp);
//                                viewer.data().set_mesh(TV,TF); // face inverted
                                viewer.data().set_face_based(true);
                            }
                            
                            IglUtils::saveTetMesh((meshFolderPath + meshName + ".msh").c_str(),
                                                  TV, TT);
                            
                            viewer.launch();
                            
                            break;
                        }
                            
                        case 4: { // visualize input tet mesh and check inversion
                            // Compute barycenters
                            Eigen::MatrixXd B;
                            igl::barycenter(TV, TT, B);
                            
                            unsigned char key = '5';
                            if (key >= '1' && key <= '9')
                            {
                                double t = double((key - '1')+1) / 9.0;
                                
                                Eigen::VectorXd v = B.col(2).array() - B.col(2).minCoeff();
                                v /= v.col(0).maxCoeff();
                                
                                std::vector<int> s;
                                
                                for (unsigned i=0; i<v.size();++i)
                                    if (v(i) < t)
                                        s.push_back(i);
                                
                                Eigen::MatrixXd V_temp(s.size()*4,3);
                                Eigen::MatrixXi F_temp(s.size()*4,3);
                                
                                for (unsigned i=0; i<s.size();++i)
                                {
                                    V_temp.row(i*4+0) = TV.row(TT(s[i],0));
                                    V_temp.row(i*4+1) = TV.row(TT(s[i],1));
                                    V_temp.row(i*4+2) = TV.row(TT(s[i],2));
                                    V_temp.row(i*4+3) = TV.row(TT(s[i],3));
                                    F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
                                    F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
                                    F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
                                    F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
                                }
                                
                                viewer.data().clear();
                                viewer.data().set_mesh(V_temp,F_temp);
                                viewer.data().set_face_based(true);
                            }
                            
                            bool inversion = false;
                            for(int elemI = 0; elemI < TT.rows(); elemI++) {
                                const Eigen::RowVector4i& tetVInd = TT.row(elemI);
                                Eigen::Matrix3d m;
                                m.col(0) = (TV.row(tetVInd(1)) - TV.row(tetVInd(0))).transpose();
                                m.col(1) = (TV.row(tetVInd(2)) - TV.row(tetVInd(0))).transpose();
                                m.col(2) = (TV.row(tetVInd(3)) - TV.row(tetVInd(0))).transpose();
                                double det = m.determinant();
                                if(det <= 0.0) {
                                    std::cout << "element inversion detected " << det << std::endl;
                                    inversion = true;
                                }
                            }
                            if(!inversion) {
                                std::cout << "no element inversion " << std::endl;
                            }
                            
                            viewer.launch();
                            
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
