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
                    IglUtils::readTetMesh(meshPath, TV, TT, F);
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
                            
                        case 3: { // tetrahedralize the input surface mesh
                            // Tetrahedralized interior
                            Eigen::MatrixXd TV;
                            Eigen::MatrixXi TT;
                            Eigen::MatrixXi TF;
                            
                            using namespace Eigen;
                            using namespace std;
                            
                            // Tetrahedralize the interior
                            std::string flag("pq1.414");
                            if(argc > 4) {
                                double maxElemVol = std::stod(argv[4]);
                                if(maxElemVol > 0.0) {
                                    flag += std::string("a") + std::string(argv[4]);
                                }
                            }
                            if(argc > 5) {
                                int addSteinerPoints = stod(argv[5]);
                                if(!addSteinerPoints) {
                                    flag += 'Y';
                                }
                            }
                            else {
                                flag += 'Y';
                            }
                            igl::copyleft::tetgen::tetrahedralize(V,F, flag.c_str(), TV,TT,TF);
                            
                            TriangleSoup<DIM> tetMesh(TV, TT, TV);
                            std::cout << "minVol = " << tetMesh.triArea.minCoeff() << std::endl;
                            std::cout << "maxVol = " << tetMesh.triArea.maxCoeff() << std::endl;
                            std::cout << "avgVol = " << tetMesh.surfaceArea / TT.rows() << std::endl;
                            
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
                        
                        case 5: {
                            // output c++ format of an obj file
                            FILE *out = fopen((meshFolderPath + "/" +
                                               meshName + ".txt").c_str(), "w");
                            assert(out);
                            
                            fprintf(out, "%le, %le, %le",
                                    V(0, 0), V(0, 1), V(0, 2));
                            for(int vI = 1; vI < V.rows(); vI++) {
                                fprintf(out, ", %le, %le, %le",
                                        V(vI, 0), V(vI, 1), V(vI, 2));
                            }
                            fprintf(out, "\n");
                            
                            fprintf(out, "%d, %d, %d",
                                    F(0, 0), F(0, 1), F(0, 2));
                            for(int fI = 1; fI < F.rows(); fI++) {
                                fprintf(out, ", %d, %d, %d",
                                        F(fI, 0), F(fI, 1), F(fI, 2));
                            }
                            fprintf(out, "\n");
                            
                            fclose(out);
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
