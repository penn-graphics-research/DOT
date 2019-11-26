//
//  MeshProcessing.hpp
//  DOT
//
//  Created by Minchen Li on 1/31/18.
//

#ifndef MeshProcessing_hpp
#define MeshProcessing_hpp

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#include "SIMD_DOUBLE_MACROS.hpp"

#include "SVD_EFTYCHIOS/PTHREAD_QUEUE.h"
#include "SVD_EFTYCHIOS/Singular_Value_Decomposition_Helper.h"

#include "Timer.hpp"

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/euler_characteristic.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <immintrin.h>

//#include <mkl.h>

#include <cstdio>


extern std::string outputFolderPath;

extern igl::opengl::glfw::Viewer viewer;

extern Timer timer_step, timer_temp, timer_temp2;

extern void initSIMD(const DOT::Mesh<DIM>* temp);


namespace DOT {
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
                    int procMode = std::stoi(argv[3]);
                    bool findSurface = true;
                    if((procMode == 3) || (procMode == 4)) {
                        findSurface = false;
                    }
                    IglUtils::readTetMesh(meshPath, TV, TT, F, findSurface);
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
                            
//                            Mesh<DIM> tetMesh(TV, TT, TV, 0.0,
//                                                      100.0, 0.4, 1.0);
//                            std::cout << "minVol = " << tetMesh.triArea.minCoeff() << std::endl;
//                            std::cout << "maxVol = " << tetMesh.triArea.maxCoeff() << std::endl;
//                            std::cout << "avgVol = " << tetMesh.surfaceArea / TT.rows() << std::endl;
                            
//                            // Compute barycenters
//                            Eigen::MatrixXd B;
//                            igl::barycenter(TV,TT,B);
//
//                            unsigned char key = '5';
//                            if (key >= '1' && key <= '9')
//                            {
//                                double t = double((key - '1')+1) / 9.0;
//
//                                VectorXd v = B.col(2).array() - B.col(2).minCoeff();
//                                v /= v.col(0).maxCoeff();
//
//                                vector<int> s;
//
//                                for (unsigned i=0; i<v.size();++i)
//                                    if (v(i) < t)
//                                        s.push_back(i);
//
//                                MatrixXd V_temp(s.size()*4,3);
//                                MatrixXi F_temp(s.size()*4,3);
//
//                                for (unsigned i=0; i<s.size();++i)
//                                {
//                                    V_temp.row(i*4+0) = TV.row(TT(s[i],0));
//                                    V_temp.row(i*4+1) = TV.row(TT(s[i],1));
//                                    V_temp.row(i*4+2) = TV.row(TT(s[i],2));
//                                    V_temp.row(i*4+3) = TV.row(TT(s[i],3));
//                                    F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
//                                    F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
//                                    F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
//                                    F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
//                                }
//
//                                viewer.data().clear();
//                                viewer.data().set_mesh(V_temp,F_temp);
////                                viewer.data().set_mesh(TV,TF); // face inverted
//                                viewer.data().set_face_based(true);
//                            }
                            
                            IglUtils::saveTetMesh((meshFolderPath + meshName + ".msh").c_str(),
                                                  TV, TT);
                            
//                            viewer.launch();
                            
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
                            
                        case 10: { // a unit test for profiling energy value computaiton
                            assert(DIM == 3);
                            
                            Timer timer;
                            timer.new_activity("SVD");
                            timer.new_activity("extract");
                            
                            int iter = 1000;
                            std::vector<Eigen::Matrix3d> testF(iter), R_SVD(iter), R_extract(iter);
                            for(int i = 0; i < iter; ++i) {
                                testF[i] = Eigen::Matrix3d::Random() / 2 + Eigen::Matrix3d::Identity();
                            }
                            
                            timer.start(0);
                            for(int i = 0; i < iter; ++i) {
                                AutoFlipSVD<Eigen::Matrix3d> svd(testF[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
                                R_SVD[i] = svd.matrixU() * svd.matrixV().transpose();
                            }
                            timer.stop();
                            
                            timer.start(1);
                            for(int i = 0; i < iter; ++i) {
                                Eigen::Quaterniond q(1.0, 0.0, 0.0, 0.0);
                                IglUtils::extractRotation(testF[i], q, 3);
                                R_extract[i] = q.matrix();
                            }
                            timer.stop();
                            
                            double err = 0;
                            for(int i = 0; i < iter; ++i) {
                                err += (R_extract[i] - R_SVD[i]).norm() / R_SVD[i].norm();
                                
//                                std::cout << F << std::endl;
//                                std::cout << R_extract << std::endl;
//                                std::cout << R_SVD << std::endl;
                            }
                            err /= iter;
                            std::cout << err << std::endl;
                            timer.print();
                            
                            break;
                        }

                        case 11: { // a unit test for profiling AVX2 SVD
                            assert(DIM == 3);

                            Timer timer;
                            timer.new_activity("SVD");
                            timer.new_activity("extract");
                            timer.new_activity("AVX2_pre");
                            timer.new_activity("AVX2");
                            timer.new_activity("AVX2_post");

                            int iter = 1024*1024;
                            std::vector<Eigen::Matrix3d> testF(iter), R_SVD(iter), R_extract(iter);
                            for(int i = 0; i < iter; ++i) {
                                testF[i] = Eigen::Matrix3d::Random() / 2 + Eigen::Matrix3d::Identity();
                            }

                            timer.start(0);
                            for(int i = 0; i < iter; ++i) {
                                AutoFlipSVD<Eigen::Matrix3d> svd(testF[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
                                R_SVD[i] = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
                            }
                            timer.stop();

                            timer.start(1);
                            for(int i = 0; i < iter; ++i) {
                                Eigen::Quaterniond q(1.0, 0.0, 0.0, 0.0);
                                IglUtils::extractRotation(testF[i], q, 3);
                                R_extract[i] = q.matrix();
                            }
                            timer.stop();

                            std::vector<Eigen::Matrix3d> U,V;
                            std::vector<Eigen::Vector3d> Sigma;
                            timer.start(3);
                            IglUtils::computeSVD_SIMD(testF, U, Sigma, V);
                            timer.stop();

                            //exit(0);

                            for(int i=0;i<iter;++i)
                            {
                                R_SVD[i] = U[i] * Sigma[i].asDiagonal() * V[i].transpose();
                            }

                            double err = 0;
                            for(int i = 0; i < iter; ++i) {
                                err += (testF[i] - R_SVD[i]).norm() / testF[i].norm();

//                                std::cout << F[i] << std::endl;
//                                std::cout << R_extract[i] << std::endl;
//                                std::cout << R_SVD[i] << std::endl;
                            }
                            err /= iter;
                            std::cout << "error " << err << std::endl;
                            timer.print();

                            break;
                        }

                        case 12: { // a unit test for profiling energy value computaiton
#if(DIM == 3)
                            //////////// ignore below /////////////////////
                            timer_step.new_activity("matrixComputation");
                            timer_step.new_activity("matrixAssembly");
                            timer_step.new_activity("symbolicFactorization");
                            timer_step.new_activity("numericalFactorization");
                            timer_step.new_activity("backSolve");
                            timer_step.new_activity("lineSearch_other");
                            timer_step.new_activity("modifyGrad");//previously boundarySplit
                            timer_step.new_activity("modifySearchDir");//previously interiorSplit
                            timer_step.new_activity("updateHistory");//previously cornerMerge
                            timer_step.new_activity("lineSearch_eVal");
                            timer_step.new_activity("fullyImplicit_eComp");
                            timer_step.new_activity("solve_extraComp");
                            //////////// ignore above /////////////////////


                            Timer timer;
                            timer.new_activity("gradient");
                            timer.new_activity("hessian");
                            timer.new_activity("energy");

                            Energy<DIM>* e = new StableNHEnergy<DIM>;
                            Mesh<DIM> tetMesh(TV, TT, TV, 100.0, 0.4, 1.0);
                            std::vector<Eigen::Matrix<double, DIM, DIM>> F(TT.rows()), U(TT.rows()), V(TT.rows());
                            std::vector<Eigen::Matrix<double, DIM, 1>> Sigma(TT.rows());
                            std::vector<AutoFlipSVD<Eigen::Matrix<double, DIM, DIM>>> svd(TT.rows());
                            Eigen::VectorXd gradient;
                            int iter = 100;
                            
                            initSIMD(&tetMesh);

                            tetMesh.V.col(0)*=1.1;
                            tetMesh.V.col(1)*=1.3;
                            tetMesh.V.col(2)*=0.9;
//                            for(int i=0;i<tetMesh.u.size();++i)
//                            {
//                                tetMesh.u[i]=1;
//                                tetMesh.lambda[i]=0;
//                            }

                            // precompute SVD
                            e->computeGradientByPK(tetMesh, true, svd, F, U, V, Sigma, 1.0, gradient);

                            // gradient
                            timer.start(0);
                            //for(int i = 0; i < iter; ++i) {
                                //e->computeGradientByPK(tetMesh, false, svd, F, U, V, Sigma, 1.0, gradient);
                                //if(i % 10 == 0) {
                                    //std::cout << i << "/" << iter << " done" << std::endl;
                                //}
                            //}
                            timer.stop();

#ifdef LINSYSSOLVER_USE_CHOLMOD
                            DOT::LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver = new DOT::CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
                            DOT::LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
                            DOT::LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
                            linSysSolver->set_pattern(tetMesh.vNeighbor, tetMesh.fixedVert);

                            // hessian
                            timer.start(1);
                            //for(int i = 0; i < iter; ++i) {
                                //linSysSolver->setZero();
                                //e->computeHessianByPK(tetMesh, false, svd, F, 1.0, linSysSolver, true);
                                //if(i % 10 == 0) {
                                    //std::cout << i << "/" << iter << " done" << std::endl;
                                //}
                            //}
                            timer.stop();

                            // energy
                            {
                                int size = F.size();
                                int ceiling_size = std::ceil(size / 8.) * 8;

                                using T=double;
                                // we only need mu & lambda & sigma to compute energy for fcr material
                                T *mu, *lambda, *sigma0, *sigma1, *sigma2;

                                void *buffers_raw;
                                int buffers_return;
                                buffers_return = posix_memalign(&buffers_raw, 64, ceiling_size * 8);
                                mu = reinterpret_cast<T *>(buffers_raw);
                                buffers_return = posix_memalign(&buffers_raw, 64, ceiling_size * 8);
                                lambda = reinterpret_cast<T *>(buffers_raw);
                                //mu=tetMesh.u.data();
                                //lambda=tetMesh.lambda.data();
                                buffers_return = posix_memalign(&buffers_raw, 64, ceiling_size * 8);
                                sigma0 = reinterpret_cast<T *>(buffers_raw);
                                buffers_return = posix_memalign(&buffers_raw, 64, ceiling_size * 8);
                                sigma1 = reinterpret_cast<T *>(buffers_raw);
                                buffers_return = posix_memalign(&buffers_raw, 64, ceiling_size * 8);
                                sigma2 = reinterpret_cast<T *>(buffers_raw);

                                // comput energy pre
                                for (int i = 0; i < size; ++i) {
                                    mu[i] = tetMesh.u[i]; // necessary since u.data() may not have enough memory
                                    lambda[i] = tetMesh.lambda[i];
                                    sigma0[i] = svd[i].singularValues()(0);
                                    sigma1[i] = svd[i].singularValues()(1);
                                    sigma2[i] = svd[i].singularValues()(2);
                                }
                                for (int i = size; i < ceiling_size; ++i) // for some extra
                                {
                                    mu[i] = 0.;
                                    lambda[i] = 0;
                                    sigma0[i] = 1;
                                    sigma1[i] = 1;
                                    sigma2[i] = 1;
                                }

                                timer.start(2);
                                Eigen::VectorXd energyValPerElem;
                                Eigen::VectorXd energyValPerElem0;
                                Eigen::VectorXd energyValPerElem1;
                                Eigen::VectorXd energyValPerElem2;

                                std::vector<Eigen::Vector3d> gradientXXXX;

                                const T __attribute__ ((aligned(64))) aOne[4] = {1., 1., 1., 1.};
                                const T __attribute__ ((aligned(64))) aOneHalf[4] = {.5, .5, .5, .5};
                                const T __attribute__ ((aligned(64))) aThree[4] = {3., 3., 3., 3.};
                                __m256d vOne = _mm256_load_pd(aOne);
                                __m256d vOneHalf = _mm256_load_pd(aOneHalf);
                                __m256d vThree = _mm256_load_pd(aThree);

                                energyValPerElem.resize(ceiling_size);
                                energyValPerElem0.resize(ceiling_size);
                                energyValPerElem1.resize(ceiling_size);
                                energyValPerElem2.resize(ceiling_size);
                                gradientXXXX.resize(ceiling_size);
                                for (int i = 0; i < iter; ++i) {
#if 1
                                    //e->getEnergyValPerElemBySVD(tetMesh, false, svd, F, U, V, Sigma, energyValPerElem, true);
                                    //e->computeGradientByPK(tetMesh, false, svd, F, U, V, Sigma, 1.0, gradient);
    //void StableNHEnergy<dim>::compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
                                                      //double u, double lambda,
                                                      //Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const
                                    for(int element=0; element<size;++element)
                                    {
                                        e->compute_dE_div_dsigma(svd[element].singularValues(),mu[element],lambda[element],gradientXXXX[element]);

                                    }
#else
                                    tbb::parallel_for(0, (int) ceiling_size / 4, 1, [&](int e)
                                            //for(int e=0;e<ceiling_size/4;++e) // 4 not 8 because of double
                                    {
                                        __m256d vResult;
                                        __m256d vResult0;
                                        __m256d vResult1;
                                        __m256d vResult2;
                                        //ENERGY_Stable_NeoHookean(e,vOne,vOneHalf,vThree,mu,lambda,sigma0,sigma1,sigma2,vResult);
                                        //double __attribute__ ((aligned(64))) buffer[4];
                                        //_mm256_store_pd(buffer, vResult);
                                        //energyValPerElem[e * 4 + 0] = buffer[0];
                                        //energyValPerElem[e * 4 + 1] = buffer[1];
                                        //energyValPerElem[e * 4 + 2] = buffer[2];
                                        //energyValPerElem[e * 4 + 3] = buffer[3];

                                        PHAT_Stable_NeoHookean(e, vOne, vTwo, mu, lambda, sigma0, sigma1, sigma2, vResult0, vResult1, vResult2);
                                        double __attribute__ ((aligned(64))) buffer[4];
                                        _mm256_store_pd(buffer, vResult0);
                                        energyValPerElem0[e * 4 + 0] = buffer[0];
                                        energyValPerElem0[e * 4 + 1] = buffer[1];
                                        energyValPerElem0[e * 4 + 2] = buffer[2];
                                        energyValPerElem0[e * 4 + 3] = buffer[3];
                                        _mm256_store_pd(buffer, vResult1);
                                        energyValPerElem1[e * 4 + 0] = buffer[0];
                                        energyValPerElem1[e * 4 + 1] = buffer[1];
                                        energyValPerElem1[e * 4 + 2] = buffer[2];
                                        energyValPerElem1[e * 4 + 3] = buffer[3];
                                        _mm256_store_pd(buffer, vResult2);
                                        energyValPerElem2[e * 4 + 0] = buffer[0];
                                        energyValPerElem2[e * 4 + 1] = buffer[1];
                                        energyValPerElem2[e * 4 + 2] = buffer[2];
                                        energyValPerElem2[e * 4 + 3] = buffer[3];
                                    });
#endif
                                    if (i % 10 == 0) {
                                        std::cout << i << "/" << iter << " done" << std::endl;
                                    }
                                }
                                timer.stop();

                                for (int i = 0; i < size; ++i)
                                    //std::cout << i << " energy is " << energyValPerElem[i] << std::endl;
                                    std::cout << i << " gradient is " << gradientXXXX[i][0] << " " << gradientXXXX[i][1] << " " << gradientXXXX[i][2] << std::endl;
                                    //std::cout << i << " gradient is " << energyValPerElem0[i] << " " << energyValPerElem1[i]<< " " << energyValPerElem2[i] << std::endl;
                            }
                            timer.print();
                            delete e;
#else
                            std::cout << "this only runs for DIM = 3" << std::endl;
#endif
                            break;
                        }
                            
                        case 13: { // a unit test for disabling cout
                            Timer timer;
                            timer.new_activity("cout");
                            timer.new_activity("disabling cout");

                            int iter = 1000000;

                            timer.start(0);
                            for(int i = 0; i < iter; ++i) {
                                std::cout << "something simething ssser ilninive " << i << std::endl;
                            }
                            timer.stop();

                            std::cout.setstate(std::ios_base::failbit);
                            timer.start(1);
                            for(int i = 0; i < iter; ++i) {
                                std::cout << "something simething ssser ilninive " << i << std::endl;
                            }
                            timer.stop();
                            printf("printf is not disabled this way!");
                            std::cout.clear();
                            
                            timer.print();
                            break;
                        }
                            
                        case 15: { // visualize input tet mesh output surface mesh
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
                            
                            
                            std::vector<bool> isSurfNode(TV.rows(), false);
                            for(int tI = 0; tI < F.rows(); ++tI) {
                                isSurfNode[F(tI, 0)] = true;
                                isSurfNode[F(tI, 1)] = true;
                                isSurfNode[F(tI, 2)] = true;
                            }
                            
                            std::vector<int> tetIndToSurf(TV.rows(), -1);
                            std::vector<int> surfIndToTet(TV.rows(), -1);
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
                                V_surf.row(vI) = TV.row(surfIndToTet[vI]);
                            }
                            Eigen::MatrixXi F_surf(F.rows(), 3);
                            for(int tI = 0; tI < F.rows(); ++tI) {
                                F_surf(tI, 0) = tetIndToSurf[F(tI, 0)];
                                F_surf(tI, 1) = tetIndToSurf[F(tI, 1)];
                                F_surf(tI, 2) = tetIndToSurf[F(tI, 2)];
                            }
                            igl::writeOBJ(outputFolderPath + meshName + "_surf.obj", V_surf, F_surf);
                            
                            viewer.launch();
                            
                            break;
                        }
                            
                        case 16: {
                            // turn status into obj
                            std::cout << "mesh loaded" << std::endl;
                            
                            std::vector<bool> isSurfNode(TV.rows(), false);
                            for(int tI = 0; tI < F.rows(); ++tI) {
                                isSurfNode[F(tI, 0)] = true;
                                isSurfNode[F(tI, 1)] = true;
                                isSurfNode[F(tI, 2)] = true;
                            }
                            
                            std::vector<int> tetIndToSurf(TV.rows(), -1);
                            std::vector<int> surfIndToTet(TV.rows(), -1);
                            int sVI = 0;
                            for(int vI = 0; vI < isSurfNode.size(); ++vI) {
                                if(isSurfNode[vI]) {
                                    tetIndToSurf[vI] = sVI;
                                    surfIndToTet[sVI] = vI;
                                    ++sVI;
                                }
                            }
                            
                            Eigen::MatrixXi F_surf(F.rows(), 3);
                            for(int tI = 0; tI < F.rows(); ++tI) {
                                F_surf(tI, 0) = tetIndToSurf[F(tI, 0)];
                                F_surf(tI, 1) = tetIndToSurf[F(tI, 1)];
                                F_surf(tI, 2) = tetIndToSurf[F(tI, 2)];
                            }
                            
                            Eigen::MatrixXd V_surf(sVI, 3);
                            for(int i = 0; i < 401; ++i) {
//                                std::ifstream in("/Users/mincli/Desktop/test/DOT/output/elf63K/status" + std::to_string(i));
                                std::ifstream in("/Users/mincli/Desktop/horse136K/status" + std::to_string(i));
                                assert(in.is_open());
                                std::string line;
                                while(std::getline(in, line)) {
                                    std::stringstream ss(line);
                                    std::string token;
                                    ss >> token;
                                    if(token == "position") {
                                        std::cout << "read restart position" << std::endl;
                                        int posRows, dim_in;
                                        ss >> posRows >> dim_in;
                                        assert(posRows == TV.rows());
                                        assert(dim_in == TV.cols());
                                        for(int vI = 0; vI < posRows; ++vI) {
                                            in >> TV(vI, 0) >> TV(vI, 1);
                                            if(DIM == 3) {
                                                in >> TV(vI, 2);
                                            }
                                        }
                                    }
                                }
                                in.close();
                                
                                for(int vI = 0; vI < sVI; ++vI) {
                                    V_surf.row(vI) = TV.row(surfIndToTet[vI]);
                                }
                                igl::writeOBJ(outputFolderPath + std::to_string(i) + ".obj",
                                              V_surf, F_surf);
                                
                            }
                            break;
                        }
                            
                        // ./build/DOT_bin 2 /Users/mincli/Desktop/DOT/output/Sharkey_stretch_FCR_DOT6_20190408171112t1/finalResult_mesh.obj 17 100 0
                        // compile with #define DIM 2
                        case 17: { // visualize node field
                            std::string outerI = argc > 4 ? argv[4] : "0";
                            std::string innerI = argc > 5 ? argv[5] : "0";
                            
                            Eigen::VectorXd f_n;
                            IglUtils::readVectorFromFile(meshFolderPath +
                                                         outerI + "_" + innerI, f_n);
                            assert(f_n.size() == V.rows() * DIM);
                            
                            bool elem = true;
                            Eigen::MatrixXd color;
                            if(elem) {
                                Eigen::VectorXd fNorm_e(F.rows());
                                fNorm_e.setZero();
                                for(int elemI = 0; elemI < F.rows(); ++elemI) {
                                    const Eigen::Matrix<int, 1, DIM + 1> elemVInd = F.row(elemI);
                                    for(int vI = 0; vI < DIM + 1; ++vI) {
                                        fNorm_e[elemI] += f_n.segment<DIM>(elemVInd[vI] * DIM).norm();
                                    }
                                    fNorm_e[elemI] = std::log10(fNorm_e[elemI]);
                                }
                                
                                double lowerBound = -5.65;
                                double upperBound = -2.8;
                                IglUtils::mapScalarToColor(fNorm_e, color,
                                                           lowerBound, upperBound, 1);
                                std::cout << "error range: " << fNorm_e.minCoeff() << " " <<
                                    fNorm_e.maxCoeff() << std::endl;
                                
                                viewer.data().set_face_based(true);
                            }
                            else {
                                Eigen::VectorXd fNorm_n(V.rows());
                                for(int vI = 0; vI < V.rows(); ++vI) {
                                    fNorm_n[vI] = std::log10(f_n.segment<DIM>(vI * DIM).norm());
                                }
                                
                                double lowerBound = -6;
                                double upperBound = -3;
                                IglUtils::mapScalarToColor(fNorm_n, color,
                                                           lowerBound, upperBound, 1);
                                std::cout << "error range: " << fNorm_n.minCoeff() << " " <<
                                    fNorm_n.maxCoeff() << std::endl;
                                
                                viewer.data().set_face_based(false);
                            }
                            
                            viewer.data().set_mesh(V, F);
                            viewer.data().set_colors(color);
                            viewer.data().show_lines = false;
                            viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
                            viewer.core.camera_zoom *= 2.0;
                            viewer.core.lighting_factor = 0.0;
                            
                            viewer.launch();
                            
                            break;
                        }
                            
                        case 18: { // mesh quality check - with node degree, node mass, dF/dx condition, elem area
                            Mesh<DIM> tetMesh(TV, TT, TV,
                                                      100.0, 0.4, 1.0);
                            
                            FILE *out = fopen((meshFolderPath + "nodeDeg.txt").c_str(), "w");
                            assert(out);
                            for(int i = 0; i < tetMesh.vNeighbor.size(); ++i) {
                                fprintf(out, "%lu\n", tetMesh.vNeighbor[i].size());
                            }
                            fclose(out);
                            
                            out = fopen((meshFolderPath + "nodeMass.txt").c_str(), "w");
                            assert(out);
                            double avgM = tetMesh.massMatrix.sum() / tetMesh.massMatrix.rows();
                            for(int i = 0; i < tetMesh.massMatrix.rows(); ++i) {
                                fprintf(out, "%le\n", tetMesh.massMatrix.coeff(i, i) / avgM);
                            }
                            fclose(out);
                            
                            out = fopen((meshFolderPath + "dFdxCond.txt").c_str(), "w");
                            assert(out);
                            for(int i = 0; i < tetMesh.restTriInv.size(); ++i) {
                                double cond = tetMesh.restTriInv[i].inverse().norm() * tetMesh.restTriInv[i].norm();
                                fprintf(out, "%le\n", cond);
                            }
                            fclose(out);
                            
                            out = fopen((meshFolderPath + "elemVol.txt").c_str(), "w");
                            assert(out);
                            double avgVol = 0.0;
                            for(int i = 0; i < tetMesh.triArea.size(); ++i) {
                                avgVol += tetMesh.triArea[i];
                            }
                            avgVol /= tetMesh.triArea.size();
                            for(int i = 0; i < tetMesh.triArea.size(); ++i) {
                                fprintf(out, "%le\n", tetMesh.triArea[i] / avgVol);
                            }
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
