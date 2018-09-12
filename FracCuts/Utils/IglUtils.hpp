//
//  IglUtils.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include "TriangleSoup.hpp"

#ifdef USE_CLOSEDFORMSVD2D
#include "ClosedFormSVD2d.hpp"
#else
#include "AutoFlipSVD.hpp"
#endif

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>

namespace FracCuts {
    
    // a static class implementing basic geometry processing operations that are not provided in libIgl
    class IglUtils {
    public:
        static void computeGraphLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL);
        
        // graph laplacian with half-weighted boundary edge, the computation is also faster
        static void computeUniformLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL);
        
        static void computeMVCMtr(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& MVCMtr);
        
        static void fixedBoundaryParam_MVC(Eigen::SparseMatrix<double> A, const Eigen::VectorXi& bnd,
                                           const Eigen::MatrixXd& bnd_uv, Eigen::MatrixXd& UV_Tutte);
        
        static void mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3]);
        static void computeDeformationGradient(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& F);
        
        // to a circle with the perimeter equal to the length of the boundary on the mesh
        static void map_vertices_to_circle(const Eigen::MatrixXd& V, const Eigen::VectorXi& bnd, Eigen::MatrixXd& UV);
        
        static void mapScalarToColor_bin(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double thres);
        static void mapScalarToColor(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color,
                                     double lowerBound, double upperBound, int opt = 0);
        
        static void addBlockToMatrix(Eigen::SparseMatrix<double>& mtr, const Eigen::MatrixXd& block,
                                     const Eigen::VectorXi& index, int dim);
        static void addBlockToMatrix(const Eigen::MatrixXd& block, const Eigen::VectorXi& index, int dim,
                                     Eigen::VectorXd* V, Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL);
        static void addDiagonalToMatrix(const Eigen::VectorXd& diagonal, const Eigen::VectorXi& index, int dim,
                                     Eigen::VectorXd* V, Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL);
        
        template<typename Scalar, int rows, int cols>
        static void symmetrizeMatrix(Eigen::Matrix<Scalar, rows, cols>& mtr) {
            if(rows != cols) {
                return;
            }
            
            for(int rowI = 0; rowI < rows; rowI++) {
                for(int colI = rowI + 1; colI < cols; colI++) {
                    double &a = mtr(rowI, colI), &b = mtr(colI, rowI);
                    a = b = (a + b) / 2.0;
                }
            }
        }
        
        // project a symmetric real matrix to the nearest SPD matrix
        template<typename Scalar, int size>
        static void makePD(Eigen::Matrix<Scalar, size, size>& symMtr) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
            if(eigenSolver.eigenvalues()[0] >= 0.0) {
                return;
            }
            Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
            int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
            for(int i = 0; i < rows; i++) {
                if(D.diagonal()[i] < 0.0) {
                    D.diagonal()[i] = 0.0;
                }
                else {
                    break;
                }
            }
            symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
        }
        template<typename Scalar, int size>
        static void makePD2d(Eigen::Matrix<Scalar, size, size>& symMtr)
        {
            // based on http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
            
            if(size == Eigen::Dynamic) {
                assert(symMtr.rows() == 2);
            }
            else {
                assert(size == 2);
            }
            
            const double a = symMtr(0, 0);
            const double b = (symMtr(0, 1) + symMtr(1, 0)) / 2.0;
            const double d = symMtr(1, 1);
            
            double b2 = b * b;
            const double D = a * d - b2;
            const double T = a + d;
            const double L2 = T / 2.0 - std::sqrt(T * T / 4.0 - D);
            if(L2 < 0.0) {
                const double L1 = T / 2.0 + std::sqrt(T * T / 4.0 - D);
                if(L1 <= 0.0) {
                    symMtr.setZero();
                }
                else {
                    if(b2 == 0.0) {
                        symMtr << L1, 0.0, 0.0 ,0.0;
                    }
                    else {
                        const double L1md = L1 - d;
                        const double L1md_div_L1 = L1md / L1;
                        symMtr(0, 0) = L1md_div_L1 * L1md;
                        symMtr(0, 1) = symMtr(1, 0) = b * L1md_div_L1;
                        symMtr(1, 1) = b2 / L1;
                    }
                }
            }
        }
        
        static void writeSparseMatrixToFile(const std::string& filePath, const Eigen::SparseMatrix<double>& mtr, bool MATLAB = false);
        static void writeSparseMatrixToFile(const std::string& filePath, const Eigen::VectorXi& I, const Eigen::VectorXi& J,
                                            const Eigen::VectorXd& V, bool MATLAB = false);
        static void loadSparseMatrixFromFile(const std::string& filePath, Eigen::SparseMatrix<double>& mtr);
        
        static void sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr,
                                          Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V);
        static void sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& V);
        
        static const std::string rtos(double real);
        
        static void differentiate_normalize(const Eigen::Vector2d& var, Eigen::Matrix2d& deriv);
        static void differentiate_xxT(const Eigen::Vector2d& var, Eigen::Matrix<Eigen::RowVector2d, 2, 2>& deriv,
                                      double param = 1.0);
        
        static double computeRotAngle(const Eigen::RowVector2d& from, const Eigen::RowVector2d& to);
        
        // test wether 2D segments ab intersect with cd
        static bool Test2DSegmentSegment(const Eigen::RowVector2d& a, const Eigen::RowVector2d& b,
                                         const Eigen::RowVector2d& c, const Eigen::RowVector2d& d,
                                         double eps = 0.0);
        
        static void addThickEdge(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& UV,
                                 Eigen::MatrixXd& seamColor, const Eigen::RowVector3d& color,
                                 const Eigen::RowVector3d& v0, const Eigen::RowVector3d& v1,
                                 double halfWidth, double texScale, bool UVorSurface = false,
                                 const Eigen::RowVector3d& normal = Eigen::RowVector3d());
        
        static void saveMesh_Seamster(const std::string& filePath,
                                      const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
        
        static void smoothVertField(const TriangleSoup& mesh, Eigen::VectorXd& field);
        
        static void compute_dsigma_div_dx(const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                          const Eigen::MatrixXd& A,
                                          Eigen::MatrixXd& dsigma_div_dx);
        
        static void compute_d2sigma_div_dx2(const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                            const Eigen::MatrixXd& A,
                                            Eigen::MatrixXd& d2sigma_div_dx2);
        static void compute_d2sigma_div_dF2(const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                            Eigen::MatrixXd& d2sigma_div_dF2);
        static void compute_dU_and_dV_div_dF(const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                             Eigen::MatrixXd& dU_div_dF,
                                             Eigen::MatrixXd& dV_div_dF);
        
        static void compute_dF_div_dx(const Eigen::MatrixXd& A,
                                      Eigen::MatrixXd& dF_div_dx);
        static void dF_div_dx_mult(const Eigen::MatrixXd& right,
                                   const Eigen::MatrixXd& A,
                                   Eigen::MatrixXd& result,
                                   bool symmetric);
        static void dF_div_dx_mult(const Eigen::MatrixXd& right,
                                   const Eigen::MatrixXd& A,
                                   Eigen::VectorXd& result);
        
        static void sampleSegment(const Eigen::RowVectorXd& vs,
                                  const Eigen::RowVectorXd& ve,
                                  double spacing,
                                  Eigen::MatrixXd& inBetween);
    };
    
}

#endif /* IglUtils_hpp */
