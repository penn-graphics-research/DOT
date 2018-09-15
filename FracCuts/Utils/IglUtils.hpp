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

#include "LinSysSolver.hpp"

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
        template<int blockSize, int indexSize>
        static void addBlockToMatrix(const Eigen::Matrix<double, blockSize, blockSize>& block,
                                     const Eigen::Matrix<int, indexSize, 1>& index,
                                     int dim, Eigen::VectorXd* V,
                                     Eigen::VectorXi* I = NULL,
                                     Eigen::VectorXi* J = NULL)
        {
            assert(indexSize * dim == blockSize);
            
            int num_free = 0;
            for(int indI = 0; indI < indexSize; indI++) {
                if(index[indI] >= 0) {
                    num_free++;
                }
            }
            if(!num_free) {
                return;
            }
            
            assert(V);
            int tripletInd = static_cast<int>(V->size());
            const int entryAmt = static_cast<int>(dim * dim * num_free * num_free);
            V->conservativeResize(tripletInd + entryAmt);
            if(I) {
                assert(J);
                assert(I->size() == tripletInd);
                assert(J->size() == tripletInd);
                I->conservativeResize(tripletInd + entryAmt);
                J->conservativeResize(tripletInd + entryAmt);
            }
            
            for(int indI = 0; indI < indexSize; indI++) {
                if(index[indI] < 0) {
                    continue;
                }
                int startIndI = index[indI] * dim;
                int startIndI_block = indI * dim;
                
                for(int indJ = 0; indJ < indexSize; indJ++) {
                    if(index[indJ] < 0) {
                        continue;
                    }
                    int startIndJ = index[indJ] * dim;
                    int startIndJ_block = indJ * dim;
                    
                    for(int dimI = 0; dimI < dim; dimI++) {
                        for(int dimJ = 0; dimJ < dim; dimJ++) {
                            (*V)[tripletInd] = block(startIndI_block + dimI, startIndJ_block + dimJ);
                            if(I) {
                                (*I)[tripletInd] = startIndI + dimI;
                                (*J)[tripletInd] = startIndJ + dimJ;
                            }
                            tripletInd++;
                        }
                    }
                }
            }
            assert(tripletInd == V->size());
        }
        static void addDiagonalToMatrix(const Eigen::VectorXd& diagonal,
                                        const Eigen::VectorXi& index,
                                        int dim, Eigen::VectorXd* V,
                                        Eigen::VectorXi* I = NULL,
                                        Eigen::VectorXi* J = NULL);
        static void addBlockToMatrix(const Eigen::Matrix<double, 2, 6>& block,
                                     const Eigen::RowVector3i& index, int rowIndI,
                                     LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver);
        static void addIdBlockToMatrixDiag(const Eigen::VectorXi& index,
                                           LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver);
        
        template<typename Scalar, int size>
        static void symmetrizeMatrix(Eigen::Matrix<Scalar, size, size>& mtr) {
            if(size == Eigen::Dynamic) {
                assert(mtr.rows() == mtr.cols());
            }
            int rows = ((size == Eigen::Dynamic) ? mtr.rows() : size);
            
            for(int rowI = 0; rowI < rows; rowI++) {
                for(int colI = rowI + 1; colI < rows; colI++) {
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
            const double T_div_2 = (a + d) / 2.0;
            const double sqrtTT4D = std::sqrt(T_div_2 * T_div_2 - D);
            const double L2 = T_div_2 - sqrtTT4D;
            if(L2 < 0.0) {
                const double L1 = T_div_2 + sqrtTT4D;
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
        
        static void writeSparseMatrixToFile(const std::string& filePath,
                                            const Eigen::SparseMatrix<double>& mtr,
                                            bool MATLAB = false);
        static void writeSparseMatrixToFile(const std::string& filePath,
                                            const Eigen::VectorXi& I, const Eigen::VectorXi& J,
                                            const Eigen::VectorXd& V, bool MATLAB = false);
        static void writeSparseMatrixToFile(const std::string& filePath,
                                            const std::map<std::pair<int, int>, double>& mtr,
                                            bool MATLAB = false);
        static void loadSparseMatrixFromFile(const std::string& filePath,
                                             Eigen::SparseMatrix<double>& mtr);
        
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
        
        static void compute_dF_div_dx(const Eigen::Matrix2d& A,
                                      Eigen::Matrix<double, 6, 4>& dF_div_dx);
        template<int colSize>
        static void dF_div_dx_mult(const Eigen::Matrix<double, 4, colSize>& right,
                                   const Eigen::Matrix2d& A,
                                   Eigen::Matrix<double, 6, colSize>& result,
                                   bool symmetric)
        {
            if(colSize == Eigen::Dynamic) {
                assert(right.cols() > 0);
            }
            else {
                assert(colSize > 0);
            }
            
            if(symmetric) {
                if(colSize == Eigen::Dynamic) {
                    assert(right.cols() == 6);
                }
                else {
                    assert(colSize == 6);
                }
                // int colI = 0;
                const double _0000 = right(0, 0) * A(0, 0);
                const double _0010 = right(0, 0) * A(1, 0);
                const double _1001 = right(1, 0) * A(0, 1);
                const double _1011 = right(1, 0) * A(1, 1);
                const double _2000 = right(2, 0) * A(0, 0);
                const double _2010 = right(2, 0) * A(1, 0);
                const double _3001 = right(3, 0) * A(0, 1);
                const double _3011 = right(3, 0) * A(1, 1);
                result(0, 0) = -_0000 - _0010 - _1001 - _1011;
                result(1, 0) = result(0, 1) = -_2000 - _2010 - _3001 - _3011;
                result(2, 0) = result(0, 2) = _0000 + _1001;
                result(3, 0) = result(0, 3) = _2000 + _3001;
                result(4, 0) = result(0, 4) = _0010 + _1011;
                result(5, 0) = result(0, 5) = _2010 + _3011;
                // colI = 1;
                const double _2100 = right(2, 1) * A(0, 0);
                const double _2110 = right(2, 1) * A(1, 0);
                const double _3101 = right(3, 1) * A(0, 1);
                const double _3111 = right(3, 1) * A(1, 1);
                result(1, 1) = -_2100 - _2110 - _3101 - _3111;
                result(2, 1) = result(1, 2) = right(0, 1) * A(0, 0) + right(1, 1) * A(0, 1);
                result(3, 1) = result(1, 3) = _2100 + _3101;
                result(4, 1) = result(1, 4) = right(0, 1) * A(1, 0) + right(1, 1) * A(1, 1);
                result(5, 1) = result(1, 5) = _2110 + _3111;
                // colI = 2;
                result(2, 2) = right(0, 2) * A(0, 0) + right(1, 2) * A(0, 1);
                result(3, 2) = result(2, 3) = right(2, 2) * A(0, 0) + right(3, 2) * A(0, 1);
                result(4, 2) = result(2, 4) = right(0, 2) * A(1, 0) + right(1, 2) * A(1, 1);
                result(5, 2) = result(2, 5) = right(2, 2) * A(1, 0) + right(3, 2) * A(1, 1);
                // colI = 3;
                result(3, 3) = right(2, 3) * A(0, 0) + right(3, 3) * A(0, 1);
                result(4, 3) = result(3, 4) = right(0, 3) * A(1, 0) + right(1, 3) * A(1, 1);
                result(5, 3) = result(3, 5) = right(2, 3) * A(1, 0) + right(3, 3) * A(1, 1);
                // colI = 4;
                result(4, 4) = right(0, 4) * A(1, 0) + right(1, 4) * A(1, 1),
                result(5, 4) = result(4, 5) = right(2, 4) * A(1, 0) + right(3, 4) * A(1, 1);
                // colI = 5;
                result(5, 5) = right(2, 5) * A(1, 0) + right(3, 5) * A(1, 1);
            }
            else {
                for(int colI = 0; colI < right.cols(); colI++) {
                    const double _000 = right(0, colI) * A(0, 0);
                    const double _010 = right(0, colI) * A(1, 0);
                    const double _101 = right(1, colI) * A(0, 1);
                    const double _111 = right(1, colI) * A(1, 1);
                    const double _200 = right(2, colI) * A(0, 0);
                    const double _210 = right(2, colI) * A(1, 0);
                    const double _301 = right(3, colI) * A(0, 1);
                    const double _311 = right(3, colI) * A(1, 1);
                    result(0, colI) = -_000 - _010 - _101 - _111;
                    result(1, colI) = -_200 - _210 - _301 - _311;
                    result(2, colI) = _000 + _101;
                    result(3, colI) = _200 + _301;
                    result(4, colI) = _010 + _111;
                    result(5, colI) = _210 + _311;
                }
            }
        }
        static void dF_div_dx_mult(const Eigen::Matrix2d& right,
                                   const Eigen::Matrix2d& A,
                                   Eigen::Matrix<double, 6, 1>& result);
        
        static void sampleSegment(const Eigen::RowVectorXd& vs,
                                  const Eigen::RowVectorXd& ve,
                                  double spacing,
                                  Eigen::MatrixXd& inBetween);
    };
    
}

#endif /* IglUtils_hpp */
