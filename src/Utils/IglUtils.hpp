//
//  IglUtils.hpp
//  DOT
//
//  Created by Minchen Li on 8/30/17.
//

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include "Mesh.hpp"

#ifdef USE_CLOSEDFORMSVD2D
#include "ClosedFormSVD2d.hpp"
#else
#include "AutoFlipSVD.hpp"
#endif

#include "LinSysSolver.hpp"

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>

namespace DOT {
    
    // a static class implementing basic geometry processing operations that are not provided in libIgl
    class IglUtils {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
        static void mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3]);
        static void computeDeformationGradient(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& F);
        
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
        template<int dim>
        static void addBlockToMatrix(const Eigen::Matrix<double, dim, dim * (dim + 1)>& block,
                                     const Eigen::Matrix<int, 1, dim + 1>& index, int rowIndI,
                                     Eigen::MatrixXd& hessian)
        {
            int rowStart = index[rowIndI] * dim;
            if(rowStart < 0) {
                rowStart = -rowStart - dim;
                hessian.diagonal().segment(rowStart, dim).setOnes();
                return;
            }
            
            if(index[0] >= 0) {
                int _dimIndex0 = index[0] * dim;
                hessian.block<dim, dim>(rowStart, _dimIndex0) += block.block(0, 0, dim, dim);
            }
            
            if(index[1] >= 0) {
                int _dimIndex1 = index[1] * dim;
                hessian.block<dim, dim>(rowStart, _dimIndex1) += block.block(0, dim, dim, dim);
            }
            
            if(index[2] >= 0) {
                int _2dim = 2 * dim;
                int _dimIndex2 = index[2] * dim;
                hessian.block<dim, dim>(rowStart, _dimIndex2) += block.block(0, _2dim, dim, dim);
            }
            
            if(dim == 3) {
                if(index[3] >= 0) {
                    int _3dim = 3 * dim;
                    int _dimIndex3 = index[3] * dim;
                    hessian.block<dim, dim>(rowStart, _dimIndex3) += block.block(0, _3dim, dim, dim);
                }
            }
        }
        template<int dim>
        static void addBlockToMatrix(const Eigen::Matrix<double, dim, dim * (dim + 1)>& block,
                                     const Eigen::Matrix<int, 1, dim + 1>& index, int rowIndI,
                                     LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver)
        {
            int rowStart = index[rowIndI] * dim;
            if(rowStart < 0) {
                rowStart = -rowStart - dim;
                linSysSolver->setCoeff(rowStart, rowStart, 1.0);
                linSysSolver->setCoeff(rowStart + 1, rowStart + 1, 1.0);
                if(dim == 3) {
                    linSysSolver->setCoeff(rowStart + 2, rowStart + 2, 1.0);
                }
                return;
            }
            
            if(index[0] >= 0) {
                int _dimIndex0 = index[0] * dim;
                linSysSolver->addCoeff(rowStart, _dimIndex0, block(0, 0));
                linSysSolver->addCoeff(rowStart, _dimIndex0 + 1, block(0, 1));
                linSysSolver->addCoeff(rowStart + 1, _dimIndex0, block(1, 0));
                linSysSolver->addCoeff(rowStart + 1, _dimIndex0 + 1, block(1, 1));
                if(dim == 3) {
                    linSysSolver->addCoeff(rowStart, _dimIndex0 + 2, block(0, 2));
                    linSysSolver->addCoeff(rowStart + 1, _dimIndex0 + 2, block(1, 2));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex0, block(2, 0));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex0 + 1, block(2, 1));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex0 + 2, block(2, 2));
                }
            }
            
            if(index[1] >= 0) {
                int _dimIndex1 = index[1] * dim;
                linSysSolver->addCoeff(rowStart, _dimIndex1, block(0, dim));
                linSysSolver->addCoeff(rowStart, _dimIndex1 + 1, block(0, dim + 1));
                linSysSolver->addCoeff(rowStart + 1, _dimIndex1, block(1, dim));
                linSysSolver->addCoeff(rowStart + 1, _dimIndex1 + 1, block(1, dim + 1));
                if(dim == 3) {
                    linSysSolver->addCoeff(rowStart, _dimIndex1 + 2, block(0, dim + 2));
                    linSysSolver->addCoeff(rowStart + 1, _dimIndex1 + 2, block(1, dim + 2));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex1, block(2, dim));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex1 + 1, block(2, dim + 1));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex1 + 2, block(2, dim + 2));
                }
            }
            
            if(index[2] >= 0) {
                int _2dim = 2 * dim;
                int _dimIndex2 = index[2] * dim;
                linSysSolver->addCoeff(rowStart, _dimIndex2, block(0, _2dim));
                linSysSolver->addCoeff(rowStart, _dimIndex2 + 1, block(0, _2dim + 1));
                linSysSolver->addCoeff(rowStart + 1, _dimIndex2, block(1, _2dim));
                linSysSolver->addCoeff(rowStart + 1, _dimIndex2 + 1, block(1, _2dim + 1));
                if(dim == 3) {
                    linSysSolver->addCoeff(rowStart, _dimIndex2 + 2, block(0, _2dim + 2));
                    linSysSolver->addCoeff(rowStart + 1, _dimIndex2 + 2, block(1, _2dim + 2));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex2, block(2, _2dim));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex2 + 1, block(2, _2dim + 1));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex2 + 2, block(2, _2dim + 2));
                }
            }
            
            if(dim == 3) {
                if(index[3] >= 0) {
                    int _3dim = 3 * dim;
                    int _dimIndex3 = index[3] * dim;
                    linSysSolver->addCoeff(rowStart, _dimIndex3, block(0, _3dim));
                    linSysSolver->addCoeff(rowStart, _dimIndex3 + 1, block(0, _3dim + 1));
                    linSysSolver->addCoeff(rowStart, _dimIndex3 + 2, block(0, _3dim + 2));
                    linSysSolver->addCoeff(rowStart + 1, _dimIndex3, block(1, _3dim));
                    linSysSolver->addCoeff(rowStart + 1, _dimIndex3 + 1, block(1, _3dim + 1));
                    linSysSolver->addCoeff(rowStart + 1, _dimIndex3 + 2, block(1, _3dim + 2));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex3, block(2, _3dim));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex3 + 1, block(2, _3dim + 1));
                    linSysSolver->addCoeff(rowStart + 2, _dimIndex3 + 2, block(2, _3dim + 2));
                }
            }
        }
        template<int dim>
        static void addIdBlockToMatrixDiag(const Eigen::VectorXi& index,
                                           LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver)
        {
            for(int indI = 0; indI < index.size(); indI++) {
                int rowStart = index[indI] * dim;
                assert(rowStart >= 0);
                linSysSolver->addCoeff(rowStart, rowStart, 1.0);
                linSysSolver->addCoeff(rowStart + 1, rowStart + 1, 1.0);
                if(dim == 3) {
                    linSysSolver->addCoeff(rowStart + 2, rowStart + 2, 1.0);
                }
            }
        }
        
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
        template<typename Scalar, int size>
        static void flipDet_SVD(Eigen::Matrix<Scalar, size, size>& mtr) {
            Eigen::JacobiSVD<Eigen::Matrix<Scalar, size, size>> svd(mtr, Eigen::ComputeFullU | Eigen::ComputeFullV);

            Eigen::Matrix<Scalar, size, size> U = svd.matrixU(), V = svd.matrixV();
            if(U.determinant() < 0) {
                U.col(U.cols() - 1) *= -1.0;
            }
            if(V.determinant() < 0) {
                V.col(V.cols() - 1) *= -1.0;
            }
            mtr = U * Eigen::DiagonalMatrix<Scalar, size>(svd.singularValues()) * V.transpose();
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
        static void writeSparseMatrixToFile(const std::string& filePath,
                                            LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                            bool MATLAB = false);
        static void writeDenseMatrixToFile(const std::string& filePath,
                                           const Eigen::MatrixXd& matrix,
                                           bool MATLAB = false);
        static void loadSparseMatrixFromFile(const std::string& filePath,
                                             Eigen::SparseMatrix<double>& mtr);
        
        static void sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr,
                                          Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V);
        static void sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& V);
        
        static void writeVectorToFile(const std::string& filePath,
                                      const Eigen::VectorXd& vec);
        static void readVectorFromFile(const std::string& filePath,
                                       Eigen::VectorXd& vec);
        
        static const std::string rtos(double real);
        
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
        
        static void findSurfaceTris(const Eigen::MatrixXi& TT, Eigen::MatrixXi& F);
        static void buildSTri2Tet(const Eigen::MatrixXi& F, const Eigen::MatrixXi& SF,
                                  std::vector<int>& sTri2Tet);
        
        static void saveTetMesh(const std::string& filePath,
                                const Eigen::MatrixXd& TV, const Eigen::MatrixXi& TT,
                                const Eigen::MatrixXi& F = Eigen::MatrixXi(),
                                bool findSurface = true);
        static bool readTetMesh(const std::string& filePath,
                                Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                                Eigen::MatrixXi& F, bool findSurface = true);
        static void readNodeEle(const std::string& filePath,
                                Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                                Eigen::MatrixXi& F);
        
        static void smoothVertField(const Mesh<DIM>& mesh, Eigen::VectorXd& field);
        
        static void compute_dF_div_dx(const Eigen::Matrix<double, DIM, DIM>& A,
                                      Eigen::Matrix<double, DIM * (DIM + 1), DIM * DIM>& dF_div_dx);
        template<int colSize>
        static void dF_div_dx_mult(const Eigen::Matrix<double, DIM * DIM, colSize>& right,
                                   const Eigen::Matrix<double, DIM, DIM>& A,
                                   Eigen::Matrix<double, DIM * (DIM + 1), colSize>& result,
                                   bool symmetric)
        {
            if(colSize == Eigen::Dynamic) {
                assert(right.cols() > 0);
            }
            else {
                assert(colSize > 0);
            }
#if(DIM == 2)
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
                result(2, 0) = result(0, 2) = _0000 + _1001;
                result(3, 0) = result(0, 3) = _2000 + _3001;
                result(4, 0) = result(0, 4) = _0010 + _1011;
                result(5, 0) = result(0, 5) = _2010 + _3011;
                result(0, 0) = -result(2, 0) - result(4, 0);
                result(1, 0) = result(0, 1) = -result(3, 0) - result(5, 0);
                // colI = 1;
                const double _2100 = right(2, 1) * A(0, 0);
                const double _2110 = right(2, 1) * A(1, 0);
                const double _3101 = right(3, 1) * A(0, 1);
                const double _3111 = right(3, 1) * A(1, 1);
                result(2, 1) = result(1, 2) = right(0, 1) * A(0, 0) + right(1, 1) * A(0, 1);
                result(3, 1) = result(1, 3) = _2100 + _3101;
                result(4, 1) = result(1, 4) = right(0, 1) * A(1, 0) + right(1, 1) * A(1, 1);
                result(5, 1) = result(1, 5) = _2110 + _3111;
                result(1, 1) = -result(3, 1) - result(5, 1);
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
                    
                    result(2, colI) = _000 + _101;
                    result(3, colI) = _200 + _301;
                    result(4, colI) = _010 + _111;
                    result(5, colI) = _210 + _311;
                    result(0, colI) = -result(2, colI) - result(4, colI);
                    result(1, colI) = -result(3, colI) - result(5, colI);
                }
            }
#else
            //TODO: use symmetric
            for(int colI = 0; colI < right.cols(); colI++) {
                result(3, colI) = (A.row(0) * right.block(0, colI, DIM, 1))[0];
                result(4, colI) = (A.row(0) * right.block(DIM, colI, DIM, 1))[0];
                result(5, colI) = (A.row(0) * right.block(DIM * 2, colI, DIM, 1))[0];
                result(6, colI) = (A.row(1) * right.block(0, colI, DIM, 1))[0];
                result(7, colI) = (A.row(1) * right.block(DIM, colI, DIM, 1))[0];
                result(8, colI) = (A.row(1) * right.block(DIM * 2, colI, DIM, 1))[0];
                result(9, colI) = (A.row(2) * right.block(0, colI, DIM, 1))[0];
                result(10, colI) = (A.row(2) * right.block(DIM, colI, DIM, 1))[0];
                result(11, colI) = (A.row(2) * right.block(DIM * 2, colI, DIM, 1))[0];
                result(0, colI) = - result(3, colI) - result(6, colI) - result(9, colI);
                result(1, colI) = - result(4, colI) - result(7, colI) - result(10, colI);
                result(2, colI) = - result(5, colI) - result(8, colI) - result(11, colI);
            }
#endif
        }
        static void dF_div_dx_mult(const Eigen::Matrix<double, DIM, DIM>& right,
                                   const Eigen::Matrix<double, DIM, DIM>& A,
                                   Eigen::Matrix<double, DIM * (DIM + 1), 1>& result);
        template<int dim>
        static void computeCofactorMtr(const Eigen::Matrix<double, dim, dim>& F,
                                       Eigen::Matrix<double, dim, dim>& A)
        {
            switch(dim) {
                case 2:
                    A(0, 0) = F(1, 1);
                    A(0, 1) = -F(1, 0);
                    A(1, 0) = -F(0, 1);
                    A(1, 1) = F(0, 0);
                    break;
                    
                case 3:
                    A(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
                    A(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
                    A(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
                    A(1, 0) = F(0, 2) * F(2, 1) - F(0, 1) * F(2, 2);
                    A(1, 1) = F(0, 0) * F(2, 2) - F(0, 2) * F(2, 0);
                    A(1, 2) = F(0, 1) * F(2, 0) - F(0, 0) * F(2, 1);
                    A(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
                    A(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
                    A(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
                    break;
                    
                default:
                    assert(0 && "dim not 2 or 3");
                    break;
            }
        }
        
        static void extractRotation(const Eigen::Matrix3d &A,
                                    Eigen::Quaterniond &q,
                                    const unsigned int maxIter);
        
        static void sampleSegment(const Eigen::RowVectorXd& vs,
                                  const Eigen::RowVectorXd& ve,
                                  double spacing,
                                  Eigen::MatrixXd& inBetween);
        
        static void findBorderVerts(const Eigen::MatrixXd& V,
                                    std::vector<std::vector<int>>& borderVerts,
                                    double ratio);

        static void computeSVD_SIMD(std::vector<Eigen::Matrix3d>& testF,
                                    std::vector<Eigen::Matrix3d>& U, std::vector<Eigen::Vector3d>& Sigma, std::vector<Eigen::Matrix3d>& V);

        static void matrixProduct(const std::vector<Eigen::Matrix3d>& left,
                                  const std::vector<Eigen::Matrix3d>& right,
                                  std::vector<Eigen::Matrix3d>& result);

        static void matrixVectorMatrixTProduct(const std::vector<Eigen::Matrix3d>& left,
                                                const std::vector<Eigen::Vector3d>& vec,
                                                const std::vector<Eigen::Matrix3d>& right,
                                                std::vector<Eigen::Matrix3d>& result);
    };

}

#endif /* IglUtils_hpp */
