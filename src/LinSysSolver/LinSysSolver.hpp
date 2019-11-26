//
//  LinSysSolver.hpp
//  DOT
//
//  Created by Minchen Li on 6/30/18.
//

#ifndef LinSysSolver_hpp
#define LinSysSolver_hpp

#include "Types.hpp"

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <set>
#include <map>
#include <iostream>

namespace DOT {
    
    template <typename vectorTypeI, typename vectorTypeS>
    class LinSysSolver
    {
    protected:
        int numRows;
        Eigen::VectorXi ia, ja;
        std::vector<std::map<int, int>> IJ2aI;
        Eigen::VectorXd a;
        
    public:
        virtual ~LinSysSolver(void) {};
        
    public:
        virtual void set_type(int threadAmt, int _mtype, bool is_upper_half = false) = 0;
        
        virtual void set_pattern(const std::vector<std::set<int>>& vNeighbor,
                                 const std::set<int>& fixedVert)
        {
            numRows = static_cast<int>(vNeighbor.size()) * DIM;
            ia.resize(vNeighbor.size() * DIM + 1);
            ia[0] = 1; // 1 + nnz above row i
            ja.resize(0); // colI of each element
            IJ2aI.resize(0); // map from matrix index to ja index
            IJ2aI.resize(vNeighbor.size() * DIM);
            for(int rowI = 0; rowI < vNeighbor.size(); rowI++) {
                if(fixedVert.find(rowI) == fixedVert.end()) {
                    int oldSize_ja = static_cast<int>(ja.size());
                    IJ2aI[rowI * DIM][rowI * DIM] = oldSize_ja;
                    IJ2aI[rowI * DIM][rowI * DIM + 1] = oldSize_ja + 1;
                    if(DIM == 3) {
                        IJ2aI[rowI * DIM][rowI * DIM + 2] = oldSize_ja + 2;
                    }
                    ja.conservativeResize(oldSize_ja + DIM);
                    ja[oldSize_ja] = rowI * DIM + 1;
                    ja[oldSize_ja + 1] = rowI * DIM + 2;
                    if(DIM == 3) {
                        ja[oldSize_ja + 2] = rowI * DIM + 3;
                    }
                    
                    int nnz_rowI = 1;
                    for(const auto& colI : vNeighbor[rowI]) {
                        if(fixedVert.find(colI) == fixedVert.end()) {
                            if(colI > rowI) {
                                // only the lower-left part
                                // colI > rowI means upper-right, but we are preparing CSR here
                                // in a row-major manner and CHOLMOD is actually column-major
                                int oldSize_ja_temp = static_cast<int>(ja.size());
                                IJ2aI[rowI * DIM][colI * DIM] = oldSize_ja_temp;
                                IJ2aI[rowI * DIM][colI * DIM + 1] = oldSize_ja_temp + 1;
                                if(DIM == 3) {
                                    IJ2aI[rowI * DIM][colI * DIM + 2] = oldSize_ja_temp + 2;
                                }
                                ja.conservativeResize(oldSize_ja_temp + DIM);
                                ja[oldSize_ja_temp] = colI * DIM + 1;
                                ja[oldSize_ja_temp + 1] = colI * DIM + 2;
                                if(DIM == 3) {
                                    ja[oldSize_ja_temp + 2] = colI * DIM + 3;
                                }
                                nnz_rowI++;
                            }
                        }
                    }
                    
                    // another row for y,
                    // excluding the left-bottom entry on the diagonal band
                    IJ2aI[rowI * DIM + 1] = IJ2aI[rowI * DIM];
                    for(auto& IJ2aI_newRow : IJ2aI[rowI * DIM + 1]) {
                        IJ2aI_newRow.second += nnz_rowI * DIM - 1;
                    }
                    ja.conservativeResize(ja.size() + nnz_rowI * DIM - 1);
                    ja.bottomRows(nnz_rowI * DIM - 1) = ja.block(oldSize_ja + 1, 0, nnz_rowI * DIM - 1, 1);
                    
                    if(DIM == 3) {
                        // third row for z
                        IJ2aI[rowI * DIM + 2] = IJ2aI[rowI * DIM + 1];
                        for(auto& IJ2aI_newRow : IJ2aI[rowI * DIM + 2]) {
                            IJ2aI_newRow.second += nnz_rowI * DIM - 2;
                        }
                        ja.conservativeResize(ja.size() + nnz_rowI * DIM - 2);
                        ja.bottomRows(nnz_rowI * DIM - 2) = ja.block(oldSize_ja + 2, 0, nnz_rowI * DIM - 2, 1);
                        
                        IJ2aI[rowI * DIM + 2].erase(rowI * DIM);
                        IJ2aI[rowI * DIM + 2].erase(rowI * DIM + 1);
                    }
                    IJ2aI[rowI * DIM + 1].erase(rowI * DIM);
                    
                    ia[rowI * DIM + 1] = ia[rowI * DIM] + nnz_rowI * DIM;
                    ia[rowI * DIM + 2] = ia[rowI * DIM + 1] + nnz_rowI * DIM - 1;
                    if(DIM == 3) {
                        ia[rowI * DIM + 3] = ia[rowI * DIM + 2] + nnz_rowI * DIM - 2;
                    }
                }
                else {
                    int oldSize_ja = static_cast<int>(ja.size());
                    IJ2aI[rowI * DIM][rowI * DIM] = oldSize_ja;
                    IJ2aI[rowI * DIM + 1][rowI * DIM + 1] = oldSize_ja + 1;
                    if(DIM == 3) {
                        IJ2aI[rowI * DIM + 2][rowI * DIM + 2] = oldSize_ja + 2;
                    }
                    ja.conservativeResize(oldSize_ja + DIM);
                    ja[oldSize_ja] = rowI * DIM + 1;
                    ja[oldSize_ja + 1] = rowI * DIM + 2;
                    if(DIM == 3) {
                        ja[oldSize_ja + 2] = rowI * DIM + 3;
                    }
                    ia[rowI * DIM + 1] = ia[rowI * DIM] + 1;
                    ia[rowI * DIM + 2] = ia[rowI * DIM + 1] + 1;
                    if(DIM == 3) {
                        ia[rowI * DIM + 3] = ia[rowI * DIM + 2] + 1;
                    }
                }
            }
            a.resize(ja.size());
        }
        virtual void set_pattern(const Eigen::SparseMatrix<double>& mtr) {
            //NOTE: mtr must be SPD
            
            numRows = static_cast<int>(mtr.rows());
            
            ja.conservativeResize(mtr.nonZeros());
            memcpy(ja.data(), mtr.innerIndexPtr(),
                   mtr.nonZeros() * sizeof(mtr.innerIndexPtr()[0]));
            
            ia.conservativeResize(numRows + 1);
            memcpy(ia.data(), mtr.outerIndexPtr(),
                   (numRows + 1) * sizeof(mtr.outerIndexPtr()[0]));
            
            a.conservativeResize(mtr.nonZeros());
            memcpy(a.data(), mtr.valuePtr(),
                   mtr.nonZeros() * sizeof(mtr.valuePtr()[0]));
        }
        
        virtual void update_a(const vectorTypeI &II,
                              const vectorTypeI &JJ,
                              const vectorTypeS &SS)
        {
            //TODO: faster O(1) indices!!
            
            assert(II.size() == JJ.size());
            assert(II.size() == SS.size());
            
            a.setZero(ja.size());
            for(int tripletI = 0; tripletI < II.size(); tripletI++) {
                int i = II[tripletI], j = JJ[tripletI];
                if(i <= j) {
                    //        if((i <= j) && (i != 2) && (j != 2)) {
                    assert(i < IJ2aI.size());
                    const auto finder = IJ2aI[i].find(j);
                    assert(finder != IJ2aI[i].end());
                    a[finder->second] += SS[tripletI];
                }
            }
            //    a[IJ2aI[2].find(2)->second] = 1.0;
        }
        virtual void update_a(const Eigen::SparseMatrix<double>& mtr)
        {
            assert(numRows == static_cast<int>(mtr.rows()));
            assert(a.size() == mtr.nonZeros());
            memcpy(a.data(), mtr.valuePtr(), mtr.nonZeros() * sizeof(mtr.valuePtr()[0]));
        }
        
        virtual void analyze_pattern(void) = 0;
        
        virtual bool factorize(void) = 0;
        
        virtual void solve(Eigen::VectorXd &rhs,
                           Eigen::VectorXd &result) = 0;
        virtual void solve_threadSafe(Eigen::VectorXd &rhs,
                                      Eigen::VectorXd &result,
                                      int dimI)
        {
            assert(0 && "please implement in subclass!");
        }
        
        virtual void multiply(const Eigen::VectorXd& x,
                              Eigen::VectorXd& Ax)
        {
            assert(x.size() == numRows);
            assert(IJ2aI.size() == numRows);
            
            Ax.setZero(numRows);
            for(int rowI = 0; rowI < numRows; ++rowI) {
                for(const auto& colI : IJ2aI[rowI]) {
                    Ax[rowI] += a[colI.second] * x[colI.first];
                    if(rowI != colI.first) {
                        Ax[colI.first] += a[colI.second] * x[rowI];
                    }
                }
            }
        }
        
    public:
        virtual void outputFactorization(const std::string& filePath) {
            assert(0 && "please implement!");
        }
        virtual double coeffMtr(int rowI, int colI) const {
            if(rowI > colI) {
                // return only upper right part for symmetric matrix
                int temp = rowI;
                rowI = colI;
                colI = temp;
            }
            assert(rowI < IJ2aI.size());
            const auto finder = IJ2aI[rowI].find(colI);
            if(finder != IJ2aI[rowI].end()) {
                return a[finder->second];
            }
            else {
                return 0.0;
            }
        }
        virtual void getCoeffMtr(Eigen::SparseMatrix<double>& mtr) const {
            mtr.resize(numRows, numRows);
            mtr.setZero();
            mtr.reserve(a.size() * 2 - numRows);
            for(int rowI = 0; rowI < numRows; rowI++) {
                for(const auto& colIter : IJ2aI[rowI]) {
                    mtr.insert(rowI, colIter.first) = a[colIter.second];
                    if(rowI != colIter.first) {
                        mtr.insert(colIter.first, rowI) = a[colIter.second];
                    }
                }
            }
        }
        virtual void getCoeffMtr_lower(Eigen::SparseMatrix<double>& mtr) const {
            assert(numRows > 0);
            
            mtr.conservativeResize(numRows, numRows);
            mtr.reserve(a.size());
            
            memcpy(mtr.innerIndexPtr(), ja.data(), ja.size() * sizeof(ja[0]));
            memcpy(mtr.outerIndexPtr(), ia.data(), ia.size() * sizeof(ia[0]));
            memcpy(mtr.valuePtr(), a.data(), a.size() * sizeof(a[0]));
        }
        virtual void getTriplets(const Eigen::VectorXi& nodeList,
                                 std::vector<Eigen::Triplet<double>>& triplet) const
        {
            std::map<int, int> rowIMapper;
            for(int i = 0; i < nodeList.size(); ++i) {
                int startI = i * DIM;
                int startRowI = nodeList[i] * DIM;
                
                rowIMapper[startRowI] = startI;
                rowIMapper[startRowI + 1] = startI + 1;
                if(DIM == 3) {
                    rowIMapper[startRowI + 2] = startI + 2;
                }
            }
            
            triplet.resize(0);
            for(int rowI = 0; rowI < numRows; rowI++) {
                auto rowIFinder = rowIMapper.find(rowI);
                for(const auto& colIter : IJ2aI[rowI]) {
                    auto colIFinder = rowIMapper.find(colIter.first);
                    if(rowIFinder != rowIMapper.end() && colIFinder != rowIMapper.end()) {
                        triplet.emplace_back(rowIFinder->second, colIFinder->second, a[colIter.second]);
                        if(rowIFinder->second != colIFinder->second) {
                            triplet.emplace_back(colIFinder->second, rowIFinder->second, a[colIter.second]);
                        }
                    }
                }
            }
        }
        virtual void setCoeff(int rowI, int colI, double val) {
            //TODO: faster O(1) indices!!
            
            if(rowI <= colI) {
                assert(rowI < IJ2aI.size());
                const auto finder = IJ2aI[rowI].find(colI);
                assert(finder != IJ2aI[rowI].end());
                a[finder->second] = val;
            }
        }
        virtual void setCoeff(const LinSysSolver<vectorTypeI, vectorTypeS>* other,
                              double multiplier)
        {
            assert(numRows == other->numRows);
            assert(ja.size() == other->a.size());
            
            a = multiplier * other->a;
        }
        virtual void setZero(void) {
            a.setZero();
        }
        virtual void setUnit_row(int rowI) {
            assert(numRows == IJ2aI.size());
            assert(rowI < numRows);
            for(const auto& colIter : IJ2aI[rowI]) {
                a[colIter.second] = (colIter.first == rowI);
            }
        }
        virtual void setUnit_col(int colI, const std::set<int>& rowVIs) {
            assert(numRows == IJ2aI.size());
            assert(colI < numRows);
            for(const auto& rowVI : rowVIs) {
                for(int dimI = 0; dimI < DIM; ++dimI) {
                    int rowI = rowVI * DIM + dimI;
                    assert(rowI < numRows);
                    if(rowI <= colI) {
                        const auto finder = IJ2aI[rowI].find(colI);
                        if(finder != IJ2aI[rowI].end()) {
                            a[finder->second] = (rowI == colI);
                        }
                    }
                }
            }
        }
        virtual void setUnit_col_dim1(int colI, const std::set<int>& rowVIs) {
            assert(numRows == IJ2aI.size());
            assert(colI < numRows);
            for(const auto& rowI : rowVIs) {
                assert(rowI < numRows);
                if(rowI <= colI) {
                    const auto finder = IJ2aI[rowI].find(colI);
                    if(finder != IJ2aI[rowI].end()) {
                        a[finder->second] = (rowI == colI);
                    }
                }
            }
        }
        
        virtual void addCoeff(int rowI, int colI, double val) {
            //TODO: faster O(1) indices!!
            
            if(rowI <= colI) {
                assert(rowI < IJ2aI.size());
                const auto finder = IJ2aI[rowI].find(colI);
                assert(finder != IJ2aI[rowI].end());
                a[finder->second] += val;
            }
        }
        virtual void addCoeff(const LinSysSolver<vectorTypeI, vectorTypeS>* other,
                              double multiplier)
        {
            assert(numRows == other->numRows);
            assert(a.size() == other->a.size());
            
            a += multiplier * other->a;
        }
        virtual void addCoeff_globalToLocal(const LinSysSolver<vectorTypeI, vectorTypeS>* other,
                                            const std::vector<int>& localVIToGlobal,
                                            double multiplier)
        {
            assert(multiplier);
            assert(!localVIToGlobal.empty());
            assert(other->numRows >= localVIToGlobal.size() * DIM);
            assert(other->a.size() == other->ja.size());
            
            assert(numRows == localVIToGlobal.size() * DIM);
            assert(a.size() == ja.size());
            
            for(int rowVI = 0; rowVI < IJ2aI.size() / DIM; ++rowVI) {
                int globalRowIStart = localVIToGlobal[rowVI] * DIM;
                int rowIStart = rowVI * DIM;
                
                for(const auto& colIter : IJ2aI[rowIStart]) {
                    int globalColI = localVIToGlobal[colIter.first / DIM] * DIM + colIter.first % DIM;
                    if(globalColI >= globalRowIStart) {
                        auto globalEntryLoc = other->IJ2aI[globalRowIStart].find(globalColI);
                        assert(globalEntryLoc != other->IJ2aI[globalRowIStart].end());
                        a[colIter.second] += multiplier * other->a[globalEntryLoc->second];
                    }
                    else {
                        auto globalEntryLoc = other->IJ2aI[globalColI].find(globalRowIStart);
                        assert(globalEntryLoc != other->IJ2aI[globalColI].end());
                        a[colIter.second] += multiplier * other->a[globalEntryLoc->second];
                    }
                }
                
                for(const auto& colIter : IJ2aI[rowIStart + 1]) {
                    int globalColI = localVIToGlobal[colIter.first / DIM] * DIM + colIter.first % DIM;
                    if(globalColI >= globalRowIStart + 1) {
                        auto globalEntryLoc = other->IJ2aI[globalRowIStart + 1].find(globalColI);
                        assert(globalEntryLoc != other->IJ2aI[globalRowIStart + 1].end());
                        a[colIter.second] += multiplier * other->a[globalEntryLoc->second];
                    }
                    else {
                        auto globalEntryLoc = other->IJ2aI[globalColI].find(globalRowIStart + 1);
                        assert(globalEntryLoc != other->IJ2aI[globalColI].end());
                        a[colIter.second] += multiplier * other->a[globalEntryLoc->second];
                    }
                }
                
                if(DIM == 3) {
                    for(const auto& colIter : IJ2aI[rowIStart + 2]) {
                        int globalColI = localVIToGlobal[colIter.first / DIM] * DIM + colIter.first % DIM;
                        if(globalColI >= globalRowIStart + 2) {
                            auto globalEntryLoc = other->IJ2aI[globalRowIStart + 2].find(globalColI);
                            assert(globalEntryLoc != other->IJ2aI[globalRowIStart + 2].end());
                            a[colIter.second] += multiplier * other->a[globalEntryLoc->second];
                        }
                        else {
                            auto globalEntryLoc = other->IJ2aI[globalColI].find(globalRowIStart + 2);
                            assert(globalEntryLoc != other->IJ2aI[globalColI].end());
                            a[colIter.second] += multiplier * other->a[globalEntryLoc->second];
                        }
                    }
                }
            }
        }
        
        virtual int getNumRows(void) const {
            return numRows;
        }
        virtual int getNumNonzeros(void) const {
            return a.size();
        }
        virtual const std::vector<std::map<int, int>>& getIJ2aI(void) const {
            return IJ2aI;
        }
        virtual Eigen::VectorXi& get_ia(void) { return ia; }
        virtual Eigen::VectorXi& get_ja(void) { return ja; }
        virtual Eigen::VectorXd& get_a(void) { return a; }
        virtual const Eigen::VectorXd& get_a(void) const { return a; }
    };
    
}

#endif /* LinSysSolver_hpp */
