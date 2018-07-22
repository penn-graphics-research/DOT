//
//  LinSysSolver.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/30/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef LinSysSolver_hpp
#define LinSysSolver_hpp

#include <Eigen/Eigen>

#include <set>

namespace FracCuts {
    
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
        
        virtual void set_pattern(const vectorTypeI &II,
                                 const vectorTypeI &JJ,
                                 const vectorTypeS &SS,
                                 const std::vector<std::set<int>>& vNeighbor,
                                 const std::set<int>& fixedVert)
        {
            assert(II.size() == JJ.size());
            assert(II.size() == SS.size());
            
            numRows = static_cast<int>(vNeighbor.size()) * 2;
            ia.resize(vNeighbor.size() * 2 + 1);
            ia[0] = 1; // 1 + nnz above row i
            ja.resize(0); // colI of each element
            IJ2aI.resize(0); // map from matrix index to ja index
            IJ2aI.resize(vNeighbor.size() * 2);
            for(int rowI = 0; rowI < vNeighbor.size(); rowI++) {
                if(fixedVert.find(rowI) == fixedVert.end()) {
                    int oldSize_ja = static_cast<int>(ja.size());
                    IJ2aI[rowI * 2][rowI * 2] = oldSize_ja;
                    IJ2aI[rowI * 2][rowI * 2 + 1] = oldSize_ja + 1;
                    ja.conservativeResize(ja.size() + 2);
                    ja.bottomRows(2) << rowI * 2 + 1, rowI * 2 + 2;
                    
                    int nnz_rowI = 1;
                    for(const auto& colI : vNeighbor[rowI]) {
                        if(fixedVert.find(colI) == fixedVert.end()) {
                            if(colI > rowI) {
                                // only the lower-left part
                                // colI > rowI means upper-right, but we are preparing CSR here
                                // in a row-major manner and CHOLMOD is actually column-major
                                IJ2aI[rowI * 2][colI * 2] = static_cast<int>(ja.size());
                                IJ2aI[rowI * 2][colI * 2 + 1] = static_cast<int>(ja.size()) + 1;
                                ja.conservativeResize(ja.size() + 2);
                                ja.bottomRows(2) << colI * 2 + 1, colI * 2 + 2;
                                nnz_rowI++;
                            }
                        }
                    }
                    
                    // another row for y,
                    // excluding the left-bottom entry on the diagonal band
                    IJ2aI[rowI * 2 + 1] = IJ2aI[rowI * 2];
                    for(auto& IJ2aI_newRow : IJ2aI[rowI * 2 + 1]) {
                        IJ2aI_newRow.second += nnz_rowI * 2 - 1;
                    }
                    ja.conservativeResize(ja.size() + nnz_rowI * 2 - 1);
                    ja.bottomRows(nnz_rowI * 2 - 1) = ja.block(oldSize_ja + 1, 0, nnz_rowI * 2 - 1, 1);
                    
                    ia[rowI * 2 + 1] = ia[rowI * 2] + nnz_rowI * 2;
                    ia[rowI * 2 + 2] = ia[rowI * 2 + 1] + nnz_rowI * 2 - 1;
                }
                else {
                    int oldSize_ja = static_cast<int>(ja.size());
                    IJ2aI[rowI * 2][rowI * 2] = oldSize_ja;
                    IJ2aI[rowI * 2 + 1][rowI * 2 + 1] = oldSize_ja + 1;
                    ja.conservativeResize(oldSize_ja + 2);
                    ja.bottomRows(2) << rowI * 2 + 1, rowI * 2 + 2;
                    ia[rowI * 2 + 1] = ia[rowI * 2] + 1;
                    ia[rowI * 2 + 2] = ia[rowI * 2 + 1] + 1;
                }
            }
        }
        virtual void set_pattern(const Eigen::SparseMatrix<double>& mtr) = 0; //NOTE: mtr must be SPD
        
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
        
        virtual void analyze_pattern(void) = 0;
        
        virtual bool factorize(void) = 0;
        
        virtual void solve(Eigen::VectorXd &rhs,
                           Eigen::VectorXd &result) = 0;
        
    public:
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
    };
    
}

#endif /* LinSysSolver_hpp */
