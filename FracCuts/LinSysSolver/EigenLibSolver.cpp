//
//  EigenLibSolver.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/30/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "EigenLibSolver.hpp"

namespace FracCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_type(int threadAmt, int _mtype, bool is_upper_half)
    {}
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_pattern(const vectorTypeI &II,
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
        //TODO: directly save into coefMtr
        coefMtr.resize(numRows, numRows);
        coefMtr.reserve(ja.size());
        ia -= Eigen::VectorXi::Ones(ia.size()); ja -= Eigen::VectorXi::Ones(ja.size());
        memcpy(coefMtr.innerIndexPtr(), ja.data(), ja.size() * sizeof(ja[0]));
        memcpy(coefMtr.outerIndexPtr(), ia.data(), ia.size() * sizeof(ia[0]));
        
        update_a(II, JJ, SS);
    }
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_pattern(const Eigen::SparseMatrix<double>& mtr) //NOTE: mtr must be SPD
    {
        coefMtr = mtr;
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::update_a(const vectorTypeI &II,
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
        //TODO: directly save into coefMtr
        memcpy(coefMtr.valuePtr(), a.data(), a.size() * sizeof(a[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::analyze_pattern(void)
    {
        simplicialLDLT.analyzePattern(coefMtr);
        assert(simplicialLDLT.info() == Eigen::Success);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    bool EigenLibSolver<vectorTypeI, vectorTypeS>::factorize(void)
    {
        simplicialLDLT.factorize(coefMtr);
        bool succeeded = (simplicialLDLT.info() == Eigen::Success);
        assert(succeeded);
        return succeeded;
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::solve(Eigen::VectorXd &rhs,
                                                         Eigen::VectorXd &result)
    {
        result = simplicialLDLT.solve(rhs);
        assert(simplicialLDLT.info() == Eigen::Success);
    }
    
    template class EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>;
    
}
