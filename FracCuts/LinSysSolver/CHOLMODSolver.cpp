//
//  CHOLMODSolver.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "CHOLMODSolver.hpp"

#include <iostream>

namespace FracCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    CHOLMODSolver<vectorTypeI, vectorTypeS>::CHOLMODSolver(void)
    {
        cholmod_start(&cm);
        A = NULL;
        L = NULL;
        b = NULL;
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    CHOLMODSolver<vectorTypeI, vectorTypeS>::~CHOLMODSolver(void)
    {
        cholmod_free_sparse(&A, &cm);
        cholmod_free_factor(&L, &cm);
        cholmod_free_dense(&b, &cm);
        cholmod_finish(&cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::set_type(int threadAmt,
                                                           int _mtype,
                                                           bool is_upper_half)
    {
        //TODO: support more matrix types, currently only SPD
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::set_pattern(const vectorTypeI &II,
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
        //TODO: directly save into A
        if(!A) {
            A = cholmod_allocate_sparse(numRows, numRows, ja.size(), true, true, -1, CHOLMOD_REAL, &cm);
            // -1: upper right part will be ignored during computation
        }
        ia.array() -= 1; ja.array() -= 1; // CHOLMOD's index starts from 0
        memcpy(A->i, ja.data(), ja.size() * sizeof(ja[0]));
        memcpy(A->p, ia.data(), ia.size() * sizeof(ia[0]));
        
        update_a(II, JJ, SS);
    }
    template <typename vectorTypeI, typename vectorTypeS>
    void  CHOLMODSolver<vectorTypeI, vectorTypeS>::set_pattern(const Eigen::SparseMatrix<double>& mtr)
    {
        numRows = static_cast<int>(mtr.rows());
        if(!A) {
            A = cholmod_allocate_sparse(numRows, numRows, mtr.nonZeros(), true, true, -1, CHOLMOD_REAL, &cm);
            // -1: upper right part will be ignored during computation
        }
        memcpy(A->i, mtr.innerIndexPtr(), mtr.nonZeros() * sizeof(mtr.innerIndexPtr()[0]));
        memcpy(A->p, mtr.outerIndexPtr(), (numRows + 1) * sizeof(mtr.outerIndexPtr()[0]));
        memcpy(A->x, mtr.valuePtr(), mtr.nonZeros() * sizeof(mtr.valuePtr()[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::update_a(const vectorTypeI &II,
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
        //TODO: directly save into A
        memcpy(A->x, a.data(), a.size() * sizeof(a[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::analyze_pattern(void)
    {
        cholmod_free_factor(&L, &cm);
        L = cholmod_analyze(A, &cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    bool CHOLMODSolver<vectorTypeI, vectorTypeS>::factorize(void)
    {
        return !cholmod_factorize(A, L, &cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::solve(Eigen::VectorXd &rhs,
                                                        Eigen::VectorXd &result)
    {
        //TODO: directly point to rhs?
        if(!b) {
            b = cholmod_allocate_dense(numRows, 1, numRows, CHOLMOD_REAL, &cm);
        }
        memcpy(b->x, rhs.data(), rhs.size() * sizeof(rhs[0]));
        cholmod_dense *x;
        x = cholmod_solve(CHOLMOD_A, L, b, &cm);
        result.resize(rhs.size());
        memcpy(result.data(), x->x, result.size() * sizeof(result[0]));
        cholmod_free_dense(&x, &cm);
    }
    
    template class CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>;
    
}


