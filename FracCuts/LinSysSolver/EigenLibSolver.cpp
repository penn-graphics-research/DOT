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
    {
        //TODO: support more matrix types, currently only SPD
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_pattern(const vectorTypeI &II,
                                                               const vectorTypeI &JJ,
                                                               const vectorTypeS &SS,
                                                               const std::vector<std::set<int>>& vNeighbor,
                                                               const std::set<int>& fixedVert)
    {
        Base::set_pattern(II, JJ, SS, vNeighbor, fixedVert);
        
        // directly save into mtr
        coefMtr.resize(Base::numRows, Base::numRows);
        coefMtr.reserve(Base::ja.size());
        Base::ia.array() -= 1.0;
        Base::ja.array() -= 1.0;
        memcpy(coefMtr.innerIndexPtr(), Base::ja.data(), Base::ja.size() * sizeof(Base::ja[0]));
        memcpy(coefMtr.outerIndexPtr(), Base::ia.data(), Base::ia.size() * sizeof(Base::ia[0]));
        
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
        Base::update_a(II, JJ, SS);
        
        //TODO: directly save into coefMtr
        memcpy(coefMtr.valuePtr(), Base::a.data(), Base::a.size() * sizeof(Base::a[0]));
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
