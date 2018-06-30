//
//  CHOLMODSolver.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef CHOLMODSolver_hpp
#define CHOLMODSolver_hpp

#include "cholmod.h"

#include <Eigen/Eigen>

#include <vector>
#include <set>

namespace FracCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    class CHOLMODSolver
    {
    protected:
        cholmod_common cm;
        cholmod_sparse *A;
        cholmod_factor *L;
        cholmod_dense *b;
        
        int numRows;
        Eigen::VectorXi ia, ja;
        std::vector<std::map<int, int>> IJ2aI;
        Eigen::VectorXd a;
        
    public:
        CHOLMODSolver(void);
        ~CHOLMODSolver(void);
        
        void set_type(int threadAmt, int _mtype, bool is_upper_half = false);
        
        void set_pattern(const vectorTypeI &II,
                         const vectorTypeI &JJ,
                         const vectorTypeS &SS,
                         const std::vector<std::set<int>>& vNeighbor,
                         const std::set<int>& fixedVert);
        void set_pattern(const Eigen::SparseMatrix<double>& mtr); //NOTE: mtr must be SPD
        
        void update_a(const vectorTypeI &II,
                      const vectorTypeI &JJ,
                      const vectorTypeS &SS);
        
        void analyze_pattern(void);
        
        bool factorize(void);
        
        void solve(Eigen::VectorXd &rhs,
                   Eigen::VectorXd &result);
    };
    
}

#endif /* CHOLMODSolver_hpp */
