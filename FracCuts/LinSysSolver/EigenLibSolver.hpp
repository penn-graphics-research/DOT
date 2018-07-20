//
//  EigenLibSolver.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/30/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef EigenLibSolver_hpp
#define EigenLibSolver_hpp

#include "LinSysSolver.hpp"

#include <Eigen/Eigen>

#include <vector>
#include <set>

namespace FracCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    class EigenLibSolver : public LinSysSolver<vectorTypeI, vectorTypeS>
    {
    protected:
        int numRows;
        Eigen::VectorXi ia, ja;
        std::vector<std::map<int, int>> IJ2aI;
        Eigen::VectorXd a;
        
        Eigen::SparseMatrix<double> coefMtr;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> simplicialLDLT;
        
    public:
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

#endif /* EigenLibSolver_hpp */