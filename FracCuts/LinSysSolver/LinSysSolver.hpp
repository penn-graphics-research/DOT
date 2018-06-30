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
    public:
        virtual ~LinSysSolver(void) {};
        
    public:
        virtual void set_type(int threadAmt, int _mtype, bool is_upper_half = false) = 0;
        
        virtual void set_pattern(const vectorTypeI &II,
                                 const vectorTypeI &JJ,
                                 const vectorTypeS &SS,
                                 const std::vector<std::set<int>>& vNeighbor,
                                 const std::set<int>& fixedVert) = 0;
        virtual void set_pattern(const Eigen::SparseMatrix<double>& mtr) = 0; //NOTE: mtr must be SPD
        
        virtual void update_a(const vectorTypeI &II,
                              const vectorTypeI &JJ,
                              const vectorTypeS &SS) = 0;
        
        virtual void analyze_pattern(void) = 0;
        
        virtual bool factorize(void) = 0;
        
        virtual void solve(Eigen::VectorXd &rhs,
                           Eigen::VectorXd &result) = 0;
    };
    
}

#endif /* LinSysSolver_hpp */
