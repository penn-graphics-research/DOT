//
//  CHOLMODSolver.hpp
//  DOT
//
//  Created by Minchen Li on 6/22/18.
//

#ifndef CHOLMODSolver_hpp
#define CHOLMODSolver_hpp

#include "LinSysSolver.hpp"

#include "cholmod.h"

#include <Eigen/Eigen>

#include <vector>
#include <set>

namespace DOT {
    
    template <typename vectorTypeI, typename vectorTypeS>
    class CHOLMODSolver : public LinSysSolver<vectorTypeI, vectorTypeS>
    {
        typedef LinSysSolver<vectorTypeI, vectorTypeS> Base;
        
    protected:
        cholmod_common cm;
        cholmod_sparse *A;
        cholmod_factor *L;
        cholmod_dense *b, *b_threadSafe[DIM], *solution;
        cholmod_dense *x_cd, *y_cd; // for multiply
        
        void *Ai, *Ap, *Ax, *bx, *b_tsx[DIM], *solutionx, *x_cdx, *y_cdx;
        
    public:
        CHOLMODSolver(void);
        ~CHOLMODSolver(void);
        
        void set_type(int threadAmt, int _mtype, bool is_upper_half = false);
        
        void set_pattern(const std::vector<std::set<int>>& vNeighbor,
                         const std::set<int>& fixedVert);
        void set_pattern(const Eigen::SparseMatrix<double>& mtr); //NOTE: mtr must be SPD
        
        void update_a(const Eigen::SparseMatrix<double>& mtr);
        
        void analyze_pattern(void);
        
        bool factorize(void);
        
        void solve(Eigen::VectorXd &rhs,
                   Eigen::VectorXd &result);
        void solve_threadSafe(Eigen::VectorXd &rhs,
                              Eigen::VectorXd &result,
                              int dimI);
        
        virtual void multiply(const Eigen::VectorXd& x,
                              Eigen::VectorXd& Ax);
        
        virtual void outputFactorization(const std::string& filePath);
    };
    
}

#endif /* CHOLMODSolver_hpp */
