//
//  LBFGSTimeStepper.hpp
//  DOT
//
//  Created by Minchen Li on 12/14/18.
//

#ifndef LBFGSTimeStepper_hpp
#define LBFGSTimeStepper_hpp

#include "Optimizer.hpp"
#include "METIS.hpp"

#include <Eigen/SparseCholesky>
#include <Eigen/src/IterativeLinearSolvers/IncompleteCholesky.h>

#include <deque>

namespace DOT {
    
    enum D0Type {
        D0T_PD,
        D0T_RH,
        D0T_H,
        D0T_HI,
        D0T_JH
    };
    
    template<int dim>
    class LBFGSTimeStepper : public Optimizer<dim>
    {
        typedef Optimizer<dim> Base;
        
    protected: // data
        D0Type m_D0Type;
        int historySize;
        std::deque<Eigen::VectorXd> dx, dg; // s and t
        std::deque<double> dgTdx; // tTs
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *D0LinSysSolver;
        std::vector<Eigen::Matrix<double, dim, dim + 1>> D_array;
        Eigen::SparseMatrix<double> W, D;
        
        std::vector<Eigen::VectorXi> nodeLists;
        std::vector<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>*> solver; //TODO: use LinSysSolver
        std::vector<idx_t> nodePart;
        
        Eigen::SparseMatrix<double> sysMtrForIC;
        Eigen::IncompleteCholesky<double> ICSolver;
        
    public: // constructor and destructor
        LBFGSTimeStepper(const Mesh<dim>& p_data0,
                         const std::vector<Energy<dim>*>& p_energyTerms,
                         const std::vector<double>& p_energyParams,
                         D0Type p_D0Type,
                         bool p_mute = false,
                         const Config& animConfig = Config());
        ~LBFGSTimeStepper(void);
        
    public: // API
        virtual void precompute(void);
        virtual void updatePrecondMtrAndFactorize(void);
        
        virtual void getFaceFieldForVis(Eigen::VectorXd& field);
        
    protected: // overwritten helper
        virtual bool fullyImplicit(void);
        virtual bool solve_oneStep(void);
        
    protected: // own helper
    };
    
}

#endif /* LBFGSTimeStepper_hpp */
