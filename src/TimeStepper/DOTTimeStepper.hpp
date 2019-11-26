//
//  DOTTimeStepper.hpp
//  DOT
//
//  Created by Minchen Li on 12/17/18.
//

#ifndef DOTTimeStepper_hpp
#define DOTTimeStepper_hpp

#include "ADMMDDTimeStepper.hpp"

#include <deque>

namespace DOT {
    
    template<int dim>
    class DOTTimeStepper : public ADMMDDTimeStepper<dim>
    {
        typedef Optimizer<dim> Base;
        typedef ADMMDDTimeStepper<dim> Super;
        
    protected: // data
        int historySize;
        std::deque<Eigen::VectorXd> dx, dg; // s and t
        std::deque<double> dgTdx; // tTs
        std::deque<double> ERecord;

        std::vector<int> dup;
        
        std::vector<Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)>> elemHessians;
        std::vector<Eigen::Matrix<int, 1, dim + 1>> vInds;
        std::vector<Eigen::Matrix<double, dim * (dim + 1), 1>> elemGradient;
        Eigen::VectorXd elemEnergyVal;
        Eigen::Matrix<double, Eigen::Dynamic, 3> ETerms;
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* H_tr;
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> H_tr_sbd;
        double curEDec_batch, lastGradSqNorm;
        std::vector<double> gradSqNorms;
        
//        std::vector<std::vector<Eigen::Triplet<double>>> triplet_sbd;
//        std::vector<Eigen::SparseMatrix<double>> WTW_sbd;
//        std::vector<Eigen::Triplet<double>> triplet_z;
//        Eigen::SparseMatrix<double> WTW_z;
//        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_z;
        
    protected: // GS DD LBFGS
        std::vector<std::deque<Eigen::VectorXd>> dx_sbd, dg_sbd; // s and t
        std::vector<std::deque<double>> dgTdx_sbd; // tTs
        std::vector<std::vector<int>> elemListOv;
        std::vector<std::map<int, int>> globalElemI2LocalOv_sbd;
        
    public: // constructor and destructor
        DOTTimeStepper(const Mesh<dim>& p_data0,
                            const std::vector<Energy<dim>*>& p_energyTerms,
                            const std::vector<double>& p_energyParams,
                            bool p_mute = false,
                            const Config& animConfig = Config());
        ~DOTTimeStepper(void);
        
    public: // overwritten API
        virtual void precompute(void);
        virtual void updatePrecondMtrAndFactorize(void);
        
    protected: // overwritten helper
        virtual bool fullyImplicit(void);
        virtual bool solve_oneStep(void);
        virtual void computeEnergyVal(const Mesh<dim>& data, int redoSVD, double& energyVal);
        
    protected: // own helper
        bool solve_oneStep_GSDD(void);
        void computeHElemAndFillIn(LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver);
        void fillInDecomposedHessians(std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*>& linSysSolver_sbd);
        void computeDecomposedHessians(void);
        
        void updateHessianAndFactor(void);

        void computeGradient_extract(int sbdI, Eigen::VectorXd& grad);
    };
    
}

#endif /* DOTTimeStepper_hpp */
