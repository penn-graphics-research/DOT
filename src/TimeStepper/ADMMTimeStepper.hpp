//
//  ADMMTimeStepper.hpp
//  DOT
//
//  Created by Minchen Li on 6/28/18.
//

#ifndef ADMMTimeStepper_hpp
#define ADMMTimeStepper_hpp

#include "Optimizer.hpp"

#include <map>

namespace DOT {
    
    template<int dim>
    class ADMMTimeStepper : public Optimizer<dim>
    {
        typedef Optimizer<dim> Base;
        
    protected:
        Eigen::MatrixXd z, u, dz, du;
        std::vector<Eigen::Matrix<double, dim * dim, dim * dim>> GW; //TODO: change to use vector if only diagonal weights
        std::vector<Eigen::MatrixXd> D_array; // maps xi to zi
        
        Eigen::VectorXd rhs_xUpdate, M_mult_xHat, x_solved;
        Eigen::MatrixXd D_mult_x;
        Eigen::SparseMatrix<double> D, W;
        std::vector<double*> W_elemPtr;
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *globalLinSysSolver;
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> linSysSolverPtrs;
        std::vector<std::map<int, double>> offset_fixVerts; // for modifying the linSys to fix vertices
        
    public:
        ADMMTimeStepper(const Mesh<dim>& p_data0,
                        const std::vector<Energy<dim>*>& p_energyTerms,
                        const std::vector<double>& p_energyParams,
                        bool p_mute = false,
                        const Config& animConfig = Config());
        ~ADMMTimeStepper(void);
        
    public:
        virtual void precompute(void);
        
        virtual void updatePrecondMtrAndFactorize(void);
        
        virtual void getFaceFieldForVis(Eigen::VectorXd& field);
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void zuUpdate(void); // local solve
        void zuUpdate_SV(void);
        void checkRes(void);
        void xUpdate(void); // global solve
        
        void compute_Di_mult_xi(int elemI);
        void initWeights(void);
        void updateWeights(double multiplier);
        void initGlobalLinSysSolver(void);
        
        // local energy computation
        //TODO: less SVD
        void computeEnergyVal_zUpdate(int triI,
                                      const Eigen::RowVectorXd& zi,
                                      double& Ei) const;
        void computeGradient_zUpdate(int triI,
                                     const Eigen::RowVectorXd& zi,
                                     Eigen::Matrix<double, dim * dim, 1>& g) const;
        void computeHessianProxy_zUpdate(int triI,
                                         const Eigen::RowVectorXd& zi,
                                         Eigen::Matrix<double, dim * dim, dim * dim>& P) const;
        // SVD space
        void computeEnergyVal_zUpdate_SV(int triI,
                                         const Eigen::Matrix<double, dim, 1>& sigma_triI,
                                         const Eigen::Matrix<double, dim, 1>& sigma_Dx_plus_u,
                                         double& Ei) const;
        void computeGradient_zUpdate_SV(int triI,
                                        const Eigen::Matrix<double, dim, 1>& sigma_triI,
                                        const Eigen::Matrix<double, dim, 1>& sigma_Dx_plus_u,
                                        Eigen::Matrix<double, dim, 1>& g) const;
        void computeHessianProxy_zUpdate_SV(int triI,
                                            const Eigen::Matrix<double, dim, 1>& sigma_triI,
                                            const Eigen::Matrix<double, dim, 1>& sigma_Dx_plus_u,
                                            Eigen::Matrix<double, dim, dim>& P) const;
    };
    
}

#endif /* ADMMTimeStepper_hpp */
