//
//  DADMMTimeStepper.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/26/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef DADMMTimeStepper_hpp
#define DADMMTimeStepper_hpp

#include "Optimizer.hpp"

namespace FracCuts {
    
    class DADMMTimeStepper : public Optimizer
    {
    protected:
        Eigen::MatrixXd z, u, dz;
        Eigen::VectorXd weights, weights2;
        
        Eigen::VectorXd rhs_xUpdate, M_mult_xHat, coefMtr_diag;
        Eigen::MatrixXd D_mult_x;
        
    public:
        DADMMTimeStepper(const TriangleSoup& p_data0,
                         const std::vector<Energy*>& p_energyTerms,
                         const std::vector<double>& p_energyParams,
                         int p_propagateFracture = 1,
                         bool p_mute = false,
                         bool p_scaffolding = false,
                         const Eigen::MatrixXd& UV_bnds = Eigen::MatrixXd(),
                         const Eigen::MatrixXi& E = Eigen::MatrixXi(),
                         const Eigen::VectorXi& bnd = Eigen::VectorXi(),
                         AnimScriptType animScriptType = AST_NULL);
        
    public:
        virtual void precompute(void);
        
        virtual void getFaceFieldForVis(Eigen::VectorXd& field) const;
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void zuUpdate(void); // local solve
        void checkRes(void);
        void xUpdate(void); // global solve
        
        void compute_Di_mult_xi(int elemI);
        
        // local energy computation
        void computeEnergyVal_zUpdate(int triI,
                                      const Eigen::RowVectorXd& zi,
                                      double& Ei) const;
        void computeGradient_zUpdate(int triI,
                                     const Eigen::RowVectorXd& zi,
                                     Eigen::VectorXd& g) const;
        void computeHessianProxy_zUpdate(int triI,
                                         const Eigen::RowVectorXd& zi,
                                         Eigen::MatrixXd& P) const;
    };
}

#endif /* DADMMTimeStepper_hpp */
