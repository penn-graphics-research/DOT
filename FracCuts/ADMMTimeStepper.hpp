//
//  ADMMTimeStepper.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/28/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef ADMMTimeStepper_hpp
#define ADMMTimeStepper_hpp

#include "Optimizer.hpp"

#include <map>

namespace FracCuts {
    
    class ADMMTimeStepper : public Optimizer
    {
    protected:
        Eigen::MatrixXd z, u;
        Eigen::VectorXd weights, weights2;
        std::vector<Eigen::MatrixXd> D_array; // maps xi to zi
        
        Eigen::VectorXd rhs_xUpdate, M_mult_xHat, x_solved;
        Eigen::MatrixXd D_mult_x;
        std::vector<std::map<int, double>> offset_fixVerts; // for modifying the linSys to fix vertices
        
    public:
        ADMMTimeStepper(const TriangleSoup& p_data0,
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
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void zuUpdate(void); // local solve
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

#endif /* ADMMTimeStepper_hpp */
