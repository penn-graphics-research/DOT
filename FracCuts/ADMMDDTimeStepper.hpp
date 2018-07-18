//
//  ADMMDDTimeStepper.hpp
//  FracCuts
//
//  Created by Minchen Li on 7/17/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef ADMMDDTimeStepper_hpp
#define ADMMDDTimeStepper_hpp

#include "Optimizer.hpp"

namespace FracCuts {
    
    class ADMMDDTimeStepper : public Optimizer
    {
    protected:
        std::vector<TriangleSoup> mesh_subdomain;
        std::vector<std::map<int, int>> globalVIToLocal_subdomain;
        std::vector<std::map<int, int>> globalVIToDual_subdomain;
        std::vector<Eigen::MatrixXd> u_subdomain;
        std::vector<Eigen::VectorXd> weights_subdomain;
        std::vector<Eigen::MatrixXd> xHat_subdomain;
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> linSysSolver_subdomain;
        //TODO: reorganize folder, cmake
        
        Eigen::VectorXd weightSum;
        
    public:
        ADMMDDTimeStepper(const TriangleSoup& p_data0,
                          const std::vector<Energy*>& p_energyTerms,
                          const std::vector<double>& p_energyParams,
                          int p_propagateFracture = 1,
                          bool p_mute = false,
                          bool p_scaffolding = false,
                          const Eigen::MatrixXd& UV_bnds = Eigen::MatrixXd(),
                          const Eigen::MatrixXi& E = Eigen::MatrixXi(),
                          const Eigen::VectorXi& bnd = Eigen::VectorXi(),
                          AnimScriptType animScriptType = AST_NULL);
        ~ADMMDDTimeStepper(void);
        
    public:
        virtual void precompute(void);
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void subdomainSolve(void); // local solve
        void checkRes(void);
        void boundaryConsensusSolve(void); // global solve
        
        // subdomain energy computation
        void computeEnergyVal_subdomain(int subdomainI, double& Ei) const;
        void computeGradient_subdomain(int subdomainI, Eigen::VectorXd& g) const;
        void computeHessianProxy_subdomain(int subdomainI, Eigen::VectorXd& V,
                                           Eigen::VectorXi& I, Eigen::VectorXi& J) const;
    };
}

#endif /* ADMMDDTimeStepper_hpp */
