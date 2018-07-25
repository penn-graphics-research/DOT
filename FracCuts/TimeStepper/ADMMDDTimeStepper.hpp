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
        std::vector<Eigen::VectorXi> elemList_subdomain;
        std::vector<TriangleSoup> mesh_subdomain;
        std::vector<std::map<int, int>> globalVIToLocal_subdomain;
        std::vector<std::map<int, int>> globalVIToDual_subdomain;
        std::vector<Eigen::MatrixXd> xHat_subdomain;
        int dualDim;
        std::vector<Eigen::MatrixXd> u_subdomain, du_subdomain, dz_subdomain;
        std::vector<Eigen::MatrixXd> weights_subdomain;
        Eigen::MatrixXd weightSum;
        std::vector<bool> isSharedVert;
        Eigen::VectorXi sharedVerts;
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> linSysSolver_subdomain;
        
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
        
        virtual void getFaceFieldForVis(Eigen::VectorXd& field) const;
        virtual void getSharedVerts(Eigen::VectorXi& sharedVerts) const;
        
        virtual void writeMeshToFile(const std::string& filePath_pre) const;
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void initPrimal(void);
        void initWeights(void);
        
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
