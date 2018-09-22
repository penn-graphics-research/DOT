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

#include <Eigen/Cholesky>

namespace FracCuts {
    
    template<int dim>
    class ADMMDDTimeStepper : public Optimizer<dim>
    {
        typedef Optimizer<dim> Base;
        
    protected:
        std::vector<Eigen::VectorXi> elemList_subdomain;
        std::vector<TriangleSoup<dim>> mesh_subdomain;
        std::vector<std::map<int, int>> globalVIToLocal_subdomain;
        std::vector<std::vector<int>> localVIToGlobal_subdomain;
        std::vector<std::map<int, int>> globalTriIToLocal_subdomain;
        std::vector<std::map<int, int>> globalVIToDual_subdomain;
        std::vector<std::vector<int>> dualIndIToLocal_subdomain; //TODO: use it to simplify index querying
        std::vector<std::vector<int>> dualIndIToShared_subdomain;
        std::vector<Eigen::MatrixXd> xHat_subdomain;
        int dualDim;
        std::vector<Eigen::MatrixXd> u_subdomain, du_subdomain, dz_subdomain;
        std::vector<Eigen::MatrixXd> weights_subdomain;
        std::vector<std::map<std::pair<int, int>, double>> weightMtr_subdomain;
        std::vector<std::map<std::pair<int, int>, double>> weightMtrFixed_subdomain;
        Eigen::MatrixXd weightSum;
        std::vector<bool> isSharedVert;
        std::map<int, int> globalVIToShared;
        Eigen::VectorXi sharedVerts;
        Eigen::MatrixXd consensusMtr;
        Eigen::LDLT<Eigen::MatrixXd> consensusSolver;
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> linSysSolver_subdomain;
        std::vector<std::vector<AutoFlipSVD<Eigen::Matrix2d>>> svd_subdomain;
        std::vector<std::vector<Eigen::Matrix2d>> F_subdomain;
        
    public:
        ADMMDDTimeStepper(const TriangleSoup<dim>& p_data0,
                          const std::vector<Energy<dim>*>& p_energyTerms,
                          const std::vector<double>& p_energyParams,
                          int p_propagateFracture = 1,
                          bool p_mute = false,
                          bool p_scaffolding = false,
                          const Eigen::MatrixXd& UV_bnds = Eigen::MatrixXd(),
                          const Eigen::MatrixXi& E = Eigen::MatrixXi(),
                          const Eigen::VectorXi& bnd = Eigen::VectorXi(),
                          const Config& animConfig = Config());
        ~ADMMDDTimeStepper(void);
        
    public:
        virtual void precompute(void);
        
        virtual void getFaceFieldForVis(Eigen::VectorXd& field) const;
        virtual void getSharedVerts(Eigen::VectorXi& sharedVerts) const;
        
        virtual void writeMeshToFile(const std::string& filePath_pre) const;
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void initPrimal(int option); // 0: last timestep; 1: explicit Euler; 2: xHat; 3: Symplectic Euler; 4: uniformly accelerated motion approximation; 5: Jacobi
        void initDual(void);
        void initWeights(void);
        void initConsensusSolver(void);
        
        void subdomainSolve(void); // local solve
        void checkRes(void);
        void boundaryConsensusSolve(void); // global solve
        
        // subdomain energy computation
        void computeEnergyVal_subdomain(int subdomainI, bool redoSVD, double& Ei);
        void computeGradient_subdomain(int subdomainI, bool redoSVD, Eigen::VectorXd& g);
        void computeHessianProxy_subdomain(int subdomainI, bool redoSVD);
    };
}

#endif /* ADMMDDTimeStepper_hpp */
