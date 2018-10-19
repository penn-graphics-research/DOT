//
//  Optimizer.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef Optimizer_hpp
#define Optimizer_hpp

#include "Types.hpp"
#include "Energy.hpp"
#include "Scaffold.hpp"
#include "AnimScripter.hpp"
#include "Config.hpp"
#include "LinSysSolver.hpp"
#include "OSQP.h"

#include <fstream>

namespace FracCuts {
    
    // a class for solving an optimization problem
    template<int dim>
    class Optimizer {
        friend class TriangleSoup<dim>;
        
    protected: // referenced data
        const TriangleSoup<dim>& data0; // initial guess
        const std::vector<Energy<dim>*>& energyTerms; // E_0, E_1, E_2, ...
        const std::vector<double>& energyParams; // a_0, a_1, a_2, ...
        // E = \Sigma_i a_i E_i
        const Config& animConfig;
        
    protected: // owned data
        int propagateFracture;
        bool fractureInitiated = false;
        bool allowEDecRelTol;
        bool mute;
        bool pardisoThreadAmt;
        bool needRefactorize;
        int globalIterNum;
        int topoIter;
        double relGL2Tol, energyParamSum;
        double sqnorm_H_rest, sqnorm_l;
        TriangleSoup<dim> result; // intermediate results of each iteration
        TriangleSoup<dim> data_findExtrema; // intermediate results for deciding the cuts in each topology step
        bool scaffolding; // whether to enable bijectivity parameterization
        double w_scaf;
        Scaffold scaffold; // air meshes to enforce bijectivity
        
        // SPD solver for solving the linear system for search directions
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *linSysSolver;
        
        bool solveQP;
        OSQP QPSolver;
        Eigen::SparseMatrix<double> P_OSQP;
        std::vector<double*> elemPtr_P_OSQP;
        Eigen::VectorXd l_OSQP, u_OSQP, dual_OSQP;
        Eigen::SparseMatrix<double> A_OSQP;
        
        Eigen::VectorXd gradient; // energy gradient computed in each iteration
        Eigen::VectorXd searchDir; // search direction comptued in each iteration
        double lastEnergyVal; // for output and line search
        double lastEDec;
        double targetGRes;
        std::vector<Eigen::VectorXd> gradient_ET;
        Eigen::VectorXd gradient_scaffold;
        std::vector<double> energyVal_ET;
        double energyVal_scaffold;
        
        Eigen::MatrixXd UV_bnds_scaffold;
        Eigen::MatrixXi E_scaffold;
        Eigen::VectorXi bnd_scaffold;
        std::vector<std::set<int>> vNeighbor_withScaf;
        std::set<int> fixedV_withScaf;
        
        std::ostringstream buffer_energyValPerIter;
        std::ostringstream buffer_gradientPerIter;
        std::ofstream file_energyValPerIter;
        std::ofstream file_gradientPerIter;
        
        std::ofstream file_iterStats;
        
//        std::map<int, double> directionFix;
//        double lambda_df;
        
    protected: // dynamic information
        Eigen::VectorXd velocity;
        Eigen::MatrixXd resultV_n, xTilta;
        double dt, dtSq;
        Eigen::Matrix<double, dim, 1> gravity, gravityDtSq;
        int frameAmt;
        AnimScripter<dim> animScripter;
        int innerIterAmt;
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd;
        std::vector<Eigen::Matrix<double, dim, dim>> F;
        
    public: // constructor and destructor
        Optimizer(const TriangleSoup<dim>& p_data0,
                  const std::vector<Energy<dim>*>& p_energyTerms,
                  const std::vector<double>& p_energyParams,
                  int p_propagateFracture = 1, bool p_mute = false, bool p_scaffolding = false,
                  const Eigen::MatrixXd& UV_bnds = Eigen::MatrixXd(),
                  const Eigen::MatrixXi& E = Eigen::MatrixXi(),
                  const Eigen::VectorXi& bnd = Eigen::VectorXi(),
                  const Config& animConfig = Config());
        ~Optimizer(void);
        
    public: // API
        virtual void setTime(double duration, double dt);
        virtual void updateTargetGRes(double p_tol = 1.0e-10);
        // precompute preconditioning matrix and factorize for fast solve, prepare initial guess
        virtual void precompute(void);
        
//        void fixDirection(void);
        
        // solve the optimization problem that minimizes E using a hill-climbing method,
        // the final result will be in result
        virtual int solve(int maxIter = 100);
        
        virtual void updatePrecondMtrAndFactorize(void);
        
        virtual void updateEnergyData(bool updateEVal = true, bool updateGradient = true, bool updateHessian = true);
        virtual void setConfig(const TriangleSoup<dim>& config, int iterNum, int p_topoIter);
        virtual void setPropagateFracture(bool p_prop);
        virtual void setScaffolding(bool p_scaffolding);
        
        virtual void computeLastEnergyVal(void);
        
        virtual void getGradientVisual(Eigen::MatrixXd& arrowVec) const;
        virtual void getFaceFieldForVis(Eigen::VectorXd& field) const;
        virtual void getSharedVerts(Eigen::VectorXi& sharedVerts) const;
        virtual TriangleSoup<dim>& getResult(void);
        virtual const Scaffold& getScaffold(void) const;
        virtual const TriangleSoup<dim>& getAirMesh(void) const;
        virtual bool isScaffolding(void) const;
        virtual const TriangleSoup<dim>& getData_findExtrema(void) const;
        virtual int getIterNum(void) const;
        virtual int getTopoIter(void) const;
        virtual int getInnerIterAmt(void) const;
        virtual void setRelGL2Tol(double p_relTol);
        virtual void setAllowEDecRelTol(bool p_allowEDecRelTol);
        virtual double getDt(void) const;
        virtual void setAnimScriptType(AnimScriptType animScriptType);
        
        virtual void flushEnergyFileOutput(void);
        virtual void flushGradFileOutput(void);
        virtual void clearEnergyFileOutputBuffer(void);
        virtual void clearGradFileOutputBuffer(void);
        
    protected: // helper functions
        virtual void initX(int option);
        virtual void computeXTilta(void);
        virtual bool fullyImplicit(void);
        
        // solve for new configuration in the next iteration
        //NOTE: must compute current gradient first
        virtual bool solve_oneStep(void);
        
        virtual bool lineSearch(void);

        virtual void stepForward(const Eigen::MatrixXd& dataV0, const Eigen::MatrixXd& scaffoldV0,
                         TriangleSoup<dim>& data, Scaffold& scaffoldData, double stepSize) const;
        
        virtual void computeEnergyVal(const TriangleSoup<dim>& data, const Scaffold& scaffoldData,
                                      bool redoSVD, double& energyVal, bool excludeScaffold = false);
        virtual void computeGradient(const TriangleSoup<dim>& data, const Scaffold& scaffoldData,
                                     bool redoSVD, Eigen::VectorXd& gradient,
                                     bool excludeScaffold = false);
        virtual void computePrecondMtr(const TriangleSoup<dim>& data, const Scaffold& scaffoldData,
                                       bool redoSVD,
                                       LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *p_linSysSolver);
        virtual void computeHessian(const TriangleSoup<dim>& data, const Scaffold& scaffoldData, Eigen::SparseMatrix<double>& hessian) const;
        
        virtual void initStepSize(const TriangleSoup<dim>& data, double& stepSize) const;
        
        virtual void writeEnergyValToFile(bool flush);
        virtual void writeGradL2NormToFile(bool flush);
        
    public: // data access
        virtual double getLastEnergyVal(bool excludeScaffold = false) const;
    };
    
}

#endif /* Optimizer_hpp */
