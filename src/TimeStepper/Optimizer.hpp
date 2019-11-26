//
//  Optimizer.hpp
//  DOT
//
//  Created by Minchen Li on 8/31/17.
//

#ifndef Optimizer_hpp
#define Optimizer_hpp

#include "Types.hpp"
#include "Energy.hpp"
#include "AnimScripter.hpp"
#include "Config.hpp"
#include "LinSysSolver.hpp"

#include <fstream>

namespace DOT {
    
    // a class for solving an optimization problem
    template<int dim>
    class Optimizer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        friend class Mesh<dim>;

    protected: // referenced data
        const Mesh<dim>& data0; // initial guess
        const std::vector<Energy<dim>*>& energyTerms; // E_0, E_1, E_2, ...
        const std::vector<double>& energyParams; // a_0, a_1, a_2, ...
        // E = \Sigma_i a_i E_i
        const Config& animConfig;
        
    protected: // owned data
        bool allowEDecRelTol;
        bool mute;
        bool pardisoThreadAmt;
        bool needRefactorize;
        int globalIterNum;
        double relGL2Tol, energyParamSum;
        Mesh<dim> result; // intermediate results of each iteration
        
        // SPD solver for solving the linear system for search directions
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *linSysSolver;
        
        Eigen::VectorXd gradient; // energy gradient computed in each iteration
        Eigen::VectorXd searchDir; // search direction comptued in each iteration
        double lastEnergyVal; // for output and line search
        double lastEDec;
        double targetGRes;
        std::vector<Eigen::VectorXd> gradient_ET;
        std::vector<double> energyVal_ET;
        
        std::ofstream file_iterStats;
        
        int numOfLineSearch;
        
//        std::map<int, double> directionFix;
//        double lambda_df;
        
    protected: // dynamic information
        Eigen::VectorXd velocity;
        Eigen::MatrixXd resultV_n, xTilta, dx_Elastic, acceleration;
        double dt, dtSq;
        Eigen::Matrix<double, dim, 1> gravity, gravityDtSq;
        int frameAmt;
        AnimScripter<dim> animScripter;
        int innerIterAmt;
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd;
        std::vector<Eigen::Matrix<double, dim, dim>> F, U, V;
        std::vector<Eigen::Matrix<double, dim, 1>> Sigma;
        
    public: // constructor and destructor
        Optimizer(const Mesh<dim>& p_data0,
                  const std::vector<Energy<dim>*>& p_energyTerms,
                  const std::vector<double>& p_energyParams,
                  bool p_mute = false,
                  const Config& animConfig = Config());
        virtual ~Optimizer(void);
        
    public: // API
        virtual void setTime(double duration, double dt);
        virtual void updateTargetGRes(void);
        // precompute preconditioning matrix and factorize for fast solve, prepare initial guess
        virtual void precompute(void);
        
        // solve the optimization problem that minimizes E using a hill-climbing method,
        // the final result will be in result
        virtual int solve(int maxIter = 100);
        
        virtual void updatePrecondMtrAndFactorize(void);
        
        virtual void updateEnergyData(bool updateEVal = true, bool updateGradient = true, bool updateHessian = true);
        
        virtual void getGradientVisual(Eigen::MatrixXd& arrowVec) const;
        virtual void getFaceFieldForVis(Eigen::VectorXd& field);
        virtual void getSharedVerts(Eigen::VectorXi& sharedVerts) const;
        virtual Mesh<dim>& getResult(void);
        virtual int getIterNum(void) const;
        virtual int getInnerIterAmt(void) const;
        virtual void setRelGL2Tol(double p_relTol = 1.0e-5);
        virtual void setAllowEDecRelTol(bool p_allowEDecRelTol);
        virtual double getDt(void) const;
        virtual void setAnimScriptType(AnimScriptType animScriptType);
        
        virtual void saveStatus(void);

        virtual const Eigen::MatrixXd& getDenseMatrix(int sbdI) const;
        virtual void getGradient(int sbdI, Eigen::VectorXd& gradient);
        virtual LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* getSparseSolver(int sbdI);

    protected: // helper functions
        virtual void initX(int option);
        virtual void computeXTilta(void);
        
        virtual double computeCharNormSq(const Mesh<dim>& mesh,
                                         const Energy<dim>* energy,
                                         double epsSq_c) const;
        
        virtual bool fullyImplicit(void);
        
        // solve for new configuration in the next iteration
        //NOTE: must compute current gradient first
        virtual bool solve_oneStep(void);

        virtual bool lineSearch(double& stepSize, double armijoParam = 0.0, double lowerBound = 0.0);
        virtual void dimSeparatedSolve(LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                       const Eigen::VectorXd& rhs,
                                       Eigen::MatrixXd& V);
        virtual void dimSeparatedSolve(std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*>& linSysSolver,
                                       const Eigen::VectorXd& rhs,
                                       Eigen::MatrixXd& V);
        virtual void dimSeparatedSolve(LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                       const Eigen::VectorXd& rhs,
                                       Eigen::VectorXd& solved);

        virtual void stepForward(const Eigen::MatrixXd& dataV0, Mesh<dim>& data, double stepSize) const;
        
        virtual void computeEnergyVal(const Mesh<dim>& data, int redoSVD, double& energyVal);
        virtual void computeGradient(const Mesh<dim>& data, bool redoSVD, Eigen::VectorXd& gradient);
        virtual void computePrecondMtr(const Mesh<dim>& data, bool redoSVD,
                                       LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *p_linSysSolver);
        
        virtual void computeSystemEnergy(double& sysE);
        
        virtual void initStepSize(const Mesh<dim>& data, double& stepSize) const;
        
        virtual void checkGradient(void);
        
    public: // data access
        virtual double getLastEnergyVal(void) const;
    };
    
}

#endif /* Optimizer_hpp */
