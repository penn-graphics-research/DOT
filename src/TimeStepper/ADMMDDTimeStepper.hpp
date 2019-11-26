//
//  ADMMDDTimeStepper.hpp
//  DOT
//
//  Created by Minchen Li on 7/17/18.
//

#ifndef ADMMDDTimeStepper_hpp
#define ADMMDDTimeStepper_hpp

#include "Optimizer.hpp"

#include "Timer.hpp"

#include <Eigen/Cholesky>

namespace DOT {
    
    template<int dim>
    class ADMMDDTimeStepper : public Optimizer<dim>
    {
        typedef Optimizer<dim> Base;
        
    protected:
        std::vector<Eigen::VectorXi> elemList_subdomain;
        std::vector<Mesh<dim>> mesh_subdomain;
        std::vector<std::map<int, int>> globalVIToLocal_subdomain;
        std::vector<std::vector<int>> localVIToGlobal_subdomain;
        std::vector<std::map<int, int>> globalTriIToLocal_subdomain;
        std::vector<std::map<int, int>> globalVIToDual_subdomain;
        std::vector<std::vector<int>> dualIndIToLocal_subdomain; //TODO: use it to simplify index querying
        std::vector<std::vector<int>> dualIndIToShared_subdomain;
        std::vector<std::vector<int>> interElemGlobalI_sbd;
        std::vector<Eigen::MatrixXd> xHat_subdomain;
        std::vector<double> tol_subdomain;
        std::vector<int> localIterCount_sbd;
        int dualDim;
        std::vector<Eigen::MatrixXd> u_subdomain, du_subdomain, dz_subdomain, y_subdomain;
        std::vector<Eigen::MatrixXd> weights_subdomain;
        std::vector<std::map<std::pair<int, int>, double>> weightMtr_subdomain;
        std::vector<std::map<std::pair<int, int>, double>> weightMtrFixed_subdomain;
        Eigen::MatrixXd weightSum;
        std::vector<bool> isSharedVert;
        std::map<int, int> globalVIToShared;
        Eigen::VectorXi sharedVerts;
        
        Eigen::SparseMatrix<double> consensusMtr;
        std::vector<double*> CM_elemPtr;
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* consensusSolver;
        Eigen::VectorXd solvedSharedVerts;
        
        std::vector<bool> m_isUpdateElemHessian;
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> linSysSolver_subdomain;
        std::vector<std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>> svd_subdomain;
        std::vector<std::vector<Eigen::Matrix<double, dim, dim>>> F_subdomain, F_subdomain_last, U_sbd, V_sbd;
        std::vector<std::vector<Eigen::Matrix<double, dim, 1>>> Sigma_sbd;
        bool useDense = false;
        std::vector<Eigen::MatrixXd> hessian_sbd;
        std::vector<Eigen::LDLT<Eigen::MatrixXd>> denseSolver_sbd;
        
        double sqn_g;
        Eigen::MatrixXd resultVk; //TODO: only need the interface verts
        
        std::vector<Timer> timer_sbd;
        Timer timer_sum;
        
    public:
        ADMMDDTimeStepper(const Mesh<dim>& p_data0,
                          const std::vector<Energy<dim>*>& p_energyTerms,
                          const std::vector<double>& p_energyParams,
                          bool p_mute = false,
                          const Config& animConfig = Config());
        virtual ~ADMMDDTimeStepper(void);
        
    public:
        virtual void precompute(void);
        
        virtual void getFaceFieldForVis(Eigen::VectorXd& field);
        virtual void getSharedVerts(Eigen::VectorXi& sharedVerts) const;
        
        virtual void writeMeshToFile(const std::string& filePath_pre) const;

        virtual const Eigen::MatrixXd& getDenseMatrix(int sbdI) const;
        virtual void getGradient(int sbdI, Eigen::VectorXd& gradient);
        virtual LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* getSparseSolver(int sbdI);
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void initPrimal(int option); // 0: last timestep; 1: explicit Euler; 2: xHat; 3: Symplectic Euler; 4: uniformly accelerated motion approximation; 5: Jacobi
        void initDual(void);
        void initWeights(bool overwriteSbdH);
        void initWeights_fast(bool overwriteSbdH);
        void initConsensusSolver(void);
        void updateConsensusSolver(void);
        
        void subdomainSolve(int localMaxIter = 100,
                            int localMinIter = 0,
                            bool updateH = true,
                            bool linesearch = true); // local solve
        void checkRes(void);
        void boundaryConsensusSolve(double relaxParam = 1.0); // global solve
        void dualSolve(double stepSize = 1.0, double relaxParam = 1.0);
        
        void uToY(void);
        void yToU(void);
        
        void computeError(const Eigen::VectorXd& residual,
                          double& err_in, double& err_b) const;

        // subdomain energy computation
        void computeEnergyVal_subdomain(int subdomainI, bool redoSVD, double& Ei);
        void computeGradient_subdomain(int subdomainI, bool redoSVD, Eigen::VectorXd& g);
        void computeHessianProxy_subdomain(int subdomainI, bool redoSVD,
                                           bool augmentLag);
        
        void extract(const Eigen::VectorXd& src_g,
                     int sbdI, Eigen::VectorXd& dst_l) const;
        void fill(int sbdI, const Eigen::VectorXd& src_l,
                  Eigen::VectorXd& dst_g) const;
    };
}

#endif /* ADMMDDTimeStepper_hpp */
