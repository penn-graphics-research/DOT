//
//  Energy.hpp
//  DOT
//
//  Created by Minchen Li on 8/31/17.
//

#ifndef Energy_hpp
#define Energy_hpp

#include "Mesh.hpp"

#ifdef USE_CLOSEDFORMSVD2D
#include "ClosedFormSVD2d.hpp"
#else
#include "AutoFlipSVD.hpp"
#endif

#include "LinSysSolver.hpp"

#include <iostream>

namespace DOT {
    
    // a class for energy terms in the objective of an optimization problem
    template<int dim>
    class Energy
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
        const bool needRefactorize;
        const bool crossSigmaDervative;
        Eigen::Vector2d visRange_energyVal;
        
    public:
        Energy(bool p_needRefactorize,
               bool p_crossSigmaDervative = false,
               double visRange_min = 0.0, double visRange_max = 1.0);
        virtual ~Energy(void);
        
    public:
        bool getNeedRefactorize(void) const;
        const Eigen::Vector2d& getVisRange_energyVal(void) const;
        void setVisRange_energyVal(double visRange_min, double visRange_max);
        
    public:
        virtual void computeEnergyVal(const Mesh<dim>& data, double& energyVal, bool uniformWeight = false) const;
        virtual void getEnergyValPerElem(const Mesh<dim>& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void getEnergyValByElemID(const Mesh<dim>& data, int elemI, double& energyVal, bool uniformWeight = false) const;
        virtual void computeGradient(const Mesh<dim>& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const Mesh<dim>& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const Mesh<dim>& data, Eigen::VectorXd* V,
                                       Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const Mesh<dim>& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const;
        
        virtual void computeEnergyVal(const Mesh<dim>& data, int redoSVD,
                                      std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                      std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                      std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                      std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                      std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                      double coef,
                                      double& energyVal) const;
        virtual void computeGradient(const Mesh<dim>& data, bool redoSVD,
                                     std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                     std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                     std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                     std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                     std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                     double coef,
                                     Eigen::VectorXd& gradient) const;
        virtual void computeHessian(const Mesh<dim>& data, bool redoSVD,
                                    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                    std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                    double coef,
                                    LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                    bool projectSPD = true) const;
        virtual void computeHessian(const Mesh<dim>& data, bool redoSVD,
                                    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                    std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                    double coef,
                                    Eigen::MatrixXd& hessian,
                                    bool projectSPD = true) const;
        
        virtual void checkEnergyVal(const Mesh<dim>& data) const;
        
        virtual void checkGradient(const Mesh<dim>& data) const; // check with finite difference method, according to energyVal
        virtual void checkHessian(const Mesh<dim>& data, bool triplet = false) const; // check with finite difference method, according to gradient
        
        virtual void getEnergyValPerElemBySVD(const Mesh<dim>& data, int redoSVD,
                                              std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                              std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                              std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                            std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                            std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                              Eigen::VectorXd& energyValPerElem,
                                              bool uniformWeight = false) const;
        virtual void computeEnergyValBySVD(const Mesh<dim>& data, int redoSVD,
                                           std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                           std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                           std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                            std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                            std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                           double coef,
                                           double& energyVal) const;
        
        virtual void computeGradientByPK(const Mesh<dim>& data, bool redoSVD,
                                         std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                         std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                         std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                         std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                         std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                         double coef,
                                         Eigen::VectorXd& gradient) const;
        virtual void computeElemGradientByPK(const Mesh<dim>& data, bool redoSVD,
                                             std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                             std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                             double coef,
                                             std::vector<Eigen::Matrix<double, dim * (dim + 1), 1>>& elemGradient);
        virtual void computeHessianByPK(const Mesh<dim>& data, bool redoSVD,
                                        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                        std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                        double coef,
                                        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                        bool projectSPD = true) const;
        virtual void computeHessianByPK(const Mesh<dim>& data, bool redoSVD,
                                        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                        std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                        double coef,
                                        Eigen::MatrixXd& hessian,
                                        bool projectSPD = true) const;
        virtual void computeElemHessianByPK(const Mesh<dim>& data, bool redoSVD,
                                            std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                            std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                            double coef,
                                            const std::vector<bool>& computeElem,
                                            std::vector<Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)>>& elemHessian,
                                            std::vector<Eigen::Matrix<int, 1, dim + 1>>& vInds,
                                            bool projectSPD = true) const;
        
        virtual void computeGradientByPK(const Mesh<dim>& data,
                                         int elemI, bool redoSVD,
                                         AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                         Eigen::Matrix<double, dim, dim>& F,
                                         double coef,
                                         Eigen::Matrix<double, dim * (dim + 1), 1>& gradient) const;
        virtual void computeHessianByPK(const Mesh<dim>& data,
                                        int elemI, bool redoSVD,
                                        AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                        Eigen::Matrix<double, dim, dim>& F,
                                        double coef,
                                        Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)>& hessian,
                                        Eigen::Matrix<int, 1, dim + 1>& vInd,
                                        bool projectSPD = true) const;
#ifdef USE_SIMD
        virtual void computeElemUEnergyVal_SIMD(Eigen::VectorXd& energyValPerElem,
                                                int ceiling_size) const;
        virtual void computePHat_SIMD(std::vector<Eigen::Matrix<double, dim, 1>>& PHat,
                                      int ceiling_size) const;
        
        virtual void computeEnergyVal_SIMD(const Mesh<dim>& data,
                                           std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                           std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                           Eigen::VectorXd& energyValPerElem,
                                           bool uniformWeight = false) const;
        virtual void computeEnergyVal_SIMD(const Mesh<dim>& data,
                                           std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                           std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                           Eigen::VectorXd& energyValPerElem,
                                           bool uniformWeight = false) const;
        virtual void computeGradient_SIMD(const Mesh<dim>& data,
                                          std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                          std::vector<Eigen::Matrix<double, dim, dim>>& U,
                                          std::vector<Eigen::Matrix<double, dim, dim>>& V,
                                          std::vector<Eigen::Matrix<double, dim, 1>>& Sigma,
                                          double coef,
                                          std::vector<Eigen::Matrix<double, dim * (dim + 1), 1>>& grad_cont) const;
#endif

        virtual void computeEnergyValBySVD_F(const Mesh<dim>& data, int triI,
                                             const Eigen::Matrix<double, 1, dim * dim>& F,
                                             double& energyVal,
                                             bool uniformWeight = false) const;
        virtual void computeGradientByPK_F(const Mesh<dim>& data, int triI,
                                           const Eigen::Matrix<double, 1, dim * dim>& F,
                                           Eigen::Matrix<double, dim * dim, 1>& gradient) const;
        virtual void computeHessianByPK_F(const Mesh<dim>& data, int triI,
                                          const Eigen::Matrix<double, 1, dim * dim>& F,
                                          Eigen::Matrix<double, dim * dim, dim * dim>& hessian,
                                          bool projectSPD = true) const;
        
        virtual void compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                               double u, double lambda,
                               double& E) const;
        virtual void compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
                                           double u, double lambda,
                                           Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::Matrix<double, dim, 1>& singularValues,
                                             double u, double lambda,
                                             Eigen::Matrix<double, dim, dim>& d2E_div_dsigma2) const;
        virtual void compute_BLeftCoef(const Eigen::Matrix<double, dim, 1>& singularValues,
                                       double u, double lambda,
                                       Eigen::Matrix<double, dim * (dim - 1) / 2, 1>& BLeftCoef) const;
        virtual void compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                       const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                       double u, double lambda,
                                       Eigen::Matrix<double, dim, dim>& dE_div_dF) const;
        virtual void compute_dPHat_div_dFHat(const AutoFlipSVD<Eigen::Matrix<double, dim, dim>> &svd,
                                             double u, double lambda,
                                             Eigen::Matrix<double, dim, dim>& A,
                                             Eigen::Matrix2d B[dim * (dim - 1) / 2],
                                             double w, bool projectSPD = true) const;
        virtual void compute_dP_div_dF(const AutoFlipSVD<Eigen::Matrix<double, dim, dim>> &svd,
                                       double u, double lambda,
                                       Eigen::Matrix<double, dim * dim, dim * dim> &dP_div_dF,
                                       double w, bool projectSPD = true) const;
        
        virtual void getBulkModulus(double u, double lambda, double& bulkModulus);

        virtual void unitTest_dE_div_dsigma(std::ostream& os = std::cout) const;
        virtual void unitTest_d2E_div_dsigma2(std::ostream& os = std::cout) const;
        virtual void unitTest_BLeftCoef(std::ostream& os = std::cout) const;
        virtual void unitTest_dE_div_dF(std::ostream& os = std::cout) const;
        virtual void unitTest_dP_div_dF(std::ostream& os = std::cout) const;
    };
    
}

#endif /* Energy_hpp */
