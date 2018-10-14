//
//  Energy.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef Energy_hpp
#define Energy_hpp

#include "TriangleSoup.hpp"

#ifdef USE_CLOSEDFORMSVD2D
#include "ClosedFormSVD2d.hpp"
#else
#include "AutoFlipSVD.hpp"
#endif

#include "LinSysSolver.hpp"

#include <iostream>

namespace FracCuts {
    
    // a class for energy terms in the objective of an optimization problem
    template<int dim>
    class Energy
    {
    protected:
        const bool needRefactorize;
        const bool needElemInvSafeGuard;
        const bool crossSigmaDervative;
        Eigen::Vector2d visRange_energyVal;
        
    public:
        Energy(bool p_needRefactorize, bool p_needElemInvSafeGuard = false,
               bool p_crossSigmaDervative = false,
               double visRange_min = 0.0, double visRange_max = 1.0);
        virtual ~Energy(void);
        
    public:
        bool getNeedRefactorize(void) const;
        bool getNeedElemInvSafeGuard(void) const;
        const Eigen::Vector2d& getVisRange_energyVal(void) const;
        void setVisRange_energyVal(double visRange_min, double visRange_max);
        
    public:
        virtual void computeEnergyVal(const TriangleSoup<dim>& data, double& energyVal, bool uniformWeight = false) const;
        virtual void getEnergyValPerElem(const TriangleSoup<dim>& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void getEnergyValByElemID(const TriangleSoup<dim>& data, int elemI, double& energyVal, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup<dim>& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup<dim>& data, Eigen::VectorXd* V,
                                       Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const;
        
        virtual void computeEnergyVal(const TriangleSoup<dim>& data, bool redoSVD,
                                      std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                      std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                      double& energyVal) const;
        virtual void computeGradient(const TriangleSoup<dim>& data, bool redoSVD,
                                     std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                     std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                     Eigen::VectorXd& gradient) const;
        virtual void computeHessian(const TriangleSoup<dim>& data, bool redoSVD,
                                    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                    std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                    double coef,
                                    LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                    bool projectSPD = true) const;
        
        virtual void checkEnergyVal(const TriangleSoup<dim>& data) const;
        
        virtual void checkGradient(const TriangleSoup<dim>& data) const; // check with finite difference method, according to energyVal
        virtual void checkHessian(const TriangleSoup<dim>& data, bool triplet = false) const; // check with finite difference method, according to gradient
        
        virtual void getEnergyValPerElemBySVD(const TriangleSoup<dim>& data, bool redoSVD,
                                              std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                              std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                              Eigen::VectorXd& energyValPerElem,
                                              bool uniformWeight = false) const;
        virtual void computeEnergyValBySVD(const TriangleSoup<dim>& data, bool redoSVD,
                                           std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                           std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                           double& energyVal) const;
        virtual void computeGradientBySVD(const TriangleSoup<dim>& data, Eigen::VectorXd& gradient) const;
        virtual void computeHessianBySVD(const TriangleSoup<dim>& data, Eigen::VectorXd* V,
                                         Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL,
                                         bool projectSPD = true) const;
        
        virtual void computeGradientByPK(const TriangleSoup<dim>& data, bool redoSVD,
                                         std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                         std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                         Eigen::VectorXd& gradient) const;
        virtual void computeHessianByPK(const TriangleSoup<dim>& data, bool redoSVD,
                                        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                        std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                        double coef,
                                        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                        bool projectSPD = true) const;
        
        virtual void computeEnergyValBySVD(const TriangleSoup<dim>& data, int triI,
                                           const Eigen::VectorXd& x,
                                           double& energyVal,
                                           bool uniformWeight = false) const;
        virtual void computeGradientBySVD(const TriangleSoup<dim>& data, int triI,
                                          const Eigen::VectorXd& x,
                                          Eigen::VectorXd& gradient) const;
        virtual void computeHessianBySVD(const TriangleSoup<dim>& data, int triI,
                                         const Eigen::VectorXd& x,
                                         Eigen::MatrixXd& hessian,
                                         bool projectSPD = true) const;
        
        virtual void computeEnergyValBySVD_F(const TriangleSoup<dim>& data, int triI,
                                             const Eigen::Matrix<double, 1, dim * dim>& F,
                                             double& energyVal,
                                             bool uniformWeight = false) const;
        virtual void computeGradientBySVD_F(const TriangleSoup<dim>& data, int triI,
                                            const Eigen::Matrix<double, 1, dim * dim>& F,
                                            Eigen::Matrix<double, dim * dim, 1>& gradient) const;
        virtual void computeHessianBySVD_F(const TriangleSoup<dim>& data, int triI,
                                           const Eigen::Matrix<double, 1, dim * dim>& F,
                                           Eigen::Matrix<double, dim * dim, dim * dim>& hessian,
                                           bool projectSPD = true) const;
        virtual void computeGradientByPK_F(const TriangleSoup<dim>& data, int triI,
                                           const Eigen::Matrix<double, 1, dim * dim>& F,
                                           Eigen::Matrix<double, dim * dim, 1>& gradient) const;
        virtual void computeHessianByPK_F(const TriangleSoup<dim>& data, int triI,
                                          const Eigen::Matrix<double, 1, dim * dim>& F,
                                          Eigen::Matrix<double, dim * dim, dim * dim>& hessian,
                                          bool projectSPD = true) const;
        
        virtual void compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                               double& E) const;
        virtual void compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
                                           Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::Matrix<double, dim, 1>& singularValues,
                                             Eigen::Matrix<double, dim, dim>& d2E_div_dsigma2) const;
        virtual void compute_BLeftCoef(const Eigen::Matrix<double, dim, 1>& singularValues,
                                       Eigen::Matrix<double, dim * (dim - 1) / 2, 1>& BLeftCoef) const;
//        virtual void compute_BRightCoef(const Eigen::Matrix<double, dim, 1>& singularValues,
//                                        const Eigen::Matrix<double, dim, 1>& dE_div_dsigma,
//                                        Eigen::Matrix<double, dim * (dim- 1) / 2, 1>& BRightCoef) const;
        virtual void compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                       const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                       Eigen::Matrix<double, dim, dim>& dE_div_dF) const;
        virtual void compute_dP_div_dF(const AutoFlipSVD<Eigen::Matrix<double, dim, dim>> &svd,
                                       Eigen::Matrix<double, dim * dim, dim * dim> &dP_div_dF,
                                       double w, bool projectSPD = true) const;
        
        virtual void compute_d2E_div_dF2_rest(Eigen::Matrix<double, dim * dim, dim * dim>& d2E_div_dF2_rest) const;
        
        virtual void filterStepSize(const TriangleSoup<dim>& data,
                                    const Eigen::VectorXd& searchDir,
                                    double& stepSize) const;
        virtual void filterStepSize(const Eigen::VectorXd& V,
                                    const Eigen::VectorXd& searchDir,
                                    double& stepSize) const;
        
        virtual void getBulkModulus(double& bulkModulus);

        virtual void unitTest_dE_div_dsigma(std::ostream& os = std::cout) const;
        virtual void unitTest_d2E_div_dsigma2(std::ostream& os = std::cout) const;
        virtual void unitTest_BLeftCoef(std::ostream& os = std::cout) const;
        virtual void unitTest_dE_div_dF(std::ostream& os = std::cout) const;
        virtual void unitTest_dP_div_dF(std::ostream& os = std::cout) const;
    };
    
}

#endif /* Energy_hpp */
