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

namespace FracCuts {
    
    // a class for energy terms in the objective of an optimization problem
    template<int dim>
    class Energy {
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
        const Eigen::Vector2d& getVisRange_energyVal(void) const;
        void setVisRange_energyVal(double visRange_min, double visRange_max);
        
    public:
        virtual void computeEnergyVal(const TriangleSoup& data, double& energyVal, bool uniformWeight = false) const;
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const = 0;
        virtual void getEnergyValByElemID(const TriangleSoup& data, int elemI, double& energyVal, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                       Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const = 0;
        
        virtual void checkGradient(const TriangleSoup& data) const; // check with finite difference method, according to energyVal
        virtual void checkHessian(const TriangleSoup& data, bool triplet = false) const; // check with finite difference method, according to gradient
        
        virtual void getEnergyValPerElemBySVD(const TriangleSoup& data, bool redoSVD,
                                              std::vector<AutoFlipSVD<Eigen::Matrix2d>>& svd,
                                              std::vector<Eigen::Matrix2d>& F,
                                              Eigen::VectorXd& energyValPerElem,
                                              bool uniformWeight = false) const;
        virtual void computeEnergyValBySVD(const TriangleSoup& data, bool redoSVD,
                                           std::vector<AutoFlipSVD<Eigen::Matrix2d>>& svd,
                                           std::vector<Eigen::Matrix2d>& F,
                                           double& energyVal) const;
        virtual void computeGradientBySVD(const TriangleSoup& data, Eigen::VectorXd& gradient) const;
        virtual void computeHessianBySVD(const TriangleSoup& data, Eigen::VectorXd* V,
                                         Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL,
                                         bool projectSPD = true) const;
        
        virtual void computeGradientByPK(const TriangleSoup& data, bool redoSVD,
                                         std::vector<AutoFlipSVD<Eigen::Matrix2d>>& svd,
                                         std::vector<Eigen::Matrix2d>& F,
                                         Eigen::VectorXd& gradient) const;
        virtual void computeHessianByPK(const TriangleSoup& data, bool redoSVD,
                                        std::vector<AutoFlipSVD<Eigen::Matrix2d>>& svd,
                                        std::vector<Eigen::Matrix2d>& F,
                                        double coef,
                                        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                        bool projectSPD = true) const;
        
        virtual void computeEnergyValBySVD(const TriangleSoup& data, int triI,
                                           const Eigen::VectorXd& x,
                                           double& energyVal,
                                           bool uniformWeight = false) const;
        virtual void computeGradientBySVD(const TriangleSoup& data, int triI,
                                          const Eigen::VectorXd& x,
                                          Eigen::VectorXd& gradient) const;
        virtual void computeHessianBySVD(const TriangleSoup& data, int triI,
                                         const Eigen::VectorXd& x,
                                         Eigen::MatrixXd& hessian,
                                         bool projectSPD = true) const;
        
        virtual void computeEnergyValBySVD_F(const TriangleSoup& data, int triI,
                                             const Eigen::RowVectorXd& F,
                                             double& energyVal,
                                             bool uniformWeight = false) const;
        virtual void computeGradientBySVD_F(const TriangleSoup& data, int triI,
                                            const Eigen::RowVectorXd& F,
                                            Eigen::VectorXd& gradient) const;
        virtual void computeHessianBySVD_F(const TriangleSoup& data, int triI,
                                           const Eigen::RowVectorXd& F,
                                           Eigen::MatrixXd& hessian,
                                           bool projectSPD = true) const;
        
        virtual void compute_E(const Eigen::Vector2d& singularValues,
                               double& E) const;
        virtual void compute_dE_div_dsigma(const Eigen::Vector2d& singularValues,
                                           Eigen::Vector2d& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::Vector2d& singularValues,
                                             Eigen::Matrix2d& d2E_div_dsigma2) const;
        virtual void compute_dE_div_dF(const Eigen::Matrix2d& F,
                                       const AutoFlipSVD<Eigen::Matrix2d>& svd,
                                       Eigen::Matrix2d& dE_div_dF) const;
        
        virtual void compute_d2E_div_dF2_rest(Eigen::MatrixXd& d2E_div_dF2_rest) const;
        
        virtual void initStepSize(const TriangleSoup& data,
                                  const Eigen::VectorXd& searchDir,
                                  double& stepSize) const;
        virtual void initStepSize_preventElemInv(const TriangleSoup& data,
                                                 const Eigen::VectorXd& searchDir,
                                                 double& stepSize) const;
        
        virtual void initStepSize(const Eigen::VectorXd& V,
                                  const Eigen::VectorXd& searchDir,
                                  double& stepSize) const;
        virtual void initStepSize_preventElemInv(const Eigen::VectorXd& V,
                                                 const Eigen::VectorXd& searchDir,
                                                 double& stepSize) const;
        
        virtual void getBulkModulus(double& bulkModulus);
    };
    
}

#endif /* Energy_hpp */
