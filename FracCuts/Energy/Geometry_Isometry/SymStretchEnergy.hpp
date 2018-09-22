//
//  SymStretchEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 9/3/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef SymStretchEnergy_hpp
#define SymStretchEnergy_hpp

#include "Energy.hpp"

namespace FracCuts {
    
    template<int dim>
    class SymStretchEnergy : public Energy<dim>
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup<dim>& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void getEnergyValByElemID(const TriangleSoup<dim>& data, int elemI, double& energyVal, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup<dim>& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup<dim>& data, Eigen::VectorXd* V,
                                       Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const;
        
        virtual void getEnergyValPerVert(const TriangleSoup<dim>& data, Eigen::VectorXd& energyValPerVert) const;
        virtual void getMaxUnweightedEnergyValPerVert(const TriangleSoup<dim>& data, Eigen::VectorXd& MaxUnweightedEnergyValPerVert) const;
        virtual void computeLocalGradient(const TriangleSoup<dim>& data, Eigen::MatrixXd& localGradients) const;
        virtual void getDivGradPerElem(const TriangleSoup<dim>& data, Eigen::VectorXd& divGradPerElem) const;
        virtual void computeDivGradPerVert(const TriangleSoup<dim>& data, Eigen::VectorXd& divGradPerVert) const;
        virtual void computeLocalSearchDir(const TriangleSoup<dim>& data, Eigen::MatrixXd& localSearchDir) const;
        
        virtual void compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                               double& E) const;
        virtual void compute_dE_div_dsigma(const Eigen::Vector2d& singularValues,
                                           Eigen::Vector2d& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::Vector2d& singularValues,
                                             Eigen::Matrix2d& d2E_div_dsigma2) const;
        
        virtual void checkEnergyVal(const TriangleSoup<dim>& data) const; // check with isometric case
        
    public:
        SymStretchEnergy(void);
        
    public:
        static void computeStressTensor(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& stressTensor);
    };
    
}

#endif /* SymStretchEnergy_hpp */
