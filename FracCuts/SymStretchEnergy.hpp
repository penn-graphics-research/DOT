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
    
    class SymStretchEnergy : public Energy
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void getEnergyValByElemID(const TriangleSoup& data, int elemI, double& energyVal, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                       Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const;
        
        virtual void getEnergyValPerVert(const TriangleSoup& data, Eigen::VectorXd& energyValPerVert) const;
        virtual void getMaxUnweightedEnergyValPerVert(const TriangleSoup& data, Eigen::VectorXd& MaxUnweightedEnergyValPerVert) const;
        virtual void computeLocalGradient(const TriangleSoup& data, Eigen::MatrixXd& localGradients) const;
        virtual void getDivGradPerElem(const TriangleSoup& data, Eigen::VectorXd& divGradPerElem) const;
        virtual void computeDivGradPerVert(const TriangleSoup& data, Eigen::VectorXd& divGradPerVert) const;
        virtual void computeLocalSearchDir(const TriangleSoup& data, Eigen::MatrixXd& localSearchDir) const;
        
        virtual void compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                           Eigen::VectorXd& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                             Eigen::VectorXd& d2E_div_dsigma2) const;
        
        // to prevent element inversion
        virtual void initStepSize(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const; // check with isometric case
        
    public:
        SymStretchEnergy(void);
        
    public:
        static void computeStressTensor(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& stressTensor);
    };
    
}

#endif /* SymStretchEnergy_hpp */
