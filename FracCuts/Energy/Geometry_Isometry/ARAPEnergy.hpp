//
//  ARAPEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 9/5/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef ARAPEnergy_hpp
#define ARAPEnergy_hpp

#include "Energy.hpp"

namespace FracCuts
{
    
    template<int dim>
    class ARAPEnergy : public Energy<dim>
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup<dim>& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup<dim>& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;
        virtual void computePrecondMtr(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const;
        virtual void computeHessian(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const;
        
        virtual void checkEnergyVal(const TriangleSoup<dim>& data) const;
        
    public:
        ARAPEnergy(void);
    };
    
}

#endif /* ARAPEnergy_hpp */
