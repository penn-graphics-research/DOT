//
//  NeoHookeanEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/19/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef NeoHookeanEnergy_hpp
#define NeoHookeanEnergy_hpp

#include "Energy.hpp"

namespace FracCuts {
    
    template<int dim>
    class NeoHookeanEnergy : public Energy<dim>
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup<dim>& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        
        virtual void compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                               double& E) const;
        virtual void compute_dE_div_dsigma(const Eigen::Vector2d& singularValues,
                                           Eigen::Vector2d& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::Vector2d& singularValues,
                                             Eigen::Matrix2d& d2E_div_dsigma2) const;
        virtual void compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                       const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                       Eigen::Matrix<double, dim, dim>& dE_div_dF) const;
        
        virtual void checkEnergyVal(const TriangleSoup<dim>& data) const; // check with isometric case
        
        virtual void getBulkModulus(double& bulkModulus);
        
    public:
        NeoHookeanEnergy(double YM = 100.0, double PR = 0.4); //TODO: organize material parameters

    protected:
        const double u, lambda;
    };
    
}

#endif /* NeoHookeanEnergy_hpp */
