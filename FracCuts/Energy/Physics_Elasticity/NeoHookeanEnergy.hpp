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
    
    class NeoHookeanEnergy : public Energy
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        
        virtual void compute_E(const Eigen::VectorXd& singularValues,
                               double& E) const;
        virtual void compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                           Eigen::VectorXd& dE_div_dsigma) const;
        virtual void compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                             Eigen::MatrixXd& d2E_div_dsigma2) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const; // check with isometric case
        
    public:
        NeoHookeanEnergy(double YM = 100.0, double PR = 0.4); //TODO: organize material parameters

    protected:
        const double u, lambda;
    };
    
}

#endif /* NeoHookeanEnergy_hpp */
