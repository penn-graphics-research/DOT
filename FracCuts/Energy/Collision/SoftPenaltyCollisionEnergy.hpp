//
//  SoftPenaltyCollisionEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 10/14/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef SoftPenaltyCollisionEnergy_hpp
#define SoftPenaltyCollisionEnergy_hpp

#include "Energy.hpp"

namespace FracCuts {
    
    template<int dim>
    class SoftPenaltyCollisionEnergy : public Energy<dim>
    {
    protected:
        bool friction;
        double floorY, k;
        
    public:
        virtual void computeEnergyVal(const TriangleSoup<dim>& data, bool redoSVD,
                                      std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                      std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                      double coef,
                                      double& energyVal) const;
        virtual void computeGradient(const TriangleSoup<dim>& data, bool redoSVD,
                                     std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                     std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                     double coef,
                                     Eigen::VectorXd& gradient) const;
        virtual void computeHessian(const TriangleSoup<dim>& data, bool redoSVD,
                                    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                    std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                    double coef,
                                    LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                    bool projectSPD = true) const;
        
    public:
        SoftPenaltyCollisionEnergy(bool p_friction = false,
                                   double p_floorY = 0.0, double p_k = 1.0);
    };
}

#endif /* SoftPenaltyCollisionEnergy_hpp */
