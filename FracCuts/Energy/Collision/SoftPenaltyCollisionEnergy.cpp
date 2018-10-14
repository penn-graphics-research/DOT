//
//  SoftPenaltyCollisionEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 10/14/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "SoftPenaltyCollisionEnergy.hpp"

namespace FracCuts {
    
    template<int dim>
    void SoftPenaltyCollisionEnergy<dim>::
    computeEnergyVal(const TriangleSoup<dim>& data, bool redoSVD,
                     std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                     std::vector<Eigen::Matrix<double, dim, dim>>& F,
                     double& energyVal) const
    {
        energyVal = 0.0;
        if(friction) {
            for(int vI = 0; vI < data.V.rows(); vI++) {
                if(data.V(vI, 1) <= floorY) {
                    Eigen::Matrix<double, 1, dim> p = data.V.row(vI);
                    p[1] = floorY;
                    energyVal += 0.5 * k * (data.V.row(vI) - p).squaredNorm();
                }
            }
        }
        else {
            for(int vI = 0; vI < data.V.rows(); vI++) {
                if(data.V(vI, 1) <= floorY) {
                    energyVal += 0.5 * k * (data.V(vI, 1) - floorY) * (data.V(vI, 1) - floorY);
                }
            }
        }
    }
    
    template<int dim>
    void SoftPenaltyCollisionEnergy<dim>::
    computeGradient(const TriangleSoup<dim>& data, bool redoSVD,
                    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                    std::vector<Eigen::Matrix<double, dim, dim>>& F,
                    Eigen::VectorXd& gradient) const
    {
        gradient.conservativeResize(data.V.rows() * dim);
        gradient.setZero();
        if(friction) {
            for(int vI = 0; vI < data.V.rows(); vI++) {
                if(data.V(vI, 1) <= floorY) {
                    Eigen::Matrix<double, 1, dim> p = data.V.row(vI);
                    p[1] = floorY;
                    gradient.segment<dim>(vI * dim) += k * (data.V.row(vI) - p).transpose();
                }
            }
        }
        else {
            for(int vI = 0; vI < data.V.rows(); vI++) {
                if(data.V(vI, 1) <= floorY) {
                    gradient[vI * dim + 1] += k * (data.V(vI, 1) - floorY);
                }
            }
        }
    }
    
    template<int dim>
    void SoftPenaltyCollisionEnergy<dim>::
    computeHessian(const TriangleSoup<dim>& data, bool redoSVD,
                   std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                   std::vector<Eigen::Matrix<double, dim, dim>>& F,
                   double coef,
                   LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                   bool projectSPD) const
    {
        double diagVal = k * coef;
        if(friction) {
            for(int vI = 0; vI < data.V.rows(); vI++) {
                if(data.V(vI, 1) <= floorY) {
                    int ind0 = vI * dim;
                    int ind1 = ind0 + 1;
                    linSysSolver->addCoeff(ind0, ind0, diagVal);
                    linSysSolver->addCoeff(ind1, ind1, diagVal);
                    if(dim == 3) {
                        int ind2 = ind0 + 2;
                        linSysSolver->addCoeff(ind2, ind2, diagVal);
                    }
                }
            }
        }
        else {
            for(int vI = 0; vI < data.V.rows(); vI++) {
                if(data.V(vI, 1) <= floorY) {
                    int ind1 = vI * dim + 1;
                    linSysSolver->addCoeff(ind1, ind1, diagVal);
                }
            }
        }
    }
    
    template<int dim>
    SoftPenaltyCollisionEnergy<dim>::
    SoftPenaltyCollisionEnergy(bool p_friction,
                               double p_floorY, double p_k) :
    Energy<dim>(true), friction(p_friction), floorY(p_floorY), k(p_k)
    {
    }
    
    template class SoftPenaltyCollisionEnergy<DIM>;
}
