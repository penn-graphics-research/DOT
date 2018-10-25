//
//  Sphere.hpp
//  FracCuts
//
//  Created by Minchen Li on 10/24/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Sphere_hpp
#define Sphere_hpp

#include "CollisionObject.h"

#include "Energy.hpp"

namespace FracCuts {
    
    template<int dim>
    class Sphere : public CollisionObject<dim>
    {
        typedef CollisionObject<dim> Base;
        
    protected:
        double radius, radius2;
        
    public:
        Sphere(const Eigen::Matrix<double, dim, 1>& origin,
               double p_radius, double p_stiffness, double p_friction);
        
    public:
        virtual void updateConstraints_OSQP(const TriangleSoup<dim>& mesh,
                                            std::vector<Eigen::Triplet<double>>& A_triplet,
                                            Eigen::VectorXd& l) const;
        
        virtual void evaluateConstraints(const TriangleSoup<dim>& mesh,
                                         Eigen::VectorXd& val, double coef = 1.0) const;
        virtual void evaluateConstraints_all(const TriangleSoup<dim>& mesh,
                                             Eigen::VectorXd& val, double coef = 1.0) const;
        
        virtual void leftMultiplyConstraintJacobianT(const TriangleSoup<dim>& mesh,
                                                     const Eigen::VectorXd& input,
                                                     Eigen::VectorXd& output_incremental) const;
        
        virtual void filterSearchDir_OSQP(const TriangleSoup<dim>& mesh,
                                          Eigen::VectorXd& searchDir);
        
        virtual void addSoftPenalty(std::vector<double>& energyParams,
                                    std::vector<Energy<DIM>*>& energTerms) const;
        
        virtual void outputConfig(std::ostream& os) const;
        
        virtual void initRenderingData(double extensionScale = 1.0);
    };
    
}

#endif /* Sphere_hpp */
