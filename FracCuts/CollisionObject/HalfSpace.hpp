//
//  HalfSpace.hpp
//  FracCuts
//
//  Created by Minchen Li on 10/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef HalfSpace_hpp
#define HalfSpace_hpp

#include "CollisionObject.h"

#include "Energy.hpp"

namespace FracCuts {
    
    template<int dim>
    class HalfSpace : public CollisionObject<dim>
    {
        typedef CollisionObject<dim> Base;
        
    protected:
        Eigen::Matrix<double, dim, 1> normal;
        double D;
        // normal[0] x + normal[1] y + normal[2] z + D = 0
        
        Eigen::Matrix<double, dim, dim> rotMtr; // for visualization
        
    public:
        HalfSpace(const Eigen::Matrix<double, dim, 1>& origin,
                  const Eigen::Matrix<double, dim, 1>& normal,
                  double p_stiffness, double p_friction);
        
        HalfSpace(double p_Y, double p_stiffness, double p_friction);
        
        void init(const Eigen::Matrix<double, dim, 1>& origin,
                  const Eigen::Matrix<double, dim, 1>& normal,
                  double p_stiffness, double p_friction);
        
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
                                                     Eigen::VectorXd& output) const;
        
        virtual void filterSearchDir_OSQP(const TriangleSoup<dim>& mesh,
                                          Eigen::VectorXd& searchDir);
        
        virtual void addSoftPenalty(std::vector<double>& energyParams,
                                    std::vector<Energy<DIM>*>& energTerms) const;
        
        virtual void outputConfig(std::ostream& os) const;
        
        virtual void draw(Eigen::MatrixXd& V,
                          Eigen::MatrixXi& F,
                          Eigen::MatrixXd& color,
                          double extensionScale = 1.0) const;
    };
    
}


#endif /* HalfSpace_hpp */
