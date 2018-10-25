//
//  CollisionObject.h
//  FracCuts
//
//  Created by Minchen Li on 10/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef CollisionObject_h
#define CollisionObject_h

#include "Energy.hpp"
#include "TriangleSoup.hpp"

#include <Eigen/Eigen>

#include <vector>

namespace FracCuts {
    
    template<int dim>
    class CollisionObject
    {
    public:
        Eigen::Matrix<double, dim, 1> origin;
        double stiffness;
        double friction;
        std::set<int> activeSet, activeSet_next;
        
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        
    public:
        virtual ~CollisionObject(void) {};
        
    public:
        virtual void clearActiveSet(void) {
            activeSet.clear();
            activeSet_next.clear();
        }
        virtual void updateActiveSet(void) {
            activeSet = activeSet_next;
        }
        
    public:
        virtual void updateConstraints_OSQP(const TriangleSoup<dim>& mesh,
                                            std::vector<Eigen::Triplet<double>>& A_triplet,
                                            Eigen::VectorXd& l) const = 0;
        
        virtual void evaluateConstraints(const TriangleSoup<dim>& mesh,
                                         Eigen::VectorXd& val, double coef = 1.0) const = 0;
        virtual void evaluateConstraints_all(const TriangleSoup<dim>& mesh,
                                             Eigen::VectorXd& val, double coef = 1.0) const = 0;
        
        virtual void leftMultiplyConstraintJacobianT(const TriangleSoup<dim>& mesh,
                                                     const Eigen::VectorXd& input,
                                                     Eigen::VectorXd& output) const = 0;
        
        virtual void filterSearchDir_OSQP(const TriangleSoup<dim>& mesh,
                                          Eigen::VectorXd& searchDir) = 0;
        
        virtual void addSoftPenalty(std::vector<double>& energyParams,
                                    std::vector<Energy<DIM>*>& energTerms) const = 0;
        
        virtual void outputConfig(std::ostream& os) const = 0;
        
        virtual void initRenderingData(double extensionScale = 1.0) = 0;
        
        virtual void draw(Eigen::MatrixXd& p_V,
                          Eigen::MatrixXi& p_F,
                          Eigen::MatrixXd& color,
                          double extensionScale = 1.0) const
        {
            int oldVSize = p_V.rows();
            p_V.conservativeResize(oldVSize + V.rows(), 3);
            p_V.bottomRows(V.rows()) = V;
            
            int oldFSize = p_F.rows();
            p_F.conservativeResize(oldFSize + F.rows(), 3);
            p_F.bottomRows(F.rows()) = F;
            p_F.bottomRows(F.rows()).array() += oldVSize;
            
            color.conservativeResize(color.rows() + F.rows(), 3);
            color.bottomRows(F.rows()).setConstant(0.9);
        }
    };
    
}


#endif /* CollisionObject_h */
