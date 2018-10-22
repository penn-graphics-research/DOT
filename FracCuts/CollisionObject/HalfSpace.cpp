//
//  HalfSpace.cpp
//  FracCuts
//
//  Created by Minchen Li on 10/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "HalfSpace.hpp"

#include "SoftPenaltyCollisionEnergy.hpp"

namespace FracCuts {
    
    template<int dim>
    HalfSpace<dim>::HalfSpace(double p_Y, double p_stiffness, double p_friction)
    {
        Base::origin.setZero();
        Base::origin[1] = p_Y;
        
        Base::stiffness = p_stiffness;
        
        Base::friction = p_friction;
    }
    
    template<int dim>
    void HalfSpace<dim>::updateConstraints_OSQP(const TriangleSoup<dim>& mesh,
                                                std::vector<Eigen::Triplet<double>>& A_triplet,
                                                Eigen::VectorXd& l) const
    {
        int oldConstraintSize = l.size();
        
        A_triplet.reserve(A_triplet.size() + mesh.V.rows());
        for(int vI = 0; vI < mesh.V.rows(); ++vI) {
            A_triplet.emplace_back(oldConstraintSize + vI, vI * dim + 1, 1.0);
        }
        
        evaluateConstraints(mesh, l, -1.0);
    }
    
    template<int dim>
    void HalfSpace<dim>::evaluateConstraints(const TriangleSoup<dim>& mesh,
                                             Eigen::VectorXd& val, double coef) const
    {
        int oldConstraintSize = val.size();
        
        val.conservativeResize(oldConstraintSize + mesh.V.rows());
        for(int vI = 0; vI < mesh.V.rows(); ++vI) {
            val[oldConstraintSize + vI] = coef * (mesh.V(vI, 1) - Base::origin[1]);
        }
    }
    
    template<int dim>
    void HalfSpace<dim>::leftMultiplyConstraintJacobianT(const TriangleSoup<dim>& mesh,
                                                         const Eigen::VectorXd& input,
                                                         Eigen::VectorXd& output_incremental) const
    {
        assert(input.size() == mesh.V.rows());
        assert(output_incremental.size() == mesh.V.rows() * dim);
        
        for(int vI = 0; vI < mesh.V.rows(); ++vI) {
            output_incremental[vI * dim + 1] += input[vI];
        }
    }
    
    template<int dim>
    void HalfSpace<dim>::addSoftPenalty(std::vector<double>& energyParams,
                                        std::vector<Energy<DIM>*>& energTerms) const
    {
        energyParams.emplace_back(1.0);
        energTerms.emplace_back(new SoftPenaltyCollisionEnergy<DIM>
                                (Base::friction, Base::origin[1], Base::stiffness));
    }
    
    template<int dim>
    void HalfSpace<dim>::outputConfig(std::ostream& os) const
    {
        os << "ground " << Base::friction << " " <<
            Base::origin[1] << " " << Base::stiffness << std::endl;
    }
    
    template<int dim>
    void HalfSpace<dim>::draw(Eigen::MatrixXd& V,
                              Eigen::MatrixXi& F,
                              Eigen::MatrixXd& color,
                              double extensionScale) const
    {
        Eigen::MatrixXd V_floor;
        Eigen::MatrixXi F_floor;
        
        double size = extensionScale * 10.0;
        int elemAmt = 200;
        double spacing = size / std::sqrt(elemAmt / 2.0);
        assert(size >= spacing);
        int gridSize = static_cast<int>(size / spacing) + 1;
        spacing = size / (gridSize - 1);
        
        V_floor.resize(gridSize * gridSize, 3);
        for(int rowI = 0; rowI < gridSize; rowI++)
        {
            for(int colI = 0; colI < gridSize; colI++)
            {
                int vI = rowI * gridSize + colI;
                V_floor.row(vI) = Eigen::RowVector3d(spacing * colI - size / 2.0,
                                                     Base::origin[1],
                                                     spacing * rowI - size / 2.0);
            }
        }
        
        F_floor.resize((gridSize - 1) * (gridSize - 1) * 2, 3);
        for(int rowI = 0; rowI < gridSize - 1; rowI++)
        {
            for(int colI = 0; colI < gridSize - 1; colI++)
            {
                int squareI = rowI * (gridSize - 1) + colI;
                F_floor.row(squareI * 2) = Eigen::Vector3i(rowI * gridSize + colI,
                                                           (rowI + 1) * gridSize + colI,
                                                           (rowI + 1) * gridSize + colI + 1);
                F_floor.row(squareI * 2 + 1) = Eigen::Vector3i(rowI * gridSize + colI,
                                                               (rowI + 1) * gridSize + colI + 1,
                                                               rowI * gridSize + colI + 1);
            }
        }
        
        int oldVSize = V.rows();
        V.conservativeResize(oldVSize + V_floor.rows(), 3);
        V.bottomRows(V_floor.rows()) = V_floor;
        
        int oldFSize = F.rows();
        F.conservativeResize(oldFSize + F_floor.rows(), 3);
        F.bottomRows(F_floor.rows()) = F_floor;
        F.bottomRows(F_floor.rows()).array() += oldVSize;
        
        color.conservativeResize(color.rows() + F_floor.rows(), 3);
        color.bottomRows(F_floor.rows()).setConstant(0.9);
    }

    template class HalfSpace<DIM>;
}
