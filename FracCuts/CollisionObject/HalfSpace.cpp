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
    HalfSpace<dim>::HalfSpace(const Eigen::Matrix<double, dim, 1>& p_origin,
                              const Eigen::Matrix<double, dim, 1>& p_normal,
                              double p_stiffness, double p_friction)
    {
        init(p_origin, p_normal, p_stiffness, p_friction);
    }
    
    template<int dim>
    HalfSpace<dim>::HalfSpace(double p_Y, double p_stiffness, double p_friction)
    {
        Eigen::Matrix<double, dim, 1> p_origin, p_normal;
        p_origin.setZero();
        p_origin[1] = p_Y;
        p_normal.setZero();
        p_normal[1] = 1.0;
        
        init(p_origin, p_normal, p_stiffness, p_friction);
    }
    
    template<int dim>
    void HalfSpace<dim>::init(const Eigen::Matrix<double, dim, 1>& p_origin,
                              const Eigen::Matrix<double, dim, 1>& p_normal,
                              double p_stiffness, double p_friction)
    {
        Base::origin = p_origin;
        normal = p_normal;
        normal.normalize();
        D = -normal.dot(Base::origin);
        
        Eigen::Matrix<double, dim, 1> defaultN;
        defaultN.setZero();
        defaultN[1] = 1.0;
        double rotAngle = std::acos(std::max(-1.0, std::min(1.0, normal.dot(defaultN))));
        if(rotAngle < 1.0e-3) {
            rotMtr.setIdentity();
        }
        else {
            rotMtr = Eigen::AngleAxisd(rotAngle, (defaultN.cross(normal)).normalized());
        }
        
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
            A_triplet.emplace_back(oldConstraintSize + vI, vI * dim, normal[0]);
            A_triplet.emplace_back(oldConstraintSize + vI, vI * dim + 1, normal[1]);
            if(dim == 3) {
                A_triplet.emplace_back(oldConstraintSize + vI, vI * dim + 2, normal[2]);
            }
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
            val[oldConstraintSize + vI] = coef * (normal.transpose().dot(mesh.V.row(vI)) + D);
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
            output_incremental.segment<dim>(vI * dim) += input[vI] * normal;
        }
    }
    
    template<int dim>
    void HalfSpace<dim>::addSoftPenalty(std::vector<double>& energyParams,
                                        std::vector<Energy<DIM>*>& energTerms) const
    {
        energyParams.emplace_back(1.0);
        energTerms.emplace_back(new SoftPenaltyCollisionEnergy<DIM>
                                (Base::friction, Base::origin[1], Base::stiffness));
        //TODO: different penalty term
    }
    
    template<int dim>
    void HalfSpace<dim>::outputConfig(std::ostream& os) const
    {
        os << "ground " << Base::friction << " " <<
            Base::origin[1] << " " << Base::stiffness << std::endl;
        //TODO: different input
    }
    
    template<int dim>
    void HalfSpace<dim>::draw(Eigen::MatrixXd& V,
                              Eigen::MatrixXi& F,
                              Eigen::MatrixXd& color,
                              double extensionScale) const
    {
        //TODO: transform rather than redraw everytime
        
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
                Eigen::RowVector3d coord(spacing * colI - size / 2.0,
                                         Base::origin[1],
                                         spacing * rowI - size / 2.0);
                V_floor.row(vI) = (rotMtr * (coord.transpose() - Base::origin) +
                                   Base::origin).transpose();
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
