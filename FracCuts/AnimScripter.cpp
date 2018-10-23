//
//  AnimScripter.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "AnimScripter.hpp"

namespace FracCuts {
    
    template<int dim>
    const std::vector<std::string> AnimScripter<dim>::animScriptTypeStrs = {
        "null", "hang", "stretch", "squash", "bend", "onepoint", "random", "fall"
    };
    
    template<int dim>
    AnimScripter<dim>::AnimScripter(AnimScriptType p_animScriptType) :
        animScriptType(p_animScriptType)
    {}
    
    template<int dim>
    void AnimScripter<dim>::initAnimScript(TriangleSoup<dim>& mesh)
    {
        switch (animScriptType) {
            case AST_NULL:
                break;
                
            case AST_HANG:
                mesh.resetFixedVert();
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI.back());
                }
                break;
                
            case AST_STRETCH: {
                mesh.resetFixedVert();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(const auto bVI : borderI) {
                        velocity_handleVerts[bVI].setZero();
                        velocity_handleVerts[bVI][0] = std::pow(-1.0, bI) * -0.1;
                    }
                    bI++;
                }
                break;
            }
                
            case AST_SQUASH: {
                mesh.resetFixedVert();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(const auto bVI : borderI) {
                        velocity_handleVerts[bVI].setZero();
                        velocity_handleVerts[bVI][0] = std::pow(-1.0, bI) * 0.03;
                    }
                    bI++;
                }
                break;
            }
                
            case AST_BEND: {
                mesh.resetFixedVert();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(int bVI = 0; bVI + 1 < borderI.size(); bVI++) {
                        angVel_handleVerts[borderI[bVI]] = std::pow(-1.0, bI) * -0.05 * M_PI;
                        rotCenter_handleVerts[borderI[bVI]] = mesh.V.row(borderI.back()).transpose();
                    }
                    bI++;
                }
                break;
            }
                
            case AST_ONEPOINT: {
                const Eigen::RowVector3d center = mesh.bbox.colwise().mean();
                mesh.V.rowwise() = center.leftCols(dim);
                mesh.V.col(1).array() += (mesh.bbox(1, 1) - mesh.bbox(0, 1)) / 2.0;
                break;
            }
                
            case AST_RANDOM: {
                mesh.V.setRandom();
                mesh.V /= 2.0;
                Eigen::RowVector3d offset = mesh.bbox.colwise().mean();
                offset[1] += (mesh.bbox(1, 1) - mesh.bbox(0, 1)) / 2.0;
                offset.leftCols(dim) -= mesh.V.row(0);
                mesh.V.rowwise() += offset.leftCols(dim);
                break;
            }
                
            case AST_FALL: {
                mesh.V.col(1).array() += 0.5 * (mesh.V.colwise().maxCoeff() -
                                                mesh.V.colwise().minCoeff()).norm();
                mesh.resetFixedVert();
                break;
            }
                
            default:
                assert(0 && "invalid animScriptType");
                break;
        }
    }
    
    template<int dim>
    void AnimScripter<dim>::stepAnimScript(TriangleSoup<dim>& mesh, double dt,
                                           const std::vector<Energy<dim>*>& energyTerms)
    {
        Eigen::VectorXd searchDir(mesh.V.rows() * dim);
        searchDir.setZero();
        switch (animScriptType) {
            case AST_NULL:
                break;
                
            case AST_HANG:
                break;
                
            case AST_STRETCH:
            case AST_SQUASH:
                for(const auto& movingVerts : velocity_handleVerts) {
                    searchDir.segment<dim>(movingVerts.first * dim) =
                        movingVerts.second * dt;
                }
                break;
                
            case AST_BEND:
                for(const auto& movingVerts : angVel_handleVerts) {
                    const Eigen::Matrix3d rotMtr =
                        Eigen::AngleAxis<double>(movingVerts.second * dt,
                                                 Eigen::Vector3d::UnitZ()).toRotationMatrix();
                    const auto rotCenter = rotCenter_handleVerts.find(movingVerts.first);
                    assert(rotCenter != rotCenter_handleVerts.end());
                    
                    searchDir.segment<dim>(movingVerts.first * dim) = (rotMtr.block<dim, dim>(0, 0) * (mesh.V.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh.V.row(movingVerts.first).transpose();
                }
                break;
                
            case AST_ONEPOINT:
                break;
                
            case AST_RANDOM:
                break;
                
            case AST_FALL:
                break;
                
            default:
                assert(0 && "invalid animScriptType");
                break;
        }
        
        double stepSize = 1.0;
        for(const auto& energyTermI : energyTerms) {
            energyTermI->filterStepSize(mesh, searchDir, stepSize);
        }
        
        for(int vI = 0; vI < mesh.V.rows(); vI++) {
            mesh.V.row(vI) += stepSize * searchDir.segment<dim>(vI * dim).transpose();
        }
        //TODO: continue to move in each Newton's iteration if not at destiny
    }
    
    template<int dim>
    void AnimScripter<dim>::setAnimScriptType(AnimScriptType p_animScriptType)
    {
        animScriptType = p_animScriptType;
    }
    
    template<int dim>
    AnimScriptType AnimScripter<dim>::getAnimScriptTypeByStr(const std::string& str)
    {
        for(int i = 0; i < animScriptTypeStrs.size(); i++) {
            if(str == animScriptTypeStrs[i]) {
                return AnimScriptType(i);
            }
        }
        return AST_NULL;
    }
    template<int dim>
    std::string AnimScripter<dim>::getStrByAnimScriptType(AnimScriptType animScriptType)
    {
        assert(animScriptType < animScriptTypeStrs.size());
        return animScriptTypeStrs[animScriptType];
    }
    
    template class AnimScripter<DIM>;
    
}
