//
//  AnimScripter.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "AnimScripter.hpp"

namespace FracCuts {
    
    const std::vector<std::string> AnimScripter::animScriptTypeStrs = {
        "null", "hang", "stretch", "squash", "bend", "onepoint", "random"
    };
    
    AnimScripter::AnimScripter(AnimScriptType p_animScriptType) :
        animScriptType(p_animScriptType)
    {}
    
    void AnimScripter::initAnimScript(TriangleSoup& mesh)
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
                        velocity_handleVerts[bVI] << std::pow(-1.0, bI) * -0.1, 0.0;
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
                        velocity_handleVerts[bVI] << std::pow(-1.0, bI) * 0.03, 0.0;
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
                mesh.V.rowwise() = center.leftCols(2);
                mesh.V.col(1).array() += (mesh.bbox(1, 1) - mesh.bbox(0, 1)) / 2.0;
                break;
            }
                
            case AST_RANDOM: {
                mesh.V.setRandom();
                mesh.V /= 2.0;
                Eigen::RowVector3d offset = mesh.bbox.colwise().mean();
                offset[1] += (mesh.bbox(1, 1) - mesh.bbox(0, 1)) / 2.0;
                offset.leftCols(2) -= mesh.V.row(0);
                mesh.V.rowwise() += offset.leftCols(2);
                break;
            }
                
            default:
                assert(0 && "invalid animScriptType");
                break;
        }
    }
    
    void AnimScripter::stepAnimScript(TriangleSoup& mesh, double dt)
    {
        switch (animScriptType) {
            case AST_NULL:
                break;
                
            case AST_HANG:
                break;
                
            case AST_STRETCH:
            case AST_SQUASH:
                for(const auto& movingVerts : velocity_handleVerts) {
                    mesh.V.row(movingVerts.first) += movingVerts.second.transpose() * dt;
                }
                break;
                
            case AST_BEND:
                for(const auto& movingVerts : angVel_handleVerts) {
                    const Eigen::Matrix2d rotMtr =
                        Eigen::Rotation2D<double>(movingVerts.second * dt).toRotationMatrix();
                    const auto rotCenter = rotCenter_handleVerts.find(movingVerts.first);
                    assert(rotCenter != rotCenter_handleVerts.end());
                    mesh.V.row(movingVerts.first) = rotMtr * (mesh.V.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second;
                }
                break;
                
            case AST_ONEPOINT:
                break;
                
            case AST_RANDOM:
                break;
                
            default:
                assert(0 && "invalid animScriptType");
                break;
        }
    }
    
    void AnimScripter::setAnimScriptType(AnimScriptType p_animScriptType)
    {
        animScriptType = p_animScriptType;
    }
    
    AnimScriptType AnimScripter::getAnimScriptTypeByStr(const std::string& str)
    {
        for(int i = 0; i < animScriptTypeStrs.size(); i++) {
            if(str == animScriptTypeStrs[i]) {
                return AnimScriptType(i);
            }
        }
        return AST_NULL;
    }
    std::string AnimScripter::getStrByAnimScriptType(AnimScriptType animScriptType)
    {
        assert(animScriptType < animScriptTypeStrs.size());
        return animScriptTypeStrs[animScriptType];
    }
    
}
