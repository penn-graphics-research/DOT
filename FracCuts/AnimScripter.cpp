//
//  AnimScripter.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "AnimScripter.hpp"

namespace FracCuts {
    
    AnimScripter::AnimScripter(AnimScriptType p_animScriptType) :
        animScriptType(p_animScriptType)
    {}
    
    void AnimScripter::initAnimScript(TriangleSoup& mesh)
    {
        switch (animScriptType) {
            case AST_NULL:
                break;
                
            case AST_HANG:
                mesh.fixedVert.clear();
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.fixedVert.insert(borderI.back());
                }
                break;
                
            case AST_STRETCH: {
                mesh.fixedVert.clear();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.fixedVert.insert(borderI.begin(), borderI.end());
                    handleVerts.emplace_back(borderI);
                    for(const auto bVI : borderI) {
                        velocity_handleVerts[bVI] << std::pow(-1.0, bI) * -0.1, 0.0;
                    }
                    bI++;
                }
                break;
            }
                
            case AST_SQUASH: {
                mesh.fixedVert.clear();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.fixedVert.insert(borderI.begin(), borderI.end());
                    handleVerts.emplace_back(borderI);
                    for(const auto bVI : borderI) {
                        velocity_handleVerts[bVI] << std::pow(-1.0, bI) * 0.03, 0.0;
                    }
                    bI++;
                }
                break;
            }
                
            case AST_BEND: {
                mesh.fixedVert.clear();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.fixedVert.insert(borderI.begin(), borderI.end());
                    handleVerts.emplace_back(borderI);
                    for(int bVI = 0; bVI + 1 < borderI.size(); bVI++) {
                        angVel_handleVerts[borderI[bVI]] = std::pow(-1.0, bI) * -0.05 * M_PI;
                        rotCenter_handleVerts[borderI[bVI]] = mesh.V.row(borderI.back()).transpose();
                    }
                    bI++;
                }
                break;
            }
                
            case AST_ONEPOINT:
                mesh.V.setZero();
                break;
                
            case AST_RANDOM: {
                mesh.V.setRandom();
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
    
}
