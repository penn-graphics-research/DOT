//
//  AnimScripter.cpp
//  DOT
//
//  Created by Minchen Li on 6/20/18.
//

#include "AnimScripter.hpp"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

namespace DOT {
    
    template<int dim>
    const std::vector<std::string> AnimScripter<dim>::animScriptTypeStrs = {
        "null", "scaleF", "hang", "stretch", "squash", "stretchnsquash",
        "bend", "twist", "twistnstretch", "twistnsns", "twistnsns_old",
        "rubberBandPull", "onepoint", "random", "fall"
    };
    
    template<int dim>
    AnimScripter<dim>::AnimScripter(AnimScriptType p_animScriptType) :
        animScriptType(p_animScriptType)
    {}
    
    template<int dim>
    void AnimScripter<dim>::initAnimScript(Mesh<dim>& mesh)
    {
        switch (animScriptType) {
            case AST_NULL:
                break;
                
            case AST_SCALEF: {
                mesh.resetFixedVert();
//                handleVerts.resize(0);
//                int bI = 0;
//                for(const auto borderI : mesh.borderVerts_primitive) {
//                    mesh.addFixedVert(borderI);
//                    handleVerts.emplace_back(borderI);
//                    bI++;
//                }
                
                Eigen::Matrix3d M;
//                M << 1.5, 0.5, 0.0,
//                0.0, 0.5, -0.5,
//                0.0, 0.0, 1.0;
                M <<
                1.5, 0.0, 0.0,
                0.0, 1.5, 0.0,
                0.0, 0.0, 1.5;
                for(int i = 0; i < mesh.V.rows(); ++i) {
                    mesh.V.row(i) = (M * mesh.V.row(i).transpose()).transpose();
                }
                
                break;
            }
                
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
                
            case AST_STRETCHNSQUASH: {
                mesh.resetFixedVert();
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(const auto bVI : borderI) {
                        velocity_handleVerts[bVI].setZero();
                        velocity_handleVerts[bVI][0] = std::pow(-1.0, bI) * -0.9;
                    }
                    bI++;
                }
                
                velocityTurningPoints.first = mesh.borderVerts_primitive[0].front();
                velocityTurningPoints.second(0, 0) =
                    mesh.V(velocityTurningPoints.first, 0) - 0.8;
                velocityTurningPoints.second(0, 1) =
                    mesh.V(velocityTurningPoints.first, 0) + 0.4;
                
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
                
            case AST_TWIST: {
                mesh.resetFixedVert();
                
                const Eigen::RowVector3d rotCenter = mesh.bbox.colwise().mean();
                
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(int bVI = 0; bVI < borderI.size(); bVI++) {
                        angVel_handleVerts[borderI[bVI]] = std::pow(-1.0, bI) * -0.1 * M_PI;
                        rotCenter_handleVerts[borderI[bVI]] = rotCenter.transpose().topRows(dim);
                    }
                    bI++;
                }
                break;
            }
                
            case AST_TWISTNSTRETCH: {
                mesh.resetFixedVert();
                
                const Eigen::RowVector3d rotCenter = mesh.bbox.colwise().mean();
                
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(int bVI = 0; bVI < borderI.size(); bVI++) {
                        angVel_handleVerts[borderI[bVI]] = std::pow(-1.0, bI) * -0.1 * M_PI;
                        rotCenter_handleVerts[borderI[bVI]] = rotCenter.transpose().topRows(dim);
                        
                        velocity_handleVerts[borderI[bVI]].setZero();
                        velocity_handleVerts[borderI[bVI]][0] = std::pow(-1.0, bI) * -0.1;
                    }
                    bI++;
                }
                break;
            }
                
            case AST_TWISTNSNS_OLD:
            case AST_TWISTNSNS: {
                mesh.resetFixedVert();
                
                const Eigen::RowVector3d rotCenter = mesh.bbox.colwise().mean();
                
                handleVerts.resize(0);
                int bI = 0;
                for(const auto borderI : mesh.borderVerts_primitive) {
                    mesh.addFixedVert(borderI);
                    handleVerts.emplace_back(borderI);
                    for(int bVI = 0; bVI < borderI.size(); bVI++) {
                        angVel_handleVerts[borderI[bVI]] = std::pow(-1.0, bI) * -0.4 * M_PI;
                        rotCenter_handleVerts[borderI[bVI]] = rotCenter.transpose().topRows(dim);
                        
                        velocity_handleVerts[borderI[bVI]].setZero();
                        if(animScriptType == AST_TWISTNSNS) {
                            velocity_handleVerts[borderI[bVI]][0] = std::pow(-1.0, bI) * -1.2;
                        }
                        else if (animScriptType == AST_TWISTNSNS_OLD) {
                            velocity_handleVerts[borderI[bVI]][0] = std::pow(-1.0, bI) * -0.9;
                        }
                    }
                    bI++;
                }
                
                velocityTurningPoints.first = mesh.borderVerts_primitive[0].front();
                if(animScriptType == AST_TWISTNSNS) {
                    velocityTurningPoints.second(0, 0) =
                        mesh.V(velocityTurningPoints.first, 0) - 1.2;
                }
                else {
                    velocityTurningPoints.second(0, 0) =
                        mesh.V(velocityTurningPoints.first, 0) - 0.8;
                }
                velocityTurningPoints.second(0, 1) =
                    mesh.V(velocityTurningPoints.first, 0) + 0.4;
                break;
            }
                
            case AST_RUBBERBANDPULL: {
                mesh.resetFixedVert();
                handleVerts.resize(0);
                handleVerts.resize(2);
                
                // grab top, waist, and bottom:
                Eigen::RowVectorXd bottomLeft = mesh.V.colwise().minCoeff();
                Eigen::RowVectorXd topRight = mesh.V.colwise().maxCoeff();
                Eigen::RowVectorXd range = topRight - bottomLeft;
                bool turningPointSet = false;
                for(int vI = 0; vI < mesh.V.rows(); ++vI) {
                    if(mesh.V(vI, 1) < bottomLeft[1] + range[1] * 0.02) {
                        mesh.addFixedVert(vI);
                        handleVerts[1].emplace_back(vI);
                        velocity_handleVerts[vI].setZero();
                        velocity_handleVerts[vI][1] = -0.2;
                    }
                    else if(mesh.V(vI, 1) > topRight[1] - range[1] * 0.02) {
                        mesh.addFixedVert(vI);
                        handleVerts[1].emplace_back(vI);
                        velocity_handleVerts[vI].setZero();
                        velocity_handleVerts[vI][1] = 0.2;
                    }
                    else if((mesh.V(vI, 1) < topRight[1] - range[1] * 0.48) &&
                            (mesh.V(vI, 1) > bottomLeft[1] + range[1] * 0.48))
                    {
                        mesh.addFixedVert(vI);
                        handleVerts[0].emplace_back(vI);
                        velocity_handleVerts[vI].setZero();
                        velocity_handleVerts[vI][0] = -2.5; // previously -2.0
                        if(!turningPointSet) {
                            turningPointSet = true;
                            velocityTurningPoints.first = vI;
                            velocityTurningPoints.second(0, 0) = mesh.V(vI, 0) - 5.0;
                        }
                    }
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
    int AnimScripter<dim>::stepAnimScript(Mesh<dim>& mesh, double dt,
                                           const std::vector<Energy<dim>*>& energyTerms)
    {
        Eigen::VectorXd searchDir(mesh.V.rows() * dim);
        searchDir.setZero();
        int returnFlag = 0;
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
                
            case AST_STRETCHNSQUASH: {
                bool flip = false;
                if((mesh.V(velocityTurningPoints.first, 0) <=
                    velocityTurningPoints.second(0, 0)) ||
                   (mesh.V(velocityTurningPoints.first, 0) >=
                    velocityTurningPoints.second(0, 1)))
                {
                    flip = true;
                }
                for(auto& movingVerts : velocity_handleVerts) {
                    if(flip) {
                        movingVerts.second[0] *= -1.0;
                    }
                    searchDir.segment<dim>(movingVerts.first * dim) =
                        movingVerts.second * dt;
                }
                break;
            }
                
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
                
            case AST_TWIST:
                for(const auto& movingVerts : angVel_handleVerts) {
                    const Eigen::Matrix3d rotMtr =
                    Eigen::AngleAxis<double>(movingVerts.second * dt,
                                             Eigen::Vector3d::UnitX()).toRotationMatrix();
                    const auto rotCenter = rotCenter_handleVerts.find(movingVerts.first);
                    assert(rotCenter != rotCenter_handleVerts.end());
                    
                    searchDir.segment<dim>(movingVerts.first * dim) = (rotMtr.block<dim, dim>(0, 0) * (mesh.V.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh.V.row(movingVerts.first).transpose();
                }
                break;
                
            case AST_TWISTNSTRETCH: {
                for(const auto& movingVerts : angVel_handleVerts) {
                    const Eigen::Matrix3d rotMtr =
                    Eigen::AngleAxis<double>(movingVerts.second * dt,
                                             Eigen::Vector3d::UnitX()).toRotationMatrix();
                    const auto rotCenter = rotCenter_handleVerts.find(movingVerts.first);
                    assert(rotCenter != rotCenter_handleVerts.end());
                    
                    searchDir.segment<dim>(movingVerts.first * dim) = (rotMtr.block<dim, dim>(0, 0) * (mesh.V.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh.V.row(movingVerts.first).transpose();
                }
                for(const auto& movingVerts : velocity_handleVerts) {
                    searchDir.segment<dim>(movingVerts.first * dim) += movingVerts.second * dt;
                }
                break;
            }
                
            case AST_TWISTNSNS_OLD:
            case AST_TWISTNSNS: {
                bool flip = false;
                if((mesh.V(velocityTurningPoints.first, 0) <=
                    velocityTurningPoints.second(0, 0)) ||
                   (mesh.V(velocityTurningPoints.first, 0) >=
                    velocityTurningPoints.second(0, 1)))
                {
                    flip = true;
                }
                
                for(auto& movingVerts : angVel_handleVerts) {
//                    if(flip) {
//                        movingVerts.second *= -1.0;
//                    }
                    
                    const Eigen::Matrix3d rotMtr =
                    Eigen::AngleAxis<double>(movingVerts.second * dt,
                                             Eigen::Vector3d::UnitX()).toRotationMatrix();
                    const auto rotCenter = rotCenter_handleVerts.find(movingVerts.first);
                    assert(rotCenter != rotCenter_handleVerts.end());
                    
                    searchDir.segment<dim>(movingVerts.first * dim) = (rotMtr.block<dim, dim>(0, 0) * (mesh.V.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh.V.row(movingVerts.first).transpose();
                }
                for(auto& movingVerts : velocity_handleVerts) {
                    if(flip) {
                        movingVerts.second[0] *= -1.0;
                    }
                    searchDir.segment<dim>(movingVerts.first * dim) += movingVerts.second * dt;
                }
                break;
            }
                
            case AST_RUBBERBANDPULL: {
                if(mesh.V(velocityTurningPoints.first, 0) <=
                   velocityTurningPoints.second(0, 0))
                {
                    velocityTurningPoints.second(0, 0) = -__DBL_MAX__;
                    for(const auto& vI : handleVerts[0]) {
                        mesh.removeFixedVert(vI);
                        velocity_handleVerts[vI].setZero();
                    }
                    for(const auto& vI : handleVerts[1]) {
                        velocity_handleVerts[vI].setZero();
                    }
                    returnFlag = 1;
                }
                for(const auto& movingVerts : velocity_handleVerts) {
                    searchDir.segment<dim>(movingVerts.first * dim) =
                        movingVerts.second * dt;
                }
                break;
            }
                
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
#ifdef USE_TBB
        tbb::parallel_for(0, (int)mesh.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < mesh.V.rows(); ++vI)
#endif
        {
            mesh.V.row(vI) += stepSize * searchDir.segment<dim>(vI * dim).transpose();
        }
#ifdef USE_TBB
        );
#endif
        
        return returnFlag;
    }
    
    template<int dim>
    void AnimScripter<dim>::setAnimScriptType(AnimScriptType p_animScriptType)
    {
        animScriptType = p_animScriptType;
    }
    
    template<int dim>
    const std::vector<std::vector<int>>& AnimScripter<dim>::getHandleVerts(void) const
    {
        return handleVerts;
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
