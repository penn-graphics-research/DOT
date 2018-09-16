//
//  Optimizer.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "SeparationEnergy.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
    #include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
    #include "PardisoSolver.hpp"
#else
    #include "EigenLibSolver.hpp"
#endif

#include "IglUtils.hpp"
#include "Timer.hpp"

#include <igl/avg_edge_length.h>
#include <igl/edge_lengths.h>

#ifdef USE_TBB
    #include <tbb/tbb.h>
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <numeric>

extern FracCuts::MethodType methodType;
extern const std::string outputFolderPath;
extern const bool fractureMode;

extern std::ofstream logFile;
extern Timer timer, timer_step, timer_temp;

namespace FracCuts {
    
    Optimizer::Optimizer(const TriangleSoup& p_data0,
                         const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams,
                         int p_propagateFracture, bool p_mute, bool p_scaffolding,
                         const Eigen::MatrixXd& UV_bnds, const Eigen::MatrixXi& E, const Eigen::VectorXi& bnd,
                         AnimScriptType animScriptType) :
        data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams),
        gravity(0.0, -9.80665)
    {
        assert(energyTerms.size() == energyParams.size());
        
        energyParamSum = 0.0;
        for(const auto& ePI : energyParams) {
            energyParamSum += ePI;
        }
        
        gradient_ET.resize(energyTerms.size());
        energyVal_ET.resize(energyTerms.size());
        
        allowEDecRelTol = true;
        propagateFracture = p_propagateFracture;
        mute = p_mute;
        
        if(!mute) {
            file_energyValPerIter.open(outputFolderPath + "energyValPerIter.txt");
            file_gradientPerIter.open(outputFolderPath + "gradientPerIter.txt");
            file_iterStats.open(outputFolderPath + "iterStats.txt");
        }
        
        if(!data0.checkInversion()) {
            std::cout << "element inverted in the initial mesh!" << std::endl;
//            exit(-1);
        }
        
        globalIterNum = 0;
        relGL2Tol = 1.0e-2;
        topoIter = 0;
        innerIterAmt = 0;
        
        Eigen::MatrixXd d2E_div_dF2_rest;
        energyTerms[0]->compute_d2E_div_dF2_rest(d2E_div_dF2_rest);
        sqnorm_H_rest = d2E_div_dF2_rest.squaredNorm();
        Eigen::MatrixXd edgeLens;
        igl::edge_lengths(data0.V_rest, data0.F, edgeLens);
        Eigen::VectorXd ls;
        ls.resize(data0.V_rest.rows());
        ls.setZero();
        for(int triI = 0; triI < data0.F.rows(); triI++) {
            for(int i = 0; i < 3; i++) {
                ls[data0.F(triI, i)] += edgeLens(triI, i);
            }
        }
        sqnorm_l = ls.squaredNorm();
        
        needRefactorize = false;
        for(const auto& energyTermI : energyTerms) {
            if(energyTermI->getNeedRefactorize()) {
                needRefactorize = true;
                break;
            }
        }
        
//        pardisoThreadAmt = 0;
        pardisoThreadAmt = 1; //TODO: use more threads!
        
        scaffolding = p_scaffolding;
        UV_bnds_scaffold = UV_bnds;
        E_scaffold = E;
        bnd_scaffold = bnd;
        w_scaf = energyParams[0] * 0.01;
        
#ifndef STATIC_SOLVE
        setTime(10.0, 0.025);
#else
        dt = dtSq = 1.0;
#endif
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        linSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        linSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        
        setAnimScriptType(animScriptType);
        
        result = data0;
        svd.resize(result.F.rows());
        F.resize(result.F.rows());
        animScripter.initAnimScript(result);
        resultV_n = result.V;
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
        
        lastEDec = 0.0;
        data_findExtrema = data0;
        updateTargetGRes();
        velocity = Eigen::VectorXd::Zero(result.V.rows() * 2);
        computeXTilta();
        computeEnergyVal(result, scaffold, true, lastEnergyVal);
        if(!mute) {
            writeEnergyValToFile(true);
            std::cout << "E_initial = " << lastEnergyVal << std::endl;
        }
    }
    
    Optimizer::~Optimizer(void)
    {
        if(file_energyValPerIter.is_open()) {
            file_energyValPerIter.close();
        }
        if(file_gradientPerIter.is_open()) {
            file_gradientPerIter.close();
        }
        if(file_iterStats.is_open()) {
            file_iterStats.close();
        }
        delete linSysSolver;
    }
    
    void Optimizer::computeLastEnergyVal(void)
    {
        computeEnergyVal(result, scaffold, true, lastEnergyVal);
    }
    
    TriangleSoup& Optimizer::getResult(void) {
        return result;
    }
    
    const Scaffold& Optimizer::getScaffold(void) const {
        return scaffold;
    }
    
    const TriangleSoup& Optimizer::getAirMesh(void) const {
        return scaffold.airMesh;
    }
    
    bool Optimizer::isScaffolding(void) const {
        return scaffolding;
    }
    
    const TriangleSoup& Optimizer::getData_findExtrema(void) const {
        return data_findExtrema;
    }
    
    int Optimizer::getIterNum(void) const {
        return globalIterNum;
    }
    int Optimizer::getTopoIter(void) const {
        return topoIter;
    }
    int Optimizer::getInnerIterAmt(void) const {
        return innerIterAmt;
    }
    
    void Optimizer::setRelGL2Tol(double p_relTol)
    {
        assert(p_relTol > 0.0);
        relGL2Tol = p_relTol;
        updateTargetGRes();
    }
    
    void Optimizer::setAllowEDecRelTol(bool p_allowEDecRelTol)
    {
        allowEDecRelTol = p_allowEDecRelTol;
    }
    
    double Optimizer::getDt(void) const
    {
        return dt;
    }
    
    void Optimizer::setAnimScriptType(AnimScriptType animScriptType)
    {
        animScripter.setAnimScriptType(animScriptType);
    }
    
    void Optimizer::setTime(double duration, double dt)
    {
        this->dt = dt;
        dtSq = dt * dt;
        frameAmt = duration / dt;
        updateTargetGRes();
        gravityDtSq = dtSq * gravity;
    }
    
//    void Optimizer::fixDirection(void)
//    {
//        assert(result.V.rows() == result.V_rest.rows());
//        
//        directionFix.clear();
//        const Eigen::RowVector2d& v0 = result.V.row(0);
//        int nbVI = *result.vNeighbor[0].begin();
//        const Eigen::RowVector2d& vi = result.V.row(nbVI);
//        Eigen::RowVector2d dif = vi - v0;
//        if(std::abs(dif[0]) > std::abs(dif[1])) {
//            double coef = dif[1] / dif[0];
//            directionFix[0] = coef;
//            directionFix[1] = -1.0;
//            directionFix[nbVI * 2] = -coef;
//            directionFix[nbVI * 2 + 1] = 1.0;
//        }
//        else {
//            double coef = dif[0] / dif[1];
//            directionFix[0] = -1.0;
//            directionFix[1] = coef;
//            directionFix[nbVI * 2] = 1.0;
//            directionFix[nbVI * 2 + 1] = -coef;
//        }
//    }
    
    void Optimizer::precompute(void)
    {
        if(!mute) { timer_step.start(1); }
        linSysSolver->set_type(pardisoThreadAmt, 2);
//            linSysSolver->set_pattern(I_mtr, J_mtr, V_mtr);
        linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                  scaffolding ? fixedV_withScaf : result.fixedVert);
        if(!mute) { timer_step.stop(); }
        computePrecondMtr(result, scaffold, true, linSysSolver);
        if(!mute) { timer_step.start(2); }
        linSysSolver->analyze_pattern();
        if(!mute) { timer_step.stop(); }
        if(!needRefactorize) {
            try {
                if(!mute) { timer_step.start(3); }
                linSysSolver->factorize();
                if(!mute) { timer_step.stop(); }
            }
            catch(std::exception e) {
//                IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr_factorizeFail", I_mtr, J_mtr, V_mtr, true);
                assert(0);
            }
        }
    }
    
    int Optimizer::solve(int maxIter)
    {
        static bool lastPropagate = false;
        int returnFlag = 0;
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
#ifndef STATIC_SOLVE
            animScripter.stepAnimScript(result, dt);
#endif
            if(!mute) { timer.start(1); }
            computeGradient(result, scaffold, true, gradient);
            const double sqn_g = gradient.squaredNorm();
            if(!mute) {
                std::cout << "||gradient||^2 = " << sqn_g << ", targetGRes = " << targetGRes << std::endl;
                writeGradL2NormToFile(false);
            }
#ifdef STATIC_SOLVE
            if(sqn_g < targetGRes) {
#else
            if(globalIterNum >= frameAmt) {
#endif
                // converged
                lastEDec = 0.0;
                globalIterNum++;
                if(!mute) { timer.stop(); }
                return 1;
            }
            else {
#ifdef STATIC_SOLVE
                if(solve_oneStep()) {
                    globalIterNum++;
                    if(!mute) { timer.stop(); }
                    return 1;
#else
                if(fullyImplicit()) {
                    returnFlag = 2;
#endif
                }
#ifndef STATIC_SOLVE
                velocity = Eigen::Map<Eigen::MatrixXd>(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>((result.V - resultV_n).array() / dt).data(),
                                                       velocity.rows(), 1);
                resultV_n = result.V;
                computeXTilta();
#endif
            }
            globalIterNum++;
            if(!mute) { timer.stop(); }
//            //DEBUG
//            if(globalIterNum > 1220) {
//                result.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_afterPN.obj");
//                scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_afterPN_AM.obj");
//            }
            
            if(propagateFracture > 0) {
                if(!createFracture(lastEDec, propagateFracture)) {
//                    propagateFracture = 0;
                    // always perform the one decreasing E_w more
                    if(scaffolding) {
                        scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                        result.scaffold = &scaffold;
                        scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                        scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
                    }
                    
                    if(lastPropagate) {
                        lastPropagate = false;
                        return 2; // for saving screenshots
                    }
                }
                else {
                    lastPropagate = true;
                }
                // for alternating propagation with lambda updates
//                if(createFracture(lastEDec, propagateFracture)) {
//                    return 2;
//                }
            }
            else {
                if(scaffolding) {
                    scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                    result.scaffold = &scaffold;
                    scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                    scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
                }
            }
        }
        return returnFlag;
    }
    
    void Optimizer::updatePrecondMtrAndFactorize(void)
    {
        if(needRefactorize) {
            // don't need to call this function
            return;
        }
        
        if(!mute) {
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
        }
        computePrecondMtr(result, scaffold, true, linSysSolver);
        
        if(!mute) { timer_step.start(3); }
        linSysSolver->factorize();
        if(!mute) { timer_step.stop(); }
    }
    
    void Optimizer::setConfig(const TriangleSoup& config, int iterNum, int p_topoIter)
    {
        topoIter = p_topoIter;
        globalIterNum = iterNum;
        result = config; //!!! is it able to copy all?
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
        
        updateEnergyData();
    }
    
    void Optimizer::setPropagateFracture(bool p_prop)
    {
        propagateFracture = p_prop;
    }
    
    void Optimizer::setScaffolding(bool p_scaffolding)
    {
        scaffolding = p_scaffolding;
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
    }
    
    void Optimizer::updateEnergyData(bool updateEVal, bool updateGradient, bool updateHessian)
    {
        energyParamSum = 0.0;
        for(const auto& ePI : energyParams) {
            energyParamSum += ePI;
        }
        updateTargetGRes();
        
        if(updateEVal) {
            // compute energy and output
            computeEnergyVal(result, scaffold, true, lastEnergyVal);
        }
        
        if(updateGradient) {
            // compute gradient and output
            computeGradient(result, scaffold, false, gradient);
            if(gradient.squaredNorm() < targetGRes) {
                logFile << "||g||^2 = " << gradient.squaredNorm() << " after fracture initiation!" << std::endl;
            }
        }
        
        if(updateHessian) {
            // for the changing hessian
            if(!mute) {
                std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
            }
            
            if(!mute) { timer_step.start(1); }
//                linSysSolver->set_pattern(I_mtr, J_mtr, V_mtr);
            linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                      scaffolding ? fixedV_withScaf : result.fixedVert);
            if(!mute) { timer_step.stop(); }
            
            computePrecondMtr(result, scaffold, false, linSysSolver);
            
            if(!mute) { timer_step.start(2); }
            linSysSolver->analyze_pattern();
            if(!mute) { timer_step.stop(); }
            if(!needRefactorize) {
                if(!mute) { timer_step.start(3); }
                linSysSolver->factorize();
                if(!mute) { timer_step.stop(); }
            }
        }
    }
    
    bool Optimizer::createFracture(int opType, const std::vector<int>& path, const Eigen::MatrixXd& newVertPos, bool allowPropagate)
    {
        assert(methodType == MT_OURS);
        
        topoIter++;
//        //DEBUG
//        if(globalIterNum > 520) {
//            result.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_preTopo.obj");
//            scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_preTopo_AM.obj");
//        }
        
        timer.start(0);
        bool isMerge = false;
        data_findExtrema = result; // potentially time-consuming
        switch(opType) {
            case 0: // boundary split
                std::cout << "boundary split without querying again" << std::endl;
                result.splitEdgeOnBoundary(std::pair<int, int>(path[0], path[1]), newVertPos);
                logFile << "boundary edge splitted without querying again" << std::endl;
                //TODO: process fractail here!
                result.updateFeatures();
                break;
                
            case 1: // interior split
                std::cout << "Interior split without querying again" << std::endl;
                result.cutPath(path, true, 1, newVertPos);
                logFile << "interior edge splitted without querying again" << std::endl;
                result.fracTail.insert(path[0]);
                result.fracTail.insert(path[2]);
                result.curInteriorFracTails.first = path[0];
                result.curInteriorFracTails.second = path[2];
                result.curFracTail = -1;
                break;
                
            case 2: // merge
                std::cout << "corner edge merged without querying again" << std::endl;
                result.mergeBoundaryEdges(std::pair<int, int>(path[0], path[1]),
                                          std::pair<int, int>(path[1], path[2]), newVertPos.row(0));
                logFile << "corner edge merged without querying again" << std::endl;
                
                result.computeFeatures(); //TODO: only update locally
                isMerge = true;
                break;
                
            default:
                assert(0);
                break;
        }
        timer.stop();
//        logFile << result.V.rows() << std::endl;
//        logFile << result.F << std::endl; //DEBUG
//        logFile << result.cohE << std::endl; //DEBUG
        
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
//            //DEBUG
//            if(globalIterNum > 10) {
//                result.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_postTopo.obj");
//                scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_postTopo_AM.obj");
//            }
        
        timer.start(3);
        updateEnergyData(true, false, true);
        timer.stop();
        fractureInitiated = true;
        if(!mute) {
            writeEnergyValToFile(false);
        }
        
        if(allowPropagate) {
            propagateFracture = 1 + isMerge;
        }

        return true;
    }
    
    bool Optimizer::createFracture(double stressThres, int propType, bool allowPropagate, bool allowInSplit)
    {
        if(propType == 0) {
            topoIter++;
        }
//        //DEBUG
//        if(globalIterNum > 520) {
//            result.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_preTopo.obj");
//            scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_preTopo_AM.obj");
//        }
        
        timer.start(0);
        bool changed = false;
        bool isMerge = false;
        switch(methodType) {
            case MT_OURS_FIXED:
            case MT_OURS: {
                data_findExtrema = result;
                switch(propType) {
                    case 0: // initiation
                        changed = result.splitOrMerge(1.0 - energyParams[0], stressThres, false, allowInSplit, isMerge);
//                        //DEBUG:
//                        if(allowInSplit) {
//                            changed = false;
//                        }
//                        else {
//                            changed = result.splitOrMerge(1.0 - energyParams[0], stressThres, false, allowInSplit, isMerge);
//                        }
                        break;
                        
                    case 1: // propagate split
                        changed = result.splitEdge(1.0 - energyParams[0], stressThres, true, allowInSplit);
                        break;
                        
                    case 2: // propagate merge
                        changed = result.mergeEdge(1.0 - energyParams[0], stressThres, true);
//                        changed = false;
                        isMerge = true;
                        break;
                }
                break;
            }
                
            case MT_GEOMIMG:
                result.geomImgCut(data_findExtrema);
                allowPropagate = false;
                changed = true;
                break;
                
            default:
                assert(0 && "Fracture forbiddened for current method type!");
                break;
        }
        timer.stop();
//        logFile << result.V.rows() << std::endl;
        if(changed) {
            // In fact currently it will always change
            // because we are doing it anyway and roll back
            // if it increase E_w
//            logFile << result.F << std::endl; //DEBUG
//            logFile << result.cohE << std::endl; //DEBUG
            
            if(scaffolding) {
                scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                result.scaffold = &scaffold;
                scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
            }
//            //DEBUG
//            if(globalIterNum > 10) {
//                result.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_postTopo.obj");
//                scaffold.airMesh.save("/Users/mincli/Desktop/meshes/test"+std::to_string(globalIterNum)+"_postTopo_AM.obj");
//            }
            
            timer.start(3);
            updateEnergyData(true, false, true);
            timer.stop();
            fractureInitiated = true;
            if((!mute) && (propType == 0)) {
                writeEnergyValToFile(false);
            }
            
            if(allowPropagate && (propType == 0)) {
//                solve(1);
                propagateFracture = 1 + isMerge;
            }
        }
        return changed;
    }
    
    void Optimizer::initX(int option)
    {
        // global:
        searchDir.resize(result.V.rows() * 2);
        switch(option) {
            case 0:
                // already at last timestep config
                return;
                
            case 1: // explicit Euler
#ifdef USE_TBB
                tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < result.V.rows(); vI++)
#endif
                {
                    if(result.isFixedVert[vI]) {
                        searchDir.segment(vI * 2, 2).setZero();

                    }
                    else {
                        searchDir.segment(vI * 2, 2) = dt * velocity.segment(vI * 2, 2);
                    }
                }
#ifdef USE_TBB
                );
#endif
                break;
                
            case 2: // xHat
#ifdef USE_TBB
                tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < result.V.rows(); vI++)
#endif
                {
                    if(result.isFixedVert[vI]) {
                        searchDir.segment(vI * 2, 2).setZero();
                    }
                    else {
                        searchDir.segment(vI * 2, 2) = dt * velocity.segment(vI * 2, 2) + gravityDtSq;
                    }
                }
#ifdef USE_TBB
                );
#endif
                break;
                
            case 3: { // Symplectic Euler
                Eigen::VectorXd f;
                energyTerms[0]->computeGradientBySVD(result, f);
#ifdef USE_TBB
                tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < result.V.rows(); vI++)
#endif
                {
                    if(result.isFixedVert[vI]) {
                        searchDir.segment(vI * 2, 2).setZero();
                    }
                    else {
                        double mass = result.massMatrix.coeff(vI, vI);
                        searchDir.segment(vI * 2, 2) = (dt * velocity.segment(vI * 2, 2) +
                                                        dtSq * (gravity - f.segment(vI * 2, 2) / mass));
                    }
                }
#ifdef USE_TBB
                );
#endif
                break;
            }
                
            case 4: { // uniformly accelerated motion approximation
                Eigen::VectorXd f;
                energyTerms[0]->computeGradientBySVD(result, f);
#ifdef USE_TBB
                tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < result.V.rows(); vI++)
#endif
                {
                    if(result.isFixedVert[vI]) {
                        searchDir.segment(vI * 2, 2).setZero();
                    }
                    else {
                        double mass = result.massMatrix.coeff(vI, vI);
                        searchDir.segment(vI * 2, 2) = (dt * velocity.segment(vI * 2, 2) +
                                                        dtSq / 2.0 * (gravity - f.segment(vI * 2, 2) / mass));
                    }
                }
#ifdef USE_TBB
                );
#endif
                break;
            }
                
            case 5: { // Jacobi
                Eigen::VectorXd g;
                computeGradient(result, scaffold, true, g);
                
                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                computePrecondMtr(result, scaffold, false, linSysSolver);
                
#ifdef USE_TBB
                tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < result.V.rows(); vI++)
#endif
                {
                    if(result.isFixedVert[vI]) {
                        searchDir.segment(vI * 2, 2).setZero();
                    }
                    else {
                        searchDir[vI * 2] = -g[vI * 2] / linSysSolver->coeffMtr(vI * 2, vI * 2);
                        searchDir[vI * 2 + 1] = -g[vI * 2 + 1] / linSysSolver->coeffMtr(vI * 2 + 1, vI * 2 + 1);
                    }
                }
#ifdef USE_TBB
                );
#endif
                
                break;
            }
                
            default:
                std::cout << "unkown primal initialization type, use last timestep instead" << std::endl;
                break;
        }
        
//        Eigen::MatrixXd V0 = result.V;
//        double E0;
//        computeEnergyVal(result, scaffold, E0);
//        double stepSize = 1.0;
//        energyTerms[0]->initStepSize(result, searchDir, stepSize);
//
//        double E;
//        stepForward(V0, Eigen::MatrixXd(), result, scaffold, stepSize);
//        computeEnergyVal(result, scaffold, E);
//        while(E > E0) {
//            stepSize /= 2.0;
//            stepForward(V0, Eigen::MatrixXd(), result, scaffold, stepSize);
//            computeEnergyVal(result, scaffold, E);
//        }
//        std::cout << "primal init step size = " << stepSize << std::endl;
        
        double stepSize = 1.0;
        energyTerms[0]->initStepSize(result, searchDir, stepSize);
        if(stepSize < 1.0) {
            stepSize *= 0.5;
        }
        stepForward(result.V, Eigen::MatrixXd(), result, scaffold, stepSize);
    }
    
    void Optimizer::computeXTilta(void)
    {
        xTilta.conservativeResize(result.V.rows(), 2);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < result.V.rows(); vI++)
#endif
        {
            if(result.isFixedVert[vI]) {
                xTilta.row(vI) = resultV_n.row(vI);
            }
            else {
                xTilta.row(vI) = (resultV_n.row(vI) +
                                  (velocity.segment<2>(vI * 2) * dt +
                                   gravityDtSq).transpose());
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    
    bool Optimizer::fullyImplicit(void)
    {
        initX(0);
        
        double sqn_g = __DBL_MAX__;
        computeEnergyVal(result, scaffold, false, lastEnergyVal);
        do {
            if(solve_oneStep()) {
                std::cout << "\tline search with Armijo's rule failed!!!" << std::endl;
                logFile << "\tline search with Armijo's rule failed!!!" << std::endl;
                return true;
            }
            innerIterAmt++;
            computeGradient(result, scaffold, false, gradient);
            sqn_g = gradient.squaredNorm();
            if(!mute) {
                std::cout << "\t||gradient||^2 = " << sqn_g << std::endl;
            }
            file_iterStats << globalIterNum << " " << sqn_g << std::endl;
        } while(sqn_g > targetGRes);
        
        logFile << "Timestep" << globalIterNum << " innerIterAmt = " << innerIterAmt << std::endl;
        
        return false;
    }
    
    bool Optimizer::solve_oneStep(void)
    {
        if(needRefactorize) {
            // for the changing hessian
            if(!fractureInitiated) {
                if(scaffolding) {
                    if(!mute) { timer_step.start(1); }
//                        linSysSolver->set_pattern(I_mtr, J_mtr, V_mtr);
                    linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                              scaffolding ? fixedV_withScaf : result.fixedVert);
                    if(!mute) { timer_step.stop(); }
                    
                    computePrecondMtr(result, scaffold, false, linSysSolver);
                    
                    if(!mute) {
                        std::cout << "symbolically factorizing proxy/Hessian matrix..." << std::endl;
                    }
                    if(!mute) { timer_step.start(2); }
                    linSysSolver->analyze_pattern();
                    if(!mute) { timer_step.stop(); }
                }
                else {
                    if(!mute) {
                        std::cout << "updating matrix entries..." << std::endl;
                    }
                    computePrecondMtr(result, scaffold, false, linSysSolver);
                }
            }
            try {
                if(!mute) {
                    std::cout << "numerically factorizing Hessian/Proxy matrix..." << std::endl;
                }
                if(!mute) { timer_step.start(3); }
                linSysSolver->factorize();
                if(!mute) { timer_step.stop(); }
            }
            catch(std::exception e) {
//                IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr", I_mtr, J_mtr, V_mtr, true);
                assert(0);
            }
        }
        
        Eigen::VectorXd minusG = -gradient;
        if(!mute) {
            std::cout << "back solve..." << std::endl;
        }
        if(!mute) { timer_step.start(4); }
        linSysSolver->solve(minusG, searchDir);
        if(!mute) { timer_step.stop(); }
        
        fractureInitiated = false;
        
        if(!mute) { timer_step.start(5); }
        bool stopped = lineSearch();
        if(!mute) { timer_step.stop(); }
        if(stopped) {
//            IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_stopped_" + std::to_string(globalIterNum), precondMtr);
//            logFile << "descent step stopped at overallIter" << globalIterNum << " for no prominent energy decrease." << std::endl;
        }
        return stopped;
    }
    
    bool Optimizer::lineSearch(void)
    {
        bool stopped = false;
        double stepSize = 1.0;
        initStepSize(result, stepSize);
        stepSize *= 0.99; // producing degenerated element is not allowed
        if(!mute) {
            std::cout << "stepSize: " << stepSize << " -> ";
        }
        
        double lastEnergyVal_scaffold = 0.0;
        const double m = searchDir.dot(gradient);
        const double c1m = 1.0e-4 * m;
        Eigen::MatrixXd resultV0 = result.V;
//        TriangleSoup temp = result; //TEST
        Eigen::MatrixXd scaffoldV0;
        if(scaffolding) {
//            Scaffold tempp = scaffold;
            scaffoldV0 = scaffold.airMesh.V;
            computeEnergyVal(result, scaffold, true, lastEnergyVal); // this update is necessary since scaffold changes
            lastEnergyVal_scaffold = energyVal_scaffold;
        }
        stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
        double testingE;
//        Eigen::VectorXd testingG;
        computeEnergyVal(result, scaffold, true, testingE);
//        computeGradient(testingData, testingG);
#define ARMIJO_RULE
#ifdef ARMIJO_RULE
//        while((testingE > lastEnergyVal + stepSize * c1m) ||
//              (searchDir.dot(testingG) < c2m)) // Wolfe condition
        while(testingE > lastEnergyVal + stepSize * c1m) // Armijo condition
//        while(0)
        {
            stepSize /= 2.0;
            if(stepSize == 0.0) {
                stopped = true;
                if(!mute) {
                    logFile << "testingE" << globalIterNum << " " << testingE << " > " << lastEnergyVal << " " << stepSize * c1m << std::endl;
//                    logFile << "testingG" << globalIterNum << " " << searchDir.dot(testingG) << " < " << c2m << std::endl;
                }
                break;
            }
            
            stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
            computeEnergyVal(result, scaffold, true, testingE);
//            computeGradient(testingData, testingG);
        }
#endif
        if(!mute) {
            std::cout << stepSize << "(armijo) ";
        }

//        while((!result.checkInversion()) ||
//              ((scaffolding) && (!scaffold.airMesh.checkInversion())))
//        {
//            assert(0 && "element inversion after armijo shouldn't happen!");
//
//            stepSize /= 2.0;
//            if(stepSize == 0.0) {
//                assert(0 && "line search failed!");
//                stopped = true;
//                break;
//            }
//
//            stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
//            computeEnergyVal(result, scaffold, testingE);
//        }
        
        lastEDec = lastEnergyVal - testingE;
        if(scaffolding) {
            lastEDec += (-lastEnergyVal_scaffold + energyVal_scaffold);
        }
//        lastEDec = (lastEnergyVal - testingE) / stepSize;
        if(allowEDecRelTol && (lastEDec / lastEnergyVal / stepSize < 1.0e-6)) {
//        if(allowEDecRelTol && (lastEDec / lastEnergyVal < 1.0e-6)) {
            // no prominent energy decrease, stop for accelerating the process
            stopped = true;
        }
        lastEnergyVal = testingE;
        
        if(!mute) {
            std::cout << stepSize << std::endl;
            std::cout << "stepLen = " << (stepSize * searchDir).squaredNorm() << std::endl;
            std::cout << "E_cur_smooth = " << testingE - energyVal_scaffold << std::endl;

            if(!stopped) {
                writeEnergyValToFile(false);
            }
        }
        
        return stopped;
    }
    
    void Optimizer::stepForward(const Eigen::MatrixXd& dataV0, const Eigen::MatrixXd& scaffoldV0,
                                TriangleSoup& data, Scaffold& scaffoldData, double stepSize) const
    {
        assert(dataV0.rows() == data.V.rows());
        if(scaffolding) {
            assert(data.V.rows() + scaffoldData.airMesh.V.rows() - scaffoldData.bnd.size() == searchDir.size() / 2);
        }
        else {
            assert(data.V.rows() * 2 == searchDir.size());
        }
        assert(data.V.rows() == result.V.rows());
        
        for(int vI = 0; vI < data.V.rows(); vI++) {
            data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];
            data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];
        }
        if(scaffolding) {
            scaffoldData.stepForward(scaffoldV0, searchDir, stepSize);
        }
    }
    
    void Optimizer::updateTargetGRes(void)
    {
//        targetGRes = energyParamSum * (data0.V_rest.rows() - data0.fixedVert.size()) * relGL2Tol * data0.avgEdgeLen * data0.avgEdgeLen;
//        targetGRes = energyParamSum * static_cast<double>(data0.V_rest.rows() - data0.fixedVert.size()) / static_cast<double>(data0.V_rest.rows()) * relGL2Tol;
//        targetGRes *= data0.surfaceArea * data0.surfaceArea;
//        targetGRes = relGL2Tol * sqnorm_H_rest * sqnorm_l * (data0.V_rest.rows() - data0.fixedVert.size()) / data0.V_rest.rows() * energyParamSum * energyParamSum;
        double m_sqn = 0.0;
        for(int vI = 0; vI < data0.V_rest.rows(); vI++) {
            if(data0.fixedVert.find(vI) == data0.fixedVert.end()) {
                double m = data0.massMatrix.coeff(vI, vI);
                m_sqn += m * m;
            }
        }
        assert(energyParamSum == 1.0);
        targetGRes = relGL2Tol * m_sqn * gravity.squaredNorm();
#ifndef STATIC_SOLVE
        targetGRes *= dtSq * dtSq;
#endif
        targetGRes = 1.0e-10;
    }
    
    void Optimizer::getGradientVisual(Eigen::MatrixXd& arrowVec) const
    {
        assert(result.V.rows() * 2 == gradient.size());
        arrowVec.resize(result.V.rows(), result.V.cols());
        for(int vI = 0; vI < result.V.rows(); vI++) {
            arrowVec(vI, 0) = gradient[vI * 2];
            arrowVec(vI, 1) = gradient[vI * 2 + 1];
            arrowVec.row(vI).normalize();
        }
        arrowVec *= igl::avg_edge_length(result.V, result.F);
    }
            
    void Optimizer::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field = Eigen::VectorXd::Zero(result.F.rows());
    }
    void Optimizer::getSharedVerts(Eigen::VectorXi& sharedVerts) const
    {
        sharedVerts.resize(0);
    }
    
    void Optimizer::initStepSize(const TriangleSoup& data, double& stepSize) const
    {
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->initStepSize(data, searchDir, stepSize);
        }
        
        if(scaffolding) {
            Eigen::VectorXd searchDir_scaffold;
            scaffold.wholeSearchDir2airMesh(searchDir, searchDir_scaffold);
            SymStretchEnergy SD;
            SD.initStepSize(scaffold.airMesh, searchDir_scaffold, stepSize);
        }
    }
    
    void Optimizer::writeEnergyValToFile(bool flush)
    {
        double E_se;
        result.computeSeamSparsity(E_se, !fractureMode);
        E_se /= result.virtualRadius;
        
        if(fractureMode) {
            buffer_energyValPerIter << lastEnergyVal + (1.0 - energyParams[0]) * E_se;
        }
        else {
            buffer_energyValPerIter << lastEnergyVal;
        }
        
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            buffer_energyValPerIter << " " << energyVal_ET[eI];
        }
        
        buffer_energyValPerIter << " " << E_se << " " << energyParams[0] << "\n";
        
        if(flush) {
            flushEnergyFileOutput();
        }
    }
    void Optimizer::writeGradL2NormToFile(bool flush)
    {
        buffer_gradientPerIter << gradient.squaredNorm();
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            buffer_gradientPerIter << " " << gradient_ET[eI].squaredNorm();
        }
        buffer_gradientPerIter << "\n";
        
        if(flush) {
            flushGradFileOutput();
        }
    }
    void Optimizer::flushEnergyFileOutput(void)
    {
        file_energyValPerIter << buffer_energyValPerIter.str();
        file_energyValPerIter.flush();
        clearEnergyFileOutputBuffer();
    }
    void Optimizer::flushGradFileOutput(void)
    {
        file_gradientPerIter << buffer_gradientPerIter.str();
        file_gradientPerIter.flush();
        clearGradFileOutputBuffer();
    }
    void Optimizer::clearEnergyFileOutputBuffer(void)
    {
        buffer_energyValPerIter.str("");
        buffer_energyValPerIter.clear();
    }
    void Optimizer::clearGradFileOutputBuffer(void)
    {
        buffer_gradientPerIter.str("");
        buffer_gradientPerIter.clear();
    }
    
    void Optimizer::computeEnergyVal(const TriangleSoup& data, const Scaffold& scaffoldData,
                                     bool redoSVD, double& energyVal, bool excludeScaffold)
    {
        if(!mute) { timer_step.start(0); }
        
        energyTerms[0]->computeEnergyValBySVD(data, redoSVD, svd, F, energyVal_ET[0]);
        energyVal = dtSq * energyParams[0] * energyVal_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeEnergyValBySVD(data, redoSVD, svd, F, energyVal_ET[eI]);
            energyVal += dtSq * energyParams[eI] * energyVal_ET[eI];
        }
        
        if(scaffolding && (!excludeScaffold)) {
            SymStretchEnergy SD;
            SD.computeEnergyVal(scaffoldData.airMesh, energyVal_scaffold, true);
            energyVal_scaffold *= w_scaf / scaffold.airMesh.F.rows();
            energyVal += energyVal_scaffold;
        }
        else {
            energyVal_scaffold = 0.0;
        }
        
#ifndef STATIC_SOLVE
        timer_temp.start(4);
        Eigen::VectorXd energyVals(data.V.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            energyVals[vI] = ((data.V.row(vI) - xTilta.row(vI)).squaredNorm() *
                              data.massMatrix.coeff(vI, vI) / 2.0);
        }
#ifdef USE_TBB
        );
#endif
        energyVal += energyVals.sum();
        //TODO: mass of negative space vertices
        timer_temp.stop();
#endif
        if(!mute) { timer_step.stop(); }
    }
    void Optimizer::computeGradient(const TriangleSoup& data, const Scaffold& scaffoldData,
                                    bool redoSVD, Eigen::VectorXd& gradient, bool excludeScaffold)
    {
        if(!mute) { timer_step.start(0); }
        
        energyTerms[0]->computeGradientByPK(data, redoSVD, svd, F, gradient_ET[0]);
        gradient = dtSq * energyParams[0] * gradient_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeGradientByPK(data, redoSVD, svd, F, gradient_ET[eI]);
            gradient += dtSq * energyParams[eI] * gradient_ET[eI];
        }
        
        if(scaffolding) {
            SymStretchEnergy SD;
            SD.computeGradient(scaffoldData.airMesh, gradient_scaffold, true);
            scaffoldData.augmentGradient(gradient, gradient_scaffold, (excludeScaffold ? 0.0 : (w_scaf / scaffold.airMesh.F.rows())));
        }
        
#ifndef STATIC_SOLVE
        timer_temp.start(5);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            if(!data.isFixedVert[vI]) {
                gradient.segment<2>(vI * 2) += (data.massMatrix.coeff(vI, vI) *
                                                (data.V.row(vI) - xTilta.row(vI)).transpose());
            }
        }
#ifdef USE_TBB
        );
#endif
        //TODO: mass of negative space vertices
        timer_temp.stop();
#endif
        if(!mute) { timer_step.stop(); }
    }
    void Optimizer::computePrecondMtr(const TriangleSoup& data, const Scaffold& scaffoldData,
                                      bool redoSVD,
                                      LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *p_linSysSolver)
    {
        if(!mute) { timer_step.start(0); }
        
        p_linSysSolver->setZero();
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeHessianByPK(data, redoSVD, svd, F,
                                                energyParams[eI] * dtSq,
                                                p_linSysSolver);
        }
        
//        if(scaffolding) {
//            SymStretchEnergy SD;
//            Eigen::VectorXi I_scaf, J_scaf;
//            Eigen::VectorXd V_scaf;
//            SD.computePrecondMtr(scaffoldData.airMesh, &V_scaf, &I_scaf, &J_scaf, true);
//            scaffoldData.augmentProxyMatrix(I, J, V, I_scaf, J_scaf, V_scaf, w_scaf / scaffold.airMesh.F.rows() * dtSq);
//        }
//            IglUtils::writeSparseMatrixToFile("/Users/mincli/Desktop/FracCuts/mtr", I_mtr, J_mtr, V_mtr, true);
        
#ifndef STATIC_SOLVE
        timer_temp.start(6);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            if(!data.isFixedVert[vI]) {
                double massI = data.massMatrix.coeff(vI, vI);
                int ind0 = vI * 2;
                int ind1 = ind0 + 1;
                p_linSysSolver->addCoeff(ind0, ind0, massI);
                p_linSysSolver->addCoeff(ind1, ind1, massI);
            }
        }
#ifdef USE_TBB
        );
#endif
        //TODO: mass of negative space vertices
        timer_temp.stop();
#endif
        if(!mute) { timer_step.stop(); }
    }
    void Optimizer::computeHessian(const TriangleSoup& data, const Scaffold& scaffoldData, Eigen::SparseMatrix<double>& hessian) const
    {
        energyTerms[0]->computeHessian(data, hessian);
        hessian *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::SparseMatrix<double> hessianI;
            energyTerms[eI]->computeHessian(data, hessianI);
            hessian += energyParams[eI] * hessianI;
        }
        
        //TODO: SCAFFOLDING
    }
    
    double Optimizer::getLastEnergyVal(bool excludeScaffold) const
    {
        return ((excludeScaffold && scaffolding) ?
                (lastEnergyVal - energyVal_scaffold) :
                lastEnergyVal);
    }
}
