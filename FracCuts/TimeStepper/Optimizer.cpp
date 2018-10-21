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

extern std::ofstream logFile;
extern Timer timer, timer_step, timer_temp;

namespace FracCuts {
    
    template<int dim>
    Optimizer<dim>::Optimizer(const TriangleSoup<dim>& p_data0,
                         const std::vector<Energy<dim>*>& p_energyTerms, const std::vector<double>& p_energyParams,
                         int p_propagateFracture, bool p_mute, bool p_scaffolding,
                         const Eigen::MatrixXd& UV_bnds, const Eigen::MatrixXi& E, const Eigen::VectorXi& bnd,
                         const Config& p_animConfig) :
        data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams), animConfig(p_animConfig),
        QPSolver(false)
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
        else {
            std::cout << "no element inversion detected!" << std::endl;
        }
        
        globalIterNum = 0;
        relGL2Tol = 1.0e-2;
        topoIter = 0;
        innerIterAmt = 0;
        
//        Eigen::MatrixXd d2E_div_dF2_rest;
//        energyTerms[0]->compute_d2E_div_dF2_rest(d2E_div_dF2_rest);
//        sqnorm_H_rest = d2E_div_dF2_rest.squaredNorm();
//        Eigen::MatrixXd edgeLens;
//        igl::edge_lengths(data0.V_rest, data0.F, edgeLens);
//        Eigen::VectorXd ls;
//        ls.resize(data0.V_rest.rows());
//        ls.setZero();
//        for(int triI = 0; triI < data0.F.rows(); triI++) {
//            for(int i = 0; i < 3; i++) {
//                ls[data0.F(triI, i)] += edgeLens(triI, i);
//            }
//        }
//        sqnorm_l = ls.squaredNorm();
        
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
        gravity.setZero();
        gravity[1] = -9.80665;
        setTime(10.0, 0.025);
#else
        dt = dtSq = 1.0;
#endif
        std::cout << "dt and tol initialized" << std::endl;
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        linSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        linSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        
        setAnimScriptType(animConfig.animScriptType);
        solveQP = (animConfig.isConstrained && (animConfig.constraintSolverType == CST_QP));
        
        result = data0;
        svd.resize(result.F.rows());
        F.resize(result.F.rows());
        animScripter.initAnimScript(result);
        if(energyTerms[0]->getNeedElemInvSafeGuard()) {
            if(!result.checkInversion()) {
                std::cout << "scripted motion causes element inversion, end process" << std::endl;
                exit(-1);
            }
        }
        std::cout << "animScriptor set" << std::endl;
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
        velocity = Eigen::VectorXd::Zero(result.V.rows() * dim);
        computeXTilta();
        computeEnergyVal(result, scaffold, true, lastEnergyVal);
        if(!mute) {
            writeEnergyValToFile(true);
            std::cout << "E_initial = " << lastEnergyVal << std::endl;
        }
        std::cout << "Newton's solver for Backward Euler constructed" << std::endl;
    }
    
    template<int dim>
    Optimizer<dim>::~Optimizer(void)
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
    
    template<int dim>
    void Optimizer<dim>::computeLastEnergyVal(void)
    {
        computeEnergyVal(result, scaffold, true, lastEnergyVal);
    }
    
    template<int dim>
    TriangleSoup<dim>& Optimizer<dim>::getResult(void) {
        return result;
    }
    
    template<int dim>
    const Scaffold& Optimizer<dim>::getScaffold(void) const {
        return scaffold;
    }
    
    template<int dim>
    const TriangleSoup<dim>& Optimizer<dim>::getAirMesh(void) const {
        return scaffold.airMesh;
    }
    
    template<int dim>
    bool Optimizer<dim>::isScaffolding(void) const {
        return scaffolding;
    }
    
    template<int dim>
    const TriangleSoup<dim>& Optimizer<dim>::getData_findExtrema(void) const {
        return data_findExtrema;
    }
    
    template<int dim>
    int Optimizer<dim>::getIterNum(void) const {
        return globalIterNum;
    }
    template<int dim>
    int Optimizer<dim>::getTopoIter(void) const {
        return topoIter;
    }
    template<int dim>
    int Optimizer<dim>::getInnerIterAmt(void) const {
        return innerIterAmt;
    }
    
    template<int dim>
    void Optimizer<dim>::setRelGL2Tol(double p_relTol)
    {
        assert(p_relTol > 0.0);
        relGL2Tol = p_relTol;
        updateTargetGRes();
    }
    
    template<int dim>
    void Optimizer<dim>::setAllowEDecRelTol(bool p_allowEDecRelTol)
    {
        allowEDecRelTol = p_allowEDecRelTol;
    }
    
    template<int dim>
    double Optimizer<dim>::getDt(void) const
    {
        return dt;
    }
    
    template<int dim>
    void Optimizer<dim>::setAnimScriptType(AnimScriptType animScriptType)
    {
        animScripter.setAnimScriptType(animScriptType);
    }
    
    template<int dim>
    void Optimizer<dim>::setTime(double duration, double dt)
    {
        this->dt = dt;
        dtSq = dt * dt;
        frameAmt = duration / dt;
        updateTargetGRes();
        gravityDtSq = dtSq * gravity;
    }
    
//    void Optimizer<dim>::fixDirection(void)
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
    
    template<int dim>
    void Optimizer<dim>::precompute(void)
    {
        std::cout << "precompute: start" << std::endl;
        if(!mute) { timer_step.start(1); }
        linSysSolver->set_type(pardisoThreadAmt, 2);
//            linSysSolver->set_pattern(I_mtr, J_mtr, V_mtr);
        linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                  scaffolding ? fixedV_withScaf : result.fixedVert);
        if(!mute) { timer_step.stop(); }
        std::cout << "precompute: sparse matrix allocated" << std::endl;
        computePrecondMtr(result, scaffold, true, linSysSolver);
        std::cout << "precompute: sparse matrix entry computed" << std::endl;
        if(!mute) { timer_step.start(2); }
        linSysSolver->analyze_pattern();
        if(!mute) { timer_step.stop(); }
        std::cout << "precompute: pattern analyzed" << std::endl;
        if(!needRefactorize) {
            try {
                if(!mute) { timer_step.start(3); }
                linSysSolver->factorize();
                if(!mute) { timer_step.stop(); }
            }
            catch(std::exception e) {
                IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr_numFacFail",
                                                  linSysSolver, true);
                std::cout << "numerical factorization failed, " <<
                    "matrix written into " << outputFolderPath + "mtr_numFacFail" << std::endl;
                exit(-1);
            }
        }
        std::cout << "precompute: factorized" << std::endl;
        
        if(solveQP) {
            //TODO: faster matrix set pattern! or use symmetric format in OSQP
            P_OSQP.resize(result.V.rows() * dim, result.V.rows() * dim);
            P_OSQP.setZero();
            P_OSQP.reserve(linSysSolver->getNumNonzeros() * 2 - gradient.size());
            for(int rowI = 0; rowI < linSysSolver->getNumRows(); rowI++) {
                for(const auto& colIter : linSysSolver->getIJ2aI()[rowI]) {
                    P_OSQP.insert(rowI, colIter.first) = 0.0;
                    if(rowI != colIter.first) {
                        P_OSQP.insert(colIter.first, rowI) = 0.0;
                    }
                }
            }
            P_OSQP.makeCompressed();
            elemPtr_P_OSQP.resize(0);
            elemPtr_P_OSQP.reserve(P_OSQP.nonZeros());
            for(int rowI = 0; rowI < linSysSolver->getNumRows(); rowI++) {
                for(const auto& colIter : linSysSolver->getIJ2aI()[rowI]) {
                    elemPtr_P_OSQP.emplace_back(&P_OSQP.coeffRef(rowI, colIter.first));
                    if(rowI != colIter.first) {
                        elemPtr_P_OSQP.emplace_back(&P_OSQP.coeffRef(colIter.first, rowI));
                    }
                }
            }
            
            l_OSQP.resize(result.V.rows());
            u_OSQP.resize(result.V.rows());
            u_OSQP.setConstant(INFINITY);
            dual_OSQP.resize(result.V.rows());
            
            //TODO: faster matrix set pattern
            A_OSQP.resize(result.V.rows(), result.V.rows() * dim);
            A_OSQP.setZero();
            A_OSQP.reserve(result.V.rows());
            for(int vI = 0; vI < result.V.rows(); ++vI) {
                A_OSQP.insert(vI, vI * dim + 1) = 1.0;
            }
            A_OSQP.makeCompressed();
        }
    }
    
    template<int dim>
    int Optimizer<dim>::solve(int maxIter)
    {
        static bool lastPropagate = false;
        int returnFlag = 0;
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
#ifndef STATIC_SOLVE
            animScripter.stepAnimScript(result, dt, energyTerms);
            if(energyTerms[0]->getNeedElemInvSafeGuard()) {
                if(!result.checkInversion()) {
                    std::cout << "scripted motion causes element inversion, end process" << std::endl;
                    exit(-1);
                }
            }
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
            
            assert(propagateFracture == 0);
            if(scaffolding) {
                scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                result.scaffold = &scaffold;
                scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
            }
        }
        return returnFlag;
    }
    
    template<int dim>
    void Optimizer<dim>::updatePrecondMtrAndFactorize(void)
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
    
    template<int dim>
    void Optimizer<dim>::setConfig(const TriangleSoup<dim>& config, int iterNum, int p_topoIter)
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
    
    template<int dim>
    void Optimizer<dim>::setPropagateFracture(bool p_prop)
    {
        propagateFracture = p_prop;
    }
    
    template<int dim>
    void Optimizer<dim>::setScaffolding(bool p_scaffolding)
    {
        scaffolding = p_scaffolding;
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
    }
    
    template<int dim>
    void Optimizer<dim>::updateEnergyData(bool updateEVal, bool updateGradient, bool updateHessian)
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
    
    template<int dim>
    void Optimizer<dim>::initX(int option)
    {
        //TODO: compute energy gradient and Hessian by PK
        
        // global:
        searchDir.resize(result.V.rows() * dim);
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
                        searchDir.segment<dim>(vI * dim).setZero();

                    }
                    else {
                        searchDir.segment<dim>(vI * dim) = dt * velocity.segment<dim>(vI * dim);
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
                        searchDir.segment<dim>(vI * dim).setZero();
                    }
                    else {
                        searchDir.segment<dim>(vI * dim) = dt * velocity.segment<dim>(vI * dim) + gravityDtSq;
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
                        searchDir.segment<dim>(vI * dim).setZero();
                    }
                    else {
                        double mass = result.massMatrix.coeff(vI, vI);
                        searchDir.segment<dim>(vI * dim) = (dt * velocity.segment<dim>(vI * dim) +
                                                        dtSq * (gravity - f.segment<dim>(vI * dim) / mass));
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
                        searchDir.segment<dim>(vI * dim).setZero();
                    }
                    else {
                        double mass = result.massMatrix.coeff(vI, vI);
                        searchDir.segment<dim>(vI * dim) = (dt * velocity.segment<dim>(vI * dim) +
                                                        dtSq / 2.0 * (gravity - f.segment<dim>(vI * dim) / mass));
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
                        searchDir.segment<dim>(vI * dim).setZero();
                    }
                    else {
                        searchDir[vI * dim] = -g[vI * dim] / linSysSolver->coeffMtr(vI * dim, vI * dim);
                        searchDir[vI * dim + 1] = -g[vI * dim + 1] / linSysSolver->coeffMtr(vI * dim + 1, vI * dim + 1);
                        searchDir[vI * dim + 2] = -g[vI * dim + 2] / linSysSolver->coeffMtr(vI * dim + 2, vI * dim + 2);
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
//        energyTerms[0]->filterStepSize(result, searchDir, stepSize);
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
        energyTerms[0]->filterStepSize(result, searchDir, stepSize);
        if(stepSize < 1.0) {
            //TODO: necessary?
            stepSize *= 0.5;
        }
        stepForward(result.V, Eigen::MatrixXd(), result, scaffold, stepSize);
    }
    
    template<int dim>
    void Optimizer<dim>::computeXTilta(void)
    {
        xTilta.conservativeResize(result.V.rows(), dim);
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
                                  (velocity.segment<dim>(vI * dim) * dt +
                                   gravityDtSq).transpose());
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    bool Optimizer<dim>::fullyImplicit(void)
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
            if(solveQP) {
                Eigen::VectorXd grad_KKT = gradient;
                Eigen::VectorXd fb(result.V.rows()); // Fischer-Burmeister measure for complementarity
                for(int vI = 0; vI < result.V.rows(); ++vI) {
                    const double &dual = dual_OSQP[vI];
                    grad_KKT[vI * dim + 1] += dual;
                    const double &constraint = result.V(vI, 1);
                    fb[vI] = dual + constraint - std::sqrt(dual * dual + constraint * constraint);
                    // nonlinear constraint would be different
                }
                sqn_g = grad_KKT.squaredNorm();
                
                std::cout << "FB norm: inf = " << fb.cwiseAbs().maxCoeff() <<
                    ", l2 = " << fb.norm() << std::endl;
            }
            else {
                sqn_g = gradient.squaredNorm();
            }
            if(!mute) {
                std::cout << "\t||gradient||^2 = " << sqn_g << std::endl;
            }
            file_iterStats << globalIterNum << " " << sqn_g << std::endl;
        } while(sqn_g > targetGRes);
        
        logFile << "Timestep" << globalIterNum << " innerIterAmt = " << innerIterAmt << std::endl;
        
        return false;
    }
    
    template<int dim>
    bool Optimizer<dim>::solve_oneStep(void)
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
                IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr_numFacFail",
                                                  linSysSolver, true);
                std::cout << "numerical factorization failed, " <<
                    "matrix written into " << outputFolderPath + "mtr_numFacFail" << std::endl;
                exit(-1);
            }
        }
        
        if(!mute) {
            std::cout << "back solve..." << std::endl;
        }
        if(!mute) { timer_step.start(4); }
        if(solveQP) {
            int elemPtrI = 0;
            for(int rowI = 0; rowI < linSysSolver->getNumRows(); rowI++) {
                for(const auto& colIter : linSysSolver->getIJ2aI()[rowI]) {
                    *elemPtr_P_OSQP[elemPtrI++] = linSysSolver->get_a()[colIter.second];
                    if(rowI != colIter.first) {
                        *elemPtr_P_OSQP[elemPtrI++] = linSysSolver->get_a()[colIter.second];
                    }
                }
            }
            
            for(int vI = 0; vI < result.V.rows(); vI++) {
                l_OSQP[vI] = -result.V(vI, 1) + animConfig.groundY;
            }
            
            QPSolver.setup(P_OSQP.valuePtr(), c_int(P_OSQP.nonZeros()),
                           P_OSQP.innerIndexPtr(), P_OSQP.outerIndexPtr(),
                           gradient.data(),
                           A_OSQP.valuePtr(), c_int(A_OSQP.nonZeros()),
                           A_OSQP.innerIndexPtr(), A_OSQP.outerIndexPtr(),
                           l_OSQP.data(), u_OSQP.data(), c_int(gradient.size()), c_int(result.V.rows()));
            searchDir.resize(gradient.size());
            memcpy(searchDir.data(), QPSolver.solve(), searchDir.size() * sizeof(searchDir[0]));
            memcpy(dual_OSQP.data(), QPSolver.getDual(), dual_OSQP.size() * sizeof(dual_OSQP[0]));
        }
        else {
            Eigen::VectorXd minusG = -gradient;
            linSysSolver->solve(minusG, searchDir);
        }
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
    
    template<int dim>
    bool Optimizer<dim>::lineSearch(void)
    {
        bool stopped = false;
        double stepSize;
        initStepSize(result, stepSize);
        if(!mute) {
            std::cout << "stepSize: " << stepSize << " -> ";
        }
        
        double lastEnergyVal_scaffold = 0.0;
        const double m = searchDir.dot(gradient);
        const double c1m = 1.0e-4 * m;
//        const double c1m = 0.0;
        Eigen::MatrixXd resultV0 = result.V;
//        TriangleSoup<dim> temp = result; //TEST
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
            //TODO: for nonlinear constraint also needs to enforce feasibility here!
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
        
        assert(result.checkInversion());

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
        
//        // projection method for collision handling
//        for(int vI = 0; vI < result.V.rows(); vI++) {
//            if(result.V(vI, 1) < 0.0) {
//                result.V(vI, 1) = 0.0;
//            }
//        }
//        computeEnergyVal(result, scaffold, true, testingE);
        
        lastEDec = lastEnergyVal - testingE;
        if(scaffolding) {
            lastEDec += (-lastEnergyVal_scaffold + energyVal_scaffold);
        }
//        lastEDec = (lastEnergyVal - testingE) / stepSize;
//        if(allowEDecRelTol && (lastEDec / lastEnergyVal / stepSize < 1.0e-6)) {
        if(allowEDecRelTol && (lastEDec / lastEnergyVal < 1.0e-3)) {
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
    
    template<int dim>
    void Optimizer<dim>::stepForward(const Eigen::MatrixXd& dataV0,
                                     const Eigen::MatrixXd& scaffoldV0,
                                     TriangleSoup<dim>& data,
                                     Scaffold& scaffoldData,
                                     double stepSize) const
    {
        assert(dataV0.rows() == data.V.rows());
        if(scaffolding) {
            assert(data.V.rows() + scaffoldData.airMesh.V.rows() - scaffoldData.bnd.size() == searchDir.size() / dim);
        }
        else {
            assert(data.V.rows() * dim == searchDir.size());
        }
        assert(data.V.rows() == result.V.rows());
        
        for(int vI = 0; vI < data.V.rows(); vI++) {
            data.V.row(vI) = dataV0.row(vI) + stepSize * searchDir.segment<dim>(vI * dim).transpose();
        }
        if(scaffolding) {
            scaffoldData.stepForward(scaffoldV0, searchDir, stepSize);
        }
    }
    
    template<int dim>
    void Optimizer<dim>::updateTargetGRes(double p_tol)
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
//        assert(energyParamSum == 1.0);
        targetGRes = relGL2Tol * m_sqn * gravity.squaredNorm();
#ifndef STATIC_SOLVE
        targetGRes *= dtSq * dtSq;
#endif
        targetGRes = p_tol;
    }
    
    template<int dim>
    void Optimizer<dim>::getGradientVisual(Eigen::MatrixXd& arrowVec) const
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
     
    template<int dim>
    void Optimizer<dim>::getFaceFieldForVis(Eigen::VectorXd& field) const
    {
        field = Eigen::VectorXd::Zero(result.F.rows());
    }
    template<int dim>
    void Optimizer<dim>::getSharedVerts(Eigen::VectorXi& sharedVerts) const
    {
        sharedVerts.resize(0);
    }
    
    template<int dim>
    void Optimizer<dim>::initStepSize(const TriangleSoup<dim>& data, double& stepSize) const
    {
        stepSize = 1.0;
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->filterStepSize(data, searchDir, stepSize);
        }
        
        if(scaffolding) {
            Eigen::VectorXd searchDir_scaffold;
            scaffold.wholeSearchDir2airMesh(searchDir, searchDir_scaffold);
            SymStretchEnergy<dim> SD;
            SD.filterStepSize(scaffold.airMesh, searchDir_scaffold, stepSize);
        }
    }
    
    template<int dim>
    void Optimizer<dim>::writeEnergyValToFile(bool flush)
    {
        double E_se = 0.0;
        
        buffer_energyValPerIter << lastEnergyVal + (1.0 - energyParams[0]) * E_se;
        
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            buffer_energyValPerIter << " " << energyVal_ET[eI];
        }
        
        buffer_energyValPerIter << " " << E_se << " " << energyParams[0] << "\n";
        
        if(flush) {
            flushEnergyFileOutput();
        }
    }
    template<int dim>
    void Optimizer<dim>::writeGradL2NormToFile(bool flush)
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
    template<int dim>
    void Optimizer<dim>::flushEnergyFileOutput(void)
    {
        file_energyValPerIter << buffer_energyValPerIter.str();
        file_energyValPerIter.flush();
        clearEnergyFileOutputBuffer();
    }
    template<int dim>
    void Optimizer<dim>::flushGradFileOutput(void)
    {
        file_gradientPerIter << buffer_gradientPerIter.str();
        file_gradientPerIter.flush();
        clearGradFileOutputBuffer();
    }
    template<int dim>
    void Optimizer<dim>::clearEnergyFileOutputBuffer(void)
    {
        buffer_energyValPerIter.str("");
        buffer_energyValPerIter.clear();
    }
    template<int dim>
    void Optimizer<dim>::clearGradFileOutputBuffer(void)
    {
        buffer_gradientPerIter.str("");
        buffer_gradientPerIter.clear();
    }
    
    template<int dim>
    void Optimizer<dim>::computeEnergyVal(const TriangleSoup<dim>& data, const Scaffold& scaffoldData,
                                     bool redoSVD, double& energyVal, bool excludeScaffold)
    {
        //TODO: write inertia and augmented Lagrangian term into energyTerms
        if(!mute) { timer_step.start(0); }
        
        energyTerms[0]->computeEnergyVal(data, redoSVD, svd, F,
                                         dtSq * energyParams[0], energyVal_ET[0]);
        energyVal =  energyVal_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeEnergyVal(data, redoSVD, svd, F,
                                              dtSq * energyParams[eI], energyVal_ET[eI]);
            energyVal += energyVal_ET[eI];
        }
        
//        if(scaffolding && (!excludeScaffold)) {
//            SymStretchEnergy<dim> SD;
//            SD.computeEnergyVal(scaffoldData.airMesh, energyVal_scaffold, true);
//            energyVal_scaffold *= w_scaf / scaffold.airMesh.F.rows();
//            energyVal += energyVal_scaffold;
//        }
//        else {
            energyVal_scaffold = 0.0;
//        }
        
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
    template<int dim>
    void Optimizer<dim>::computeGradient(const TriangleSoup<dim>& data, const Scaffold& scaffoldData,
                                    bool redoSVD, Eigen::VectorXd& gradient, bool excludeScaffold)
    {
        if(!mute) { timer_step.start(0); }
        
        energyTerms[0]->computeGradient(data, redoSVD, svd, F,
                                        dtSq * energyParams[0], gradient_ET[0]);
        gradient = gradient_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeGradient(data, redoSVD, svd, F,
                                             dtSq * energyParams[eI], gradient_ET[eI]);
            gradient += gradient_ET[eI];
        }
        
//        if(scaffolding) {
//            SymStretchEnergy<dim> SD;
//            SD.computeGradient(scaffoldData.airMesh, gradient_scaffold, true);
//            scaffoldData.augmentGradient(gradient, gradient_scaffold, (excludeScaffold ? 0.0 : (w_scaf / scaffold.airMesh.F.rows())));
//        }
        
#ifndef STATIC_SOLVE
        timer_temp.start(5);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            if(!data.isFixedVert[vI]) {
                gradient.segment<dim>(vI * dim) += (data.massMatrix.coeff(vI, vI) *
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
    template<int dim>
    void Optimizer<dim>::computePrecondMtr(const TriangleSoup<dim>& data,
                                           const Scaffold& scaffoldData,
                                           bool redoSVD,
                                           LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *p_linSysSolver)
    {
        if(!mute) { timer_step.start(0); }
        
        p_linSysSolver->setZero();
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeHessian(data, redoSVD, svd, F,
                                            energyParams[eI] * dtSq,
                                            p_linSysSolver);
        }
//        Eigen::VectorXi I, J;
//        Eigen::VectorXd V;
//        energyTerms[0]->computeHessianBySVD(data, &V, &I, &J, true);
//        V *= energyParams[0] * dtSq;
//        p_linSysSolver->update_a(I, J, V);
        
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
                int ind0 = vI * dim;
                int ind1 = ind0 + 1;
                p_linSysSolver->addCoeff(ind0, ind0, massI);
                p_linSysSolver->addCoeff(ind1, ind1, massI);
                if(dim == 3) {
                    int ind2 = ind0 + 2;
                    p_linSysSolver->addCoeff(ind2, ind2, massI);
                }
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
    template<int dim>
    void Optimizer<dim>::computeHessian(const TriangleSoup<dim>& data, const Scaffold& scaffoldData, Eigen::SparseMatrix<double>& hessian) const
    {
//        energyTerms[0]->computeHessian(data, hessian);
//        hessian *= energyParams[0];
//        for(int eI = 1; eI < energyTerms.size(); eI++) {
//            Eigen::SparseMatrix<double> hessianI;
//            energyTerms[eI]->computeHessian(data, hessianI);
//            hessian += energyParams[eI] * hessianI;
//        }
//        
//        //TODO: SCAFFOLDING
    }
    
    template<int dim>
    double Optimizer<dim>::getLastEnergyVal(bool excludeScaffold) const
    {
        return ((excludeScaffold && scaffolding) ?
                (lastEnergyVal - energyVal_scaffold) :
                lastEnergyVal);
    }
            
    template class Optimizer<DIM>;
}
