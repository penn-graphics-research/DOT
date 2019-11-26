//
//  Optimizer.cpp
//  DOT
//
//  Created by Minchen Li on 8/31/17.
//

#include "Optimizer.hpp"

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
#include <igl/face_areas.h>
#include <igl/edge_lengths.h>
#include <igl/writeOBJ.h>

#ifdef USE_TBB
    #include <tbb/tbb.h>
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <numeric>

#include <sys/stat.h> // for mkdir

extern const std::string outputFolderPath;

extern std::ofstream logFile;
extern Timer timer, timer_step, timer_temp, timer_temp2;

extern Eigen::MatrixXi SF;
extern std::vector<bool> isSurfNode;
extern std::vector<int> tetIndToSurf;
extern std::vector<int> surfIndToTet;
extern Eigen::MatrixXd V_surf;
extern Eigen::MatrixXi F_surf;

namespace DOT {
    
    template<int dim>
    Optimizer<dim>::Optimizer(const Mesh<dim>& p_data0,
                         const std::vector<Energy<dim>*>& p_energyTerms, 
                         const std::vector<double>& p_energyParams,
                         bool p_mute, const Config& p_animConfig) :
        data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams), animConfig(p_animConfig)
    {
        assert(energyTerms.size() == energyParams.size());

        {
            int ceil_t_size = std::ceil(p_data0.F.rows() / 4.f) * 4;
            U.resize(ceil_t_size);
            V.resize(ceil_t_size);
            Sigma.resize(ceil_t_size);
        }

        energyParamSum = 0.0;
        for(const auto& ePI : energyParams) {
            energyParamSum += ePI;
        }
        
        gradient_ET.resize(energyTerms.size());
        energyVal_ET.resize(energyTerms.size());
        
        allowEDecRelTol = true;
        mute = p_mute;
        
        if(!mute) {
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
        relGL2Tol = 1.0e-8;
        innerIterAmt = 0;
        
        
        needRefactorize = false;
        for(const auto& energyTermI : energyTerms) {
            if(energyTermI->getNeedRefactorize()) {
                needRefactorize = true;
                break;
            }
        }
        
        
//        pardisoThreadAmt = 0;
        pardisoThreadAmt = 1;
        
        gravity.setZero();
        if(animConfig.withGravity) {
            gravity[1] = -9.80665;
        }
        setTime(10.0, 0.025);
        std::cout << "dt and tol initialized" << std::endl;
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        linSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        linSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        
        setAnimScriptType(animConfig.animScriptType);
        
        result = data0;
        animScripter.initAnimScript(result); //TODO: check compatibility with restart
        if(animConfig.restart) {
            std::ifstream in(animConfig.statusPath.c_str());
            assert(in.is_open());
            std::string line;
            while(std::getline(in, line)) {
                std::stringstream ss(line);
                std::string token;
                ss >> token;
                if(token == "timestep") {
                    ss >> globalIterNum;
                }
                else if(token == "position") {
                    std::cout << "read restart position" << std::endl;
                    int posRows, dim_in;
                    ss >> posRows >> dim_in;
                    assert(posRows == result.V.rows());
                    assert(dim_in == result.V.cols());
                    for(int vI = 0; vI < posRows; ++vI) {
                        in >> result.V(vI, 0) >> result.V(vI, 1);
                        if(dim == 3) {
                            in >> result.V(vI, 2);
                        }
                    }
                }
                else if(token == "velocity") {
                    std::cout << "read restart velocity" << std::endl;
                    int velDim;
                    ss >> velDim;
                    assert(velDim == result.V.rows() * dim);
                    velocity.conservativeResize(velDim);
                    for(int velI = 0; velI < velDim; ++velI) {
                        in >> velocity[velI];
                    }
                }
                else if(token == "dx_Elastic") {
                    std::cout << "read restart dx_Elastic" << std::endl;
                    int dxERows, dim_in;
                    ss >> dxERows >> dim_in;
                    assert(dxERows == result.V.rows());
                    assert(dim_in == dim);
                    dx_Elastic.conservativeResize(dxERows, dim);
                    for(int vI = 0; vI < dxERows; ++vI) {
                        in >> dx_Elastic(vI, 0) >> dx_Elastic(vI, 1);
                        if(dim == 3) {
                            in >> dx_Elastic(vI, 2);
                        }
                    }
                }
            }
            in.close();
            //TODO: load acceleration, also save acceleration in the saveStatus() function
        }
        else {
            velocity = Eigen::VectorXd::Zero(result.V.rows() * dim);
            dx_Elastic = Eigen::MatrixXd::Zero(result.V.rows(), dim);
            acceleration = Eigen::MatrixXd::Zero(result.V.rows(), dim);
        }
        resultV_n = result.V;
        computeXTilta();
        
        svd.resize(result.F.rows());
        F.resize(result.F.rows());
        std::cout << "animScriptor set" << std::endl;
        
        lastEDec = 0.0;
        updateTargetGRes();
        
        std::cout << "Newton's solver for Backward Euler constructed" << std::endl;
        
        numOfLineSearch = 0;
    }
    
    template<int dim>
    Optimizer<dim>::~Optimizer(void)
    {
        if(file_iterStats.is_open()) {
            file_iterStats.close();
        }
        delete linSysSolver;
    }
    
    template<int dim>
    Mesh<dim>& Optimizer<dim>::getResult(void) {
        return result;
    }
    
    template<int dim>
    int Optimizer<dim>::getIterNum(void) const {
        return globalIterNum;
    }
    template<int dim>
    int Optimizer<dim>::getInnerIterAmt(void) const {
        return innerIterAmt;
    }
    
    template<int dim>
    void Optimizer<dim>::setRelGL2Tol(double p_relTol)
    {
        assert(p_relTol > 0.0);
        relGL2Tol = p_relTol * p_relTol;
        updateTargetGRes();
        logFile << globalIterNum << "th tol: " << targetGRes << std::endl;
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
        computeXTilta();
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
        linSysSolver->set_pattern(result.vNeighbor, result.fixedVert);
        if(!mute) { timer_step.stop(); }
        std::cout << "precompute: sparse matrix allocated" << std::endl;
        computePrecondMtr(result, true, linSysSolver);
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
        
        computeEnergyVal(result, false, lastEnergyVal);
        if(!mute) {
            std::cout << "E_initial = " << lastEnergyVal << std::endl;
        }
        
        double sysE;
        computeSystemEnergy(sysE);
        logFile << "sysE = " << sysE << std::endl;
    }
    
    template<int dim>
    int Optimizer<dim>::solve(int maxIter)
    {
        static bool lastPropagate = false;
        int returnFlag = 0;
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
            timer_step.start(11);
            if(animScripter.stepAnimScript(result, dt, energyTerms)) {
                updatePrecondMtrAndFactorize();
            }
            timer_step.stop();
            if(!mute) { timer.start(0); }

            if(globalIterNum >= frameAmt) {
                // converged
                lastEDec = 0.0;
                globalIterNum++;
                if(!mute) { timer.stop(); }
                return 1;
            }
            else {

                if(fullyImplicit()) {
                    returnFlag = 2;
                }
                
                timer_step.start(11);
                switch(animConfig.timeIntegrationType) {
                    case TIT_BE:
                        dx_Elastic = result.V - xTilta;
                        velocity = Eigen::Map<Eigen::MatrixXd>(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>((result.V - resultV_n).array() / dt).data(), velocity.rows(), 1);
                        resultV_n = result.V;
                        computeXTilta();
                        break;
                }
                timer_step.stop();
            }
            globalIterNum++;
            if(!mute) { timer.stop(); }
        }
        return returnFlag;
    }

    template<int dim>
    void Optimizer<dim>::updatePrecondMtrAndFactorize(void)
    {
//        if(needRefactorize) {
//            // don't need to call this function
//            return;
//        }

        if(!mute) {
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
        }
        
        if(!mute) { timer_step.start(1); }
        linSysSolver->set_pattern(result.vNeighbor, result.fixedVert);
        if(!mute) { timer_step.start(2); }
        linSysSolver->analyze_pattern();
        if(!mute) { timer_step.start(3); }
        
        computePrecondMtr(result, true, linSysSolver);

        if(!mute) { timer_step.start(3); }
        linSysSolver->factorize();
        if(!mute) { timer_step.stop(); }
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
            computeEnergyVal(result, true, lastEnergyVal);
        }

        if(updateGradient) {
            // compute gradient and output
            computeGradient(result, false, gradient);
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
            linSysSolver->set_pattern(result.vNeighbor, result.fixedVert);
            if(!mute) { timer_step.stop(); }

            computePrecondMtr(result, false, linSysSolver);

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
        // global:
        searchDir.conservativeResize(result.V.rows() * dim);
        switch(option) {
            case 0:
                // already at last timestep config
                searchDir.setZero();
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
                switch(animConfig.timeIntegrationType) {
                    case TIT_BE:
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
                }
                break;

            case 3: { // Symplectic Euler
                switch(animConfig.timeIntegrationType) {
                    case TIT_BE:
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
                                searchDir.segment<dim>(vI * dim) = (dt * velocity.segment<dim>(vI * dim) +
                                                                    gravityDtSq +
                                                                    dx_Elastic.row(vI).transpose());
                            }
                        }
#ifdef USE_TBB
                        );
#endif
                        break;
                }
                break;
            }

            case 4: { // uniformly accelerated motion approximation
                switch(animConfig.timeIntegrationType) {
                    case TIT_BE:
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
                                searchDir.segment<dim>(vI * dim) = (dt * velocity.segment<dim>(vI * dim) + (gravityDtSq + 0.5 * dx_Elastic.row(vI).transpose()));
                            }
                        }
#ifdef USE_TBB
                        );
#endif
                        break;
                }
                break;
            }

            case 5: { // Jacobi
                Eigen::VectorXd g;
                computeGradient(result, true, g);

                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                computePrecondMtr(result, false, linSysSolver);

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

        double stepSize = 1.0;
        stepForward(result.V, result, stepSize);
    }

    template<int dim>
    void Optimizer<dim>::computeXTilta(void)
    {
        xTilta.conservativeResize(result.V.rows(), dim);
        switch(animConfig.timeIntegrationType) {
            case TIT_BE:
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
                break;
        }
    }
        
    template<int dim>
    double Optimizer<dim>::computeCharNormSq(const Mesh<dim>& mesh,
                                             const Energy<dim>* energy,
                                             double epsSq_c) const
    {
        Eigen::Matrix<double, dim * dim, dim * dim> d2E_div_dF2_rest;
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd_iden(Eigen::Matrix<double, dim, dim>::
                                                              Identity(),
                                                              Eigen::ComputeFullU |
                                                              Eigen::ComputeFullV);
        energy->compute_dP_div_dF(svd_iden,
                                  mesh.u[0], mesh.lambda[0], // not for inhomogenous material!
                                  d2E_div_dF2_rest,
                                  1.0, false);
        double sqnorm_H_rest = d2E_div_dF2_rest.squaredNorm();
        
        Eigen::MatrixXd edgeLens;
        if(dim == 2) {
            igl::edge_lengths(mesh.V_rest, mesh.F, edgeLens);
        }
        else {
            igl::face_areas(mesh.V_rest, mesh.F, edgeLens);
        }
        Eigen::VectorXd ls;
        ls.resize(mesh.V_rest.rows());
        ls.setZero();
        for(int triI = 0; triI < mesh.F.rows(); triI++) {
            for(int i = 0; i < dim + 1; i++) {
                ls[mesh.F(triI, i)] += edgeLens(triI, i);
            }
        }
        double sqnorm_l = ls.squaredNorm();
        
        double charNormSq = (epsSq_c * sqnorm_H_rest * sqnorm_l *
                             (mesh.V_rest.rows() - mesh.fixedVert.size()) / mesh.V_rest.rows() *
                             energyParamSum * energyParamSum);
        charNormSq *= dtSq * dtSq;
        
        return charNormSq;
    }

    template<int dim>
    bool Optimizer<dim>::fullyImplicit(void)
    {
        timer_step.start(10);
        initX(animConfig.warmStart);

        double sqn_g = __DBL_MAX__;
        computeEnergyVal(result, true, lastEnergyVal);
        computeGradient(result, false, gradient);
        timer_step.stop();
        std::cout << "after initX E = " << lastEnergyVal <<
            " ||g||^2 = " << gradient.squaredNorm() << ", tol = " <<
            targetGRes << std::endl;
        file_iterStats << globalIterNum << " 0 " << lastEnergyVal << " " << gradient.squaredNorm() << " 0" << std::endl;
        Eigen::VectorXd fb;
        int iterCap = 10000, curIterI = 0;
        do {
            file_iterStats << globalIterNum << " ";

            if(solve_oneStep()) {
                std::cout << "\tline search with Armijo's rule failed!!!" << std::endl;
                logFile << "\tline search with Armijo's rule failed!!!" << std::endl;
                return true;
            }
            innerIterAmt++;

            timer_step.start(10);
            sqn_g = gradient.squaredNorm();
            if(!mute) {
                std::cout << "\t||gradient||^2 = " << sqn_g << std::endl;
            }
            file_iterStats << lastEnergyVal << " " << sqn_g << " " << 0 << std::endl;
            timer_step.stop();
            
            if(++curIterI >= iterCap) {
                break;
            }
        } while(sqn_g > targetGRes);

        logFile << "Timestep" << globalIterNum << " innerIterAmt = " << innerIterAmt <<
            ", accumulated line search steps " << numOfLineSearch << std::endl;
        
        double sysE;
        computeSystemEnergy(sysE);
        logFile << "sysE = " << sysE << std::endl;

        return (curIterI >= iterCap);
    }
    
    template<int dim>
    bool Optimizer<dim>::solve_oneStep(void)
    {
        if(needRefactorize) {
            // for the changing hessian
            if(!mute) {
                std::cout << "updating matrix entries..." << std::endl;
            }
            computePrecondMtr(result, false, linSysSolver);

            try {
                if(!mute) {
                    std::cout << "numerically factorizing Hessian/Proxy matrix..." << std::endl;
                }
                if(!mute) { timer_step.start(3); }
                linSysSolver->factorize();
                // output factorization and exit
//                    linSysSolver->outputFactorization("/Users/mincli/Desktop/test/DOT/output/L");
//                    std::cout << "matrix written" << std::endl;
//                    exit(0);
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
        Eigen::VectorXd minusG = -gradient;
        linSysSolver->solve(minusG, searchDir);
        if(!mute) { timer_step.stop(); }
        
        double stepSize;
        bool stopped = lineSearch(stepSize);
        if(stopped) {
//            IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_stopped_" + std::to_string(globalIterNum), precondMtr);
//            logFile << "descent step stopped at overallIter" << globalIterNum << " for no prominent energy decrease." << std::endl;
        }
        computeGradient(result, false, gradient);
        return stopped;
    }
    
    template<int dim>
    bool Optimizer<dim>::lineSearch(double& stepSize,
                                    double armijoParam,
                                    double lowerBound)
    {
        timer_step.start(5);
        
        bool outputLineSearch = false;
        if(outputLineSearch) {
            result.saveAsMesh(outputFolderPath +
                              ((dim == 2) ? "lineSearchBase.obj" : "lineSearchBase.msh"));
        }
        
        bool stopped = false;
        
        initStepSize(result, stepSize);
        if(!mute) {
            std::cout << "stepSize: " << stepSize << " -> ";
        }
        
//        const double m = searchDir.dot(gradient);
//        const double c1m = 1.0e-4 * m;
        double c1m = 0.0;
        if(armijoParam > 0.0) {
            c1m = armijoParam * searchDir.dot(gradient);
        }
        Eigen::MatrixXd resultV0 = result.V;
//        Mesh<dim> temp = result; //TEST
        
        if(outputLineSearch) {
            Eigen::VectorXd energyValPerElem;
            energyTerms[0]->getEnergyValPerElemBySVD(result, false, svd, F, U, V, Sigma,
                                                     energyValPerElem, false);
            IglUtils::writeVectorToFile(outputFolderPath + "E0.txt", energyValPerElem);
        }
        
        stepForward(resultV0, result, stepSize);
        double testingE;
//        Eigen::VectorXd testingG;
        timer_step.start(9);
        computeEnergyVal(result, 2, testingE);
        timer_step.start(5);
//        computeGradient(testingData, testingG);
        
        if(outputLineSearch) {
            Eigen::VectorXd energyValPerElem;
            energyTerms[0]->getEnergyValPerElemBySVD(result, false, svd, F, U, V, Sigma,
                                                     energyValPerElem, false);
            IglUtils::writeVectorToFile(outputFolderPath + "E.txt", energyValPerElem);
        }

#define ARMIJO_RULE
#ifdef ARMIJO_RULE
//        while((testingE > lastEnergyVal + stepSize * c1m) ||
//              (searchDir.dot(testingG) < c2m)) // Wolfe condition
        while((testingE > lastEnergyVal + stepSize * c1m) && // Armijo condition
              (stepSize > lowerBound))
//        while(0)
        {
            if(stepSize == 1.0) {
                // can try cubic interpolation here
                stepSize /= 2.0;
            }
            else {
                stepSize /= 2.0;
            }

            ++numOfLineSearch;
            if(stepSize == 0.0) {
                stopped = true;
                if(!mute) {
                    logFile << "testingE" << globalIterNum << " " << testingE << " > " << lastEnergyVal << " " << stepSize * c1m << std::endl;
//                    logFile << "testingG" << globalIterNum << " " << searchDir.dot(testingG) << " < " << c2m << std::endl;
                }
                break;
            }

            stepForward(resultV0, result, stepSize);
            timer_step.start(9);
            computeEnergyVal(result, 2, testingE);
            timer_step.start(5);
//            computeGradient(testingData, testingG);
        }
#endif
        
        if(!mute) {
            std::cout << stepSize << "(armijo) ";
        }

#ifdef USE_SIMD

#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < result.F.rows(); triI++)
#endif
        {
            svd[triI].set(U[triI], Sigma[triI], V[triI]);
        }
#ifdef USE_TBB
        );
#endif

#endif
        
        lastEDec = lastEnergyVal - testingE;
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
            std::cout << "E_cur_smooth = " << testingE << std::endl;
        }
        file_iterStats << stepSize << " ";
        
        if(outputLineSearch) {
            IglUtils::writeVectorToFile(outputFolderPath + "searchDir.txt",
                                        searchDir);
            exit(0);
        }
        
        timer_step.stop();
        
        return stopped;
    }
     
    template<int dim>
    void Optimizer<dim>::dimSeparatedSolve(std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*>& linSysSolver,
                                   const Eigen::VectorXd& rhs,
                                   Eigen::MatrixXd& V)
    {
        assert(linSysSolver[0]->getNumRows() == V.rows());
        assert(linSysSolver[1]->getNumRows() == V.rows());
        if(dim == 3) {
            assert(linSysSolver[2]->getNumRows() == V.rows());
        }
        assert(rhs.size() % dim == 0);
        assert(rhs.size() / dim == V.rows());
        
        Eigen::VectorXd srhs[dim];
        srhs[0].conservativeResize(V.rows());
        srhs[1].conservativeResize(V.rows());
        if(dim == 3) {
            srhs[2].conservativeResize(V.rows());
        }
#ifdef USE_TBB
        tbb::parallel_for(0, (int)V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < V.rows(); ++vI)
#endif
        {
            srhs[0][vI] = rhs[vI * dim];
            srhs[1][vI] = rhs[vI * dim + 1];
            if(dim == 3) {
                srhs[2][vI] = rhs[vI * dim + 2];
            }
        }
#ifdef USE_TBB
        );
#endif
        
        Eigen::VectorXd solved[dim];
#if defined(USE_TBB) && !defined(LINSYSSOLVER_USE_PARDISO)
        tbb::parallel_for(0, (int)dim, 1, [&](int dimI)
        {
            linSysSolver[dimI]->solve_threadSafe(srhs[dimI], solved[dimI], dimI);
#else
        for(int dimI = 0; dimI < dim; ++dimI)
        {
            linSysSolver[dimI]->solve(srhs[dimI], solved[dimI]);
#endif
        }
#if defined(USE_TBB) && !defined(LINSYSSOLVER_USE_PARDISO)
        );
#endif
                              
#ifdef USE_TBB
        tbb::parallel_for(0, (int)V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < V.rows(); vI++)
#endif
        {
            V(vI, 0) = solved[0][vI];
            V(vI, 1) = solved[1][vI];
            if(dim == 3) {
                V(vI, 2) = solved[2][vI];
            }
        }
#ifdef USE_TBB
        );
#endif
    }
       
    template<int dim>
    void Optimizer<dim>::dimSeparatedSolve(LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                           const Eigen::VectorXd& rhs,
                                           Eigen::MatrixXd& V)
    {
        std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*> linSysSolvers(3, linSysSolver);
        dimSeparatedSolve(linSysSolvers, rhs, V);
    }
    //TODO: merge with above
    template<int dim>
    void Optimizer<dim>::dimSeparatedSolve(LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                           const Eigen::VectorXd& rhs,
                                           Eigen::VectorXd& p_solved)
    {
        assert(linSysSolver->getNumRows() == rhs.size() / dim);
        assert(rhs.size() % dim == 0);
        int nV = linSysSolver->getNumRows();
        
        Eigen::VectorXd srhs[dim];
        srhs[0].conservativeResize(nV);
        srhs[1].conservativeResize(nV);
        if(dim == 3) {
            srhs[2].conservativeResize(nV);
        }
#ifdef USE_TBB
        tbb::parallel_for(0, (int)nV, 1, [&](int vI)
#else
        for(int vI = 0; vI < nV; ++vI)
#endif
        {
            srhs[0][vI] = rhs[vI * dim];
            srhs[1][vI] = rhs[vI * dim + 1];
            if(dim == 3) {
                srhs[2][vI] = rhs[vI * dim + 2];
            }
        }
#ifdef USE_TBB
        );
#endif
        
        Eigen::VectorXd solved[dim];
#if defined(USE_TBB) && !defined(LINSYSSOLVER_USE_PARDISO)
        tbb::parallel_for(0, (int)dim, 1, [&](int dimI)
        {
            linSysSolver->solve_threadSafe(srhs[dimI], solved[dimI], dimI);
#else
        for(int dimI = 0; dimI < dim; ++dimI)
        {
            linSysSolver->solve(srhs[dimI], solved[dimI]);
#endif
        }
#if defined(USE_TBB) && !defined(LINSYSSOLVER_USE_PARDISO)
        );
#endif
                              
#ifdef USE_TBB
        tbb::parallel_for(0, (int)nV, 1, [&](int vI)
#else
        for(int vI = 0; vI < nV; vI++)
#endif
        {
            p_solved[vI * dim] = solved[0][vI];
            p_solved[vI * dim + 1] = solved[1][vI];
            if(dim == 3) {
                p_solved[vI * dim + 2] = solved[2][vI];
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    void Optimizer<dim>::stepForward(const Eigen::MatrixXd& dataV0,
                                     Mesh<dim>& data,
                                     double stepSize) const
    {
        assert(dataV0.rows() == data.V.rows());
        assert(data.V.rows() * dim == searchDir.size());
        assert(data.V.rows() == result.V.rows());
        
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            data.V.row(vI) = dataV0.row(vI) + stepSize * searchDir.segment<dim>(vI * dim).transpose();
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template<int dim>
    void Optimizer<dim>::updateTargetGRes(void)
    {
        targetGRes = computeCharNormSq(data0, energyTerms[0], relGL2Tol);
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
    void Optimizer<dim>::getFaceFieldForVis(Eigen::VectorXd& field)
    {
//        field = Eigen::VectorXd::Zero(result.F.rows());
        field = result.u;
    }
    template<int dim>
    void Optimizer<dim>::getSharedVerts(Eigen::VectorXi& sharedVerts) const
    {
        sharedVerts.resize(0);
    }
    
    template<int dim>
    void Optimizer<dim>::initStepSize(const Mesh<dim>& data, double& stepSize) const
    {
#ifdef ALPHAINIT
        if(animConfig.timeStepperType == TST_DOT) {
            assert(linSysSolver);

            Eigen::VectorXd Hp;
            linSysSolver->multiply(searchDir, Hp);
            stepSize = std::max(0.1, std::min(1.0, -searchDir.dot(gradient) / searchDir.dot(Hp)));
//            stepSize = std::min(1.0, -searchDir.dot(gradient) / searchDir.dot(Hp));
        }
        else {
#endif
            stepSize = 1.0;
#ifdef ALPHAINIT
        }
#endif
    }
            
    template<int dim>
    void Optimizer<dim>::saveStatus(void)
    {
        FILE *out = fopen((outputFolderPath + "status" + std::to_string(globalIterNum)).c_str(), "w");
        assert(out);
        
        fprintf(out, "timestep %d\n", globalIterNum);
        
        fprintf(out, "\nposition %ld %ld\n", result.V.rows(), result.V.cols());
        for(int vI = 0; vI < result.V.rows(); ++vI) {
            fprintf(out, "%le %le", result.V(vI, 0),
                    result.V(vI, 1));
            if(dim == 3) {
                fprintf(out, " %le\n", result.V(vI, 2));
            }
            else {
                fprintf(out, "\n");
            }
        }
        
        fprintf(out, "\nvelocity %ld\n", velocity.size());
        for(int velI = 0; velI < velocity.size(); ++velI) {
            fprintf(out, "%le\n", velocity[velI]);
        }
        
        fprintf(out, "\ndx_Elastic %ld %d\n", dx_Elastic.rows(), dim);
        for(int velI = 0; velI < dx_Elastic.rows(); ++velI) {
            fprintf(out, "%le %le", dx_Elastic(velI, 0),
                    dx_Elastic(velI, 1));
            if(dim == 3) {
                fprintf(out, " %le\n", dx_Elastic(velI, 2));
            }
            else {
                fprintf(out, "\n");
            }
        }
        
        fclose(out);
        
        // surface mesh
#if (DIM == 3)

#ifdef USE_TBB
        tbb::parallel_for(0, (int)V_surf.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < V_surf.rows(); ++vI)
#endif
        {
            V_surf.row(vI) = result.V.row(surfIndToTet[vI]);
        }
#ifdef USE_TBB
        );
#endif
        igl::writeOBJ(outputFolderPath +
                      std::to_string(globalIterNum) +
                      ".obj", V_surf, F_surf);

#else

        Eigen::MatrixXd V_output = result.V;
        V_output.conservativeResize(V_output.rows(), 3);
        V_output.col(2).setZero();
        igl::writeOBJ(outputFolderPath +
                      std::to_string(globalIterNum) +
                      ".obj", V_output, result.F);

#endif
    }

    template<int dim>
    const Eigen::MatrixXd& Optimizer<dim>::getDenseMatrix(int sbdI) const
    {
        assert(0 && "please implement!");
    }

    template<int dim>
    void Optimizer<dim>::getGradient(int sbdI, Eigen::VectorXd& gradient)
    {
        assert(0 && "please implement!");
    }

    template<int dim>
    LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* Optimizer<dim>::getSparseSolver(int sbdI)
    {
        assert(0 && "please implement!");
    }
    
    template<int dim>
    void Optimizer<dim>::computeEnergyVal(const Mesh<dim>& data,
                                          int redoSVD, double& energyVal)
    {
        //TODO: write inertia and augmented Lagrangian term into energyTerms
//        if(!mute) { timer_step.start(0); }
        switch(animConfig.timeIntegrationType) {
            case TIT_BE: {
                energyTerms[0]->computeEnergyVal(data, redoSVD, svd, F, U, V, Sigma,
                                                 dtSq * energyParams[0], energyVal_ET[0]);
                energyVal =  energyVal_ET[0];
                for(int eI = 1; eI < energyTerms.size(); eI++) {
                    energyTerms[eI]->computeEnergyVal(data, redoSVD, svd, F, U, V, Sigma,
                                                      dtSq * energyParams[eI], energyVal_ET[eI]);
                    energyVal += energyVal_ET[eI];
                }
                break;
            }
        }
        
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
        
//        if(!mute) { timer_step.stop(); }
    }
    template<int dim>
    void Optimizer<dim>::computeGradient(const Mesh<dim>& data,
                                         bool redoSVD, Eigen::VectorXd& gradient)
    {
//        if(!mute) { timer_step.start(0); }
        
        switch(animConfig.timeIntegrationType) {
            case TIT_BE: {
                energyTerms[0]->computeGradient(data, redoSVD, svd, F, U, V, Sigma,
                                                dtSq * energyParams[0], gradient_ET[0]);
                gradient = gradient_ET[0];
                for(int eI = 1; eI < energyTerms.size(); eI++) {
                    energyTerms[eI]->computeGradient(data, redoSVD, svd, F, U, V, Sigma,
                                                     dtSq * energyParams[eI], gradient_ET[eI]);
                    gradient += gradient_ET[eI];
                }
                break;
            }
        }
        
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

//        if(!mute) { timer_step.stop(); }
    }
    template<int dim>
    void Optimizer<dim>::computePrecondMtr(const Mesh<dim>& data,
                                           bool redoSVD,
                                           LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *p_linSysSolver)
    {
        if(!mute) { timer_step.start(0); }
        
        p_linSysSolver->setZero();
        switch(animConfig.timeIntegrationType) {
            case TIT_BE: {
                for(int eI = 0; eI < energyTerms.size(); eI++) {
                    energyTerms[eI]->computeHessian(data, redoSVD, svd, F,
                                                    energyParams[eI] * dtSq,
                                                    p_linSysSolver);
                }
                break;
            }
        }
        
//        Eigen::VectorXi I, J;
//        Eigen::VectorXd V;
//        energyTerms[0]->computeHessianBySVD(data, &V, &I, &J, true);
//        V *= energyParams[0] * dtSq;
//        p_linSysSolver->update_a(I, J, V);

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
        
        if(!mute) { timer_step.stop(); }
        // output matrix and exit
//        IglUtils::writeSparseMatrixToFile("/Users/mincli/Desktop/DOT/output/A", p_linSysSolver, true);
//        std::cout << "matrix written" << std::endl;
//        exit(0);
    }
                          
    template<int dim>
    void Optimizer<dim>::computeSystemEnergy(double& sysE)
    {
        energyTerms[0]->computeEnergyVal(result, false, svd, F, U, V, Sigma, 1.0, sysE);
        
        Eigen::VectorXd energyVals(result.V.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < result.V.rows(); vI++)
#endif
        {
            energyVals[vI] = result.massMatrix.coeff(vI, vI) * ((result.V.row(vI) - resultV_n.row(vI)).squaredNorm() / dtSq / 2.0 - gravity.dot(result.V.row(vI).transpose()));
        }
#ifdef USE_TBB
        );
#endif
        sysE += energyVals.sum();
    }
          
    template<int dim>
    void Optimizer<dim>::checkGradient(void)
    {
        std::cout << "checking energy gradient computation..." << std::endl;
        
        double energyVal0;
        computeEnergyVal(result, true, energyVal0);
        
        const double h = 1.0e-6 * igl::avg_edge_length(result.V, result.F);
        Mesh<dim> perturbed = result;
        Eigen::VectorXd gradient_finiteDiff;
        gradient_finiteDiff.resize(result.V.rows() * dim);
        for(int vI = 0; vI < result.V.rows(); vI++)
        {
            for(int dimI = 0; dimI < dim; dimI++) {
                perturbed.V = result.V;
                perturbed.V(vI, dimI) += h;
                double energyVal_perturbed;
                computeEnergyVal(perturbed, true, energyVal_perturbed);
                gradient_finiteDiff[vI * dim + dimI] = (energyVal_perturbed - energyVal0) / h;
            }
            
            if(((vI + 1) % 100) == 0) {
                std::cout << vI + 1 << "/" << result.V.rows() << " vertices computed" << std::endl;
            }
        }
        for(const auto fixedVI : result.fixedVert) {
            gradient_finiteDiff.segment<dim>(dim * fixedVI).setZero();
        }
        
        Eigen::VectorXd gradient_symbolic;
        computeGradient(result, true, gradient_symbolic);
        
        Eigen::VectorXd difVec = gradient_symbolic - gradient_finiteDiff;
        const double dif_L2 = difVec.norm();
        const double relErr = dif_L2 / gradient_finiteDiff.norm();
        
        std::cout << "L2 dist = " << dif_L2 << ", relErr = " << relErr << std::endl;
        
        logFile << "check gradient:" << std::endl;
        logFile << "g_symbolic =\n" << gradient_symbolic << std::endl;
        logFile << "g_finiteDiff = \n" << gradient_finiteDiff << std::endl;
    }
    
    template<int dim>
    double Optimizer<dim>::getLastEnergyVal(void) const
    {
        return lastEnergyVal;
    }
            
    template class Optimizer<DIM>;
}
