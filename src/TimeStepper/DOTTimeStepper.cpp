//
//  DOTTimeStepper.cpp
//  DOT
//
//  Created by Minchen Li on 12/17/18.
//

#include "DOTTimeStepper.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#include "IglUtils.hpp"
#include "Timer.hpp"

#include <igl/writeOBJ.h>

#include <tbb/tbb.h>


extern const std::string outputFolderPath;

extern std::ofstream logFile;
extern Timer timer, timer_step, timer_temp;

extern Eigen::MatrixXi SF;


namespace DOT {
    
    // ------  constructor and destructor ------
    template<int dim>
    DOTTimeStepper<dim>::DOTTimeStepper(const Mesh<dim>& p_data0,
                                                  const std::vector<Energy<dim>*>& p_energyTerms,
                                                  const std::vector<double>& p_energyParams,
                                                  bool p_mute,
                                                  const Config& animConfig) :
        Super(p_data0, p_energyTerms, p_energyParams, p_mute, animConfig)
    {
        historySize = 5;

        dup.resize(Base::result.V.rows(), 0);
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
        {
            for(int localVI = 0; localVI < Super::mesh_subdomain[subdomainI].V.rows();
                ++localVI)
            {
                int globalVI = Super::localVIToGlobal_subdomain[subdomainI][localVI];
                ++dup[globalVI];
            }
        }
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        H_tr = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        H_tr = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        H_tr = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        H_tr->set_type(Base::pardisoThreadAmt, 2);
        H_tr->set_pattern(Base::result.vNeighbor, Base::result.fixedVert);
        
        H_tr_sbd.resize(Super::mesh_subdomain.size());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int sbdI)
#else
        for(int sbdI = 0; sbdI < Super::mesh_subdomain.size(); ++sbdI)
#endif
        {
            // for enhancing local Hessian with missing information at interfaces
            std::vector<std::set<int>> vNeighborExt = Super::mesh_subdomain[sbdI].vNeighbor;
            for(const auto& dualMapperI : Super::globalVIToDual_subdomain[sbdI]) {
                int localVI = Super::globalVIToLocal_subdomain[sbdI][dualMapperI.first];
                
                for(const auto& nbVI_g : Base::result.vNeighbor[dualMapperI.first]) {
                    const auto localFinder = Super::globalVIToLocal_subdomain[sbdI].find(nbVI_g);
                    if(localFinder != Super::globalVIToLocal_subdomain[sbdI].end()) {
                        vNeighborExt[localVI].insert(localFinder->second);
                    }
                }
            }
            
#ifdef LINSYSSOLVER_USE_CHOLMOD
            H_tr_sbd[sbdI] = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
            H_tr_sbd[sbdI] = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
            H_tr_sbd[sbdI] = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
            H_tr_sbd[sbdI]->set_type(Base::pardisoThreadAmt, 2);
            H_tr_sbd[sbdI]->set_pattern(vNeighborExt, Super::mesh_subdomain[sbdI].fixedVert);
        }
#ifdef USE_TBB
        );
#endif
        
        if(Base::animConfig.timeStepperType == TST_LBFGS_GSDD) {
            dx_sbd.resize(Base::animConfig.partitionAmt);
            dg_sbd.resize(Base::animConfig.partitionAmt);
            dgTdx_sbd.resize(Base::animConfig.partitionAmt);
            elemListOv.resize(Base::animConfig.partitionAmt);
            globalElemI2LocalOv_sbd.resize(Base::animConfig.partitionAmt);
        
            // construct elemListOv
#ifdef USE_TBB
            tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int sbdI)
#else
            for(int sbdI = 0; sbdI < Super::mesh_subdomain.size(); ++sbdI)
#endif
            {
                std::set<int> elemSetOv;
                elemSetOv.insert(Super::elemList_subdomain[sbdI].data(),
                                 Super::elemList_subdomain[sbdI].data() +
                                 Super::elemList_subdomain[sbdI].size());
                for(const auto& dualMapperI : Super::globalVIToDual_subdomain[sbdI]) {
                    for(const auto& incTriI : Base::result.vFLoc[dualMapperI.first]) {
                        elemSetOv.insert(incTriI.first);
                    }
                }
                elemListOv[sbdI].resize(elemSetOv.size());
                std::copy(elemSetOv.begin(), elemSetOv.end(), elemListOv[sbdI].begin());
                
                int elemII = 0;
                for(const auto& elemI : elemListOv[sbdI]) {
                    globalElemI2LocalOv_sbd[sbdI][elemI] = elemII;
                    ++elemII;
                }
            }
#ifdef USE_TBB
            );
#endif
        }
    }
    template<int dim>
    DOTTimeStepper<dim>::~DOTTimeStepper(void)
    {
        delete H_tr;
        for(auto& H_tr_sbdI : H_tr_sbd) {
            delete H_tr_sbdI;
        }
    }
    
    // ------ API ------
    template<int dim>
    void DOTTimeStepper<dim>::precompute(void)
    {
        Super::precompute();
        
        computeDecomposedHessians();
        timer_step.start(3);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
#endif
        {
//            Super::computeHessianProxy_subdomain(subdomainI, true, true);
            if(Super::useDense) {
                Super::denseSolver_sbd[subdomainI].compute(Super::hessian_sbd[subdomainI]);
            }
            else {
                Super::linSysSolver_subdomain[subdomainI]->factorize();
            }
        
            Super::F_subdomain[subdomainI].resize(Super::mesh_subdomain[subdomainI].F.rows());
            for(const auto& triIMapperI : Super::globalTriIToLocal_subdomain[subdomainI]) {
                Super::F_subdomain[subdomainI][triIMapperI.second] = Base::F[triIMapperI.first];
            }
//            Super::u_subdomain[subdomainI].setZero();
        }
#ifdef USE_TBB
        );
#endif
        Super::F_subdomain_last = Super::F_subdomain;
        
        timer_step.stop();
    }
    
    template<int dim>
    void DOTTimeStepper<dim>::updatePrecondMtrAndFactorize(void)
    {
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
#endif
        {
            std::set<int> updatedFixedVert_local;
            for(const auto& localMapperI : Super::globalVIToLocal_subdomain[subdomainI]) {
                if(Base::result.isFixedVert[localMapperI.first]) {
                    updatedFixedVert_local.insert(localMapperI.second);
                }
            }
            Super::mesh_subdomain[subdomainI].resetFixedVert(updatedFixedVert_local);
        }
#ifdef USE_TBB
        );
#endif
        
        timer_step.start(2);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); subdomainI++)
#endif
        {
            // for enhancing local Hessian with missing information at interfaces
            std::vector<std::set<int>> vNeighborExt = Super::mesh_subdomain[subdomainI].vNeighbor;
            for(const auto& dualMapperI : Super::globalVIToDual_subdomain[subdomainI]) {
                int localVI = Super::globalVIToLocal_subdomain[subdomainI][dualMapperI.first];
                
                for(const auto& nbVI_g : Base::result.vNeighbor[dualMapperI.first]) {
                    const auto localFinder = Super::globalVIToLocal_subdomain[subdomainI].find(nbVI_g);
                    if(localFinder != Super::globalVIToLocal_subdomain[subdomainI].end()) {
                        vNeighborExt[localVI].insert(localFinder->second);
                    }
                }
            }
            
            Super::linSysSolver_subdomain[subdomainI]->set_pattern(vNeighborExt,
                                                                    Super::mesh_subdomain[subdomainI].fixedVert);
            
            Super::linSysSolver_subdomain[subdomainI]->analyze_pattern();
            
            H_tr_sbd[subdomainI]->set_pattern(vNeighborExt,
                                              Super::mesh_subdomain[subdomainI].fixedVert);
        }
#ifdef USE_TBB
        );
#endif
        timer_step.stop();
        
        //        timer_step.start(1);
        // for weights computation
        Base::linSysSolver->set_pattern(Base::result.vNeighbor, Base::result.fixedVert);
        H_tr->set_pattern(Base::result.vNeighbor, Base::result.fixedVert);
        //        timer_step.stop();
        
        Super::m_isUpdateElemHessian.resize(0);
        Super::m_isUpdateElemHessian.resize(Base::result.F.rows(), true);
        computeDecomposedHessians();
        timer_step.start(3);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
#endif
        {
            if(Super::useDense) {
                Super::denseSolver_sbd[subdomainI].compute(Super::hessian_sbd[subdomainI]);
            }
            else {
                Super::linSysSolver_subdomain[subdomainI]->factorize();
            }
            
            for(const auto& triIMapperI : Super::globalTriIToLocal_subdomain[subdomainI]) {
                Super::F_subdomain[subdomainI][triIMapperI.second] = Base::F[triIMapperI.first];
            }
        }
#ifdef USE_TBB
        );
#endif
        Super::F_subdomain_last = Super::F_subdomain;
        timer_step.stop();
    }
    
    template<int dim>
    bool DOTTimeStepper<dim>::fullyImplicit(void)
    {
        dx.resize(0);
        dg.resize(0);
        dgTdx.resize(0);
        if(Base::animConfig.timeStepperType == TST_LBFGS_GSDD) {
            //TODO: parallelize
            for(int sbdI = 0; sbdI < Base::animConfig.partitionAmt; ++sbdI) {
                dx_sbd[sbdI].resize(0);
                dg_sbd[sbdI].resize(0);
                dgTdx_sbd[sbdI].resize(0);
            }
        }

        
        timer_step.start(10);
        Base::initX(Base::animConfig.warmStart);
        
        double sqn_g = __DBL_MAX__;
        Base::computeEnergyVal(Base::result, true, Base::lastEnergyVal);
        Base::computeGradient(Base::result, false, Base::gradient);
        timer_step.stop();
        std::cout << "after initX E = " << Base::lastEnergyVal <<
            " ||g||^2 = " << Base::gradient.squaredNorm() << ", tol = " <<
            Base::targetGRes << std::endl;
        double E_last = Base::lastEnergyVal;
        Base::file_iterStats << Base::globalIterNum << " 0 " << Base::lastEnergyVal << " " << Base::gradient.squaredNorm() << std::endl;
        gradSqNorms.resize(0);
        lastGradSqNorm = Base::gradient.squaredNorm();
        int iterCap = 10000, curIterI = 0;
        do {
            Base::file_iterStats << Base::globalIterNum << " ";
            
            bool stopped;
            if(Base::animConfig.timeStepperType == TST_LBFGS_GSDD) {
                stopped = solve_oneStep_GSDD();
            }
            else {
                stopped = solve_oneStep();
            }
            if(stopped) {
                std::cout << "\tline search with Armijo's rule failed!!!" << std::endl;
                logFile << "\tline search with Armijo's rule failed!!!" << std::endl;
                return true;
            }
            Base::innerIterAmt++;
            
            timer_step.start(10);
            
            sqn_g = Base::gradient.squaredNorm();
            gradSqNorms.emplace_back(sqn_g);
            
            if(!Base::mute) {
                std::cout << "\t||gradient||^2 = " << sqn_g << std::endl;
            }
            
            Base::file_iterStats << Base::lastEnergyVal << " " << sqn_g << std::endl;
            
            timer_step.stop();
            
            if(++curIterI >= iterCap) {
                break;
            }
        } while(sqn_g > Base::targetGRes);
        
        logFile << "Timestep" << Base::globalIterNum << " innerIterAmt = " << Base::innerIterAmt <<
            ", accumulated line search steps " << Base::numOfLineSearch << std::endl;
        
        bool returnFlag = (curIterI >= iterCap);

        updateHessianAndFactor();

        return returnFlag;
    }
    
    template<int dim>
    void DOTTimeStepper<dim>::updateHessianAndFactor(void)
    {
        timer_step.start(1);
            
        computeHElemAndFillIn(Base::linSysSolver);
        
        bool update = true;
        timer_step.stop();
        
        if(update) {
            fillInDecomposedHessians(Super::linSysSolver_subdomain);

            timer_step.start(3);
#ifdef USE_TBB
            tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int subdomainI)
#else
            for (int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
#endif
            {
                if(Super::useDense) {
                    Super::denseSolver_sbd[subdomainI].compute(Super::hessian_sbd[subdomainI]);
                }
                else {
                    Super::linSysSolver_subdomain[subdomainI]->factorize();
                }
            }
#ifdef USE_TBB
            );
#endif
            timer_step.stop();
        }
    }

    // ------  overwritten helper ------
    template<int dim>
    bool DOTTimeStepper<dim>::solve_oneStep(void)
    {
        timer_step.start(6);
        // assemble L-BFGS modified -gradient, from latest to old
        //NOTE: boundary condition handled in gradient
        Eigen::VectorXd minusG_LBFGS = -Base::gradient;
        std::vector<double> ksi(historySize);
        for(int historyI = (int)dx.size() - 1; historyI >= (int)dx.size() - historySize; --historyI) {
            if(historyI < 0) {
                break;
            }
            
            ksi[historyI] = dx[historyI].dot(minusG_LBFGS) / dgTdx[historyI];
            minusG_LBFGS -= ksi[historyI] * dg[historyI];
        }
        //NOTE: this is order dependent, probably also try Anderson Acceleration
        timer_step.stop();
        
        
        if(!Base::mute) { std::cout << "back solve..." << std::endl; }
        if(!Base::mute) { timer_step.start(4); }
        
        std::vector<Eigen::VectorXd> p_sbd(Super::mesh_subdomain.size());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int subdomainI)
#else
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
#endif
        {
            // no Hessian computation and factorization, no line search, 1 iteration
            Eigen::VectorXd rhs(Super::mesh_subdomain[subdomainI].V.rows() * dim);
            for(int localVI = 0; localVI < Super::mesh_subdomain[subdomainI].V.rows();
                ++localVI)
            {
                rhs.segment<dim>(localVI * dim) =
                    minusG_LBFGS.segment<dim>(Super::localVIToGlobal_subdomain[subdomainI][localVI] * dim);
            }
            
            if(Super::useDense) {
                p_sbd[subdomainI] = Super::denseSolver_sbd[subdomainI].solve(rhs);
            }
            else {
                Super::linSysSolver_subdomain[subdomainI]->solve(rhs, p_sbd[subdomainI]);
            }
        }
#ifdef USE_TBB
        );
#endif
        
        if(!Base::mute) { std::cout << "extract searchDir" << std::endl; }
        Base::searchDir.setZero(Base::result.V.rows() * dim);
        //TODO: only parallelize interior part
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
        {
            for(int localVI = 0; localVI < Super::mesh_subdomain[subdomainI].V.rows();
                ++localVI)
            {
                int globalVI = Super::localVIToGlobal_subdomain[subdomainI][localVI];
                Base::searchDir.segment(globalVI * dim, dim) += p_sbd[subdomainI].segment<dim>(localVI * dim);
            }
        }
        // averaging
        for(int vI = 0; vI < Base::result.V.rows(); ++vI) {
            if(dup[vI] > 1) {
                Base::searchDir.segment(vI * dim, dim) /= dup[vI];
            }
        }
        
        if(!Base::mute) { timer_step.stop(); }
        
        
        if(!Base::mute) { std::cout << "modify searchDir" << std::endl; }
        timer_step.start(7);
        // L-BFGS search direction modification, from old to latest
        //NOTE: boundary condition reflected in dx
        for(int historyI = (int)dx.size() - historySize; historyI < (int)dx.size(); ++historyI) {
            if(historyI < 0) {
                continue;
            }
            
            Base::searchDir += dx[historyI] * (ksi[historyI] -
                                               dg[historyI].dot(Base::searchDir) / dgTdx[historyI]);
        }
        timer_step.stop();
        

        double stepSize;
        bool stopped = Base::lineSearch(stepSize);
        
        
        // update s and t
        timer_step.start(8);
        dx.emplace_back(stepSize * Base::searchDir);
        dg.resize(dg.size() + 1);
        dg.back() = Base::gradient;
        Base::computeGradient(Base::result, false, Base::gradient); // SVD computed during line search
        dg.back() = Base::gradient - dg.back();
        dgTdx.emplace_back(dg.back().dot(dx.back()));
        if(dgTdx.back() <= 0.0) {
            dx.pop_back();
            dg.pop_back();
            dgTdx.pop_back();
        }
        else {
            if(dx.size() > historySize) {
                dx.pop_front();
                dg.pop_front();
                dgTdx.pop_front();
            }
        }
        timer_step.stop();
        
        
        if(stopped) {
            assert(0);
            //            IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_stopped_" + std::to_string(globalIterNum), precondMtr);
            logFile << "descent step stopped at overallIter" << Base::globalIterNum << " for no prominent energy decrease." << std::endl;
        }
        
        return stopped;
    }
    
    template<int dim>
    bool DOTTimeStepper<dim>::solve_oneStep_GSDD(void)
    {
        timer_step.start(12);
        Eigen::VectorXd minusG_LBFGS;
        Super::extract(Base::gradient, 0, minusG_LBFGS);
        minusG_LBFGS *= -1.0;
        timer_step.stop();
        
        std::vector<Eigen::VectorXd> p_sbd(Super::mesh_subdomain.size());
        bool stopped = false;
        for(int subdomainI = 0; subdomainI < Super::mesh_subdomain.size(); ++subdomainI)
        {
            timer_step.start(12);
            Eigen::VectorXd minusGrad_sbd = minusG_LBFGS;
            timer_step.stop();
            
            timer_step.start(4);
            if(Super::useDense) {
                p_sbd[subdomainI] = Super::denseSolver_sbd[subdomainI].solve(minusG_LBFGS);
            }
            else {
                Super::linSysSolver_subdomain[subdomainI]->solve(minusG_LBFGS, p_sbd[subdomainI]);
            }
            timer_step.stop();
            
            timer_step.start(5);
            Base::searchDir.setZero(Base::result.V.rows() * dim);
            Super::fill(subdomainI, p_sbd[subdomainI], Base::searchDir);
            timer_step.stop();
            
            //TODO: only search on subdomain mesh but also involve the interface band of the element, needs SIMD
            double stepSize;
            stopped = stopped || Base::lineSearch(stepSize);
            
            //TODO: needs to use SIMD
            timer_step.start(12);
            computeGradient_extract(subdomainI, minusG_LBFGS);
            timer_step.stop();
            
            if(subdomainI + 1 < Super::mesh_subdomain.size()) {
                timer_step.start(12);
                computeGradient_extract(subdomainI + 1, minusG_LBFGS);
                minusG_LBFGS *= -1.0;
                timer_step.stop();
            }
        }
        
        timer_step.start(12);
        Base::computeGradient(Base::result, false, Base::gradient);
        timer_step.stop();
        
        if(stopped) {
            assert(0);
            //            IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_stopped_" + std::to_string(globalIterNum), precondMtr);
            logFile << "descent step stopped at overallIter" << Base::globalIterNum << " for no prominent energy decrease." << std::endl;
        }
        
        return stopped;
    }

    template<int dim>
    void DOTTimeStepper<dim>::computeEnergyVal(const Mesh<dim>& data, int redoSVD, double& energyVal)
    {
        Base::computeEnergyVal(data, redoSVD, energyVal);
    }
    
    template<int dim>
    void DOTTimeStepper<dim>::computeHElemAndFillIn(LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver)
    {
        timer_step.start(0);
        switch(Base::animConfig.timeIntegrationType) {
            case TIT_BE:
                Base::energyTerms[0]->computeElemHessianByPK(Base::result, false,
                                                             Base::svd, Base::F,
                                                             Base::energyParams[0] * Base::dtSq,
                                                             Super::m_isUpdateElemHessian,
                                                             elemHessians, vInds, true);
                break;
        }
        
        // Note that if an element Hessian is not updated, it is the value from last update
        linSysSolver->setZero();
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < Base::result.V.rows(); vI++)
#endif
        {
            for(const auto FLocI : Base::result.vFLoc[vI]) {
                IglUtils::addBlockToMatrix<dim>(elemHessians[FLocI.first].block(FLocI.second * dim, 0, dim, dim * (dim + 1)), vInds[FLocI.first], FLocI.second, linSysSolver);
            }
            
            if(!Base::result.isFixedVert[vI]) {
                double massI = Base::result.massMatrix.coeff(vI, vI);
                int ind0 = vI * dim;
                int ind1 = ind0 + 1;
                linSysSolver->addCoeff(ind0, ind0, massI);
                linSysSolver->addCoeff(ind1, ind1, massI);
                if(dim == 3) {
                    int ind2 = ind0 + 2;
                    linSysSolver->addCoeff(ind2, ind2, massI);
                }
            }
        }
#ifdef USE_TBB
        );
#endif
        
        timer_step.stop();
    }
    
    template<int dim>
    void DOTTimeStepper<dim>::fillInDecomposedHessians(std::vector<LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>*>& linSysSolver_sbd)
    {
        //TODO: consider directly extract entries from global Hessian
        
        timer_step.start(1);
        for(int sbdI = 0; sbdI < Super::mesh_subdomain.size(); ++sbdI) {
            std::vector<Eigen::Matrix<int, 1, dim + 1>> vInds_l(Super::mesh_subdomain[sbdI].F.rows());
#ifdef USE_TBB
            tbb::parallel_for(0, (int)Super::mesh_subdomain[sbdI].F.rows(), 1, [&](int elemI)
#else
            for(int elemI = 0; elemI < Super::mesh_subdomain[sbdI].F.rows(); ++elemI)
#endif
            {
                const Eigen::Matrix<int, 1, dim + 1>& elemVInds_l = Super::mesh_subdomain[sbdI].F.row(elemI);
                
                vInds_l[elemI][0] = (Super::mesh_subdomain[sbdI].isFixedVert[elemVInds_l[0]] ? (-elemVInds_l[0] - 1) : elemVInds_l[0]);
                vInds_l[elemI][1] = (Super::mesh_subdomain[sbdI].isFixedVert[elemVInds_l[1]] ? (-elemVInds_l[1] - 1) : elemVInds_l[1]);
                vInds_l[elemI][2] = (Super::mesh_subdomain[sbdI].isFixedVert[elemVInds_l[2]] ? (-elemVInds_l[2] - 1) : elemVInds_l[2]);
                if(dim == 3) {
                    vInds_l[elemI][3] = (Super::mesh_subdomain[sbdI].isFixedVert[elemVInds_l[3]] ? (-elemVInds_l[3] - 1) : elemVInds_l[3]);
                }
            }
#ifdef USE_TBB
            );
#endif
            
            if(Super::useDense) {
                Super::hessian_sbd[sbdI].setZero();
#ifdef USE_TBB
                tbb::parallel_for(0, (int)Super::mesh_subdomain[sbdI].V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < Super::mesh_subdomain[sbdI].V.rows(); vI++)
#endif
                {
                    for(const auto& FLocI : Super::mesh_subdomain[sbdI].vFLoc[vI]) {
                        int elemI_g = Super::elemList_subdomain[sbdI][FLocI.first];
                        IglUtils::addBlockToMatrix<dim>(elemHessians[elemI_g].block(FLocI.second * dim, 0, dim, dim * (dim + 1)), vInds_l[FLocI.first], FLocI.second, Super::hessian_sbd[sbdI]);
                    }
                    
                    double massI = Super::mesh_subdomain[sbdI].massMatrix.coeff(vI, vI);
                    int ind0 = vI * dim;
                    Super::hessian_sbd[sbdI].block(ind0, ind0, dim, dim).diagonal() += Eigen::Matrix<double, dim, 1>::Constant(massI);
                }
#ifdef USE_TBB
            );
#endif
            }
            else {
                linSysSolver_sbd[sbdI]->setZero();
#ifdef USE_TBB
                tbb::parallel_for(0, (int)Super::mesh_subdomain[sbdI].V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < Super::mesh_subdomain[sbdI].V.rows(); vI++)
#endif
                {
                    for(const auto& FLocI : Super::mesh_subdomain[sbdI].vFLoc[vI]) {
                        int elemI_g = Super::elemList_subdomain[sbdI][FLocI.first];
                        IglUtils::addBlockToMatrix<dim>(elemHessians[elemI_g].block(FLocI.second * dim, 0, dim, dim * (dim + 1)), vInds_l[FLocI.first], FLocI.second, linSysSolver_sbd[sbdI]);
                    }
                    
                    double massI = Super::mesh_subdomain[sbdI].massMatrix.coeff(vI, vI);
                    int ind0 = vI * dim;
                    int ind1 = ind0 + 1;
                    linSysSolver_sbd[sbdI]->addCoeff(ind0, ind0, massI);
                    linSysSolver_sbd[sbdI]->addCoeff(ind1, ind1, massI);
                    if(dim == 3) {
                        int ind2 = ind0 + 2;
                        linSysSolver_sbd[sbdI]->addCoeff(ind2, ind2, massI);
                    }
                }
#ifdef USE_TBB
                );
#endif
            }
        }
        
        // missing Hessian information - fullW
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain.size(), 1, [&](int sbdI)
#else
        for(int sbdI = 0; sbdI < Super::mesh_subdomain.size(); ++sbdI)
#endif
        {
            for(const auto& dualMapperI : Super::globalVIToDual_subdomain[sbdI]) {
                if(!Base::result.isFixedVert[dualMapperI.first]) {
                    int localVI = Super::globalVIToLocal_subdomain[sbdI][dualMapperI.first];
                    int startInd = localVI * dim;
                    int startIndp1 = startInd + 1;
                    int startIndp2 = startInd + 2;
                    
                    // add missing mass
                    double massDif = (Base::result.massMatrix.coeff(dualMapperI.first,
                                                                    dualMapperI.first) -
                                        Super::mesh_subdomain[sbdI].massMatrix.coeff(localVI,
                                                                                    localVI));
                    if(Super::useDense) {
                        Super::hessian_sbd[sbdI].block(startInd, startInd, dim, dim).diagonal() += Eigen::Matrix<double, dim, 1>::Constant(massDif);
                    }
                    else {
                        linSysSolver_sbd[sbdI]->addCoeff(startInd, startInd, massDif);
                        linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startIndp1,
                                                                        massDif);
                        if(dim == 3) {
                            linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startIndp2,
                                                                        massDif);
                        }
                    }
                    
                    // check whether any elements are missing
                    for(const auto& FLocI_g : Base::result.vFLoc[dualMapperI.first]) {
                        const auto elemFinder = Super::globalTriIToLocal_subdomain[sbdI].find(FLocI_g.first);
                        if(elemFinder == Super::globalTriIToLocal_subdomain[sbdI].end()) {
                            // for missing element, add the missing Hessian

                            const Eigen::Matrix<double, dim, dim>& hessianI = elemHessians[FLocI_g.first].block(FLocI_g.second * dim, FLocI_g.second * dim, dim, dim);
                            if(Super::useDense) {
                                Super::hessian_sbd[sbdI].block(startInd, startInd, dim, dim) += hessianI;
                            }
                            else {
                                linSysSolver_sbd[sbdI]->addCoeff(startInd, startInd, hessianI(0, 0));
                                linSysSolver_sbd[sbdI]->addCoeff(startInd, startIndp1, hessianI(0, 1));
                                linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startInd, hessianI(1, 0));
                                linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startIndp1, hessianI(1, 1));
                                if(dim == 3) {
                                    linSysSolver_sbd[sbdI]->addCoeff(startInd, startIndp2, hessianI(0, 2));
                                    linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startIndp2, hessianI(1, 2));
                                    
                                    linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startInd, hessianI(2, 0));
                                    linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startIndp1, hessianI(2, 1));
                                    linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startIndp2, hessianI(2, 2));
                                }
                            }
                            
                            // check for off-diagonal blocks
                            const Eigen::Matrix<int, 1, dim + 1>& elemVInds_g = Base::result.F.row(FLocI_g.first);
                            for(int vI_elem = 0; vI_elem < dim + 1; ++vI_elem) {
                                if(Base::result.isFixedVert[elemVInds_g[vI_elem]] ||
                                    (vI_elem == FLocI_g.second))
                                {
                                    continue;
                                }
                                
                                const auto dualNFinder = Super::globalVIToDual_subdomain[sbdI].find(elemVInds_g[vI_elem]);
                                if(dualNFinder != Super::globalVIToDual_subdomain[sbdI].end()) {
                                    int startInd_col = Super::globalVIToLocal_subdomain[sbdI][dualNFinder->first] * dim;
                                    int startIndp1_col = startInd_col + 1;
                                    int startIndp2_col = startInd_col + 2;
                                    
                                    const Eigen::Matrix<double, dim, dim>& hessianJ = elemHessians[FLocI_g.first].block(FLocI_g.second * dim, vI_elem * dim, dim, dim);
                                    if(Super::useDense) {
                                        Super::hessian_sbd[sbdI].block(startInd, startInd_col, dim, dim) += hessianJ;
                                    }
                                    else {
                                        linSysSolver_sbd[sbdI]->addCoeff(startInd, startInd_col, hessianJ(0, 0));
                                        linSysSolver_sbd[sbdI]->addCoeff(startInd, startIndp1_col, hessianJ(0, 1));
                                        linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startInd_col, hessianJ(1, 0));
                                        linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startIndp1_col, hessianJ(1, 1));
                                        if(dim == 3) {
                                            linSysSolver_sbd[sbdI]->addCoeff(startInd, startIndp2_col, hessianJ(0, 2));
                                            linSysSolver_sbd[sbdI]->addCoeff(startIndp1, startIndp2_col, hessianJ(1, 2));
                                            
                                            linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startInd_col, hessianJ(2, 0));
                                            linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startIndp1_col, hessianJ(2, 1));
                                            linSysSolver_sbd[sbdI]->addCoeff(startIndp2, startIndp2_col, hessianJ(2, 2));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#ifdef USE_TBB
        );
#endif
        
        timer_step.stop();
    }
    
    template<int dim>
    void DOTTimeStepper<dim>::computeDecomposedHessians(void)
    {
        computeHElemAndFillIn(Base::linSysSolver);
        fillInDecomposedHessians(Super::linSysSolver_subdomain);
    }
    
    template<int dim>
    void DOTTimeStepper<dim>::computeGradient_extract(int sbdI,
                                                           Eigen::VectorXd& grad)
    {
        std::vector<Eigen::Matrix<double, dim * (dim + 1), 1>> elemGrads(elemListOv[sbdI].size());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)elemListOv[sbdI].size(), 1, [&](int elemII)
#else
        for(int elemII = 0; elemII < elemListOv[sbdI].size(); ++elemII)
#endif
        {
            int elemI = elemListOv[sbdI][elemII];
            Base::energyTerms[0]->computeGradientByPK(Base::result, elemI,
                                                      false, Base::svd[elemI], Base::F[elemI],
                                                      Base::dtSq * Base::energyParams[0],
                                                      elemGrads[elemII]);
        }
#ifdef USE_TBB
        );
#endif
        
        grad.conservativeResize(Super::mesh_subdomain[sbdI].V.rows() * dim);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Super::mesh_subdomain[sbdI].V.rows(), 1, [&](int localVI)
#else
        for(int localVI = 0; localVI < Super::mesh_subdomain[sbdI].V.rows();
            ++localVI)
#endif
        {
            if(Super::mesh_subdomain[sbdI].isFixedVert[localVI]) {
                grad.segment<dim>(localVI * dim).setZero();
            }
            else {
                int globalVI = Super::localVIToGlobal_subdomain[sbdI][localVI];
                grad.segment<dim>(localVI * dim) = Base::result.massMatrix.coeff(globalVI, globalVI) * (Base::result.V.row(globalVI) - Base::xTilta.row(globalVI)).transpose();
                
                for(const auto& FLocI : Base::result.vFLoc[globalVI]) {
                    grad.segment<dim>(localVI * dim) +=
                        elemGrads[globalElemI2LocalOv_sbd[sbdI][FLocI.first]].segment(FLocI.second * dim, dim);
                }
            }
        }
#ifdef USE_TBB
        );
#endif
    }
    
    template class DOTTimeStepper<DIM>;
    
}
