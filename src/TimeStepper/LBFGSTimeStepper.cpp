//
//  LBFGSTimeStepper.cpp
//  DOT
//
//  Created by Minchen Li on 12/14/18.
//

#include "LBFGSTimeStepper.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#include "IglUtils.hpp"
#include "Timer.hpp"

#include <tbb/tbb.h>


extern const std::string outputFolderPath;

extern std::ofstream logFile;
extern Timer timer, timer_step, timer_temp;


namespace DOT {
    
    // ------  constructor and destructor ------
    template<int dim>
    LBFGSTimeStepper<dim>::LBFGSTimeStepper(const Mesh<dim>& p_data0,
                                            const std::vector<Energy<dim>*>& p_energyTerms,
                                            const std::vector<double>& p_energyParams,
                                            D0Type p_D0Type,
                                            bool p_mute,
                                            const Config& animConfig) :
        Optimizer<dim>(p_data0, p_energyTerms, p_energyParams, p_mute, animConfig)
    {
        m_D0Type = p_D0Type;
        historySize = 5;
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        D0LinSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        D0LinSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        D0LinSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
        
        // initialize D_array
        //TODO: directly construct global matrix?
        D_array.resize(Base::result.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)Base::result.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < Base::result.F.rows(); ++triI)
#endif
        {
            const Eigen::Matrix<double, dim, dim>& A = Base::result.restTriInv[triI];
            D_array[triI].block(0, 1, dim, dim) = A.transpose();
            D_array[triI].leftCols(1) = -A.colwise().sum().transpose();
        }
#ifdef USE_TBB
        );
#endif
        
        if(m_D0Type == D0T_JH) {
            // node partitioning
            METIS<dim> partitions(Base::result);
            partitions.partMesh_nodes(Base::animConfig.partitionAmt);
            nodePart = partitions.getNpart();
            nodeLists.resize(Base::animConfig.partitionAmt);
            Eigen::VectorXi checkSum(Base::result.V.rows());
            checkSum.setZero();
            solver.resize(Base::animConfig.partitionAmt);
            for(int i = 0; i < Base::animConfig.partitionAmt; ++i) {
                partitions.getNodeList(i, nodeLists[i]);
                for(int j = 0; j < nodeLists[i].size(); ++j) {
                    if(checkSum[nodeLists[i][j]] != 0) {
                        std::cout << "overlapping nodes detected!" << std::endl;
                        exit(-1);
                    }
                    checkSum[nodeLists[i][j]] = 1;
                }
                solver[i] = new Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
            }
            if(checkSum.sum() != checkSum.size()) {
                std::cout << "void nodes detected!" << std::endl;
                exit(-2);
            }
            std::cout << "Block Jacobi LBFGSH initialized" << std::endl;
        }
    }
    template<int dim>
    LBFGSTimeStepper<dim>::~LBFGSTimeStepper(void)
    {
        delete D0LinSysSolver;
        for(auto& i : solver) {
            delete i;
        }
    }
    
    // ------ API ------
    template<int dim>
    void LBFGSTimeStepper<dim>::precompute(void)
    {
        Base::computeEnergyVal(Base::result, true, Base::lastEnergyVal);
        
        switch (m_D0Type) {
            case D0T_PD: {
                // assemble matrix M + dtSq L
                //TODO: directly compute each entry of global matrix in parallel
                //TODO: faster matrix sparsity structure construction
                
                timer_step.start(0);
                // initialize D operator
                std::vector<Eigen::Triplet<double>> triplet;
                triplet.reserve(Base::result.F.rows() * dim * (dim + 1));
                for(int triI = 0; triI < Base::result.F.rows(); triI++) {
                    const Eigen::Matrix<int, 1, dim + 1>& triVInd = Base::result.F.row(triI);
                    for(int localVI = 0; localVI < dim + 1; localVI++) {
                        //TODO: simplify!
                        triplet.emplace_back(triI * dim, triVInd[localVI],
                                             D_array[triI](0, localVI));
                        triplet.emplace_back(triI * dim + 1, triVInd[localVI],
                                             D_array[triI](1, localVI));
                        if(dim == 3) {
                            triplet.emplace_back(triI * dim + 2, triVInd[localVI],
                                                 D_array[triI](2, localVI));
                        }
                    }
                }
                D.resize(Base::result.F.rows() * dim, Base::result.V.rows());
                D.setZero();
                D.setFromTriplets(triplet.begin(), triplet.end());
                
                // initialize weights
                double EHCoeff = Base::dtSq;
                triplet.resize(0);
                triplet.reserve(Base::result.F.rows() * dim);
                for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
                    double w = EHCoeff * Base::result.triArea[elemI] * (2 * Base::result.u[elemI] +
                                                                           Base::result.lambda[elemI]);
                    int rowIStart = elemI * dim;
                    triplet.emplace_back(rowIStart, rowIStart, w);
                    triplet.emplace_back(rowIStart + 1, rowIStart + 1, w);
                    if(dim == 3) {
                        triplet.emplace_back(rowIStart + 2, rowIStart + 2, w);
                    }
                }
                W.resize(Base::result.F.rows() * dim, Base::result.F.rows() * dim);
                W.setZero();
                W.setFromTriplets(triplet.begin(), triplet.end());
                
                // assemble matrix and handle boundary condition
                Eigen::SparseMatrix<double> B = Base::result.massMatrix + D.transpose() * W * D;
                for (int k = 0; k < B.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it)
                    {
                        bool fixed_rowV = (Base::result.fixedVert.find(it.row()) !=
                                           Base::result.fixedVert.end());
                        if(fixed_rowV) {
                            it.valueRef() = 0.0;
                        }
                        else {
                            bool fixed_colV = (Base::result.fixedVert.find(it.col()) !=
                                               Base::result.fixedVert.end());
                            if(fixed_colV) {
                                it.valueRef() = 0.0;
                            }
                        }
                    }
                }
                for(const auto& fixedVI : Base::result.fixedVert) {
                    B.coeffRef(fixedVI, fixedVI) = 1.0;
                }
                B.makeCompressed();
                timer_step.stop();
                
                // pre-factorize linear solver
                D0LinSysSolver->set_type(1, 2);
                timer_step.start(1);
                D0LinSysSolver->set_pattern(B); // with entries filled in
                timer_step.start(2);
                D0LinSysSolver->analyze_pattern();
                timer_step.start(3);
                D0LinSysSolver->factorize();
                timer_step.stop();
                
                break;
            }
                
            case D0T_RH:
            case D0T_H: {
                Base::linSysSolver->set_type(1, 2);
                timer_step.start(1);
                Base::linSysSolver->set_pattern(Base::result.vNeighbor,
                                                Base::result.fixedVert);
                timer_step.start(2);
                Base::linSysSolver->analyze_pattern();
                timer_step.stop();
                
                Base::computePrecondMtr(Base::result, true, Base::linSysSolver);
                timer_step.start(3);
                Base::linSysSolver->factorize();
                timer_step.stop();
                
                break;
            }
                
            case D0T_HI: {
                Base::linSysSolver->set_type(1, 2);
                timer_step.start(1);
                Base::linSysSolver->set_pattern(Base::result.vNeighbor,
                                                Base::result.fixedVert);
                timer_step.stop();
                
                Base::computePrecondMtr(Base::result,
                                        true, Base::linSysSolver);
                
                Base::linSysSolver->getCoeffMtr_lower(sysMtrForIC);
                
                timer_step.start(2);
                ICSolver.analyzePattern(sysMtrForIC);
                timer_step.start(3);
                ICSolver.factorize(sysMtrForIC);
                timer_step.stop();
                
                break;
            }
                
            case D0T_JH: { //TODO: direct matrix assembly from elemHessians
                Base::linSysSolver->set_type(1, 2);
                
                timer_step.start(1);
                Base::linSysSolver->set_pattern(Base::result.vNeighbor,
                                                Base::result.fixedVert);
                timer_step.stop();
                
                Base::computePrecondMtr(Base::result, true, Base::linSysSolver);
                
                //TODO: parallelize
                for(int i = 0; i < Base::animConfig.partitionAmt; ++i) {
                    std::vector<Eigen::Triplet<double>> triplet;
                    Base::linSysSolver->getTriplets(nodeLists[i], triplet);
                    
                    Eigen::SparseMatrix<double> mtr(nodeLists[i].size() * DIM,
                                                    nodeLists[i].size() * DIM);
                    mtr.setFromTriplets(triplet.begin(), triplet.end());
                    
                    timer_step.start(2);
                    solver[i]->analyzePattern(mtr);
                    timer_step.start(3);
                    solver[i]->factorize(mtr);
                    timer_step.stop();
                }
                
                break;
            }
        }
    }
    
    template<int dim>
    void LBFGSTimeStepper<dim>::updatePrecondMtrAndFactorize(void)
    {
        precompute();
    }
    
    template<int dim>
    void LBFGSTimeStepper<dim>::getFaceFieldForVis(Eigen::VectorXd& field)
    {
        if(m_D0Type == D0T_JH) { //TODO: only compute the field once
            field.resize(Base::result.F.rows());
            field.setZero();
            for(int elemI = 0; elemI < Base::result.F.rows(); ++elemI) {
                const Eigen::Matrix<int, 1, dim + 1> elemVInd = Base::result.F.row(elemI);
                field[elemI] += nodePart[elemVInd[0]];
                field[elemI] += nodePart[elemVInd[1]];
                field[elemI] += nodePart[elemVInd[2]];
                if(dim == 3) {
                    field[elemI] += nodePart[elemVInd[3]];
                }
            }
        }
        else {
            field = Base::result.u;
        }
    }
    
    template<int dim>
    bool LBFGSTimeStepper<dim>::fullyImplicit(void)
    {
        dx.resize(0);
        dg.resize(0);
        dgTdx.resize(0);

        bool returnFlag = Base::fullyImplicit();
        
        if(m_D0Type == D0T_H) {
            Base::computePrecondMtr(Base::result, true, Base::linSysSolver);
            timer_step.start(3);
            Base::linSysSolver->factorize();
            timer_step.stop();
        }
        else if(m_D0Type == D0T_HI) {
            Base::computePrecondMtr(Base::result, true, Base::linSysSolver);
            
            Base::linSysSolver->getCoeffMtr_lower(sysMtrForIC);
            
            timer_step.start(3);
            ICSolver.factorize(sysMtrForIC);
            timer_step.stop();
        }
        else if(m_D0Type == D0T_JH) { //TODO: direct update from elemHessian into existing solver/matrix data structure
            Base::computePrecondMtr(Base::result, true, Base::linSysSolver);
            //TODO: parallelize
            for(int i = 0; i < Base::animConfig.partitionAmt; ++i) {
                std::vector<Eigen::Triplet<double>> triplet;
                Base::linSysSolver->getTriplets(nodeLists[i], triplet);
                
                Eigen::SparseMatrix<double> mtr(nodeLists[i].size() * DIM,
                                                nodeLists[i].size() * DIM);
                mtr.setFromTriplets(triplet.begin(), triplet.end());
                
                timer_step.start(2);
                solver[i]->factorize(mtr);
                timer_step.stop();
            }
        }
        
        return returnFlag;
    }
    
    // ------  overwritten helper ------
    template<int dim>
    bool LBFGSTimeStepper<dim>::solve_oneStep(void)
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
        
        switch (m_D0Type) {
            case D0T_PD: {
                Base::dimSeparatedSolve(D0LinSysSolver,
                                        minusG_LBFGS,
                                        Base::searchDir);
                break;
            }
                
            case D0T_RH:
            case D0T_H: {
                Base::linSysSolver->solve(minusG_LBFGS,
                                          Base::searchDir);
                break;
            }
                
            case D0T_HI: {
                Base::searchDir = ICSolver.solve(minusG_LBFGS);
            }
                
            case D0T_JH: {
                //TODO: parallelize
                for(int i = 0; i < Base::animConfig.partitionAmt; ++i) {
                    Eigen::VectorXd rhs(nodeLists[i].size() * dim);
                    for(int j = 0; j < nodeLists[i].size(); ++j) {
                        rhs.segment<dim>(j * dim) = minusG_LBFGS.segment<dim>(nodeLists[i][j] * dim);
                    }
                    
                    Eigen::VectorXd x = solver[i]->solve(rhs);
                    for(int j = 0; j < nodeLists[i].size(); ++j) {
                        Base::searchDir.segment(nodeLists[i][j] * dim, dim) = x.segment<dim>(j * dim);
                    }
                }
                break;
            }
        }
        
        if(!Base::mute) { timer_step.stop(); }
        
        
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
        
        
        timer_step.start(8);
        // update s and t
        dx.emplace_back(stepSize * Base::searchDir);
        dg.resize(dg.size() + 1);
        dg.back() = Base::gradient;
        Base::computeGradient(Base::result, false, Base::gradient); // SVD computed during line search
        dg.back() = Base::gradient - dg.back();
        
        // blending dg with Hs
//        if(m_D0Type == D0T_H) {
//            Eigen::VectorXd Hs;
//            Base::linSysSolver->multiply(dx.back(), Hs);
//
//            double beta = std::max(0.0, std::min(1.0, dg.back().dot(Hs) / Hs.squaredNorm()));
//            dg.back() *= (1.0 - beta);
//            dg.back() += beta * Hs;
//        }
        
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
    
    template class LBFGSTimeStepper<DIM>;
    
}
