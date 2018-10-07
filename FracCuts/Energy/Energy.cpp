//
//  Energy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/4/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Energy.hpp"
#include "get_feasible_steps.hpp"
#include "IglUtils.hpp"
#include "Timer.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#include <igl/avg_edge_length.h>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <fstream>
#include <iostream>

extern std::string outputFolderPath;
extern std::ofstream logFile;
extern Timer timer_temp, timer_temp2;

namespace FracCuts {
    
    template<int dim>
    Energy<dim>::Energy(bool p_needRefactorize,
                   bool p_needElemInvSafeGuard,
                   bool p_crossSigmaDervative,
                   double visRange_min,
                   double visRange_max) :
        needRefactorize(p_needRefactorize),
        needElemInvSafeGuard(p_needElemInvSafeGuard),
        crossSigmaDervative(p_crossSigmaDervative),
        visRange_energyVal(visRange_min, visRange_max)
    {}
    
    template<int dim>
    Energy<dim>::~Energy(void)
    {}
    
    template<int dim>
    bool Energy<dim>::getNeedRefactorize(void) const
    {
        return needRefactorize;
    }
    
    template<int dim>
    bool Energy<dim>::getNeedElemInvSafeGuard(void) const
    {
        return needElemInvSafeGuard;
    }
    
    template<int dim>
    const Eigen::Vector2d& Energy<dim>::getVisRange_energyVal(void) const
    {
        return visRange_energyVal;
    }
    template<int dim>
    void Energy<dim>::setVisRange_energyVal(double visRange_min, double visRange_max)
    {
        visRange_energyVal << visRange_min, visRange_max;
    }
    
    template<int dim>
    void Energy<dim>::computeEnergyVal(const TriangleSoup<dim>& data, double& energyVal, bool uniformWeight) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElem(data, energyValPerElem, uniformWeight);
        energyVal = energyValPerElem.sum();
    }
    
    template<int dim>
    void Energy<dim>::getEnergyValByElemID(const TriangleSoup<dim>& data, int elemI, double& energyVal, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    template<int dim>
    void Energy<dim>::computeGradient(const TriangleSoup<dim>& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    template<int dim>
    void Energy<dim>::computePrecondMtr(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    template<int dim>
    void Energy<dim>::computePrecondMtr(const TriangleSoup<dim>& data, Eigen::VectorXd* V,
                                   Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    template<int dim>
    void Energy<dim>::computeHessian(const TriangleSoup<dim>& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    template<int dim>
    void Energy<dim>::checkGradient(const TriangleSoup<dim>& data) const
    {
        std::cout << "checking energy gradient computation..." << std::endl;
        
        std::vector<Eigen::Matrix<double, dim, dim>> F(data.F.rows());
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd(data.F.rows());
        
        double energyVal0;
        computeEnergyValBySVD(data, true, svd, F, energyVal0);
        const double h = 1.0e-8 * igl::avg_edge_length(data.V, data.F);
        TriangleSoup<dim> perturbed = data;
        Eigen::VectorXd gradient_finiteDiff;
        gradient_finiteDiff.resize(data.V.rows() * dim);
        for(int vI = 0; vI < data.V.rows(); vI++)
        {
            for(int dimI = 0; dimI < dim; dimI++) {
                perturbed.V = data.V;
                perturbed.V(vI, dimI) += h;
                double energyVal_perturbed;
                computeEnergyValBySVD(perturbed, true, svd, F, energyVal_perturbed);
                gradient_finiteDiff[vI * dim + dimI] = (energyVal_perturbed - energyVal0) / h;
            }
            
            if(((vI + 1) % 100) == 0) {
                std::cout << vI + 1 << "/" << data.V.rows() << " vertices computed" << std::endl;
            }
        }
        for(const auto fixedVI : data.fixedVert) {
            gradient_finiteDiff.segment<dim>(dim * fixedVI).setZero();
        }
        
        Eigen::VectorXd gradient_symbolic;
//        computeGradient(data, gradient_symbolic);
//        computeGradientBySVD(data, gradient_symbolic);
        computeGradientByPK(data, true, svd, F, gradient_symbolic);
        
        Eigen::VectorXd difVec = gradient_symbolic - gradient_finiteDiff;
        const double dif_L2 = difVec.norm();
        const double relErr = dif_L2 / gradient_finiteDiff.norm();
        
        std::cout << "L2 dist = " << dif_L2 << ", relErr = " << relErr << std::endl;
        
        logFile << "check gradient:" << std::endl;
        logFile << "g_symbolic =\n" << gradient_symbolic << std::endl;
        logFile << "g_finiteDiff = \n" << gradient_finiteDiff << std::endl;
    }
    
    template<int dim>
    void Energy<dim>::checkHessian(const TriangleSoup<dim>& data, bool triplet) const
    {
        std::cout << "checking energy hessian computation..." << std::endl;
        
        std::vector<Eigen::Matrix<double, dim, dim>> F(data.F.rows());
        std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>> svd(data.F.rows());
        
        Eigen::VectorXd gradient0;
//        computeGradientBySVD(data, gradient0);
        computeGradientByPK(data, true, svd, F, gradient0);
        const double h = 1.0e-4 * igl::avg_edge_length(data.V, data.F);
        TriangleSoup<dim> perturbed = data;
        Eigen::SparseMatrix<double> hessian_finiteDiff;
        hessian_finiteDiff.resize(data.V.rows() * dim, data.V.rows() * dim);
        for(int vI = 0; vI < data.V.rows(); vI++)
        {
            if(data.fixedVert.find(vI) != data.fixedVert.end()) {
                hessian_finiteDiff.insert(vI * dim, vI * dim) = 1.0;
                hessian_finiteDiff.insert(vI * dim + 1, vI * dim + 1) = 1.0;
                if(dim == 3) {
                    hessian_finiteDiff.insert(vI * dim + 2, vI * dim + 2) = 1.0;
                }
                continue;
            }
            
            for(int dimI = 0; dimI < dim; dimI++) {
                perturbed.V = data.V;
                perturbed.V(vI, dimI) += h;
                Eigen::VectorXd gradient_perturbed;
//                computeGradientBySVD(perturbed, gradient_perturbed);
                computeGradientByPK(perturbed, true, svd, F, gradient_perturbed);
                Eigen::VectorXd hessian_colI = (gradient_perturbed - gradient0) / h;
                int colI = vI * dim + dimI;
                for(int rowI = 0; rowI < data.V.rows() * dim; rowI++) {
                    if((data.fixedVert.find(rowI / dim) == data.fixedVert.end()) &&
                       (hessian_colI[rowI] != 0.0))
                    {
                        hessian_finiteDiff.insert(rowI, colI) = hessian_colI[rowI];
                    }
                }
            }
            
            if(((vI + 1) % 100) == 0) {
                std::cout << vI + 1 << "/" << data.V.rows() << " vertices computed" << std::endl;
            }
        }
        
        Eigen::SparseMatrix<double> hessian_symbolicPK, hessian_symbolicSVD;
        if(triplet) {
            Eigen::VectorXi I, J;
            Eigen::VectorXd V;
//            computePrecondMtr(data, &V, &I, &J); //TODO: change name to Hessian! don't project!
            computeHessianBySVD(data, &V, &I, &J, false);
            std::vector<Eigen::Triplet<double>> triplet(V.size());
            for(int entryI = 0; entryI < V.size(); entryI++) {
                triplet[entryI] = Eigen::Triplet<double>(I[entryI], J[entryI], V[entryI]);
            }
            hessian_symbolicSVD.resize(data.V.rows() * dim, data.V.rows() * dim);
            hessian_symbolicSVD.setFromTriplets(triplet.begin(), triplet.end());
            
            LinSysSolver<Eigen::VectorXi, Eigen::VectorXd> *linSysSolver;
#ifdef LINSYSSOLVER_USE_CHOLMOD
            linSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
            linSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
            linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
            linSysSolver->set_type(1, 2);
            linSysSolver->set_pattern(data.vNeighbor, data.fixedVert);
            linSysSolver->setZero();
            computeHessianByPK(data, true, svd, F, 1.0, linSysSolver, false);
            linSysSolver->getCoeffMtr(hessian_symbolicPK);
        }
        else {
            computeHessian(data, hessian_symbolicSVD); //TODO: don't project
        }
        
        Eigen::SparseMatrix<double> difMtrPK = hessian_symbolicPK - hessian_finiteDiff;
        const double difPK_L2 = difMtrPK.norm();
        const double relErrPK = difPK_L2 / hessian_finiteDiff.norm();
        std::cout << "PK L2 dist = " << difPK_L2 << ", relErr = " << relErrPK << std::endl;
        IglUtils::writeSparseMatrixToFile(outputFolderPath + "H_symbolicPK", hessian_symbolicPK, true);

        Eigen::SparseMatrix<double> difMtrSVD = hessian_symbolicSVD - hessian_finiteDiff;
        const double difSVD_L2 = difMtrSVD.norm();
        const double relErrSVD = difSVD_L2 / hessian_finiteDiff.norm();
        std::cout << "SVD L2 dist = " << difSVD_L2 << ", relErr = " << relErrSVD << std::endl;
        IglUtils::writeSparseMatrixToFile(outputFolderPath + "H_symbolicSVD", hessian_symbolicSVD, true);

        IglUtils::writeSparseMatrixToFile(outputFolderPath + "H_finiteDiff", hessian_finiteDiff, true);
    }
    
    template<int dim>
    void Energy<dim>::getEnergyValPerElemBySVD(const TriangleSoup<dim>& data, bool redoSVD,
                                               std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                               std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                               Eigen::VectorXd& energyValPerElem,
                                               bool uniformWeight) const
    {
        energyValPerElem.resize(data.F.rows());
//        FILE *out = fopen("/Users/mincli/Desktop/OptCuts_dynamic/output/F", "w");
//        assert(out);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            if(redoSVD) {
                timer_temp2.start(4);
                timer_temp.start(1);
                const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
                Eigen::Matrix<double, dim, dim> Xt;
                Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
                Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
                if(dim == 3) {
                    Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
                }
                F[triI] = Xt * data.restTriInv[triI];
                timer_temp2.stop();
                
                timer_temp.start(0);
                svd[triI].compute(F[triI], Eigen::ComputeFullU | Eigen::ComputeFullV);
//                Eigen::Matrix2d F = Xt * A;
//                fprintf(out, "%le %le %le %le\n", F(0, 0), F(0, 1), F(1, 0), F(1, 1));
            }
            
            timer_temp2.start(4);
            timer_temp.start(1);
            compute_E(svd[triI].singularValues(), energyValPerElem[triI]);
            if(!uniformWeight) {
                energyValPerElem[triI] *= data.triWeight[triI] * data.triArea[triI];
            }
            timer_temp.stop();
            timer_temp2.stop();
        }
#ifdef USE_TBB
        );
#endif
//        fclose(out);
    }
    
    template<int dim>
    void Energy<dim>::computeEnergyValBySVD(const TriangleSoup<dim>& data, bool redoSVD,
                                            std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                            std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                            double& energyVal) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElemBySVD(data, redoSVD, svd, F, energyValPerElem);
        energyVal = energyValPerElem.sum();
    }
    
    template<int dim>
    void Energy<dim>::computeGradientBySVD(const TriangleSoup<dim>& data, Eigen::VectorXd& gradient) const
    {
        gradient.resize(data.V.rows() * dim);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
            
            Eigen::Matrix<double, dim, dim> Xt, A;
            Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
            Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            if(dim == 3) {
                Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
            }
            A = data.restTriInv[triI]; //TODO: this only need to be computed once
            
            timer_temp.start(0);
            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            timer_temp.stop();
            
            timer_temp.start(1);
            Eigen::Matrix<double, dim * (dim + 1), dim> dsigma_div_dx;
            IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
            
            Eigen::Matrix<double, dim, 1> dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            for(int triVI = 0; triVI < dim + 1; triVI++) {
                gradient.segment(triVInd[triVI] * dim, dim) += w * dsigma_div_dx.block(triVI * dim, 0, dim, dim) * dE_div_dsigma;
            }
            timer_temp.stop();
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient.segment<dim>(dim * fixedVI).setZero();
        }
    }
    
    template<int dim>
    void Energy<dim>::computeHessianBySVD(const TriangleSoup<dim>& data, Eigen::VectorXd* V,
                                          Eigen::VectorXi* I, Eigen::VectorXi* J,
                                          bool projectSPD) const
    {
        std::vector<Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)>> triHessians(data.F.rows());
        std::vector<Eigen::Matrix<int, dim + 1, 1>> vInds(data.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
            
            Eigen::Matrix<double, dim, dim> Xt, A;
            Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
            Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            if(dim == 3) {
                Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
            }
            A = data.restTriInv[triI]; //TODO: this only need to be computed once
            
            timer_temp.start(0);
            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            timer_temp.stop();
            
            timer_temp.start(1);
            // right term:
            Eigen::Matrix<double, dim, 1> dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            
            Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1) * dim> d2sigma_div_dx2;
            IglUtils::compute_d2sigma_div_dx2(svd, A, d2sigma_div_dx2);
            
            Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)> d2E_div_dx2_right = d2sigma_div_dx2.block(0, 0, dim * (dim + 1), dim * (dim + 1)) * dE_div_dsigma[0] +
                d2sigma_div_dx2.block(0, dim * (dim + 1), dim * (dim + 1), dim * (dim + 1)) * dE_div_dsigma[1];
            if(dim == 3) {
                d2E_div_dx2_right += d2sigma_div_dx2.block(0, dim * (dim + 1) * 2, dim * (dim + 1), dim * (dim + 1)) * dE_div_dsigma[2];
            }
            
            // left term:
            Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
            compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
            
            Eigen::Matrix<double, dim * (dim + 1), dim> dsigma_div_dx;
            IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
            
            Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)> d2E_div_dx2_left =
                d2E_div_dsigma2(0, 0) * dsigma_div_dx.col(0) * dsigma_div_dx.col(0).transpose() +
                d2E_div_dsigma2(1, 1) * dsigma_div_dx.col(1) * dsigma_div_dx.col(1).transpose();
            if(dim == 3) {
                d2E_div_dx2_left += d2E_div_dsigma2(2, 2) * dsigma_div_dx.col(2) * dsigma_div_dx.col(2).transpose();
            }
            
            // cross sigma derivative
            if(crossSigmaDervative) {
                for(int sigmaI = 0; sigmaI < dim; sigmaI++) {
                    for(int sigmaJ = sigmaI + 1; sigmaJ < dim; sigmaJ++) {
                        const Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)> m =
                            dsigma_div_dx.col(sigmaJ) * dsigma_div_dx.col(sigmaI).transpose();
                        d2E_div_dx2_left += d2E_div_dsigma2(sigmaI, sigmaJ) * m +
                            d2E_div_dsigma2(sigmaJ, sigmaI) * m.transpose();
                    }
                }
            }
            
            // add up left term and right term
            const double w = data.triWeight[triI] * data.triArea[triI];
            triHessians[triI] = w * (d2E_div_dx2_left + d2E_div_dx2_right);
            timer_temp.stop();
            
            if(projectSPD) {
                timer_temp.start(2);
                IglUtils::makePD(triHessians[triI]);
                timer_temp.stop();
            }
            
            Eigen::Matrix<int, dim + 1, 1>& vInd = vInds[triI];
            vInd = triVInd;
            for(int vI = 0; vI < dim + 1; vI++) {
                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
                    vInd[vI] = -1;
                }
            }
        }
#ifdef USE_TBB
        );
#endif
        timer_temp.start(3);
        for(int triI = 0; triI < data.F.rows(); triI++) {
            IglUtils::addBlockToMatrix(triHessians[triI], vInds[triI], dim, V, I, J);
        }
        
        Eigen::VectorXi fixedVertInd;
        fixedVertInd.resize(data.fixedVert.size());
        int fVI = 0;
        for(const auto fixedVI : data.fixedVert) {
            fixedVertInd[fVI++] = fixedVI;
        }
        IglUtils::addDiagonalToMatrix(Eigen::VectorXd::Ones(data.fixedVert.size() * dim),
                                      fixedVertInd, dim, V, I, J);
        timer_temp.stop();
    }
    
    template<int dim>
    void Energy<dim>::computeGradientByPK(const TriangleSoup<dim>& data, bool redoSVD,
                                          std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                          std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                          Eigen::VectorXd& gradient) const
    {
        std::vector<Eigen::Matrix<double, dim * (dim + 1), 1>> gradient_cont(data.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            const Eigen::Matrix<double, dim, dim>& A = data.restTriInv[triI];
            
            if(redoSVD) {
                timer_temp2.start(5);
                timer_temp.start(1);
                const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
                
                Eigen::Matrix<double, dim, dim> Xt;
                Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
                Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
                if(dim == 3) {
                    Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
                }
                
                F[triI] = Xt * A;
                timer_temp2.stop();
                
                timer_temp.start(0);
                svd[triI].compute(F[triI], Eigen::ComputeFullU | Eigen::ComputeFullV);
                timer_temp.start(1);
            }
            
            timer_temp2.start(6);
            Eigen::Matrix<double, dim, dim> P;
            compute_dE_div_dF(F[triI], svd[triI], P);
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            P *= w;
            
            timer_temp2.start(7);
            IglUtils::dF_div_dx_mult(P, A, gradient_cont[triI]);
            timer_temp.stop();
            timer_temp2.stop();
        }
#ifdef USE_TBB
        );
#endif
        timer_temp2.start(8);
        gradient.conservativeResize(data.V.rows() * dim);
        gradient.setZero();
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            int _dimVI = vI * dim;
            for(const auto FLocI : data.vFLoc[vI]) {
                gradient.segment<dim>(_dimVI) +=
                    gradient_cont[FLocI.first].segment(FLocI.second * dim, dim);
            }
        }
#ifdef USE_TBB
        );
#endif
        timer_temp2.stop();
        
        for(const auto fixedVI : data.fixedVert) {
            gradient.segment<dim>(dim * fixedVI).setZero();
        }
    }
    template<int dim>
    void Energy<dim>::computeHessianByPK(const TriangleSoup<dim>& data, bool redoSVD,
                                         std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
                                         std::vector<Eigen::Matrix<double, dim, dim>>& F,
                                         double coef,
                                         LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                         bool projectSPD) const
    {
        std::vector<Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1)>> triHessians(data.F.rows());
        std::vector<Eigen::Matrix<int, 1, dim + 1>> vInds(data.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            timer_temp.start(1);
            const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
            
            const Eigen::Matrix<double, dim, dim>& A = data.restTriInv[triI];
            
            if(redoSVD) {
                timer_temp.start(0);
                Eigen::Matrix<double, dim, dim> Xt;
                Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
                Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
                if(dim == 3) {
                    Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
                }
                F[triI] = Xt * A;
                svd[triI].compute(F[triI], Eigen::ComputeFullU | Eigen::ComputeFullV);
                timer_temp.start(1);
            }
            const Eigen::Matrix<double, dim, 1>& sigma = svd[triI].singularValues();
            
            // compute A
            timer_temp2.start(0);
            Eigen::Matrix<double, dim, 1> dE_div_dsigma;
            compute_dE_div_dsigma(sigma, dE_div_dsigma);
            Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
            compute_d2E_div_dsigma2(sigma, d2E_div_dsigma2);
            timer_temp2.stop();
            if(projectSPD) {
                timer_temp.start(2);
#if(DIM == 2)
                IglUtils::makePD2d(d2E_div_dsigma2);
#else
                IglUtils::makePD(d2E_div_dsigma2); //TODO: use implicit QR to accelerate
#endif
                timer_temp.start(1);
            }
            
            // compute B
            const int Cdim2 = dim * (dim - 1) / 2;
            Eigen::Matrix<double, Cdim2, 1> BLeftCoef;
            compute_BLeftCoef(sigma, BLeftCoef);
            Eigen::Matrix2d B[Cdim2];
            for(int cI = 0; cI < Cdim2; cI++) {
                timer_temp2.start(1);
                int cI_post = (cI + 1) % Cdim2;
                
                double rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
                double sum_sigma = sigma[cI] + sigma[cI_post];
                const double eps = 1.0e-6;
                if(sum_sigma < eps) {
                    rightCoef /= 2.0 * eps;
                }
                else {
                    rightCoef /= 2.0 * sum_sigma;
                }
            
                const double& leftCoef = BLeftCoef[cI];
                B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
                B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
                timer_temp2.stop();
                if(projectSPD) {
                    timer_temp.start(2);
                    IglUtils::makePD2d(B[cI]);
                    timer_temp.start(1);
                }
            }
            
            // compute M using A(d2E_div_dsigma2) and B
            const double w = coef * data.triWeight[triI] * data.triArea[triI];
            timer_temp2.start(2);
            Eigen::Matrix<double, dim * dim, dim * dim> M;
            M.setZero();
            if(dim == 2) {
                M(0, 0) = w * d2E_div_dsigma2(0, 0);
                M(0, 3) = w * d2E_div_dsigma2(0, 1);
                M.block(1, 1, 2, 2) = w * B[0];
                M(3, 0) = w * d2E_div_dsigma2(1, 0);
                M(3, 3) = w * d2E_div_dsigma2(1, 1);
            }
            else {
                // A
                M(0, 0) = w * d2E_div_dsigma2(0, 0);
                M(0, 4) = w * d2E_div_dsigma2(0, 1);
                M(0, 8) = w * d2E_div_dsigma2(0, 2);
                M(4, 0) = w * d2E_div_dsigma2(1, 0);
                M(4, 4) = w * d2E_div_dsigma2(1, 1);
                M(4, 8) = w * d2E_div_dsigma2(1, 2);
                M(8, 0) = w * d2E_div_dsigma2(2, 0);
                M(8, 4) = w * d2E_div_dsigma2(2, 1);
                M(8, 8) = w * d2E_div_dsigma2(2, 2);
                // B01
                M(1, 1) = w * B[0](0, 0);
                M(1, 3) = w * B[0](0, 1);
                M(3, 1) = w * B[0](1, 0);
                M(3, 3) = w * B[0](1, 1);
                // B12
                M(5, 5) = w * B[1](0, 0);
                M(5, 7) = w * B[1](0, 1);
                M(7, 5) = w * B[1](1, 0);
                M(7, 7) = w * B[1](1, 1);
                // B20
                M(2, 2) = w * B[2](1, 1);
                M(2, 6) = w * B[2](1, 0);
                M(6, 2) = w * B[2](0, 1);
                M(6, 6) = w * B[2](0, 0);
            }
            
            // compute dP_div_dF
            Eigen::Matrix<double, dim * dim, dim * dim> wdP_div_dF;
            const Eigen::Matrix<double, dim, dim>& U = svd[triI].matrixU();
            const Eigen::Matrix<double, dim, dim>& V = svd[triI].matrixV();
            for(int i = 0; i < dim; i++) {
                int _dim_i = i * dim;
                for(int j = 0; j < dim; j++) {
                    int ij = _dim_i + j;
                    for(int r = 0; r < dim; r++) {
                        int _dim_r = r * dim;
                        for(int s = 0; s < dim; s++) {
                            int rs = _dim_r + s;
                            if(ij < rs) {
                                // upper right
                                if(dim == 2) {
                                    wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                                }
                                else {
                                    wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                                        M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                                        M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                                        M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                                        M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                                        M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                                        M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                                        M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                                        M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                                        M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                                        M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                                        M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                                        M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                                }
                            }
                            else if(ij == rs) {
                                // diagonal
                                if(dim == 2) {
                                    wdP_div_dF(ij, rs) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                                }
                                else {
                                    wdP_div_dF(ij, rs) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                                        M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                                        M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                                        M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                                        M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                                        M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                                        M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                                        M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                                        M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                                        M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                                        M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                                        M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                                        M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                                }
                            }
                            else {
                                // bottom left, same as upper right
                                continue;
                            }
                        }
                    }
                }
            }
            timer_temp2.stop();
            
            Eigen::Matrix<double, dim * (dim + 1), dim * dim> wdP_div_dx;
            timer_temp2.start(3);
            IglUtils::dF_div_dx_mult<dim * dim>(wdP_div_dF.transpose(), A, wdP_div_dx, false);
            IglUtils::dF_div_dx_mult<dim * (dim + 1)>(wdP_div_dx.transpose(), A, triHessians[triI], true);
            
            Eigen::Matrix<int, 1, dim + 1>& vInd = vInds[triI];
            vInd[0] = (data.isFixedVert[triVInd[0]] ? (-triVInd[0] - 1) : triVInd[0]);
            vInd[1] = (data.isFixedVert[triVInd[1]] ? (-triVInd[1] - 1) : triVInd[1]);
            vInd[2] = (data.isFixedVert[triVInd[2]] ? (-triVInd[2] - 1) : triVInd[2]);
            if(dim == 3) {
                vInd[3] = (data.isFixedVert[triVInd[3]] ? (-triVInd[3] - 1) : triVInd[3]);
            }
            timer_temp.stop();
        }
#ifdef USE_TBB
        );
#endif
        timer_temp.start(3);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.V.rows(), 1, [&](int vI)
#else
        for(int vI = 0; vI < data.V.rows(); vI++)
#endif
        {
            for(const auto FLocI : data.vFLoc[vI]) {
                IglUtils::addBlockToMatrix<dim>(triHessians[FLocI.first].block(FLocI.second * dim, 0,
                                                                          dim, dim * (dim + 1)),
                                                vInds[FLocI.first], FLocI.second, linSysSolver);
            }
        }
#ifdef USE_TBB
        );
#endif
        timer_temp.stop();
    }
    
    template<int dim>
    void Energy<dim>::computeEnergyValBySVD(const TriangleSoup<dim>& data, int triI,
                                            const Eigen::VectorXd& x,
                                            double& energyVal,
                                            bool uniformWeight) const
    {
        assert(dim == 2);
#if(DIM == 4)
        const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
        
        Eigen::Vector3d x0_3D[dim + 1] = {
            data.V_rest.row(triVInd[0]),
            data.V_rest.row(triVInd[1]),
            data.V_rest.row(triVInd[2])
        };
        if(dim == 3) {
            x0_3D[3] = data.V_rest.row(triVInd[3]);
        }
        Eigen::Matrix<double, dim, 1> x0[dim + 1];
        if(dim == 3) {
            x0[0] = x0_3D[0];
            x0[1] = x0_3D[1];
            x0[2] = x0_3D[2];
            x0[3] = x0_3D[3];
        }
        else {
            IglUtils::mapTriangleTo2D(x0_3D, x0);
        }
        
        Eigen::Matrix2d X0, Xt, A;
        X0 << x0[1] - x0[0], x0[2] - x0[0];
        Xt << x.segment(2, 2) - x.segment(0, 2), x.segment(4, 2) - x.segment(0, 2);
        A = X0.inverse(); //TODO: this only need to be computed once
        
        AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
        
        compute_E(svd.singularValues(), energyVal);
        if(!uniformWeight) {
            energyVal *= data.triWeight[triI] * data.triArea[triI];
        }
#endif
    }
    template<int dim>
    void Energy<dim>::computeGradientBySVD(const TriangleSoup<dim>& data, int triI,
                                      const Eigen::VectorXd& x,
                                      Eigen::VectorXd& gradient) const
    {
        assert(dim == 2);
#if(DIM == 4)
        const Eigen::RowVector3i& triVInd = data.F.row(triI);
        
        Eigen::Vector3d x0_3D[3] = {
            data.V_rest.row(triVInd[0]),
            data.V_rest.row(triVInd[1]),
            data.V_rest.row(triVInd[2])
        };
        Eigen::Vector2d x0[3];
        IglUtils::mapTriangleTo2D(x0_3D, x0);
        
        Eigen::Matrix2d X0, Xt, A;
        X0 << x0[1] - x0[0], x0[2] - x0[0];
        Xt << x.segment(2, 2) - x.segment(0, 2), x.segment(4, 2) - x.segment(0, 2);
        A = X0.inverse(); //TODO: this only need to be computed once
        
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        Eigen::Matrix<double, dim * (dim + 1), dim> dsigma_div_dx;
        IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
        
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        gradient.resize(6);
        const double w = data.triWeight[triI] * data.triArea[triI];
        for(int triVI = 0; triVI < 3; triVI++) {
            gradient.segment(triVI * 2, 2) = w * dsigma_div_dx.block(triVI * 2, 0, 2, 2) * dE_div_dsigma;
        }
#endif
    }
    template<int dim>
    void Energy<dim>::computeHessianBySVD(const TriangleSoup<dim>& data, int triI,
                                     const Eigen::VectorXd& x,
                                     Eigen::MatrixXd& hessian,
                                     bool projectSPD) const
    {
        assert(dim == 2);
#if(DIM == 4)
        const Eigen::RowVector3i& triVInd = data.F.row(triI);
        
        Eigen::Vector3d x0_3D[3] = {
            data.V_rest.row(triVInd[0]),
            data.V_rest.row(triVInd[1]),
            data.V_rest.row(triVInd[2])
        };
        Eigen::Vector2d x0[3];
        IglUtils::mapTriangleTo2D(x0_3D, x0);
        
        Eigen::Matrix2d X0, Xt, A;
        X0 << x0[1] - x0[0], x0[2] - x0[0];
        Xt << x.segment(2, 2) - x.segment(0, 2), x.segment(4, 2) - x.segment(0, 2);
        A = X0.inverse(); //TODO: this only need to be computed once
        
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        // right term:
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        Eigen::Matrix<double, dim * (dim + 1), dim * (dim + 1) * dim> d2sigma_div_dx2;
        IglUtils::compute_d2sigma_div_dx2(svd, A, d2sigma_div_dx2);
        
        Eigen::MatrixXd d2E_div_dx2_right = d2sigma_div_dx2.block(0, 0, 6, 6) * dE_div_dsigma[0] +
        d2sigma_div_dx2.block(0, 6, 6, 6) * dE_div_dsigma[1];
        
        // left term:
        Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
        compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
        
        Eigen::Matrix<double, dim * (dim + 1), dim> dsigma_div_dx;
        IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
        
        Eigen::MatrixXd d2E_div_dx2_left = d2E_div_dsigma2(0, 0) * dsigma_div_dx.col(0) * dsigma_div_dx.col(0).transpose() +
        d2E_div_dsigma2(1, 1) * dsigma_div_dx.col(1) * dsigma_div_dx.col(1).transpose();
        
        // cross sigma derivative
        if(crossSigmaDervative) {
            for(int sigmaI = 0; sigmaI < svd.singularValues().size(); sigmaI++) {
                for(int sigmaJ = sigmaI + 1; sigmaJ < svd.singularValues().size(); sigmaJ++) {
                    const Eigen::MatrixXd m = dsigma_div_dx.col(sigmaJ) * dsigma_div_dx.col(sigmaI).transpose();
                    d2E_div_dx2_left += d2E_div_dsigma2(sigmaI, sigmaJ) * m +
                    d2E_div_dsigma2(sigmaJ, sigmaI) * m.transpose();
                }
            }
        }
        
        // add up left term and right term
        const double w = data.triWeight[triI] * data.triArea[triI];
        hessian = w * (d2E_div_dx2_left + d2E_div_dx2_right);
        
        if(projectSPD) {
            IglUtils::makePD(hessian);
        }
#endif
    }
    
    template<int dim>
    void Energy<dim>::computeEnergyValBySVD_F(const TriangleSoup<dim>& data, int triI,
                                              const Eigen::Matrix<double, 1, dim * dim>& F,
                                              double& energyVal,
                                              bool uniformWeight) const
    {
        Eigen::Matrix<double, dim, dim> F_mtr;
        F_mtr.row(0) = F.segment(0, dim);
        F_mtr.row(1) = F.segment(dim, dim);
        if(dim == 3) {
            F_mtr.row(2) = F.segment(dim * 2, dim);
        }
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F_mtr); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
        
        compute_E(svd.singularValues(), energyVal);
        if(!uniformWeight) {
            energyVal *= data.triWeight[triI] * data.triArea[triI];
        }
    }
    template<int dim>
    void Energy<dim>::computeGradientBySVD_F(const TriangleSoup<dim>& data, int triI,
                                             const Eigen::Matrix<double, 1, dim * dim>& F,
                                             Eigen::Matrix<double, dim * dim, 1>& gradient) const
    {
        Eigen::Matrix<double, dim, dim> F_mtr;
        F_mtr.row(0) = F.segment(0, dim);
        F_mtr.row(1) = F.segment(dim, dim);
        if(dim == 3) {
            F_mtr.row(2) = F.segment(dim * 2, dim);
        }
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F_mtr, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        gradient.setZero();
        for(int dimI = 0; dimI < dim; dimI++) {
            Eigen::Matrix<double, dim, dim> dsigma_div_dF = svd.matrixU().col(dimI) * svd.matrixV().col(dimI).transpose();
            Eigen::Matrix<double, 1, dim * dim> dsigma_div_dF_vec;
            dsigma_div_dF_vec.segment(0, dim) = dsigma_div_dF.row(0);
            dsigma_div_dF_vec.segment(dim, dim) = dsigma_div_dF.row(1);
            if(dim == 3) {
                dsigma_div_dF_vec.segment(dim * 2, dim) = dsigma_div_dF.row(2);
            }
            gradient += dsigma_div_dF_vec.transpose() * dE_div_dsigma[dimI];
        }
        
        const double w = data.triWeight[triI] * data.triArea[triI];
        gradient *= w;
    }
    template<int dim>
    void Energy<dim>::computeHessianBySVD_F(const TriangleSoup<dim>& data, int triI,
                                            const Eigen::Matrix<double, 1, dim * dim>& F,
                                            Eigen::Matrix<double, dim * dim, dim * dim>& hessian,
                                            bool projectSPD) const
    {
        Eigen::Matrix<double, dim, dim> F_mtr;
        F_mtr.row(0) = F.segment(0, dim);
        F_mtr.row(1) = F.segment(dim, dim);
        if(dim == 3) {
            F_mtr.row(2) = F.segment(dim * 2, dim);
        }
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F_mtr, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        // right term:
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        Eigen::Matrix<double, dim * dim, dim * dim * dim> d2sigma_div_dF2;
        IglUtils::compute_d2sigma_div_dF2(svd, d2sigma_div_dF2);
        
        Eigen::Matrix<double, dim * dim, dim * dim> d2E_div_dF2_right =
            d2sigma_div_dF2.block(0, 0, dim * dim, dim * dim) * dE_div_dsigma[0] +
            d2sigma_div_dF2.block(0, dim * dim, dim * dim, dim * dim) * dE_div_dsigma[1];
        if(dim == 3) {
            d2E_div_dF2_right +=
                d2sigma_div_dF2.block(0, dim * dim * 2, dim * dim, dim * dim) * dE_div_dsigma[2];
        }
        
        // left term:
        Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
        compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
        
        Eigen::Matrix<double, dim * dim, dim> dsigma_div_dF;
        for(int dimI = 0; dimI < dim; dimI++) {
            Eigen::Matrix<double, dim, dim> dsigmai_div_dF = svd.matrixU().col(dimI) * svd.matrixV().col(dimI).transpose();
            dsigma_div_dF.block(0, dimI, dim, 1) = dsigmai_div_dF.row(0).transpose();
            dsigma_div_dF.block(dim, dimI, dim, 1) = dsigmai_div_dF.row(1).transpose();
            if(dim == 3) {
                dsigma_div_dF.block(dim * 2, dimI, dim, 1) = dsigmai_div_dF.row(2).transpose();
            }
        }
        
        Eigen::Matrix<double, dim * dim, dim * dim> d2E_div_dF2_left =
            d2E_div_dsigma2(0, 0) * dsigma_div_dF.col(0) * dsigma_div_dF.col(0).transpose() +
            d2E_div_dsigma2(1, 1) * dsigma_div_dF.col(1) * dsigma_div_dF.col(1).transpose();
        if(dim == 3) {
            d2E_div_dF2_left +=
                d2E_div_dsigma2(2, 2) * dsigma_div_dF.col(2) * dsigma_div_dF.col(2).transpose();
        }
        
        // cross sigma derivative
        if(crossSigmaDervative) {
            for(int sigmaI = 0; sigmaI < dim; sigmaI++) {
                for(int sigmaJ = sigmaI + 1; sigmaJ < dim; sigmaJ++) {
                    const Eigen::Matrix<double, dim * dim, dim * dim> m =
                        dsigma_div_dF.col(sigmaJ) * dsigma_div_dF.col(sigmaI).transpose();
                    d2E_div_dF2_left +=
                        d2E_div_dsigma2(sigmaI, sigmaJ) * m +
                        d2E_div_dsigma2(sigmaJ, sigmaI) * m.transpose();
                }
            }
        }
        
        // add up left term and right term
        const double w = data.triWeight[triI] * data.triArea[triI];
        hessian = w * (d2E_div_dF2_left + d2E_div_dF2_right);
        
        if(projectSPD) {
            IglUtils::makePD(hessian);
        }
    }
    template<int dim>
    void Energy<dim>::computeGradientByPK_F(const TriangleSoup<dim>& data, int triI,
                                            const Eigen::Matrix<double, 1, dim * dim>& F,
                                            Eigen::Matrix<double, dim * dim, 1>& gradient) const
    {
        Eigen::Matrix<double, dim, dim> F_mtr;
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd;
        bool redoSVD = true;
        if(redoSVD) {
            F_mtr.row(0) = F.segment(0, dim);
            F_mtr.row(1) = F.segment(dim, dim);
            if(dim == 3) {
                F_mtr.row(2) = F.segment(dim * 2, dim);
            }
            svd.compute(F_mtr, Eigen::ComputeFullU | Eigen::ComputeFullV);
        }
        
        Eigen::Matrix<double, dim, dim> P;
        compute_dE_div_dF(F_mtr, svd, P);
        
        const double w = data.triWeight[triI] * data.triArea[triI];
        gradient.segment(0, dim) = w * P.row(0);
        gradient.segment(dim, dim) = w * P.row(1);
        if(dim == 3) {
            gradient.segment(dim * 2, dim) = w * P.row(2);
        }
    }
    template<int dim>
    void Energy<dim>::computeHessianByPK_F(const TriangleSoup<dim>& data, int triI,
                                           const Eigen::Matrix<double, 1, dim * dim>& F,
                                           Eigen::Matrix<double, dim * dim, dim * dim>& hessian,
                                           bool projectSPD) const
    {
        Eigen::Matrix<double, dim, dim> F_mtr;
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd;
        bool redoSVD = true;
        if(redoSVD) {
            F_mtr.row(0) = F.segment(0, dim);
            F_mtr.row(1) = F.segment(dim, dim);
            if(dim == 3) {
                F_mtr.row(2) = F.segment(dim * 2, dim);
            }
            svd.compute(F_mtr, Eigen::ComputeFullU | Eigen::ComputeFullV);
        }
        
        const Eigen::Matrix<double, dim, 1>& sigma = svd.singularValues();
        
        // compute A
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(sigma, dE_div_dsigma);
        Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
        compute_d2E_div_dsigma2(sigma, d2E_div_dsigma2);
        if(projectSPD) {
#if(DIM == 2)
            IglUtils::makePD2d(d2E_div_dsigma2);
#else
            IglUtils::makePD(d2E_div_dsigma2); //TODO: use implicit QR to accelerate
#endif
        }
        
        // compute B
        const int Cdim2 = dim * (dim - 1) / 2;
        Eigen::Matrix<double, Cdim2, 1> BLeftCoef;
        compute_BLeftCoef(sigma, BLeftCoef);
        Eigen::Matrix2d B[Cdim2];
        for(int cI = 0; cI < Cdim2; cI++) {
            int cI_post = (cI + 1) % Cdim2;
            
            double rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
            double sum_sigma = sigma[cI] + sigma[cI_post];
            const double eps = 1.0e-6;
            if(sum_sigma < eps) {
                rightCoef /= 2.0 * eps;
            }
            else {
                rightCoef /= 2.0 * sum_sigma;
            }
            
            const double& leftCoef = BLeftCoef[cI];
            B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
            B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
            if(projectSPD) {
                IglUtils::makePD2d(B[cI]);
            }
        }
        
        // compute M using A(d2E_div_dsigma2) and B
        const double w = data.triWeight[triI] * data.triArea[triI];
        Eigen::Matrix<double, dim * dim, dim * dim>& M = hessian;
        M.setZero();
        if(dim == 2) {
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 3) = w * d2E_div_dsigma2(0, 1);
            M.block(1, 1, 2, 2) = w * B[0];
            M(3, 0) = w * d2E_div_dsigma2(1, 0);
            M(3, 3) = w * d2E_div_dsigma2(1, 1);
        }
        else {
            // A
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 4) = w * d2E_div_dsigma2(0, 1);
            M(0, 8) = w * d2E_div_dsigma2(0, 2);
            M(4, 0) = w * d2E_div_dsigma2(1, 0);
            M(4, 4) = w * d2E_div_dsigma2(1, 1);
            M(4, 8) = w * d2E_div_dsigma2(1, 2);
            M(8, 0) = w * d2E_div_dsigma2(2, 0);
            M(8, 4) = w * d2E_div_dsigma2(2, 1);
            M(8, 8) = w * d2E_div_dsigma2(2, 2);
            // B01
            M(1, 1) = w * B[0](0, 0);
            M(1, 3) = w * B[0](0, 1);
            M(3, 1) = w * B[0](1, 0);
            M(3, 3) = w * B[0](1, 1);
            // B12
            M(5, 5) = w * B[1](0, 0);
            M(5, 7) = w * B[1](0, 1);
            M(7, 5) = w * B[1](1, 0);
            M(7, 7) = w * B[1](1, 1);
            // B20
            M(2, 2) = w * B[2](1, 1);
            M(2, 6) = w * B[2](1, 0);
            M(6, 2) = w * B[2](0, 1);
            M(6, 6) = w * B[2](0, 0);
        }
        
        // compute dP_div_dF
        Eigen::Matrix<double, dim * dim, dim * dim> wdP_div_dF;
        const Eigen::Matrix<double, dim, dim>& U = svd.matrixU();
        const Eigen::Matrix<double, dim, dim>& V = svd.matrixV();
        for(int i = 0; i < dim; i++) {
            int _dim_i = i * dim;
            for(int j = 0; j < dim; j++) {
                int ij = _dim_i + j;
                for(int r = 0; r < dim; r++) {
                    int _dim_r = r * dim;
                    for(int s = 0; s < dim; s++) {
                        int rs = _dim_r + s;
                        if(ij < rs) {
                            // upper right
                            if(dim == 2) {
                                wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                            }
                            else {
                                wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                                M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                                M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                                M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                                M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                                M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                                M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                                M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                                M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                                M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                                M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                                M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                                M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                                M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                            }
                        }
                        else if(ij == rs) {
                            // diagonal
                            if(dim == 2) {
                                wdP_div_dF(ij, rs) =
                                M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                            }
                            else {
                                wdP_div_dF(ij, rs) =
                                M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                                M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                                M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                                M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                                M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                                M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                                M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                                M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                                M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                                M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                                M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                                M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                                M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                                M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                            }
                        }
                        else {
                            // bottom left, same as upper right
                            continue;
                        }
                    }
                }
            }
        }
    }
    
    template<int dim>
    void Energy<dim>::compute_E(const Eigen::Matrix<double, dim, 1>& singularValues,
                                double& E) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    template<int dim>
    void Energy<dim>::compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
                                            Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    template<int dim>
    void Energy<dim>::compute_d2E_div_dsigma2(const Eigen::Matrix<double, dim, 1>& singularValues,
                                              Eigen::Matrix<double, dim, dim>& d2E_div_dsigma2) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    template<int dim>
    void Energy<dim>::compute_BLeftCoef(const Eigen::Matrix<double, dim, 1>& singularValues,
                                        Eigen::Matrix<double, dim * (dim - 1) / 2, 1>& BLeftCoef) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    template<int dim>
    void Energy<dim>::compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
                                        const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
                                        Eigen::Matrix<double, dim, dim>& dE_div_dF) const
    {
        assert(0 && "please implement this method in the subclass!");
    }

    template<int dim>
    void Energy<dim>::compute_dP_div_dF(const AutoFlipSVD<Eigen::Matrix<double, dim, dim>> &svd,
                                        Eigen::Matrix<double, dim * dim, dim * dim> &dP_div_dF,
                                        double w, bool projectSPD) const
    {
        // compute A
        const Eigen::Matrix<double, dim, 1>& sigma = svd.singularValues();
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(sigma, dE_div_dsigma);
        Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
        compute_d2E_div_dsigma2(sigma, d2E_div_dsigma2);
        if (projectSPD) {
#if(DIM == 2)
            IglUtils::makePD2d(d2E_div_dsigma2);
#else
            IglUtils::makePD(d2E_div_dsigma2); //TODO: use implicit QR to accelerate
#endif
        }

        // compute B
        const int Cdim2 = dim * (dim - 1) / 2;
        Eigen::Matrix<double, Cdim2, 1> BLeftCoef;
        compute_BLeftCoef(sigma, BLeftCoef);
        Eigen::Matrix2d B[Cdim2];
        for (int cI = 0; cI < Cdim2; cI++) {
            int cI_post = (cI + 1) % Cdim2;

            double rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
            double sum_sigma = sigma[cI] + sigma[cI_post];
            const double eps = 1.0e-6;
            if (sum_sigma < eps) {
                rightCoef /= 2.0 * eps;
            } else {
                rightCoef /= 2.0 * sum_sigma;
            }

            const double &leftCoef = BLeftCoef[cI];
            B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
            B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
            if (projectSPD) {
                IglUtils::makePD2d(B[cI]);
            }
        }

        // compute M using A(d2E_div_dsigma2) and B
        Eigen::Matrix<double, dim * dim, dim * dim> M;
        M.setZero();
        if (dim == 2) {
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 3) = w * d2E_div_dsigma2(0, 1);
            M.block(1, 1, 2, 2) = w * B[0];
            M(3, 0) = w * d2E_div_dsigma2(1, 0);
            M(3, 3) = w * d2E_div_dsigma2(1, 1);
        } else {
            // A
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 4) = w * d2E_div_dsigma2(0, 1);
            M(0, 8) = w * d2E_div_dsigma2(0, 2);
            M(4, 0) = w * d2E_div_dsigma2(1, 0);
            M(4, 4) = w * d2E_div_dsigma2(1, 1);
            M(4, 8) = w * d2E_div_dsigma2(1, 2);
            M(8, 0) = w * d2E_div_dsigma2(2, 0);
            M(8, 4) = w * d2E_div_dsigma2(2, 1);
            M(8, 8) = w * d2E_div_dsigma2(2, 2);
            // B01
            M(1, 1) = w * B[0](0, 0);
            M(1, 3) = w * B[0](0, 1);
            M(3, 1) = w * B[0](1, 0);
            M(3, 3) = w * B[0](1, 1);
            // B12
            M(5, 5) = w * B[1](0, 0);
            M(5, 7) = w * B[1](0, 1);
            M(7, 5) = w * B[1](1, 0);
            M(7, 7) = w * B[1](1, 1);
            // B20
            M(2, 2) = w * B[2](1, 1);
            M(2, 6) = w * B[2](1, 0);
            M(6, 2) = w * B[2](0, 1);
            M(6, 6) = w * B[2](0, 0);
        }

        // compute dP_div_dF
        Eigen::Matrix<double, dim * dim, dim * dim>& wdP_div_dF = dP_div_dF;
        const Eigen::Matrix<double, dim, dim> &U = svd.matrixU();
        const Eigen::Matrix<double, dim, dim> &V = svd.matrixV();
        for (int i = 0; i < dim; i++) {
            int _dim_i = i * dim;
            for (int j = 0; j < dim; j++) {
                int ij = _dim_i + j;
                for (int r = 0; r < dim; r++) {
                    int _dim_r = r * dim;
                    for (int s = 0; s < dim; s++) {
                        int rs = _dim_r + s;
                        if (ij < rs) {
                            // upper right
                            if (dim == 2) {
                                wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                            } else {
                                wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                                        M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                                        M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                                        M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                                        M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                                        M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                                        M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                                        M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                                        M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                                        M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                                        M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                                        M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                                        M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                            }
                        } else if (ij == rs) {
                            // diagonal
                            if (dim == 2) {
                                wdP_div_dF(ij, rs) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 3) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 2) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(2, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(2, 2) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(3, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(3, 3) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
                            } else {
                                wdP_div_dF(ij, rs) =
                                        M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) +
                                        M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) +
                                        M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) +
                                        M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) +
                                        M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) +
                                        M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) +
                                        M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) +
                                        M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) +
                                        M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) +
                                        M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) +
                                        M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) +
                                        M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) +
                                        M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) +
                                        M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) +
                                        M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) +
                                        M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) +
                                        M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) +
                                        M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) +
                                        M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) +
                                        M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) +
                                        M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
                            }
                        } else {
                            // bottom left, same as upper right
                            continue;
                        }
                    }
                }
            }
        }
    }
    
    template<int dim>
    void Energy<dim>::compute_d2E_div_dF2_rest(Eigen::Matrix<double, dim * dim, dim * dim>& d2E_div_dF2_rest) const
    {
        AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(Eigen::Matrix<double, dim, dim>::Identity(),
                                                         Eigen::ComputeFullU |
                                                         Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        // right term:
        Eigen::Matrix<double, dim, 1> dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        Eigen::Matrix<double, dim * dim, dim * dim * dim> d2sigma_div_dF2;
        IglUtils::compute_d2sigma_div_dF2(svd, d2sigma_div_dF2);
        
        Eigen::Matrix<double, dim * dim, dim * dim> d2E_div_dF2_right =
            d2sigma_div_dF2.block(0, 0, dim * dim, dim * dim) * dE_div_dsigma[0] +
            d2sigma_div_dF2.block(0, dim * dim, dim * dim, dim * dim) * dE_div_dsigma[1];
        if(dim == 3) {
            d2E_div_dF2_right += d2sigma_div_dF2.block(0, dim * dim * 2,
                                                       dim * dim, dim * dim) * dE_div_dsigma[2];
        }
        
        // left term:
        Eigen::Matrix<double, dim, dim> d2E_div_dsigma2;
        compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
        
        Eigen::Matrix<double, dim * dim, dim> dsigma_div_dF;
        for(int dimI = 0; dimI < dim; dimI++) {
            Eigen::Matrix<double, dim, dim> dsigma_div_dF_mtr = svd.matrixU().col(dimI) *
                svd.matrixV().col(dimI).transpose();
            dsigma_div_dF.block(0, dimI, dim, 1) = dsigma_div_dF_mtr.row(0).transpose();
            dsigma_div_dF.block(dim, dimI, dim, 1) = dsigma_div_dF_mtr.row(1).transpose();
            if(dim == 3) {
                dsigma_div_dF.block(dim * 2, dimI, dim, 1) = dsigma_div_dF_mtr.row(2).transpose();
            }
        }
        
        Eigen::Matrix<double, dim * dim, dim * dim> d2E_div_dF2_left =
            d2E_div_dsigma2(0, 0) * dsigma_div_dF.col(0) * dsigma_div_dF.col(0).transpose() +
            d2E_div_dsigma2(1, 1) * dsigma_div_dF.col(1) * dsigma_div_dF.col(1).transpose();
        if(dim == 3) {
            d2E_div_dF2_left +=
                d2E_div_dsigma2(2, 2) * dsigma_div_dF.col(2) * dsigma_div_dF.col(2).transpose();
        }
        
        // cross sigma derivative
        if(crossSigmaDervative) {
            for(int sigmaI = 0; sigmaI < dim; sigmaI++) {
                for(int sigmaJ = sigmaI + 1; sigmaJ < dim; sigmaJ++) {
                    const Eigen::Matrix<double, dim * dim, dim * dim> m = dsigma_div_dF.col(sigmaJ) * dsigma_div_dF.col(sigmaI).transpose();
                    d2E_div_dF2_left += d2E_div_dsigma2(sigmaI, sigmaJ) * m +
                        d2E_div_dsigma2(sigmaJ, sigmaI) * m.transpose();
                }
            }
        }
        
        d2E_div_dF2_rest = d2E_div_dF2_left + d2E_div_dF2_right;
    }
    
    template<int dim>
    void Energy<dim>::initStepSize(const TriangleSoup<dim>& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        if(needElemInvSafeGuard) {
            initStepSize_preventElemInv(data, searchDir, stepSize);
        }
    }
    
    template<int dim>
    void Energy<dim>::initStepSize_preventElemInv(const TriangleSoup<dim>& data,
                                                  const Eigen::VectorXd& searchDir,
                                                  double& stepSize) const
    {
        // TODO: make sure explosion is because of scripts
//#if(DIM == 2)
//        assert(searchDir.size() == data.V.rows() * dim);
//        assert(stepSize > 0.0);
//        
//        for(int triI = 0; triI < data.F.rows(); triI++)
//        {
//            const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
//            
//            const Eigen::Matrix<double, dim, 1>& U1 = data.V.row(triVInd[0]);
//            const Eigen::Matrix<double, dim, 1>& U2 = data.V.row(triVInd[1]);
//            const Eigen::Matrix<double, dim, 1>& U3 = data.V.row(triVInd[2]);
//            //TODO: U4? V4, ..
//            
//            const Eigen::Matrix<double, dim, 1>& V1 = searchDir.segment<dim>(triVInd[0] * dim);
//            const Eigen::Matrix<double, dim, 1>& V2 = searchDir.segment<dim>(triVInd[1] * dim);
//            const Eigen::Matrix<double, dim, 1>& V3 = searchDir.segment<dim>(triVInd[2] * dim);
//            
//            const Eigen::Matrix<double, dim, 1> U2m1 = U2 - U1;
//            const Eigen::Matrix<double, dim, 1> U3m1 = U3 - U1;
//            const Eigen::Matrix<double, dim, 1> V2m1 = V2 - V1;
//            const Eigen::Matrix<double, dim, 1> V3m1 = V3 - V1;
//            
//            const double a = V2m1[0] * V3m1[1] - V2m1[1] * V3m1[0];
//            const double b = U2m1[0] * V3m1[1] - U2m1[1] * V3m1[0] + V2m1[0] * U3m1[1] - V2m1[1] * U3m1[0];
//            const double c = U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0];
//            assert(c > 0.0);
//            const double delta = b * b - 4.0 * a * c;
//            
//            double bound = stepSize;
//            if(a > 0.0) {
//                if((b < 0.0) && (delta >= 0.0)) {
//                    bound = 2.0 * c / (-b + sqrt(delta));
//                    // (same in math as (-b - sqrt(delta)) / 2.0 / a
//                    //  but smaller numerical error when b < 0.0)
//                    assert(bound > 0.0);
//                }
//            }
//            else if(a < 0.0) {
//                assert(delta > 0.0);
//                if(b < 0.0) {
//                    bound = 2.0 * c / (-b + sqrt(delta));
//                    // (same in math as (-b - sqrt(delta)) / 2.0 / a
//                    //  but smaller numerical error when b < 0.0)
//                }
//                else {
//                    bound = (-b - sqrt(delta)) / 2.0 / a;
//                }
//                assert(bound > 0.0);
//            }
//            else {
//                if(b < 0.0) {
//                    bound = -c / b;
//                    assert(bound > 0.0);
//                }
//            }
//            
//            if(bound < stepSize) {
//                stepSize = bound;
//            }
//        }
//#endif
        Eigen::VectorXd output(data.F.rows());
#if(DIM == 2)
        computeInjectiveStepSize_2d(data.F, data.V, searchDir, 1.0e-6, output.data());
#else
        computeInjectiveStepSize_3d(data.F, data.V, searchDir, 1.0e-6, output.data());
#endif
        stepSize = std::min(1.0, 0.95 * output.minCoeff());
        std::cout << "stepSize " << stepSize << std::endl;
        //TODO: single element below, 2D use mine or ?, 3D numerical error use 0.95? optimize code?
    }
    
    template<int dim>
    void Energy<dim>::initStepSize(const Eigen::VectorXd& V,
                              const Eigen::VectorXd& searchDir,
                              double& stepSize) const
    {
        if(needElemInvSafeGuard) {
            initStepSize_preventElemInv(V, searchDir, stepSize);
        }
    }
    
    template<int dim>
    void Energy<dim>::initStepSize_preventElemInv(const Eigen::VectorXd& V,
                                                  const Eigen::VectorXd& searchDir,
                                                  double& stepSize) const
    {
        assert(V.size() == searchDir.size());
        assert(stepSize > 0.0);
        
        const Eigen::Vector2d& U1 = V.segment(0, 2);
        const Eigen::Vector2d& U2 = V.segment(2, 2);
        const Eigen::Vector2d& U3 = V.segment(4, 2);
        
        const Eigen::Vector2d V1(searchDir.segment(0, 2));
        const Eigen::Vector2d V2(searchDir.segment(2, 2));
        const Eigen::Vector2d V3(searchDir.segment(4, 2));
        
        const Eigen::Vector2d U2m1 = U2 - U1;
        const Eigen::Vector2d U3m1 = U3 - U1;
        const Eigen::Vector2d V2m1 = V2 - V1;
        const Eigen::Vector2d V3m1 = V3 - V1;
        
        const double a = V2m1[0] * V3m1[1] - V2m1[1] * V3m1[0];
        const double b = U2m1[0] * V3m1[1] - U2m1[1] * V3m1[0] + V2m1[0] * U3m1[1] - V2m1[1] * U3m1[0];
        const double c = U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0];
        assert(c > 0.0);
        const double delta = b * b - 4.0 * a * c;
        
        double bound = stepSize;
        if(a > 0.0) {
            if((b < 0.0) && (delta >= 0.0)) {
                bound = 2.0 * c / (-b + sqrt(delta));
                // (same in math as (-b - sqrt(delta)) / 2.0 / a
                //  but smaller numerical error when b < 0.0)
                assert(bound > 0.0);
            }
        }
        else if(a < 0.0) {
            assert(delta > 0.0);
            if(b < 0.0) {
                bound = 2.0 * c / (-b + sqrt(delta));
                // (same in math as (-b - sqrt(delta)) / 2.0 / a
                //  but smaller numerical error when b < 0.0)
            }
            else {
                bound = (-b - sqrt(delta)) / 2.0 / a;
            }
            assert(bound > 0.0);
        }
        else {
            if(b < 0.0) {
                bound = -c / b;
                assert(bound > 0.0);
            }
        }
        
        if(bound < stepSize) {
            stepSize = bound;
        }
    }
    
    template<int dim>
    void Energy<dim>::getBulkModulus(double& bulkModulus)
    {
        assert(0 && "please implement this method in the subclass!");
    }

    template<int dim>
    void Energy<dim>::unitTest_dE_div_dsigma(std::ostream& os) const
    {
        std::vector<Eigen::Matrix<double, dim, 1>> testSigma;

        testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Ones());
        if(needElemInvSafeGuard) {
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
        }
        else {
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Zero());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
        }

        const double h = 1.0e-6;
        int testI = 0;
        for(const auto& testSigmaI : testSigma) {
            os << "--- unitTest_dE_div_dsigma " << testI << " ---\n" << "sigma =\n" << testSigmaI << std::endl;

            double E0;
            compute_E(testSigmaI, E0);

            Eigen::Matrix<double, dim, 1> dE_div_dsigma_FD;
            for(int dimI = 0; dimI < dim; dimI++) {
                Eigen::Matrix<double, dim, 1> sigma_perterb = testSigmaI;
                sigma_perterb[dimI] += h;
                double E;
                compute_E(sigma_perterb, E);
                dE_div_dsigma_FD[dimI] = (E - E0) / h;
            }
            os << "dE_div_dsigma_FD =\n" << dE_div_dsigma_FD << std::endl;

            Eigen::Matrix<double, dim, 1> dE_div_dsigma_S;
            compute_dE_div_dsigma(testSigmaI, dE_div_dsigma_S);
            os << "dE_div_dsigma_S =\n" << dE_div_dsigma_S << std::endl;

            double err = (dE_div_dsigma_FD - dE_div_dsigma_S).norm();
            os << "err = " << err << " (" << err / dE_div_dsigma_FD.norm() * 100 << "%)" << std::endl;

            ++testI;
        }
    }

    template<int dim>
    void Energy<dim>::unitTest_d2E_div_dsigma2(std::ostream& os) const
    {
        std::vector<Eigen::Matrix<double, dim, 1>> testSigma;

        testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Ones());
        if(needElemInvSafeGuard) {
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
        }
        else {
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Zero());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
        }

        const double h = 1.0e-6;
        int testI = 0;
        for(const auto& testSigmaI : testSigma) {
            os << "--- unitTest_d2E_div_dsigma2 " << testI << " ---\n" << "sigma =\n" << testSigmaI << std::endl;

            Eigen::Matrix<double, dim, 1> dE_div_dsigma0;
            compute_dE_div_dsigma(testSigmaI, dE_div_dsigma0);

            Eigen::Matrix<double, dim, dim> d2E_div_dsigma2_FD;
            for(int dimI = 0; dimI < dim; dimI++) {
                Eigen::Matrix<double, dim, 1> sigma_perterb = testSigmaI;
                sigma_perterb[dimI] += h;
                Eigen::Matrix<double, dim, 1> dE_div_dsigma;
                compute_dE_div_dsigma(sigma_perterb, dE_div_dsigma);
                d2E_div_dsigma2_FD.row(dimI) = ((dE_div_dsigma - dE_div_dsigma0) / h).transpose();
            }
            os << "d2E_div_dsigma2_FD =\n" << d2E_div_dsigma2_FD << std::endl;

            Eigen::Matrix<double, dim, dim> d2E_div_dsigma2_S;
            compute_d2E_div_dsigma2(testSigmaI, d2E_div_dsigma2_S);
            os << "d2E_div_dsigma2_S =\n" << d2E_div_dsigma2_S << std::endl;

            double err = (d2E_div_dsigma2_FD - d2E_div_dsigma2_S).norm();
            os << "err = " << err << " (" << err / d2E_div_dsigma2_FD.norm() * 100 << "%)" << std::endl;

            ++testI;
        }
    }

    template<int dim>
    void Energy<dim>::unitTest_BLeftCoef(std::ostream& os) const
    {
        std::vector<Eigen::Matrix<double, dim, 1>> testSigma;

        if(needElemInvSafeGuard) {
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() + Eigen::Matrix<double, dim, 1>::Ones());
        }
        else {
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Zero());
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
            testSigma.emplace_back(Eigen::Matrix<double, dim, 1>::Random() * 2);
        }

        int testI = 0;
        for(const auto& testSigmaI : testSigma) {
            os << "--- unitTest_BLeftCoef " << testI << " ---\n" << "sigma =\n" << testSigmaI << std::endl;

            Eigen::Matrix<double, dim * (dim - 1) / 2, 1> BLeftCoef_div;
            Eigen::Matrix<double, dim, 1> dE_div_dsigma;
            compute_dE_div_dsigma(testSigmaI, dE_div_dsigma);
            if(dim == 2) {
                BLeftCoef_div[0] = (dE_div_dsigma[0] - dE_div_dsigma[1]) / (testSigmaI[0] - testSigmaI[1]) / 2.0;
            }
            else {
                BLeftCoef_div[0] = (dE_div_dsigma[0] - dE_div_dsigma[1]) / (testSigmaI[0] - testSigmaI[1]) / 2.0;
                BLeftCoef_div[1] = (dE_div_dsigma[1] - dE_div_dsigma[2]) / (testSigmaI[1] - testSigmaI[2]) / 2.0;
                BLeftCoef_div[2] = (dE_div_dsigma[2] - dE_div_dsigma[0]) / (testSigmaI[2] - testSigmaI[0]) / 2.0;
            }
            os << "BLeftCoef_div =\n" << BLeftCoef_div << std::endl;

            Eigen::Matrix<double, dim * (dim - 1) / 2, 1> BLeftCoef_S;
            compute_BLeftCoef(testSigmaI, BLeftCoef_S);
            os << "BLeftCoef_S =\n" << BLeftCoef_S << std::endl;

            double err = (BLeftCoef_div - BLeftCoef_S).norm();
            os << "err = " << err << " (" << err / BLeftCoef_div.norm() * 100 << "%)" << std::endl;

            ++testI;
        }
    }

    template<int dim>
    void Energy<dim>::unitTest_dE_div_dF(std::ostream& os) const
    {
        const double h = 1.0e-6;
        std::vector<Eigen::Matrix<double, dim, dim>> testF;

        testF.emplace_back(Eigen::Matrix<double, dim, dim>::Identity());
        if(needElemInvSafeGuard) {
            for(int testI = 0; testI < 6; testI++) {
                testF.emplace_back(Eigen::Matrix<double, dim, dim>::Random());
                IglUtils::flipDet_SVD(testF.back());
                testF.back() += 1.0e-2 * Eigen::Matrix<double, dim, dim>::Identity();
            }
        }
        else {
            testF.emplace_back(Eigen::Matrix<double, dim, dim>::Zero());
            for(int testI = 0; testI < 6; testI++) {
                testF.emplace_back(Eigen::Matrix<double, dim, dim>::Random());
            }
        }

        int testI = 0;
        for(const auto& testFI : testF) {
            os << "--- unitTest_dE_div_dF " << testI << " ---\n" << "F =\n" << testFI << std::endl;

            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd0(testFI, Eigen::ComputeFullU | Eigen::ComputeFullV);
            double E0;
            compute_E(svd0.singularValues(), E0);

            Eigen::Matrix<double, dim, dim> P_FD;
            for(int dimI = 0; dimI < dim; dimI++) {
                for(int dimJ = 0; dimJ < dim; dimJ++) {
                    Eigen::Matrix<double, dim, dim> F_perterb = testFI;
                    F_perterb(dimI, dimJ) += h;

                    AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F_perterb, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    double E;
                    compute_E(svd.singularValues(), E);

                    P_FD(dimI, dimJ) = (E - E0) / h;
                }
            }
            os << "P_FD =\n" << P_FD << std::endl;

            Eigen::Matrix<double, dim, dim> P_S;
            compute_dE_div_dF(testFI, svd0, P_S);
            os << "P_S =\n" << P_S << std::endl;

            double err = (P_FD - P_S).norm();
            os << "err = " << err << " (" << err / P_FD.norm() * 100 << "%)" << std::endl;

            ++testI;
        }
    }

    template<int dim>
    void Energy<dim>::unitTest_dP_div_dF(std::ostream& os) const
    {
        const double h = 1.0e-6;
        std::vector<Eigen::Matrix<double, dim, dim>> testF;

        testF.emplace_back(Eigen::Matrix<double, dim, dim>::Identity());
        if(needElemInvSafeGuard) {
            for(int testI = 0; testI < 6; testI++) {
                testF.emplace_back(Eigen::Matrix<double, dim, dim>::Random());
                IglUtils::flipDet_SVD(testF.back());
                testF.back() += 1.0e-2 * Eigen::Matrix<double, dim, dim>::Identity();
            }
        }
        else {
            testF.emplace_back(Eigen::Matrix<double, dim, dim>::Zero());
            for(int testI = 0; testI < 6; testI++) {
                testF.emplace_back(Eigen::Matrix<double, dim, dim>::Random());
            }
        }

        int testI = 0;
        for(const auto& testFI : testF) {
            os << "--- unitTest_dP_div_dF " << testI << " ---\n" << "F =\n" << testFI << std::endl;

            AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(testFI, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix<double, dim, dim> P0;
            compute_dE_div_dF(testFI, svd, P0);

            Eigen::Matrix<double, dim * dim, dim * dim> dP_div_dF_FD;
            for(int dimI = 0; dimI < dim; dimI++) {
                for(int dimJ = 0; dimJ < dim; dimJ++) {
                    Eigen::Matrix<double, dim, dim> F_perterb = testFI;
                    F_perterb(dimI, dimJ) += h;
                    Eigen::Matrix<double, dim, dim> P;
                    AutoFlipSVD<Eigen::Matrix<double, dim, dim>> svd(F_perterb, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    compute_dE_div_dF(F_perterb, svd, P);

                    Eigen::Matrix<double, dim, dim> FD = (P - P0) / h;
                    dP_div_dF_FD.block(0, dimI * dim + dimJ, dim, 1) = FD.row(0).transpose();
                    dP_div_dF_FD.block(dim, dimI * dim + dimJ, dim, 1) = FD.row(1).transpose();
                    if(dim == 3) {
                        dP_div_dF_FD.block(dim * 2, dimI * dim + dimJ, dim, 1) = FD.row(2).transpose();
                    }
                }
            }
            os << "dP_div_dF_FD =\n" << dP_div_dF_FD << std::endl;

            Eigen::Matrix<double, dim * dim, dim * dim> dP_div_dF_S;
            svd.compute(testFI, Eigen::ComputeFullU | Eigen::ComputeFullV);
            compute_dP_div_dF(svd, dP_div_dF_S, 1.0, false);
            os << "dP_div_dF_S =\n" << dP_div_dF_S << std::endl;

            double err = (dP_div_dF_FD - dP_div_dF_S).norm();
            os << "err = " << err << " (" << err / dP_div_dF_FD.norm() * 100 << "%)" << std::endl;

            ++testI;
        }

    }
    
    template class Energy<DIM>;
    
}
