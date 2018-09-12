//
//  Energy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/4/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Energy.hpp"
#include "IglUtils.hpp"
#include "Timer.hpp"

#include <igl/avg_edge_length.h>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <fstream>
#include <iostream>

extern std::ofstream logFile;
extern Timer timer_temp, timer_temp2;

namespace FracCuts {
    
    Energy::Energy(bool p_needRefactorize,
                   bool p_needElemInvSafeGuard,
                   bool p_crossSigmaDervative,
                   double visRange_min,
                   double visRange_max) :
        needRefactorize(p_needRefactorize),
        needElemInvSafeGuard(p_needElemInvSafeGuard),
        crossSigmaDervative(p_crossSigmaDervative),
        visRange_energyVal(visRange_min, visRange_max)
    {}
    
    Energy::~Energy(void)
    {}
    
    bool Energy::getNeedRefactorize(void) const
    {
        return needRefactorize;
    }
    
    const Eigen::Vector2d& Energy::getVisRange_energyVal(void) const
    {
        return visRange_energyVal;
    }
    void Energy::setVisRange_energyVal(double visRange_min, double visRange_max)
    {
        visRange_energyVal << visRange_min, visRange_max;
    }
    
    void Energy::computeEnergyVal(const TriangleSoup& data, double& energyVal, bool uniformWeight) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElem(data, energyValPerElem, uniformWeight);
        energyVal = energyValPerElem.sum();
    }
    
    void Energy::getEnergyValByElemID(const TriangleSoup& data, int elemI, double& energyVal, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                   Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::checkGradient(const TriangleSoup& data) const
    {
        std::cout << "checking energy gradient computation..." << std::endl;
        
        double energyVal0;
        computeEnergyVal(data, energyVal0);
        const double h = 1.0e-8 * igl::avg_edge_length(data.V, data.F);
        TriangleSoup perturbed = data;
        Eigen::VectorXd gradient_finiteDiff;
        gradient_finiteDiff.resize(data.V.rows() * 2);
        for(int vI = 0; vI < data.V.rows(); vI++)
        {
            for(int dimI = 0; dimI < 2; dimI++) {
                perturbed.V = data.V;
                perturbed.V(vI, dimI) += h;
                double energyVal_perturbed;
                computeEnergyVal(perturbed, energyVal_perturbed);
                gradient_finiteDiff[vI * 2 + dimI] = (energyVal_perturbed - energyVal0) / h;
            }
            
            if(((vI + 1) % 100) == 0) {
                std::cout << vI + 1 << "/" << data.V.rows() << " vertices computed" << std::endl;
            }
        }
        for(const auto fixedVI : data.fixedVert) {
            gradient_finiteDiff[2 * fixedVI] = 0.0;
            gradient_finiteDiff[2 * fixedVI + 1] = 0.0;
        }
        
        Eigen::VectorXd gradient_symbolic;
//        computeGradient(data, gradient_symbolic);
        computeGradientBySVD(data, gradient_symbolic);
        
        Eigen::VectorXd difVec = gradient_symbolic - gradient_finiteDiff;
        const double dif_L2 = difVec.norm();
        const double relErr = dif_L2 / gradient_finiteDiff.norm();
        
        std::cout << "L2 dist = " << dif_L2 << ", relErr = " << relErr << std::endl;
        
        logFile << "check gradient:" << std::endl;
        logFile << "g_symbolic =\n" << gradient_symbolic << std::endl;
        logFile << "g_finiteDiff = \n" << gradient_finiteDiff << std::endl;
    }
    
    void Energy::checkHessian(const TriangleSoup& data, bool triplet) const
    {
        std::cout << "checking energy hessian computation..." << std::endl;
        
        Eigen::VectorXd gradient0;
//        computeGradient(data, gradient0);
        computeGradientBySVD(data, gradient0);
        const double h = 1.0e-8 * igl::avg_edge_length(data.V, data.F);
        TriangleSoup perturbed = data;
        Eigen::SparseMatrix<double> hessian_finiteDiff;
        hessian_finiteDiff.resize(data.V.rows() * 2, data.V.rows() * 2);
        for(int vI = 0; vI < data.V.rows(); vI++)
        {
            if(data.fixedVert.find(vI) != data.fixedVert.end()) {
                hessian_finiteDiff.insert(vI * 2, vI * 2) = 1.0;
                hessian_finiteDiff.insert(vI * 2 + 1, vI * 2 + 1) = 1.0;
                continue;
            }
            
            for(int dimI = 0; dimI < 2; dimI++) {
                perturbed.V = data.V;
                perturbed.V(vI, dimI) += h;
                Eigen::VectorXd gradient_perturbed;
//                computeGradient(perturbed, gradient_perturbed);
                computeGradientBySVD(perturbed, gradient_perturbed);
                Eigen::VectorXd hessian_colI = (gradient_perturbed - gradient0) / h;
                int colI = vI * 2 + dimI;
                for(int rowI = 0; rowI < data.V.rows() * 2; rowI++) {
                    if(data.fixedVert.find(rowI / 2) != data.fixedVert.end()) {
                        continue;
                    }
                    
                    hessian_finiteDiff.insert(rowI, colI) = hessian_colI[rowI];
                }
            }
            
            if(((vI + 1) % 100) == 0) {
                std::cout << vI + 1 << "/" << data.V.rows() << " vertices computed" << std::endl;
            }
        }
        
        Eigen::SparseMatrix<double> hessian_symbolic;
        if(triplet) {
            Eigen::VectorXi I, J;
            Eigen::VectorXd V;
//            computePrecondMtr(data, &V, &I, &J); //TODO: change name to Hessian! don't project!
            computeHessianBySVD(data, &V, &I, &J, false);
            std::vector<Eigen::Triplet<double>> triplet(V.size());
            for(int entryI = 0; entryI < V.size(); entryI++) {
                triplet[entryI] = Eigen::Triplet<double>(I[entryI], J[entryI], V[entryI]);
            }
            hessian_symbolic.resize(data.V.rows() * 2, data.V.rows() * 2);
            hessian_symbolic.setFromTriplets(triplet.begin(), triplet.end());
        }
        else {
            computeHessian(data, hessian_symbolic); //TODO: don't project
        }
        
        Eigen::SparseMatrix<double> difMtr = hessian_symbolic - hessian_finiteDiff;
        const double dif_L2 = difMtr.norm();
        const double relErr = dif_L2 / hessian_finiteDiff.norm();
        
        std::cout << "L2 dist = " << dif_L2 << ", relErr = " << relErr << std::endl;
        
        logFile << "check hessian:" << std::endl;
        logFile << "h_symbolic =\n" << hessian_symbolic << std::endl;
        logFile << "h_finiteDiff = \n" << hessian_finiteDiff << std::endl;
    }
    
    void Energy::getEnergyValPerElemBySVD(const TriangleSoup& data, bool redoSVD,
                                          std::vector<AutoFlipSVD<Eigen::MatrixXd>>& svd,
                                          Eigen::VectorXd& energyValPerElem,
                                          bool uniformWeight) const
    {
        energyValPerElem.resize(data.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            timer_temp2.start(4);
            timer_temp.start(1);
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Matrix2d Xt, A;
            Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
            Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = data.restTriInv[triI]; //TODO: this only need to be computed once
            
            if(redoSVD) {
                timer_temp2.stop();
                timer_temp.start(0);
                svd[triI].compute(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV);
                timer_temp.start(1);
                timer_temp2.start(4);
            }
            
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
    }
    
    void Energy::computeEnergyValBySVD(const TriangleSoup& data, bool redoSVD,
                                       std::vector<AutoFlipSVD<Eigen::MatrixXd>>& svd,
                                       double& energyVal) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElemBySVD(data, redoSVD, svd, energyValPerElem);
        energyVal = energyValPerElem.sum();
    }
    
    void Energy::computeGradientBySVD(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Matrix2d Xt, A;
            Xt << (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose(),
                (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = data.restTriInv[triI]; //TODO: this only need to be computed once
            
            timer_temp.start(0);
            AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            timer_temp.stop();
            
            timer_temp.start(1);
            Eigen::MatrixXd dsigma_div_dx;
            IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
            
            Eigen::VectorXd dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            for(int triVI = 0; triVI < 3; triVI++) {
                gradient.segment(triVInd[triVI] * 2, 2) += w * dsigma_div_dx.block(triVI * 2, 0, 2, 2) * dE_div_dsigma;
            }
            timer_temp.stop();
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    
    void Energy::computeHessianBySVD(const TriangleSoup& data, Eigen::VectorXd* V,
                                     Eigen::VectorXi* I, Eigen::VectorXi* J,
                                     bool projectSPD) const
    {
        std::vector<Eigen::Matrix<double, 6, 6>> triHessians(data.F.rows());
        std::vector<Eigen::VectorXi> vInds(data.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Matrix2d Xt, A;
            Xt << (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose(),
                (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = data.restTriInv[triI]; //TODO: this only need to be computed once
            
            timer_temp.start(0);
            AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            timer_temp.stop();
            
            timer_temp.start(1);
            // right term:
            Eigen::VectorXd dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            
            Eigen::MatrixXd d2sigma_div_dx2;
            IglUtils::compute_d2sigma_div_dx2(svd, A, d2sigma_div_dx2);
            
            Eigen::MatrixXd d2E_div_dx2_right = d2sigma_div_dx2.block(0, 0, 6, 6) * dE_div_dsigma[0] +
            d2sigma_div_dx2.block(0, 6, 6, 6) * dE_div_dsigma[1];
            
            // left term:
            Eigen::MatrixXd d2E_div_dsigma2;
            compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
            
            Eigen::MatrixXd dsigma_div_dx;
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
            triHessians[triI] = w * (d2E_div_dx2_left + d2E_div_dx2_right);
            timer_temp.stop();
            
            if(projectSPD) {
                timer_temp.start(2);
                IglUtils::makePD(triHessians[triI]);
                timer_temp.stop();
            }
            
            Eigen::VectorXi& vInd = vInds[triI];
            vInd = triVInd;
            for(int vI = 0; vI < 3; vI++) {
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
            IglUtils::addBlockToMatrix(triHessians[triI], vInds[triI], 2, V, I, J);
        }
        
        Eigen::VectorXi fixedVertInd;
        fixedVertInd.resize(data.fixedVert.size());
        int fVI = 0;
        for(const auto fixedVI : data.fixedVert) {
            fixedVertInd[fVI++] = fixedVI;
        }
        IglUtils::addDiagonalToMatrix(Eigen::VectorXd::Ones(data.fixedVert.size() * 2),
                                      fixedVertInd, 2, V, I, J);
        timer_temp.stop();
    }
    
    void Energy::computeGradientByPK(const TriangleSoup& data, bool redoSVD,
                                     std::vector<AutoFlipSVD<Eigen::MatrixXd>>& svd,
                                     Eigen::VectorXd& gradient) const
    {
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            timer_temp2.start(5);
            timer_temp.start(1);
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Matrix2d Xt, A, F;
            Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
            Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = data.restTriInv[triI];
            F = Xt * A;
            timer_temp2.stop();
            
            if(redoSVD) {
                timer_temp.start(0);
                svd[triI].compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
//                svd[triI] = AutoFlipSVD<Eigen::MatrixXd>(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
                timer_temp.start(1);
            }
            
            timer_temp2.start(6);
            Eigen::MatrixXd P;
            compute_dE_div_dF(F, svd[triI], P);
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            P *= w;
            
            timer_temp2.start(7);
            Eigen::VectorXd gradient_cont;
            IglUtils::dF_div_dx_mult(P, A, gradient_cont);
            
            timer_temp2.start(8);
            gradient.segment(triVInd[0] * 2, 2) += gradient_cont.segment(0, 2);
            gradient.segment(triVInd[1] * 2, 2) += gradient_cont.segment(2, 2);
            gradient.segment(triVInd[2] * 2, 2) += gradient_cont.segment(4, 2);
            //TODO: funtionalize it
            timer_temp.stop();
            timer_temp2.stop();
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient.segment(2 * fixedVI, 2).setZero();
        }
    }
    void Energy::computeHessianByPK(const TriangleSoup& data, bool redoSVD,
                                    std::vector<AutoFlipSVD<Eigen::MatrixXd>>& svd,
                                    double coef, Eigen::VectorXd* V,
                                    Eigen::VectorXi* I, Eigen::VectorXi* J,
                                    bool projectSPD) const
    {
        std::vector<Eigen::MatrixXd> triHessians(data.F.rows());
        std::vector<Eigen::VectorXi> vInds(data.F.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < data.F.rows(); triI++)
#endif
        {
            timer_temp.start(1);
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Matrix2d Xt, A;
            Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
            Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = data.restTriInv[triI];
            
            if(redoSVD) {
                timer_temp.start(0);
                svd[triI].compute(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV);
//                svd[triI] = AutoFlipSVD<Eigen::MatrixXd>(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV);
                timer_temp.start(1);
            }
            
            // compute A
            timer_temp2.start(0);
            Eigen::VectorXd dE_div_dsigma;
            compute_dE_div_dsigma(svd[triI].singularValues(), dE_div_dsigma);
            Eigen::MatrixXd d2E_div_dsigma2;
            compute_d2E_div_dsigma2(svd[triI].singularValues(), d2E_div_dsigma2);
            timer_temp2.stop();
            if(projectSPD) {
                timer_temp.start(2);
                IglUtils::makePD2d(d2E_div_dsigma2);
                timer_temp.start(1);
            }
            
            // compute B
            timer_temp2.start(1);
            Eigen::Matrix2d B01;
            double leftCoef = d2E_div_dsigma2(0, 0);
            const double dif_sigma = svd[triI].singularValues()[0] - svd[triI].singularValues()[1];
            if(std::abs(dif_sigma) > 1.0e-6) {
                leftCoef = (dE_div_dsigma[0] - dE_div_dsigma[1]) / dif_sigma;
            }
            leftCoef /= 2.0;
            double rightCoef = dE_div_dsigma[0] + dE_div_dsigma[1];
            double sum_sigma = svd[triI].singularValues()[0] + svd[triI].singularValues()[1];
            if(sum_sigma < 1.0e-6) {
                rightCoef /= 2.0e-6;
            }
            else {
                rightCoef /= 2.0 * sum_sigma;
            }
            B01(0, 0) = B01(1, 1) = leftCoef + rightCoef;
            B01(0, 1) = B01(1, 0) = leftCoef - rightCoef;
            timer_temp2.stop();
            if(projectSPD) {
                timer_temp.start(2);
                IglUtils::makePD2d(B01);
                timer_temp.start(1);
            }
              
            // compute M using A(d2E_div_dsigma2) and B
            const double w = coef * data.triWeight[triI] * data.triArea[triI];
            timer_temp2.start(2);
            Eigen::MatrixXd M(4, 4);
            M.setZero();
            M(0, 0) = w * d2E_div_dsigma2(0, 0);
            M(0, 3) = w * d2E_div_dsigma2(0, 1);
            M.block(1, 1, 2, 2) = w * B01;
            M(3, 0) = w * d2E_div_dsigma2(1, 0);
            M(3, 3) = w * d2E_div_dsigma2(1, 1);
            
            // compute dP_div_dF
            Eigen::MatrixXd wdP_div_dF(4, 4);
            const Eigen::MatrixXd& U = svd[triI].matrixU();
            const Eigen::MatrixXd& V = svd[triI].matrixV();
            for(int i = 0; i < 2; i++) {
                int _2i = i * 2;
                for(int j = 0; j < 2; j++) {
                    int ij = _2i + j;
                    for(int r = 0; r < 2; r++) {
                        int _2r = r * 2;
                        for(int s = 0; s < 2; s++) {
                            int rs = _2r + s;
                            if(ij < rs) {
                                // upper right
                                wdP_div_dF(ij, rs) = wdP_div_dF(rs, ij) =
                                (M(0, 0) * U(i, 0) * U(r, 0) * V(s, 0) * V(j, 0) +
                                M(0, 3) * U(i, 0) * U(r, 1) * V(s, 1) * V(j, 0) +
                                M(1, 1) * U(i, 0) * U(r, 0) * V(s, 1) * V(j, 1) +
                                M(1, 2) * U(i, 0) * U(r, 1) * V(s, 0) * V(j, 1) +
                                M(2, 1) * U(i, 1) * U(r, 0) * V(s, 1) * V(j, 0) +
                                M(2, 2) * U(i, 1) * U(r, 1) * V(s, 0) * V(j, 0) +
                                M(3, 0) * U(i, 1) * U(r, 0) * V(s, 0) * V(j, 1) +
                                M(3, 3) * U(i, 1) * U(r, 1) * V(s, 1) * V(j, 1));
                            }
                            else if(ij == rs) {
                                // diagonal
                                wdP_div_dF(ij, rs) =
                                (M(0, 0) * U(i, 0) * U(r, 0) * V(s, 0) * V(j, 0) +
                                M(0, 3) * U(i, 0) * U(r, 1) * V(s, 1) * V(j, 0) +
                                M(1, 1) * U(i, 0) * U(r, 0) * V(s, 1) * V(j, 1) +
                                M(1, 2) * U(i, 0) * U(r, 1) * V(s, 0) * V(j, 1) +
                                M(2, 1) * U(i, 1) * U(r, 0) * V(s, 1) * V(j, 0) +
                                M(2, 2) * U(i, 1) * U(r, 1) * V(s, 0) * V(j, 0) +
                                M(3, 0) * U(i, 1) * U(r, 0) * V(s, 0) * V(j, 1) +
                                M(3, 3) * U(i, 1) * U(r, 1) * V(s, 1) * V(j, 1));
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
            
            Eigen::MatrixXd wdP_div_dx;
            timer_temp2.start(3);
            IglUtils::dF_div_dx_mult(wdP_div_dF.transpose(), A, wdP_div_dx, false);
            IglUtils::dF_div_dx_mult(wdP_div_dx.transpose(), A, triHessians[triI], true);
            timer_temp2.stop();
            
            Eigen::VectorXi& vInd = vInds[triI];
            vInd = triVInd;
            for(int vI = 0; vI < 3; vI++) {
                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
                    vInd[vI] = -1;
                }
            }
            timer_temp.stop();
        }
#ifdef USE_TBB
        );
#endif
        timer_temp.start(3);
        for(int triI = 0; triI < data.F.rows(); triI++) {
            IglUtils::addBlockToMatrix(triHessians[triI], vInds[triI], 2, V, I, J);
            //!!! the index can be cached if matrix structure doesn't change
        }
        
        Eigen::VectorXi fixedVertInd;
        fixedVertInd.resize(data.fixedVert.size());
        int fVI = 0;
        for(const auto fixedVI : data.fixedVert) {
            fixedVertInd[fVI++] = fixedVI;
        }
        IglUtils::addDiagonalToMatrix(Eigen::VectorXd::Ones(data.fixedVert.size() * 2),
                                      fixedVertInd, 2, V, I, J);
        //!!! the index can be cached if matrix structure doesn't change
        timer_temp.stop();
    }
    
    void Energy::computeEnergyValBySVD(const TriangleSoup& data, int triI,
                                       const Eigen::VectorXd& x,
                                       double& energyVal,
                                       bool uniformWeight) const
    {
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
        
        AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
        
        compute_E(svd.singularValues(), energyVal);
        if(!uniformWeight) {
            energyVal *= data.triWeight[triI] * data.triArea[triI];
        }
    }
    void Energy::computeGradientBySVD(const TriangleSoup& data, int triI,
                                      const Eigen::VectorXd& x,
                                      Eigen::VectorXd& gradient) const
    {
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
        
        AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        Eigen::MatrixXd dsigma_div_dx;
        IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
        
        Eigen::VectorXd dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        gradient.resize(6);
        const double w = data.triWeight[triI] * data.triArea[triI];
        for(int triVI = 0; triVI < 3; triVI++) {
            gradient.segment(triVI * 2, 2) = w * dsigma_div_dx.block(triVI * 2, 0, 2, 2) * dE_div_dsigma;
        }
    }
    void Energy::computeHessianBySVD(const TriangleSoup& data, int triI,
                                     const Eigen::VectorXd& x,
                                     Eigen::MatrixXd& hessian,
                                     bool projectSPD) const
    {
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
        
        AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        // right term:
        Eigen::VectorXd dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        Eigen::MatrixXd d2sigma_div_dx2;
        IglUtils::compute_d2sigma_div_dx2(svd, A, d2sigma_div_dx2);
        
        Eigen::MatrixXd d2E_div_dx2_right = d2sigma_div_dx2.block(0, 0, 6, 6) * dE_div_dsigma[0] +
        d2sigma_div_dx2.block(0, 6, 6, 6) * dE_div_dsigma[1];
        
        // left term:
        Eigen::MatrixXd d2E_div_dsigma2;
        compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
        
        Eigen::MatrixXd dsigma_div_dx;
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
    }
    
    void Energy::computeEnergyValBySVD_F(const TriangleSoup& data, int triI,
                                         const Eigen::RowVectorXd& F,
                                         double& energyVal,
                                         bool uniformWeight) const
    {
        Eigen::Matrix2d F_mtr;
        F_mtr << F.segment(0, 2), F.segment(2, 2);
        AutoFlipSVD<Eigen::MatrixXd> svd(F_mtr); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
        
        compute_E(svd.singularValues(), energyVal);
        if(!uniformWeight) {
            energyVal *= data.triWeight[triI] * data.triArea[triI];
        }
    }
    void Energy::computeGradientBySVD_F(const TriangleSoup& data, int triI,
                                        const Eigen::RowVectorXd& F,
                                        Eigen::VectorXd& gradient) const
    {
        Eigen::Matrix2d F_mtr;
        F_mtr << F.segment(0, 2), F.segment(2, 2);
        AutoFlipSVD<Eigen::MatrixXd> svd(F_mtr, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        Eigen::VectorXd dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        gradient = Eigen::VectorXd::Zero(4);
        for(int dimI = 0; dimI < 2; dimI++) {
            Eigen::Matrix2d dsigma_div_dF = svd.matrixU().col(dimI) * svd.matrixV().col(dimI).transpose();
            Eigen::RowVectorXd dsigma_div_dF_vec;
            dsigma_div_dF_vec.resize(4);
            dsigma_div_dF_vec << dsigma_div_dF.row(0), dsigma_div_dF.row(1);
            gradient += dsigma_div_dF_vec.transpose() * dE_div_dsigma[dimI];
        }
        
        const double w = data.triWeight[triI] * data.triArea[triI];
        gradient *= w;
    }
    void Energy::computeHessianBySVD_F(const TriangleSoup& data, int triI,
                                       const Eigen::RowVectorXd& F,
                                       Eigen::MatrixXd& hessian,
                                       bool projectSPD) const
    {
        Eigen::Matrix2d F_mtr;
        F_mtr << F.segment(0, 2), F.segment(2, 2);
        AutoFlipSVD<Eigen::MatrixXd> svd(F_mtr, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        // right term:
        Eigen::VectorXd dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        Eigen::MatrixXd d2sigma_div_dF2;
        IglUtils::compute_d2sigma_div_dF2(svd, d2sigma_div_dF2);
        
        Eigen::MatrixXd d2E_div_dF2_right = d2sigma_div_dF2.block(0, 0, 4, 4) * dE_div_dsigma[0] +
            d2sigma_div_dF2.block(0, 4, 4, 4) * dE_div_dsigma[1];
        
        // left term:
        Eigen::MatrixXd d2E_div_dsigma2;
        compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
        
        Eigen::MatrixXd dsigma_div_dF;
        dsigma_div_dF.resize(4, 2);
        for(int dimI = 0; dimI < 2; dimI++) {
            Eigen::Matrix2d dsigmai_div_dF = svd.matrixU().col(dimI) * svd.matrixV().col(dimI).transpose();
            dsigma_div_dF.col(dimI) << dsigmai_div_dF.row(0).transpose(), dsigmai_div_dF.row(1).transpose();
        }
        
        Eigen::MatrixXd d2E_div_dF2_left = d2E_div_dsigma2(0, 0) * dsigma_div_dF.col(0) * dsigma_div_dF.col(0).transpose() +
        d2E_div_dsigma2(1, 1) * dsigma_div_dF.col(1) * dsigma_div_dF.col(1).transpose();
        
        // cross sigma derivative
        if(crossSigmaDervative) {
            for(int sigmaI = 0; sigmaI < svd.singularValues().size(); sigmaI++) {
                for(int sigmaJ = sigmaI + 1; sigmaJ < svd.singularValues().size(); sigmaJ++) {
                    const Eigen::MatrixXd m = dsigma_div_dF.col(sigmaJ) * dsigma_div_dF.col(sigmaI).transpose();
                    d2E_div_dF2_left += d2E_div_dsigma2(sigmaI, sigmaJ) * m +
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
    
    void Energy::compute_E(const Eigen::VectorXd& singularValues,
                           double& E) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    void Energy::compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                       Eigen::VectorXd& dE_div_dsigma) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    void Energy::compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                         Eigen::MatrixXd& d2E_div_dsigma2) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    void Energy::compute_dE_div_dF(const Eigen::MatrixXd& F,
                                   const AutoFlipSVD<Eigen::MatrixXd>& svd,
                                   Eigen::MatrixXd& dE_div_dF) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::compute_d2E_div_dF2_rest(Eigen::MatrixXd& d2E_div_dF2_rest) const
    {
        Eigen::Matrix2d A = Eigen::Matrix2d::Identity();
        
        AutoFlipSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
        
        // right term:
        Eigen::VectorXd dE_div_dsigma;
        compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
        
        Eigen::MatrixXd d2sigma_div_dF2;
        IglUtils::compute_d2sigma_div_dF2(svd, d2sigma_div_dF2);
        
        Eigen::MatrixXd d2E_div_dF2_right = d2sigma_div_dF2.block(0, 0, 4, 4) * dE_div_dsigma[0] +
            d2sigma_div_dF2.block(0, 4, 4, 4) * dE_div_dsigma[1];
        
        // left term:
        Eigen::MatrixXd d2E_div_dsigma2;
        compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
        
        Eigen::MatrixXd dsigma_div_dF;
        dsigma_div_dF.resize(4, 2);
        for(int dimI = 0; dimI < 2; dimI++) {
            Eigen::Matrix2d dsigma_div_dF_mtr = svd.matrixU().col(dimI) *
                svd.matrixV().col(dimI).transpose();
            dsigma_div_dF.col(dimI) <<
                dsigma_div_dF_mtr.row(0).transpose(),
                dsigma_div_dF_mtr.row(1).transpose();
        }
        
        Eigen::MatrixXd d2E_div_dF2_left = d2E_div_dsigma2(0, 0) * dsigma_div_dF.col(0) * dsigma_div_dF.col(0).transpose() +
        d2E_div_dsigma2(1, 1) * dsigma_div_dF.col(1) * dsigma_div_dF.col(1).transpose();
        
        // cross sigma derivative
        if(crossSigmaDervative) {
            for(int sigmaI = 0; sigmaI < svd.singularValues().size(); sigmaI++) {
                for(int sigmaJ = sigmaI + 1; sigmaJ < svd.singularValues().size(); sigmaJ++) {
                    const Eigen::MatrixXd m = dsigma_div_dF.col(sigmaJ) * dsigma_div_dF.col(sigmaI).transpose();
                    d2E_div_dF2_left += d2E_div_dsigma2(sigmaI, sigmaJ) * m +
                    d2E_div_dsigma2(sigmaJ, sigmaI) * m.transpose();
                }
            }
        }
        
        d2E_div_dF2_rest = d2E_div_dF2_left + d2E_div_dF2_right;
    }
    
    void Energy::initStepSize(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        if(needElemInvSafeGuard) {
            initStepSize_preventElemInv(data, searchDir, stepSize);
        }
    }
    
    void Energy::initStepSize_preventElemInv(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        assert(searchDir.size() == data.V.rows() * 2);
        assert(stepSize > 0.0);
        
        for(int triI = 0; triI < data.F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d V1(searchDir[triVInd[0] * 2], searchDir[triVInd[0] * 2 + 1]);
            const Eigen::Vector2d V2(searchDir[triVInd[1] * 2], searchDir[triVInd[1] * 2 + 1]);
            const Eigen::Vector2d V3(searchDir[triVInd[2] * 2], searchDir[triVInd[2] * 2 + 1]);
            
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
    }
    
    void Energy::initStepSize(const Eigen::VectorXd& V,
                              const Eigen::VectorXd& searchDir,
                              double& stepSize) const
    {
        if(needElemInvSafeGuard) {
            initStepSize_preventElemInv(V, searchDir, stepSize);
        }
    }
    
    void Energy::initStepSize_preventElemInv(const Eigen::VectorXd& V,
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
    
    void Energy::getBulkModulus(double& bulkModulus)
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
}
