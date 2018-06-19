//
//  Energy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/4/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Energy.hpp"
#include "IglUtils.hpp"

#include <igl/avg_edge_length.h>

#include <tbb/tbb.h>

#include <fstream>
#include <iostream>

extern std::ofstream logFile;

namespace FracCuts {
    
    Energy::Energy(bool p_needRefactorize) :
        needRefactorize(p_needRefactorize)
    {
        
    }
    
    Energy::~Energy(void)
    {
        
    }
    
    bool Energy::getNeedRefactorize(void) const
    {
        return needRefactorize;
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
        computeGradient(data, gradient0);
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
                computeGradient(perturbed, gradient_perturbed);
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
//            computePrecondMtr(data, &V, &I, &J); //TODO: change name to Hessian!
            computeHessianBySVD(data, &V, &I, &J);
            std::vector<Eigen::Triplet<double>> triplet(V.size());
            for(int entryI = 0; entryI < V.size(); entryI++) {
                triplet[entryI] = Eigen::Triplet<double>(I[entryI], J[entryI], V[entryI]);
            }
            hessian_symbolic.resize(data.V.rows() * 2, data.V.rows() * 2);
            hessian_symbolic.setFromTriplets(triplet.begin(), triplet.end());
        }
        else {
            computeHessian(data, hessian_symbolic);
        }
        
        Eigen::SparseMatrix<double> difMtr = hessian_symbolic - hessian_finiteDiff;
        const double dif_L2 = difMtr.norm();
        const double relErr = dif_L2 / hessian_finiteDiff.norm();
        
        std::cout << "L2 dist = " << dif_L2 << ", relErr = " << relErr << std::endl;
        
        logFile << "check hessian:" << std::endl;
        logFile << "h_symbolic =\n" << hessian_symbolic << std::endl;
        logFile << "h_finiteDiff = \n" << hessian_finiteDiff << std::endl;
    }
    
    void Energy::computeGradientBySVD(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
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
            Xt << (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose(),
            (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = X0.inverse(); //TODO: this only need to be computed once
            
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            
            Eigen::MatrixXd dsigma_div_dx;
            IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
            
            Eigen::VectorXd dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            
            const double w = data.triArea[triI];
            for(int triVI = 0; triVI < 3; triVI++) {
                gradient.segment(triVInd[triVI] * 2, 2) += w * dsigma_div_dx.block(triVI * 2, 0, 2, 2) * dE_div_dsigma;
            }
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    
    void Energy::computeHessianBySVD(const TriangleSoup& data, Eigen::VectorXd* V,
                                     Eigen::VectorXi* I, Eigen::VectorXi* J) const
    {
        std::vector<Eigen::Matrix<double, 6, 6>> triHessians(data.F.rows());
        std::vector<Eigen::VectorXi> vInds(data.F.rows());
        //        for(int triI = 0; triI < data.F.rows(); triI++) {
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI) {
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
            Xt << (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose(),
            (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = X0.inverse(); //TODO: this only need to be computed once
            
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            
            // right term:
            Eigen::VectorXd dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            
            Eigen::MatrixXd d2sigma_div_dx2;
            IglUtils::compute_d2sigma_div_dx2(svd, A, d2sigma_div_dx2);
            
            Eigen::MatrixXd d2E_div_dx2_right = d2sigma_div_dx2.block(0, 0, 6, 6) * dE_div_dsigma[0] +
            d2sigma_div_dx2.block(0, 6, 6, 6) * dE_div_dsigma[1];
            
            // left term:
            Eigen::VectorXd d2E_div_dsigma2;
            compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
            
            Eigen::MatrixXd dsigma_div_dx;
            IglUtils::compute_dsigma_div_dx(svd, A, dsigma_div_dx);
            
            Eigen::MatrixXd d2E_div_dx2_left = d2E_div_dsigma2[0] * dsigma_div_dx.col(0) * dsigma_div_dx.col(0).transpose() +
            d2E_div_dsigma2[1] * dsigma_div_dx.col(1) * dsigma_div_dx.col(1).transpose();
            
            // add up left term and right term
            const double w = data.triArea[triI];
            triHessians[triI] = w * (d2E_div_dx2_left + d2E_div_dx2_right);
            
            IglUtils::makePD(triHessians[triI]);
            
            Eigen::VectorXi& vInd = vInds[triI];
            vInd = triVInd;
            for(int vI = 0; vI < 3; vI++) {
                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
                    vInd[vI] = -1;
                }
            }
            //        }
        });
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
    }
    
    void Energy::compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                       Eigen::VectorXd& dE_div_dsigma) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    void Energy::compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                         Eigen::VectorXd& d2E_div_dsigma2) const
    {
        assert(0 && "please implement this method in the subclass!");
    }
    
    void Energy::initStepSize(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        
    }
    
}
