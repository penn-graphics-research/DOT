//
//  Energy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/4/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Energy.hpp"

#include <igl/avg_edge_length.h>

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
        assert(0 && "please implement before use!");
    }
    
    void Energy::computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                   Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        assert(0 && "please implement this method in your subclass!");
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
        assert(0 && "please implement!");
    }
    
    void Energy::computeHessianBySVD(const TriangleSoup& data, Eigen::VectorXd* V,
                                     Eigen::VectorXi* I, Eigen::VectorXi* J) const
    {
        assert(0 && "please implement!");
    }
    
    void Energy::initStepSize(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        
    }
    
}
