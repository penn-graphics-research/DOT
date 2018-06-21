//
//  SeparationEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/8/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "SeparationEnergy.hpp"
#include "IglUtils.hpp"

#include <iostream>
#include <fstream>

extern std::ofstream logFile;

namespace FracCuts {
    
    void SeparationEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        const double normalizer_div = data.virtualRadius;
        
        energyValPerElem.resize(data.cohE.rows());
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(data.boundaryEdge[cohI]) {
                energyValPerElem[cohI] = 0.0;
            }
            else {
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / normalizer_div));
                energyValPerElem[cohI] = w * kernel((data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2))).squaredNorm());
                energyValPerElem[cohI] += w * kernel((data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3))).squaredNorm());
            }
        }
    }
    
    void SeparationEnergy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        const double normalizer_div = data.virtualRadius;
        
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / normalizer_div));
                const Eigen::Vector2d xamc = data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2));
                const Eigen::Vector2d xbmd = data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3));
                const double kG_ac = w * kernelGradient(xamc.squaredNorm());
                const double kG_bd = w * kernelGradient(xbmd.squaredNorm());
                gradient.block(data.cohE(cohI, 0) * 2, 0, 2, 1) += kG_ac * 2.0 * xamc;
                gradient.block(data.cohE(cohI, 2) * 2, 0, 2, 1) -= kG_ac * 2.0 * xamc;
                gradient.block(data.cohE(cohI, 1) * 2, 0, 2, 1) += kG_bd * 2.0 * xbmd;
                gradient.block(data.cohE(cohI, 3) * 2, 0, 2, 1) -= kG_bd * 2.0 * xbmd;
            }
        }
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    
    void SeparationEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight) const
    {
        computeHessian(data, precondMtr, uniformWeight);
    }
    
    void SeparationEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                             Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        const double normalizer_div = data.virtualRadius;
        
        Eigen::Matrix4d dt2dd2x;
        dt2dd2x <<
            2.0, 0.0, -2.0, 0.0,
            0.0, 2.0, 0.0, -2.0,
            -2.0, 0.0, 2.0, 0.0,
            0.0, -2.0, 0.0, 2.0;
        
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const Eigen::Vector2d xamc = data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2));
                const Eigen::Vector2d xbmd = data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3));
                
                //                Eigen::VectorXd dtddx_ac;
                //                dtddx_ac.resize(4);
                //                dtddx_ac << 2 * xamc, -2 * xamc;
                //                Eigen::VectorXd dtddx_bd;
                //                dtddx_bd.resize(4);
                //                dtddx_bd << 2 * xbmd, -2 * xbmd;
                
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / normalizer_div));
                
                const double sqn_xamc = xamc.squaredNorm();
                const Eigen::Matrix4d hessian_ac = (w * kernelGradient(sqn_xamc)) * dt2dd2x;
                //                const Eigen::Matrix4d hessian_ac = w * (kernelHessian(sqn_xamc) * dtddx_ac * dtddx_ac.transpose() +
                //                                                    kernelGradient(sqn_xamc) * dt2dd2x);
                const double sqn_xbmd = xbmd.squaredNorm();
                const Eigen::Matrix4d hessian_bd = (w * kernelGradient(sqn_xbmd)) * dt2dd2x;
                //                const Eigen::Matrix4d hessian_bd = w * (kernelHessian(sqn_xbmd) * dtddx_bd * dtddx_bd.transpose() +
                //                                                    kernelGradient(sqn_xbmd) * dt2dd2x);
                
                bool fixed[4];
                for(int vI = 0; vI < 4; vI++) {
                    fixed[vI] = (data.fixedVert.find(data.cohE(cohI, vI)) != data.fixedVert.end());
                }
                Eigen::VectorXi vInd;
                vInd.resize(1);
                if(fixed[0]) {
                    if(!fixed[2]) {
                        vInd[0] = data.cohE(cohI, 2);
                        IglUtils::addBlockToMatrix(hessian_ac.block(2, 2, 2, 2), vInd, 2, V, I, J);
                    }
                }
                else {
                    if(fixed[2]) {
                        vInd[0] = data.cohE(cohI, 0);
                        IglUtils::addBlockToMatrix(hessian_ac.block(0, 0, 2, 2), vInd, 2, V, I, J);
                    }
                    else {
                        IglUtils::addBlockToMatrix(hessian_ac, Eigen::Vector2i(data.cohE(cohI, 0), data.cohE(cohI, 2)), 2, V, I, J);
                    }
                }
                
                if(fixed[1]) {
                    if(!fixed[3]) {
                        vInd[0] = data.cohE(cohI, 3);
                        IglUtils::addBlockToMatrix(hessian_bd.block(2, 2, 2, 2), vInd, 2, V, I, J);
                    }
                }
                else {
                    if(fixed[3]) {
                        vInd[0] = data.cohE(cohI, 1);
                        IglUtils::addBlockToMatrix(hessian_bd.block(0, 0, 2, 2), vInd, 2, V, I, J);
                    }
                    else {
                        IglUtils::addBlockToMatrix(hessian_bd, Eigen::Vector2i(data.cohE(cohI, 1), data.cohE(cohI, 3)), 2, V, I, J);
                    }
                }
            }
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
    
    void SeparationEnergy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight) const
    {
        //TODO: use the sparsity structure from last compute
        
        const double normalizer_div = data.virtualRadius;
        
        hessian.resize(data.V.rows() * 2, data.V.rows() * 2);
        hessian.reserve(data.V.rows() * 3 * 4);
        hessian.setZero();
        
        Eigen::Matrix4d dt2dd2x;
        dt2dd2x <<
            2.0, 0.0, -2.0, 0.0,
            0.0, 2.0, 0.0, -2.0,
            -2.0, 0.0, 2.0, 0.0,
            0.0, -2.0, 0.0, 2.0;
        
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const Eigen::Vector2d xamc = data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2));
                const Eigen::Vector2d xbmd = data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3));
                
//                Eigen::VectorXd dtddx_ac;
//                dtddx_ac.resize(4);
//                dtddx_ac << 2 * xamc, -2 * xamc;
//                Eigen::VectorXd dtddx_bd;
//                dtddx_bd.resize(4);
//                dtddx_bd << 2 * xbmd, -2 * xbmd;
                
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / normalizer_div));
                
                const double sqn_xamc = xamc.squaredNorm();
                const Eigen::Matrix4d hessian_ac = (w * kernelGradient(sqn_xamc)) * dt2dd2x;
//                const Eigen::Matrix4d hessian_ac = w * (kernelHessian(sqn_xamc) * dtddx_ac * dtddx_ac.transpose() +
//                                                    kernelGradient(sqn_xamc) * dt2dd2x);
                const double sqn_xbmd = xbmd.squaredNorm();
                const Eigen::Matrix4d hessian_bd = (w * kernelGradient(sqn_xbmd)) * dt2dd2x;
//                const Eigen::Matrix4d hessian_bd = w * (kernelHessian(sqn_xbmd) * dtddx_bd * dtddx_bd.transpose() +
//                                                    kernelGradient(sqn_xbmd) * dt2dd2x);
                
                bool fixed[4];
                for(int vI = 0; vI < 4; vI++) {
                    fixed[vI] = (data.fixedVert.find(data.cohE(cohI, vI)) != data.fixedVert.end());
                }
                Eigen::VectorXi vInd;
                vInd.resize(1);
                if(fixed[0]) {
                    if(!fixed[2]) {
                        vInd[0] = data.cohE(cohI, 2);
                        IglUtils::addBlockToMatrix(hessian, hessian_ac.block(2, 2, 2, 2), vInd, 2);
                    }
                }
                else {
                    if(fixed[2]) {
                        vInd[0] = data.cohE(cohI, 0);
                        IglUtils::addBlockToMatrix(hessian, hessian_ac.block(0, 0, 2, 2), vInd, 2);
                    }
                    else {
                        IglUtils::addBlockToMatrix(hessian, hessian_ac, Eigen::Vector2i(data.cohE(cohI, 0), data.cohE(cohI, 2)), 2);
                    }
                }
                
                if(fixed[1]) {
                    if(!fixed[3]) {
                        vInd[0] = data.cohE(cohI, 3);
                        IglUtils::addBlockToMatrix(hessian, hessian_bd.block(2, 2, 2, 2), vInd, 2);
                    }
                }
                else {
                    if(fixed[3]) {
                        vInd[0] = data.cohE(cohI, 1);
                        IglUtils::addBlockToMatrix(hessian, hessian_bd.block(0, 0, 2, 2), vInd, 2);
                    }
                    else {
                        IglUtils::addBlockToMatrix(hessian, hessian_bd, Eigen::Vector2i(data.cohE(cohI, 1), data.cohE(cohI, 3)), 2);
                    }
                }
            }
        }
        for(const auto fixedVI : data.fixedVert) {
            hessian.insert(2 * fixedVI, 2 * fixedVI) = 1.0;
            hessian.insert(2 * fixedVI + 1, 2 * fixedVI + 1) = 1.0;
        }
        hessian.makeCompressed();
        
//        Eigen::BDCSVD<Eigen::MatrixXd> svd((Eigen::MatrixXd(hessian)));
//        logFile << "singular values of hessian_ES:" << std::endl << svd.singularValues() << std::endl;
//        double det = 1.0;
//        for(int i = svd.singularValues().size() - 1; i >= 0; i--) {
//            det *= svd.singularValues()[i];
//        }
//        std::cout << "det(hessian_ES) = " << det << std::endl;
    }
    
    void SeparationEnergy::checkEnergyVal(const TriangleSoup& data) const
    {
        
    }
    
    SeparationEnergy::SeparationEnergy(double p_sigma_base, double p_sigma_param) :
        sigma_base(p_sigma_base), sigma_param(p_sigma_param), Energy(true)
    {
        sigma = sigma_param * sigma_base;
        assert(sigma > 0);
    }
    
    bool SeparationEnergy::decreaseSigma(void)
    {
        if(sigma_param > 1.0e-4) {
            sigma_param /= 2.0;
            sigma = sigma_param * sigma_base;
            std::cout << "sigma decreased to ";
            std::cout << sigma_param << " * " << sigma_base << " = " << sigma << std::endl;
            return true;
        }
        else {
            std::cout << "sigma stays at it's predefined minimum ";
            std::cout << sigma_param << " * " << sigma_base << " = " << sigma << std::endl;
            return false;
        }
        
    }
    
    double SeparationEnergy::getSigmaParam(void) const
    {
        return sigma_param;
    }
    
    double SeparationEnergy::kernel(double t) const
    {
//        return t * t / (t * t + sigma);
        return t / (t + sigma);
    }
    
    double SeparationEnergy::kernelGradient(double t) const
    {
//        const double denom_sqrt = t * t + sigma;
//        return 2 * t * sigma / (denom_sqrt * denom_sqrt);
        const double denom_sqrt = t + sigma;
        return sigma / (denom_sqrt * denom_sqrt);
    }
    
    double SeparationEnergy::kernelHessian(double t) const
    {
//        const double t2 = t * t;
//        return -2 * sigma * (3 * t2 - sigma) * (t2 + sigma) / std::pow(t2 + sigma, 4);
        const double tpsigma = t + sigma;
        return -2 * sigma / std::pow(tpsigma, 3);
    }
    
}
