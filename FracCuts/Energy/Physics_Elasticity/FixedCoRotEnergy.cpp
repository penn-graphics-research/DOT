//
//  FixedCoRotEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "FixedCoRotEnergy.hpp"
#include "IglUtils.hpp"
#include "Timer.hpp"

#include <tbb/tbb.h>

extern Timer timer_temp;

namespace FracCuts {
    
    void FixedCoRotEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        Energy::getEnergyValPerElemBySVD(data, energyValPerElem, uniformWeight);
    }
    
    void FixedCoRotEnergy::compute_E(const Eigen::VectorXd& singularValues,
                                     double& E) const
    {
        const double sigmam12Sum = (singularValues - Eigen::Vector2d::Ones()).squaredNorm();
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        
        E = u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
    }
    void FixedCoRotEnergy::compute_dE_div_dsigma(const Eigen::VectorXd& singularValues,
                                                 Eigen::VectorXd& dE_div_dsigma) const
    {
        const double sigmaProdm1 = singularValues.prod() - 1.0;
        Eigen::Vector2d sigmaProd_noI = Eigen::Vector2d::Ones();
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            for(int sigmaJ = 0; sigmaJ < singularValues.size(); sigmaJ++) {
                if(sigmaJ != sigmaI) {
                    sigmaProd_noI[sigmaI] *= singularValues[sigmaJ];
                }
            }
        }
        
        dE_div_dsigma.resize(singularValues.size());
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            dE_div_dsigma[sigmaI] = 2.0 * u * (singularValues[sigmaI] - 1.0) +
                lambda * sigmaProd_noI[sigmaI] * sigmaProdm1;
        }
    }
    void FixedCoRotEnergy::compute_d2E_div_dsigma2(const Eigen::VectorXd& singularValues,
                                                   Eigen::MatrixXd& d2E_div_dsigma2) const
    {
        const double sigmaProd = singularValues.prod();
        Eigen::Vector2d sigmaProd_noI = Eigen::Vector2d::Ones();
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            for(int sigmaJ = 0; sigmaJ < singularValues.size(); sigmaJ++) {
                if(sigmaJ != sigmaI) {
                    sigmaProd_noI[sigmaI] *= singularValues[sigmaJ];
                }
            }
        }
        
        d2E_div_dsigma2.resize(singularValues.size(), singularValues.size());
        for(int sigmaI = 0; sigmaI < singularValues.size(); sigmaI++) {
            d2E_div_dsigma2(sigmaI, sigmaI) = 2.0 * u +
                lambda * sigmaProd_noI[sigmaI] * sigmaProd_noI[sigmaI];
            for(int sigmaJ = sigmaI + 1; sigmaJ < singularValues.size(); sigmaJ++) {
                d2E_div_dsigma2(sigmaI, sigmaJ) = d2E_div_dsigma2(sigmaJ, sigmaI) =
                    lambda * ((sigmaProd - 1.0) + sigmaProd_noI[sigmaI] * sigmaProd_noI[sigmaJ]);
            }
        }
    }
    
    void FixedCoRotEnergy::computeGradientByPK(const TriangleSoup& data, Eigen::VectorXd& gradient) const
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
            
            Eigen::Matrix2d X0, Xt, A, F;
            X0 << x0[1] - x0[0], x0[2] - x0[0];
            Xt << (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose(),
            (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            A = X0.inverse(); //TODO: this only need to be computed once
            F = Xt * A;
            
            timer_temp.start(0);
            AutoFlipSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            timer_temp.stop();
            
            timer_temp.start(1);
            const double J = svd.singularValues().prod();
            const double w = data.triWeight[triI] * data.triArea[triI];
            Eigen::Matrix2d JFInvT = svd.matrixU() *
                Eigen::DiagonalMatrix<double, 2>(svd.singularValues()[1], svd.singularValues()[0]) *
                svd.matrixV().transpose();
            Eigen::Matrix2d P = w * (2 * u * (F - svd.matrixU() * svd.matrixV().transpose()) +
                                     lambda * (J - 1) * JFInvT);
            const double mA11mA21 = -A(0, 0) - A(1, 0);
            const double mA12mA22 = -A(0, 1) - A(1, 1);
            gradient[triVInd[0] * 2] += P(0, 0) * mA11mA21 + P(0, 1) * mA12mA22;
            gradient[triVInd[0] * 2 + 1] += P(1, 0) * mA11mA21 + P(1, 1) * mA12mA22;
            gradient[triVInd[1] * 2] += P(0, 0) * A(0, 0) + P(0, 1) * A(0, 1);
            gradient[triVInd[1] * 2 + 1] += P(1, 0) * A(0, 0) + P(1, 1) * A(0, 1);
            gradient[triVInd[2] * 2] += P(0, 0) * A(1, 0) + P(0, 1) * A(1, 1);
            gradient[triVInd[2] * 2 + 1] += P(1, 0) * A(1, 0) + P(1, 1) * A(1, 1);
            //TODO: funtionalize it
            timer_temp.stop();
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    void FixedCoRotEnergy::computeHessianByPK(const TriangleSoup& data, Eigen::VectorXd* V,
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
            
            timer_temp.start(0);
            AutoFlipSVD<Eigen::MatrixXd> svd(Xt * A, Eigen::ComputeFullU | Eigen::ComputeFullV); //TODO: only decompose once for each element in each iteration
            timer_temp.stop();
            
            timer_temp.start(1);
            Eigen::VectorXd dE_div_dsigma;
            compute_dE_div_dsigma(svd.singularValues(), dE_div_dsigma);
            Eigen::MatrixXd d2E_div_dsigma2;
            compute_d2E_div_dsigma2(svd.singularValues(), d2E_div_dsigma2);
            if(projectSPD) {
                timer_temp.stop();
                timer_temp.start(2);
                IglUtils::makePD(d2E_div_dsigma2);
                timer_temp.stop();
                timer_temp.start(1);
            }
            //TODO: explore symmetry
            
            // compute B
            Eigen::Matrix2d B01;
            double leftCoef = d2E_div_dsigma2(0, 0);
            const double dif_sigma = svd.singularValues()[0] - svd.singularValues()[1];
            if(std::abs(dif_sigma) > 1.0e-6) {
                leftCoef = (dE_div_dsigma[0] - dE_div_dsigma[1]) / dif_sigma;
            }
            leftCoef /= 2.0;
            double rightCoef = dE_div_dsigma[0] + dE_div_dsigma[1];
            double sum_sigma = svd.singularValues()[0] + svd.singularValues()[1];
            if(sum_sigma < 1.0e-6) {
                rightCoef /= 2.0e-6;
            }
            else {
                rightCoef /= 2.0 * sum_sigma;
            }
            B01(0, 0) = B01(1, 1) = leftCoef + rightCoef;
            B01(0, 1) = B01(1, 0) = leftCoef - rightCoef;
            if(projectSPD) {
                timer_temp.stop();
                timer_temp.start(2);
                IglUtils::makePD(B01);
                timer_temp.stop();
                timer_temp.start(1);
            }
            
            // compute M using A(d2E_div_dsigma2) and B
            Eigen::MatrixXd M(4, 4);
            M.setZero();
            M(0, 0) = d2E_div_dsigma2(0, 0);
            M(0, 3) = d2E_div_dsigma2(0, 1);
            M.block(1, 1, 2, 2) = B01;
            M(3, 0) = d2E_div_dsigma2(1, 0);
            M(3, 3) = d2E_div_dsigma2(1, 1);
            
            // compute dP_div_dF
            Eigen::MatrixXd dP_div_dF(4, 4);
            const Eigen::MatrixXd& U = svd.matrixU();
            const Eigen::MatrixXd& V = svd.matrixV();
            for(int i = 0; i < 2; i++) {
                int _2i = i * 2;
                for(int j = 0; j < 2; j++) {
                    int ij = _2i + j;
                    for(int r = 0; r < 2; r++) {
                        int _2r = r * 2;
                        for(int s = 0; s < 2; s++) {
                            int rs = _2r + s;
                            dP_div_dF(ij, rs) =
                            (M(0, 0) * U(i, 0) * U(r, 0) * V(s, 0) * V(j, 0) +
                             M(0, 3) * U(i, 0) * U(r, 1) * V(s, 1) * V(j, 0) +
                             M(1, 1) * U(i, 0) * U(r, 0) * V(s, 1) * V(j, 1) +
                             M(1, 2) * U(i, 0) * U(r, 1) * V(s, 0) * V(j, 1) +
                             M(2, 1) * U(i, 1) * U(r, 0) * V(s, 1) * V(j, 0) +
                             M(2, 2) * U(i, 1) * U(r, 1) * V(s, 0) * V(j, 0) +
                             M(3, 0) * U(i, 1) * U(r, 0) * V(s, 0) * V(j, 1) +
                             M(3, 3) * U(i, 1) * U(r, 1) * V(s, 1) * V(j, 1));
                        }
                    }
                }
            }
            
            // compute dF_div_dx
            Eigen::MatrixXd dF_div_dx(6, 4);
            const double mA11mA21 = -A(0, 0) - A(1, 0);
            const double mA12mA22 = -A(0, 1) - A(1, 1);
            dF_div_dx <<
            mA11mA21, mA12mA22, 0.0, 0.0,
            0.0, 0.0, mA11mA21, mA12mA22,
            A(0, 0), A(0, 1), 0.0, 0.0,
            0.0, 0.0, A(0, 0), A(0, 1),
            A(1, 0), A(1, 1), 0.0, 0.0,
            0.0, 0.0, A(1, 0), A(1, 1);
            //TODO: functionalize to avoid multiplying zero
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            triHessians[triI] = w * (dF_div_dx * (dP_div_dF * dF_div_dx.transpose()));
            timer_temp.stop();
            
//            if(projectSPD) {
//                timer_temp.start(2);
//                IglUtils::makePD(triHessians[triI]);
//                timer_temp.stop();
//            }
            
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
    
    void FixedCoRotEnergy::checkEnergyVal(const TriangleSoup& data) const // check with isometric case
    {
        //TODO: move to super class, only provide a value
        
        std::cout << "check energyVal computation..." << std::endl;
        
        double err = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            
            Eigen::Vector3d x0_3D[3] = {
                data.V_rest.row(triVInd[0]),
                data.V_rest.row(triVInd[1]),
                data.V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x0[3];
            IglUtils::mapTriangleTo2D(x0_3D, x0);
            
            Eigen::Matrix2d X0, A;
            X0 << x0[1] - x0[0], x0[2] - x0[0];
            A = X0.inverse(); //TODO: this only need to be computed once
            
            AutoFlipSVD<Eigen::MatrixXd> svd(X0 * A); //TODO: only decompose once for each element in each iteration, would need ComputeFull U and V for derivative computations
            
            const double sigmam12Sum = (svd.singularValues() - Eigen::Vector2d::Ones()).squaredNorm();
            const double sigmaProdm1 = svd.singularValues().prod() - 1.0;
            
            const double w = data.triWeight[triI] * data.triArea[triI];
            const double energyVal = w * (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
            err += energyVal;
        }
        
        std::cout << "energyVal computation error = " << err << std::endl;
    }
    
    FixedCoRotEnergy::FixedCoRotEnergy(double YM, double PR) :
        Energy(true, false, true), u(YM / 2.0 / (1.0 + PR)), lambda(YM * PR / (1.0 + PR) / (1.0 - 2.0 * PR))
    {
        const double sigmam12Sum = 2;
        const double sigmaProdm1 = 3;
        
        const double visRange_max = (u * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1);
        Energy::setVisRange_energyVal(0.0, visRange_max);
    }
    
    void FixedCoRotEnergy::getBulkModulus(double& bulkModulus)
    {
        bulkModulus = lambda + (2.0/3.0) * u;
    }
    
}
