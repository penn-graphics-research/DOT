//
//  DADMMTimeStepper.cpp
//  FracCuts
//
//  Created by Minchen Li on 6/26/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "DADMMTimeStepper.hpp"

#include <tbb/tbb.h>

#include <iostream>

extern std::ofstream logFile;

namespace FracCuts {
    
    DADMMTimeStepper::DADMMTimeStepper(const TriangleSoup& p_data0,
                                       const std::vector<Energy*>& p_energyTerms,
                                       const std::vector<double>& p_energyParams,
                                       int p_propagateFracture,
                                       bool p_mute,
                                       bool p_scaffolding,
                                       const Eigen::MatrixXd& UV_bnds,
                                       const Eigen::MatrixXi& E,
                                       const Eigen::VectorXi& bnd,
                                       AnimScriptType animScriptType) :
        Optimizer(p_data0, p_energyTerms, p_energyParams,
                  p_propagateFracture, p_mute, p_scaffolding,
                  UV_bnds, E, bnd, animScriptType)
    {
        V_localCopy.resize(result.F.rows(), 6);
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            V_localCopy.row(triI) <<
                result.V.row(triVInd[0]),
                result.V.row(triVInd[1]),
                result.V.row(triVInd[2]);
        }
        y = Eigen::MatrixXd::Zero(result.F.rows(), 6);
        rho = 10.0;
        kappa = 0.1;
        
        incTriAmt = Eigen::VectorXi::Zero(result.V.rows());
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            incTriAmt[triVInd[0]]++;
            incTriAmt[triVInd[1]]++;
            incTriAmt[triVInd[2]]++;
        }
    }
    
    void DADMMTimeStepper::precompute(void)
    {}
    
    bool DADMMTimeStepper::fullyImplicit(void)
    {
        //!!! only need to handle position constraints here
        V_localCopy.resize(result.F.rows(), 6);
        for(int triI = 0; triI < result.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = result.F.row(triI);
            V_localCopy.row(triI) <<
            result.V.row(triVInd[0]),
            result.V.row(triVInd[1]),
            result.V.row(triVInd[2]);
        }
        
        const double dualTol = targetGRes * 6;
        const int primalMaxIter = 1; //!!! how many primal updates to run between each dual update?
        const double primalTol = targetGRes * 1000.0; //!!! use primal update tol?
        const int localMaxIter = 10; //!!! how many local copy updates to run between each global update?
        const double localTol = targetGRes / result.F.rows(); //!!! use local copy update tol?
        
        while(true) {
//            double lastAugLagE = 0.0;
            double g_global_sqn = 0.0;
            for(int i = 0; i < primalMaxIter; i++) {
                // local copy update
//                for(int triI = 0; triI < result.F.rows(); triI++) {
                tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI) {
                    Eigen::VectorXd x = V_localCopy.row(triI).transpose();
                    Eigen::VectorXd g;
                    for(int j = 0; j < localMaxIter; j++) {
                        computeGradient_decentral(result, triI, x.transpose(), y.row(triI), g);
                        if(g.squaredNorm() < localTol) {
                            break;
                        }
                        
                        Eigen::MatrixXd P;
                        computeHessianProxy_decentral(result, triI, x.transpose(), y.row(triI), P);
                        
                        // solve for search direction
                        Eigen::VectorXd p = P.ldlt().solve(-g);
                        
                        // line search init
                        double alpha = 1.0;
                        energyTerms[0]->initStepSize(x, p, alpha);
                        alpha *= 0.99;
                        
                        // Armijo's rule:
                        const double m = p.dot(g);
                        const double c1m = 1.0e-4 * m;
                        Eigen::VectorXd x0 = x;
                        double E0;
                        computeEnergyVal_decentral(result, triI, x0.transpose(), y.row(triI), E0);
                        x = x0 + alpha * p;
                        double E;
                        computeEnergyVal_decentral(result, triI, x.transpose(), y.row(triI), E);
                        while(E > E0 + alpha * c1m) {
                            alpha /= 2.0;
                            x = x0 + alpha * p;
                            computeEnergyVal_decentral(result, triI, x.transpose(), y.row(triI), E);
                        }
    //                    assert(TriangleSoup::checkInversion(x));
                    }
                    
                    V_localCopy.row(triI) = x.transpose();
//                }
                });
                
                //        //DEBUG
                //        FILE *out = fopen("/Users/mincli/Desktop/output_FracCuts/triSoup.obj", "w"); assert(out);
                //        for(int triI = 0; triI < V_localCopy.rows(); triI++) {
                //            fprintf(out, "v %le %le 0.0\n", V_localCopy(triI, 0), V_localCopy(triI, 1));
                //            fprintf(out, "v %le %le 0.0\n", V_localCopy(triI, 2), V_localCopy(triI, 3));
                //            fprintf(out, "v %le %le 0.0\n", V_localCopy(triI, 4), V_localCopy(triI, 5));
                //        }
                //        for(int triI = 0; triI < V_localCopy.rows(); triI++) {
                //            fprintf(out, "f %d %d %d\n", triI * 3 + 1, triI * 3 + 2, triI * 3 + 3);
                //        }
                //        fclose(out);
                
                // global update
                result.V.setZero();
                for(int triI = 0; triI < result.F.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd = result.F.row(triI);
                    result.V.row(triVInd[0]) += (V_localCopy.block(triI, 0, 1, 2) +
                                                 y.block(triI, 0, 1, 2) / rho);
                    result.V.row(triVInd[1]) += (V_localCopy.block(triI, 2, 1, 2) +
                                                 y.block(triI, 2, 1, 2) / rho);
                    result.V.row(triVInd[2]) += (V_localCopy.block(triI, 4, 1, 2) +
                                                 y.block(triI, 4, 1, 2) / rho);
                }
                for(int vI = 0; vI < result.V.rows(); vI++) {
                    result.V.row(vI) /= incTriAmt[vI];
                }
                
                //        out = fopen("/Users/mincli/Desktop/output_FracCuts/mesh.obj", "w"); assert(out);
                //        for(int vI = 0; vI < result.V.rows(); vI++) {
                //            fprintf(out, "v %le %le 0.0\n", result.V(vI, 0), result.V(vI, 1));
                //        }
                //        for(int triI = 0; triI < result.F.rows(); triI++) {
                //            fprintf(out, "f %d %d %d\n", result.F(triI, 0) + 1,
                //                    result.F(triI, 1) + 1, result.F(triI, 2) + 1);
                //        }
                //        fclose(out);
                //        computeEnergyVal(result, scaffold, lastEnergyVal);
                //        std::cout << "energyVal = " << lastEnergyVal << std::endl;
                
//                double augLagE = 0.0;
//                for(int triI = 0; triI < result.F.rows(); triI++) {
//                    double E_triI;
//                    computeEnergyVal_decentral(result, triI, V_localCopy.row(triI), y.row(triI), E_triI);
//                    augLagE += E_triI;
//                }
//                std::cout << "\taugLagE = " << augLagE << std::endl;
//                if(std::abs((lastAugLagE - augLagE) / augLagE) < 1.0e-6) {
//                    break;
//                }
//                lastAugLagE = augLagE;
                
                g_global_sqn = 0.0;
                for(int triI = 0; triI < result.F.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd = result.F.row(triI);
                    for(int localVI = 0; localVI < 3; localVI++) {
                        g_global_sqn += (rho * (V_localCopy.block(triI, localVI * 2, 1, 2) -
                                                result.V.row(triVInd[localVI])) +
                                         y.block(triI, localVI * 2, 1, 2)).squaredNorm();
                    }
                }
//                std::cout << "\t||g_global||^2 = " << g_global_sqn << std::endl;
                if(g_global_sqn < primalTol) {
                    break;
                }
            }
            
            // dual update
            Eigen::MatrixXd y_old = y;
            for(int triI = 0; triI < result.F.rows(); triI++) {
                //        tbb::parallel_for(0, (int)result.F.rows(), 1, [&](int triI) {
                const Eigen::RowVector3i& triVInd = result.F.row(triI);
                y.block(triI, 0, 1, 2) += kappa * (V_localCopy.block(triI, 0, 1, 2) -
                                                   result.V.row(triVInd[0]));
                y.block(triI, 2, 1, 2) += kappa * (V_localCopy.block(triI, 2, 1, 2) -
                                                   result.V.row(triVInd[1]));
                y.block(triI, 4, 1, 2) += kappa * (V_localCopy.block(triI, 4, 1, 2) -
                                                   result.V.row(triVInd[2]));
            }
            //        });
            
            std::cout << "\t||g_global||^2 = " << g_global_sqn << std::endl;
            
            double g_y_sqnorm = (y - y_old).squaredNorm() / kappa / kappa;
            std::cout << "\t||g_y||^2 = " << g_y_sqnorm << std::endl;
            
            computeGradient(result, scaffold, gradient);
            double gradient_sqn = gradient.squaredNorm();
            std::cout << "\t||gradient||^2 = " << gradient_sqn << std::endl;
            
            if((g_y_sqnorm < dualTol) && (g_global_sqn < primalTol) &&
               (gradient_sqn < targetGRes * 1000.0))
            {
                logFile << "||gradient||^2 = " << gradient_sqn << std::endl;
                break;
            }
        }
        
        return false;
    }
    
    // add dynamic and gravity
    void DADMMTimeStepper::computeEnergyVal_decentral(const TriangleSoup& globalMesh,
                                                      int partitionI,
                                                      const Eigen::RowVectorXd& localCopy,
                                                      const Eigen::RowVectorXd& dualVar,
                                                      double& E) const
    {
        energyTerms[0]->computeEnergyValBySVD(globalMesh, partitionI, localCopy.transpose(), E);
        
        // dynamics and gravity
        E *= dtSq;
        const Eigen::RowVector3i& triVInd = globalMesh.F.row(partitionI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            double massI = globalMesh.massMatrix.coeff(vI, vI);
            E += (localCopy.segment(localVI * 2, 2) - resultV_n.row(vI) - dt * velocity.segment(vI * 2, 2).transpose()  - dtSq * gravity.transpose()).squaredNorm() * massI / 2.0;
        }

        // DADMM
        E += rho / 2.0 * ((localCopy.segment(0, 2) - globalMesh.V.row(triVInd[0])).squaredNorm() +
                          (localCopy.segment(2, 2) - globalMesh.V.row(triVInd[1])).squaredNorm() +
                          (localCopy.segment(4, 2) - globalMesh.V.row(triVInd[2])).squaredNorm());
        E += (dualVar.segment(0, 2).dot(localCopy.segment(0, 2) - globalMesh.V.row(triVInd[0])) +
              dualVar.segment(2, 2).dot(localCopy.segment(2, 2) - globalMesh.V.row(triVInd[1])) +
              dualVar.segment(4, 2).dot(localCopy.segment(4, 2) - globalMesh.V.row(triVInd[2])));
    }
    
    void DADMMTimeStepper::computeGradient_decentral(const TriangleSoup& globalMesh,
                                                     int partitionI,
                                                     const Eigen::RowVectorXd& localCopy,
                                                     const Eigen::RowVectorXd& dualVar,
                                                     Eigen::VectorXd& g) const
    {
        energyTerms[0]->computeGradientBySVD(globalMesh, partitionI, localCopy.transpose(), g);
        
        // dynamics and gravity
        g *= dtSq;
        const Eigen::RowVector3i& triVInd = globalMesh.F.row(partitionI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            if(globalMesh.fixedVert.find(vI) != globalMesh.fixedVert.end()) {
                g.segment(localVI * 2, 2).setZero();
            }
            else {
                double massI = globalMesh.massMatrix.coeff(vI, vI);
                g.segment(localVI * 2, 2) += massI * (localCopy.segment(localVI * 2, 2).transpose() - resultV_n.row(vI).transpose() - dt * velocity.segment(vI * 2, 2) - dtSq * gravity);
            }
        }

        // DADMM
        g.segment(0, 2) += (dualVar.segment(0, 2) +
                            rho * (localCopy.segment(0, 2) - globalMesh.V.row(triVInd[0]))).transpose();
        g.segment(2, 2) += (dualVar.segment(2, 2) +
                            rho * (localCopy.segment(2, 2) - globalMesh.V.row(triVInd[1]))).transpose();
        g.segment(4, 2) += (dualVar.segment(4, 2) +
                            rho * (localCopy.segment(4, 2) - globalMesh.V.row(triVInd[2]))).transpose();
    }
    
    void DADMMTimeStepper::computeHessianProxy_decentral(const TriangleSoup& globalMesh,
                                                         int partitionI,
                                                         const Eigen::RowVectorXd& localCopy,
                                                         const Eigen::RowVectorXd& dualVar,
                                                         Eigen::MatrixXd& P) const
    {
        energyTerms[0]->computeHessianBySVD(globalMesh, partitionI, localCopy.transpose(), P);
        
        // dynamics and gravity
        P *= dtSq;
        const Eigen::RowVector3i& triVInd = globalMesh.F.row(partitionI);
        for(int localVI = 0; localVI < 3; localVI++) {
            int vI = triVInd[localVI];
            if(globalMesh.fixedVert.find(vI) != globalMesh.fixedVert.end()) {
                P.row(localVI * 2).setZero();
                P.row(localVI * 2 + 1).setZero();
                P.col(localVI * 2).setZero();
                P.col(localVI * 2 + 1).setZero();
                P.block(localVI * 2, localVI * 2, 2, 2).setIdentity();
            }
            else {
                double massI = globalMesh.massMatrix.coeff(vI, vI);
                P(localVI * 2, localVI * 2) += massI;
                P(localVI * 2 + 1, localVI * 2 + 1) += massI;
            }
        }

        // DADMM
        P.diagonal().array() += rho;
    }
    
}
