//
//  DADMMTimeStepper.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/26/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef DADMMTimeStepper_hpp
#define DADMMTimeStepper_hpp

#include "Optimizer.hpp"

namespace FracCuts {
    
    class DADMMTimeStepper : public Optimizer
    {
    protected:
        Eigen::MatrixXd V_localCopy;
        Eigen::MatrixXd y;
        double rho, kappa;
        
        Eigen::VectorXi incTriAmt;
        
    public:
        DADMMTimeStepper(const TriangleSoup& p_data0,
                         const std::vector<Energy*>& p_energyTerms,
                         const std::vector<double>& p_energyParams,
                         int p_propagateFracture = 1,
                         bool p_mute = false,
                         bool p_scaffolding = false,
                         const Eigen::MatrixXd& UV_bnds = Eigen::MatrixXd(),
                         const Eigen::MatrixXi& E = Eigen::MatrixXi(),
                         const Eigen::VectorXi& bnd = Eigen::VectorXi());
        
    public:
        virtual void precompute(void);
        
    protected:
        virtual bool fullyImplicit(void);
        
    protected:
        void computeEnergyVal_decentral(const TriangleSoup& globalMesh,
                                        int partitionI,
                                        const Eigen::RowVectorXd& localCopy,
                                        const Eigen::RowVectorXd& dualVar,
                                        double& E) const;
        void computeGradient_decentral(const TriangleSoup& globalMesh,
                                       int partitionI,
                                       const Eigen::RowVectorXd& localCopy,
                                       const Eigen::RowVectorXd& dualVar,
                                       Eigen::VectorXd& g) const;
        void computeHessianProxy_decentral(const TriangleSoup& globalMesh,
                                           int partitionI,
                                           const Eigen::RowVectorXd& localCopy,
                                           const Eigen::RowVectorXd& dualVar,
                                           Eigen::MatrixXd& P) const;
    };
}

#endif /* DADMMTimeStepper_hpp */
