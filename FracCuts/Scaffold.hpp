//
//  Scaffold.hpp
//  FracCuts
//
//  Created by Minchen Li on 4/5/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Scaffold_hpp
#define Scaffold_hpp

#include "TriangleSoup.hpp"

#include<Eigen/Eigen>

#include <cstdio>

namespace FracCuts {
    class Scaffold
    {
    public:
        TriangleSoup airMesh; // tessellation of voided regions
        Eigen::VectorXi bnd, localVI2Global; // map between airMesh indices to augmented system indices
        std::map<int, int> meshVI2AirMesh; // the inverse map of bnd
        int wholeMeshSize; // augmented system size
        
    public:
        Scaffold(void);
        Scaffold(const TriangleSoup& mesh, Eigen::MatrixXd UV_bnds = Eigen::MatrixXd(),
                Eigen::MatrixXi E = Eigen::MatrixXi(), const Eigen::VectorXi& p_bnd = Eigen::VectorXi());

        // augment mesh gradient with air mesh gradient with parameter w_scaf
        void augmentGradient(Eigen::VectorXd& gradient, const Eigen::VectorXd& gradient_scaf, double w_scaf) const;
        
        // augment mesh proxy matrix with air mesh proxy matrix with parameter w_scaf
        void augmentProxyMatrix(Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V,
                                const Eigen::VectorXi& I_scaf, const Eigen::VectorXi& J_scaf, const Eigen::VectorXd& V_scaf,
                                double w_scaf) const;
        
        // extract air mesh searchDir from augmented searchDir
        void wholeSearchDir2airMesh(const Eigen::VectorXd& searchDir, Eigen::VectorXd& searchDir_airMesh) const;
        
        // stepForward air mesh using augmented searchDir
        void stepForward(const TriangleSoup& airMesh0, const Eigen::VectorXd& searchDir, double stepSize);
        
        // for rendering purpose:
        void augmentUVwithAirMesh(Eigen::MatrixXd& UV, double scale) const;
        void augmentFwithAirMesh(Eigen::MatrixXi& F) const;
        void augmentFColorwithAirMesh(Eigen::MatrixXd& FColor) const;
        
        // get 1-ring airmesh loop for scaffolding optimization on local stencils
        void get1RingAirLoop(int vI,
                             Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd,
                             std::set<int>& loop_AMVI) const;
        
        bool getCornerAirLoop(const std::vector<int>& corner_mesh, const Eigen::RowVector2d& mergedPos,
                              Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd) const;
    };
}

#endif /* Scaffold_hpp */