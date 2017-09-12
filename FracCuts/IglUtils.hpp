//
//  IglUtils.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include <Eigen/Eigen>

namespace FracCuts {
    
    // a static class implementing basic geometry processing operations that are not provided in libIgl
    class IglUtils {
    public:
        static void computeGraphLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL);
        
        // graph laplacian with half-weighted boundary edge, the computation is also faster
        static void computeUniformLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL);
        
        static void mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3]);
        
        static void mapScalarToColor(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double upperBound);
        
        static void addBlockToMatrix(Eigen::SparseMatrix<double>& mtr, const Eigen::MatrixXd& block,
                                     const Eigen::VectorXi& index, int dim);
    };
    
}

#endif /* IglUtils_hpp */
