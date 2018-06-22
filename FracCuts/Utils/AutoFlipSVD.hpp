//
//  AutoFlipSVD.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/21/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef AutoFlipSVD_hpp
#define AutoFlipSVD_hpp

#include <Eigen/Eigen>

#include <iostream>

namespace FracCuts {
    
    template<typename MatrixType>
    class AutoFlipSVD : Eigen::JacobiSVD<MatrixType>
    {
    protected:
        bool flipped_U, flipped_V, flipped_sigma;
        
        typename Eigen::JacobiSVD<MatrixType>::SingularValuesType singularValues_flipped;
        MatrixType matrixU_flipped, matrixV_flipped;
        
    public:
        AutoFlipSVD(MatrixType mtr, unsigned int computationOptions = 0) :
            Eigen::JacobiSVD<MatrixType>(mtr, computationOptions),
            flipped_U(false), flipped_V(false), flipped_sigma(false)
        {
            //!!! this flip algorithm is only valid in 2D
            
            bool fullUComputed = (computationOptions & Eigen::ComputeFullU);
            bool fullVComputed = (computationOptions & Eigen::ComputeFullV);
            if(fullUComputed && fullVComputed)
            {
                if(Eigen::JacobiSVD<MatrixType>::m_matrixU.determinant() < 0.0) {
                    matrixU_flipped = Eigen::JacobiSVD<MatrixType>::m_matrixU;
                    matrixU_flipped.col(1) *= -1.0;
                    flipped_U = true;
                    
                    singularValues_flipped = Eigen::JacobiSVD<MatrixType>::m_singularValues;
                    singularValues_flipped[1] *= -1.0;
                    flipped_sigma = true;
                }
                if(Eigen::JacobiSVD<MatrixType>::m_matrixV.determinant() < 0.0) {
                    matrixV_flipped = Eigen::JacobiSVD<MatrixType>::m_matrixV;
                    matrixV_flipped.col(1) *= -1.0;
                    flipped_V = true;
                    
                    if(!flipped_sigma) {
                        singularValues_flipped = Eigen::JacobiSVD<MatrixType>::m_singularValues;
                        singularValues_flipped[1] *= -1.0;
                        flipped_sigma = true;
                    }
                    else {
                        singularValues_flipped[1] = Eigen::JacobiSVD<MatrixType>::m_singularValues[1];
                        flipped_sigma = false;
                    }
                }
            }
            else if(mtr.determinant() < 0.0) {
                singularValues_flipped = Eigen::JacobiSVD<MatrixType>::m_singularValues;
                singularValues_flipped[1] *= -1.0;
                flipped_sigma = true;
            }
        }
        
    public:
        const typename Eigen::JacobiSVD<MatrixType>::SingularValuesType& singularValues(void) const {
            if(flipped_sigma) {
                return singularValues_flipped;
            }
            else {
                return Eigen::JacobiSVD<MatrixType>::singularValues();
            }
        }
        const MatrixType& matrixU(void) const {
            if(flipped_U) {
                return matrixU_flipped;
            }
            else {
                return Eigen::JacobiSVD<MatrixType>::matrixU();
            }
        }
        const MatrixType& matrixV(void) const {
            if(flipped_V) {
                return matrixV_flipped;
            }
            else {
                return Eigen::JacobiSVD<MatrixType>::matrixV();
            }
        }
    };
    
}

#endif /* AutoFlipSVD_hpp */