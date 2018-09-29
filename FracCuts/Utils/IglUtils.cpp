//
//  IglUtils.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "IglUtils.hpp"

#include <set>

namespace FracCuts {
    void IglUtils::computeGraphLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL) {
        // compute vertex adjacency
        int vertAmt = F.maxCoeff() + 1;
        std::vector<std::set<int>> adjacentVertices(vertAmt);
        for(int rowI = 0; rowI < F.rows(); rowI++) {
            adjacentVertices[F(rowI, 0)].insert(F(rowI, 1));
            adjacentVertices[F(rowI, 1)].insert(F(rowI, 0));
            adjacentVertices[F(rowI, 1)].insert(F(rowI, 2));
            adjacentVertices[F(rowI, 2)].insert(F(rowI, 1));
            adjacentVertices[F(rowI, 2)].insert(F(rowI, 0));
            adjacentVertices[F(rowI, 0)].insert(F(rowI, 2));
        }
        
        graphL.resize(vertAmt, vertAmt);
        graphL.setZero();
        graphL.reserve(vertAmt * 7);
        for(int rowI = 0; rowI < vertAmt; rowI++) {
            graphL.insert(rowI, rowI) = -static_cast<double>(adjacentVertices[rowI].size());
            for(const auto& neighborI : adjacentVertices[rowI]) {
                graphL.insert(rowI, neighborI) = 1;
            }
        }
    }
    
    void IglUtils::computeUniformLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL) {
        int vertAmt = F.maxCoeff() + 1;
        graphL.resize(vertAmt, vertAmt);
        graphL.setZero();
        graphL.reserve(vertAmt * 7);
        for(int rowI = 0; rowI < F.rows(); rowI++) {
            graphL.coeffRef(F(rowI, 0), F(rowI, 1))++;
            graphL.coeffRef(F(rowI, 1), F(rowI, 0))++;
            graphL.coeffRef(F(rowI, 1), F(rowI, 2))++;
            graphL.coeffRef(F(rowI, 2), F(rowI, 1))++;
            graphL.coeffRef(F(rowI, 2), F(rowI, 0))++;
            graphL.coeffRef(F(rowI, 0), F(rowI, 2))++;
            
            graphL.coeffRef(F(rowI, 0), F(rowI, 0)) -= 2;
            graphL.coeffRef(F(rowI, 1), F(rowI, 1)) -= 2;
            graphL.coeffRef(F(rowI, 2), F(rowI, 2)) -= 2;
        }
    }
    
    double getHETan(const std::map<std::pair<int, int>, double>& HETan, int v0, int v1) {
        auto finder = HETan.find(std::pair<int, int>(v0, v1));
        if(finder == HETan.end()) {
            return 0.0;
        }
        else {
            return finder->second;
        }
    }
    
    void IglUtils::computeMVCMtr(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& MVCMtr)
    {
        std::map<std::pair<int, int>, double> HETan;
        std::map<std::pair<int, int>, int> thirdPoint;
        std::vector<std::set<int>> vvNeighbor(V.rows());
        for (int triI = 0; triI < F.rows(); triI++)
        {
            int v0I = F(triI, 0);
            int v1I = F(triI, 1);
            int v2I = F(triI, 2);
            
            Eigen::Vector3d e01 = V.row(v1I) - V.row(v0I);
            Eigen::Vector3d e12 = V.row(v2I) - V.row(v1I);
            Eigen::Vector3d e20 = V.row(v0I) - V.row(v2I);
            double dot0102 = -e01.dot(e20);
            double dot1210 = -e12.dot(e01);
            double dot2021 = -e20.dot(e12);
            double cos0102 = dot0102/(e01.norm()*e20.norm());
            double cos1210 = dot1210/(e01.norm()*e12.norm());
            double cos2021 = dot2021/(e12.norm()*e20.norm());
            
            HETan[std::pair<int, int>(v0I, v1I)] = sqrt(1.0 - cos0102*cos0102) / (1.0 + cos0102);
            HETan[std::pair<int, int>(v1I, v2I)] = sqrt(1.0 - cos1210*cos1210) / (1.0 + cos1210);
            HETan[std::pair<int, int>(v2I, v0I)] = sqrt(1.0 - cos2021*cos2021) / (1.0 + cos2021);
            
            thirdPoint[std::pair<int, int>(v0I, v1I)] = v2I;
            thirdPoint[std::pair<int, int>(v1I, v2I)] = v0I;
            thirdPoint[std::pair<int, int>(v2I, v0I)] = v1I;
            
            vvNeighbor[v0I].insert(v1I);
            vvNeighbor[v0I].insert(v2I);
            vvNeighbor[v1I].insert(v0I);
            vvNeighbor[v1I].insert(v2I);
            vvNeighbor[v2I].insert(v0I);
            vvNeighbor[v2I].insert(v1I);
        }
        
        MVCMtr.resize(V.rows(), V.rows());
        MVCMtr.setZero();
        MVCMtr.reserve(V.rows() * 7);
        for(int rowI = 0; rowI < V.rows(); rowI++)
        {
            for(const auto& nbVI : vvNeighbor[rowI]) {
                double weight = getHETan(HETan, rowI, nbVI);
                auto finder = thirdPoint.find(std::pair<int, int>(nbVI, rowI));
                if(finder != thirdPoint.end()) {
                    weight += getHETan(HETan, rowI, finder->second);
                }
                weight /= (V.row(rowI) - V.row(nbVI)).norm();
                
                MVCMtr.coeffRef(rowI, rowI) -= weight;
                MVCMtr.insert(rowI, nbVI) = weight;
                
//                // symmetrized version
//                MVCMtr.coeffRef(rowI, rowI) -= weight;
//                MVCMtr.coeffRef(rowI, nbVI) += weight;
//                MVCMtr.coeffRef(nbVI, nbVI) -= weight;
//                MVCMtr.coeffRef(nbVI, rowI) += weight;
            }
        }
//        writeSparseMatrixToFile("/Users/mincli/Desktop/meshes/mtr", MVCMtr);
    }
    
    void IglUtils::fixedBoundaryParam_MVC(Eigen::SparseMatrix<double> A, const Eigen::VectorXi& bnd,
                                       const Eigen::MatrixXd& bnd_uv, Eigen::MatrixXd& UV_Tutte)
    {
        assert(bnd.size() == bnd_uv.rows());
        assert(bnd.maxCoeff() < A.rows());
        assert(A.rows() == A.cols());
        
        int vN = static_cast<int>(A.rows());
        A.conservativeResize(vN + bnd.size(), vN + bnd.size());
        A.reserve(A.nonZeros() + bnd.size() * 2);
        for(int pcI = 0; pcI < bnd.size(); pcI++) {
            A.insert(vN + pcI, bnd[pcI]) = 1.0;
            A.insert(bnd[pcI], vN + pcI) = 1.0;
        }
        
        Eigen::SparseLU<Eigen::SparseMatrix<double>> spLUSolver;
        spLUSolver.compute(A);
        if(spLUSolver.info() == Eigen::Success) {
            UV_Tutte.resize(A.rows(), 2);
            Eigen::VectorXd rhs;
            rhs.resize(A.rows());
            
            for(int dimI = 0; dimI < 2; dimI++) {
                rhs << Eigen::VectorXd::Zero(vN), bnd_uv.col(dimI);
                UV_Tutte.col(dimI) = spLUSolver.solve(rhs);
                if(spLUSolver.info() != Eigen::Success) {
                    assert("LU back solve failed!");
                }
            }
            
            UV_Tutte.conservativeResize(vN, 2);
        }
        else {
            assert("LU decomposition on MVC matrix (with Langrange Multiplier) failed!");
        }
    }
    
    void IglUtils::mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3])
    {
        const Eigen::Vector3d e[2] = {
            v[1] - v[0], v[2] - v[0]
        };
        u[0] << 0.0, 0.0;
        u[1] << e[0].norm(), 0.0;
        u[2] << e[0].dot(e[1]) / u[1][0], e[0].cross(e[1]).norm() / u[1][0];
    }
    
    void IglUtils::computeDeformationGradient(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& F)
    {
        Eigen::Vector2d x[3];
        IglUtils::mapTriangleTo2D(v, x);

        const Eigen::Vector2d u01 = u[1] - u[0];
        const Eigen::Vector2d u02 = u[2] - u[0];
        const double u01Len = u01.norm();
    
        Eigen::Matrix2d U; U << u01Len, u01.dot(u02) / u01Len, 0.0, (u01[0] * u02[1] - u01[1] * u02[0]) / u01Len;
        Eigen::Matrix2d V; V << x[1], x[2];
        F = V * U.inverse();
    }
    
    void IglUtils::map_vertices_to_circle(
        const Eigen::MatrixXd& V,
        const Eigen::VectorXi& bnd,
        Eigen::MatrixXd& UV)
    {
        // Get sorted list of boundary vertices
        std::vector<int> interior,map_ij;
        map_ij.resize(V.rows());
        
        std::vector<bool> isOnBnd(V.rows(),false);
        for (int i = 0; i < bnd.size(); i++)
        {
            isOnBnd[bnd[i]] = true;
            map_ij[bnd[i]] = i;
        }
        
        for (int i = 0; i < (int)isOnBnd.size(); i++)
        {
            if (!isOnBnd[i])
            {
                map_ij[i] = static_cast<int>(interior.size());
                interior.push_back(i);
            }
        }
        
        // Map boundary to circle
        std::vector<double> len(bnd.size());
        len[0] = 0.;
        
        for (int i = 1; i < bnd.size(); i++)
        {
            len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
        }
        double total_len = len[len.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();
        
        UV.resize(bnd.size(),2);
        const double radius = total_len / 2.0 / M_PI;
        for (int i = 0; i < bnd.size(); i++)
        {
            double frac = len[i] * 2. * M_PI / total_len;
            UV.row(map_ij[bnd[i]]) << radius * cos(frac), radius * sin(frac);
        }
    }
    
    void splitRGB(char32_t color, double rgb[3]) {
        rgb[0] = static_cast<int>((0xff0000 & color) >> 16) / 255.0;
        rgb[1] = static_cast<int>((0x00ff00 & color) >> 8) / 255.0;
        rgb[2] = static_cast<int>(0x0000ff & color) / 255.0;
    }
    void getColor(double scalar, double rgb[3], double center, double halfScale, int opt = 0)
    {
        static char32_t colorMap[3][100] = {
            // red to deep blue
            {0xdd0000,0xdc0005,0xdc0009,0xdb000e,0xda0012,0xd90015,0xd80018,0xd8001b,0xd7001e,0xd60021,0xd50023,0xd50026,0xd40028,0xd3002a,0xd2002d,0xd1002f,0xd00031,0xd00033,0xcf0036,0xce0038,0xcd003a,0xcc003c,0xcb003e,0xca0040,0xca0042,0xc90044,0xc80046,0xc70048,0xc6004a,0xc5004c,0xc4004e,0xc30050,0xc20052,0xc10054,0xc00056,0xbf0058,0xbe005a,0xbd005c,0xbc005e,0xbb0060,0xba0062,0xb80064,0xb70066,0xb60068,0xb5006a,0xb4006c,0xb3006e,0xb10070,0xb00072,0xaf0074,0xae0076,0xac0078,0xab007b,0xaa007d,0xa8007f,0xa70081,0xa50083,0xa40085,0xa20087,0xa10089,0x9f008b,0x9e008d,0x9c008f,0x9b0091,0x990093,0x970095,0x950097,0x940099,0x92009b,0x90009d,0x8e009f,0x8c00a1,0x8a00a4,0x8800a6,0x8600a8,0x8300aa,0x8100ac,0x7f00ae,0x7c00b0,0x7a00b2,0x7700b4,0x7400b6,0x7100b9,0x6f00bb,0x6b00bd,0x6800bf,0x6500c1,0x6100c3,0x5d00c5,0x5900c7,0x5500ca,0x5100cc,0x4c00ce,0x4600d0,0x4000d2,0x3900d4,0x3100d6,0x2800d9,0x1a00db,0x0000dd},
            
            // red to light green
            {0xdd0000,0xdc0e00,0xdb1800,0xda1f00,0xda2500,0xd92a00,0xd82e00,0xd73200,0xd63600,0xd53a00,0xd43d00,0xd34000,0xd34300,0xd24600,0xd14900,0xd04b00,0xcf4e00,0xce5000,0xcd5300,0xcc5500,0xcb5700,0xca5900,0xc95c00,0xc85e00,0xc76000,0xc66200,0xc56400,0xc46600,0xc36800,0xc26a00,0xc16b00,0xbf6d00,0xbe6f00,0xbd7100,0xbc7300,0xbb7400,0xba7600,0xb97800,0xb77900,0xb67b00,0xb57d00,0xb47e00,0xb28000,0xb18100,0xb08300,0xaf8500,0xad8600,0xac8800,0xab8900,0xa98b00,0xa88c00,0xa68e00,0xa58f00,0xa39100,0xa29200,0xa09300,0x9f9500,0x9d9600,0x9c9800,0x9a9900,0x999a00,0x979c00,0x959d00,0x939f00,0x92a000,0x90a100,0x8ea300,0x8ca400,0x8aa500,0x88a700,0x86a800,0x84a900,0x82ab00,0x80ac00,0x7ead00,0x7cae00,0x79b000,0x77b100,0x74b200,0x72b400,0x6fb500,0x6db600,0x6ab700,0x67b900,0x64ba00,0x61bb00,0x5ebc00,0x5abd00,0x56bf00,0x53c000,0x4fc100,0x4ac200,0x45c400,0x40c500,0x3bc600,0x34c700,0x2dc800,0x24ca00,0x17cb00,0x00cc00},
            
            // blue to black to green
            {0x1e90ff,0x1f8df9,0x218af3,0x2287ee,0x2383e8,0x2480e2,0x247ddc,0x257ad7,0x2577d1,0x2674cb,0x2671c6,0x276ec0,0x276bbb,0x2768b5,0x2765b0,0x2762aa,0x275fa5,0x275ca0,0x26599a,0x265695,0x265490,0x26518a,0x254e85,0x254b80,0x24487b,0x244576,0x234371,0x22406c,0x223d67,0x213b62,0x20385d,0x1f3558,0x1f3353,0x1e304f,0x1d2d4a,0x1c2b45,0x1b2841,0x1a263c,0x182338,0x172133,0x161e2f,0x151c2a,0x141926,0x121722,0x11151e,0x0f121a,0x0c0f16,0x090b11,0x05070b,0x020204,0x010200,0x030601,0x050a02,0x060e03,0x081104,0x0a1405,0x0c1606,0x0d1807,0x0f1a08,0x101d09,0x111f09,0x12210a,0x12230b,0x12250b,0x13280c,0x132a0c,0x132c0d,0x142e0d,0x14310e,0x14330e,0x14360e,0x15380e,0x153a0e,0x153d0e,0x153f0e,0x15420e,0x15440e,0x15470e,0x15490e,0x154c0e,0x154e0d,0x15510d,0x14530d,0x14560d,0x14580c,0x135b0c,0x135d0b,0x12600b,0x12630a,0x11650a,0x106809,0x0f6a08,0x0e6d07,0x0c7006,0x0b7206,0x097505,0x077804,0x057b02,0x027d01,0x008000}
        };
        
        scalar -= center;
        scalar /= halfScale;
        
        switch(opt) {
            case 0:
                if(scalar >= 1.0) {
                    splitRGB(colorMap[0][99], rgb);
                }
                else if(scalar >= 0.0){
                    splitRGB(colorMap[0][static_cast<int>(scalar * 100.0)], rgb);
                }
                else if(scalar > -1.0) {
                    splitRGB(colorMap[1][static_cast<int>(scalar * -100.0)], rgb);
                }
                else { // scalar <= -1.0
                    splitRGB(colorMap[1][99], rgb);
                }
                break;
                
            case 1:
                splitRGB(colorMap[2][std::max(0, std::min(99, static_cast<int>(scalar * 100.0)))], rgb);
                break;
                
            default:
                assert(0 && "the given color option is not implemented!");
                break;
        }
    }
    void IglUtils::mapScalarToColor_bin(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double thres)
    {
        assert(thres > 0.0);
        color.resize(scalar.size(), 3);
        for(int elemI = 0; elemI < scalar.size(); elemI++)
        {
            if(scalar[elemI] < 0.0) {
                // boundary edge
                color.row(elemI) = Eigen::RowVector3d(0.0, 0.2, 0.8);
            }
            else {
                const double s = ((scalar[elemI] > thres) ? 1.0 : 0.0);
                color.row(elemI) = Eigen::RowVector3d(1.0 - s, 1.0 - s, 1.0 - s);
            }
        }
    }
    void IglUtils::mapScalarToColor(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color,
                                    double lowerBound, double upperBound, int opt)
    {
        double center = 0.0, halfScale = 0.0;
        switch (opt) {
            case 0:
                center = (upperBound + lowerBound) / 2.0;
                halfScale = (upperBound - lowerBound) / 2.0;
                break;
                
            case 1:
                center = lowerBound;
                halfScale = upperBound - lowerBound;
                break;
                
            default:
                assert(0 && "the given color option is not implemented!");
                break;
        }
        
        color.resize(scalar.size(), 3);
        for(int elemI = 0; elemI < scalar.size(); elemI++) {
            double rgb[3];
            getColor(scalar[elemI], rgb, center, halfScale, opt);
            color.row(elemI) << rgb[0], rgb[1], rgb[2];
        }
    }
    
    void IglUtils::addBlockToMatrix(Eigen::SparseMatrix<double>& mtr, const Eigen::MatrixXd& block,
                                 const Eigen::VectorXi& index, int dim)
    {
        assert(block.rows() == block.cols());
        assert(index.size() * dim == block.rows());
        assert(mtr.rows() == mtr.cols());
        assert(index.maxCoeff() * dim + dim - 1 < mtr.rows());
        
        for(int indI = 0; indI < index.size(); indI++) {
            if(index[indI] < 0) {
                continue;
            }
            int startIndI = index[indI] * dim;
            int startIndI_block = indI * dim;
            
            for(int indJ = 0; indJ < index.size(); indJ++) {
                if(index[indJ] < 0) {
                    continue;
                }
                int startIndJ = index[indJ] * dim;
                int startIndJ_block = indJ * dim;
                
                for(int dimI = 0; dimI < dim; dimI++) {
                    for(int dimJ = 0; dimJ < dim; dimJ++) {
                        mtr.coeffRef(startIndI + dimI, startIndJ + dimJ) += block(startIndI_block + dimI, startIndJ_block + dimJ);
                    }
                }
            }
        }
    }
    
    void IglUtils::addDiagonalToMatrix(const Eigen::VectorXd& diagonal, const Eigen::VectorXi& index, int dim,
                                    Eigen::VectorXd* V, Eigen::VectorXi* I, Eigen::VectorXi* J)
    {
        assert(index.size() * dim == diagonal.size());
        
        assert(V);
        int tripletInd = static_cast<int>(V->size());
        const int entryAmt = static_cast<int>(diagonal.size());
        V->conservativeResize(tripletInd + entryAmt);
        if(I) {
            assert(J);
            assert(I->size() == tripletInd);
            assert(J->size() == tripletInd);
            I->conservativeResize(tripletInd + entryAmt);
            J->conservativeResize(tripletInd + entryAmt);
        }
        
        for(int indI = 0; indI < index.size(); indI++) {
            if(index[indI] < 0) {
                assert(0 && "currently doesn't support fixed vertices here!");
                continue;
            }
            int startIndI = index[indI] * dim;
            int startIndI_diagonal = indI * dim;
            
            for(int dimI = 0; dimI < dim; dimI++) {
                (*V)[tripletInd] = diagonal(startIndI_diagonal + dimI);
                if(I) {
                    (*I)[tripletInd] = (*J)[tripletInd] = startIndI + dimI;
                }
                tripletInd++;
            }
        }
    }
    
    void IglUtils::writeSparseMatrixToFile(const std::string& filePath,
                                           const Eigen::VectorXi& I, const Eigen::VectorXi& J,
                                           const Eigen::VectorXd& V, bool MATLAB)
    {
        assert(I.size() == J.size());
        assert(V.size() == I.size());
        
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                out << I.maxCoeff() + 1 << " " << J.maxCoeff() + 1 << " " << I.size() << std::endl;
            }
            for (int k = 0; k < I.size(); k++) {
                out << I[k] + MATLAB << " " << J[k] + MATLAB << " " << V[k] << std::endl;
            }
            out.close();
        }
        else {
            std::cout << "writeSparseMatrixToFile failed! file open error!" << std::endl;
        }

    }
    
    void IglUtils::writeSparseMatrixToFile(const std::string& filePath, const Eigen::SparseMatrix<double>& mtr, bool MATLAB)
    {
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                out << mtr.rows() << " " << mtr.cols() << " " << mtr.nonZeros() << std::endl;
            }
            for (int k = 0; k < mtr.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(mtr, k); it; ++it)
                {
                    out << it.row() + MATLAB << " " << it.col() + MATLAB << " " << it.value() << std::endl;
                }
            }
            out.close();
        }
        else {
            std::cout << "writeSparseMatrixToFile failed! file open error!" << std::endl;
        }
    }
    
    void IglUtils::writeSparseMatrixToFile(const std::string& filePath,
                                           const std::map<std::pair<int, int>, double>& mtr,
                                           bool MATLAB)
    {
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                int mtrSize = mtr.rbegin()->first.first + 1;
                out << mtrSize << " " << mtrSize << " " << mtr.size() << std::endl;
            }
            for(const auto& entryI : mtr) {
                out << entryI.first.first + MATLAB << " "
                    << entryI.first.second + MATLAB << " "
                    << entryI.second << std::endl;
            }
            out.close();
        }
        else {
            std::cout << "writeSparseMatrixToFile failed! file open error!" << std::endl;
        }
    }
    
    void IglUtils::writeSparseMatrixToFile(const std::string& filePath,
                                           LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver,
                                           bool MATLAB)
    {
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                out << linSysSolver->getNumRows() << " " <<
                    linSysSolver->getNumRows() << " " <<
                    linSysSolver->getNumNonzeros() << std::endl;
            }
            for(int rowI = 0; rowI < linSysSolver->getNumRows(); rowI++) {
                for(const auto& colIter : linSysSolver->getIJ2aI()[rowI]) {
                    out << rowI + MATLAB << " "
                        << colIter.first + MATLAB << " "
                        << linSysSolver->coeffMtr(rowI, colIter.first) << std::endl;
                }
            }
            out.close();
        }
        else {
            std::cout << "writeSparseMatrixToFile failed! file open error!" << std::endl;
        }
    }
    
    void IglUtils::loadSparseMatrixFromFile(const std::string& filePath, Eigen::SparseMatrix<double>& mtr)
    {
        std::ifstream in;
        in.open(filePath);
        if(in.is_open()) {
            int rows, cols, nonZeroAmt;
            in >> rows >> cols >> nonZeroAmt;
            mtr.resize(rows, cols);
            std::vector<Eigen::Triplet<double>> IJV;
            IJV.reserve(nonZeroAmt);
            int i, j;
            double v;
            for(int nzI = 0; nzI < nonZeroAmt; nzI++) {
                assert(!in.eof());
                in >> i >> j >> v;
                IJV.emplace_back(Eigen::Triplet<double>(i, j, v));
            }
            in.close();
            mtr.setFromTriplets(IJV.begin(), IJV.end());
        }
        else {
            std::cout << "loadSparseMatrixToFile failed! file open error!" << std::endl;
        }
    }
    
    void IglUtils::sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr,
                                         Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V)
    {
        I.resize(mtr.nonZeros());
        J.resize(mtr.nonZeros());
        V.resize(mtr.nonZeros());
        int entryI = 0;
        for (int k = 0; k < mtr.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mtr, k); it; ++it)
            {
                I[entryI] = static_cast<int>(it.row());
                J[entryI] = static_cast<int>(it.col());
                V[entryI] = it.value();
                entryI++;
            }
        }
    }
    
    void IglUtils::sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& V)
    {
        V.resize(mtr.nonZeros());
        int entryI = 0;
        for (int k = 0; k < mtr.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mtr, k); it; ++it)
            {
                V[entryI++] = it.value();
            }
        }
    }
    
    const std::string IglUtils::rtos(double real)
    {
        std::string str_real = std::to_string(real);
        size_t pointPos = str_real.find_last_of('.');
        if(pointPos == std::string::npos) {
            return str_real;
        }
        else {
            const char* str = str_real.c_str();
            size_t cI = str_real.length() - 1;
            while((cI > pointPos) && (str[cI] == '0')) {
                cI--;
            }
            
            if(cI == pointPos) {
                return str_real.substr(0, pointPos);
            }
            else {
                return str_real.substr(0, cI + 1);
            }
        }
    }
    
    void IglUtils::differentiate_normalize(const Eigen::Vector2d& var, Eigen::Matrix2d& deriv)
    {
        const double x2 = var[0] * var[0];
        const double y2 = var[1] * var[1];
        const double mxy = -var[0] * var[1];
        deriv << y2, mxy,
            mxy, x2;
        deriv /= std::pow(x2 + y2, 1.5);
    }
    
    void IglUtils::differentiate_xxT(const Eigen::Vector2d& var, Eigen::Matrix<Eigen::RowVector2d, 2, 2>& deriv,
                                     double param)
    {
        deriv(0, 0) = param * Eigen::RowVector2d(2 * var[0], 0.0);
        deriv(0, 1) = param * Eigen::RowVector2d(var[1], var[0]);
        deriv(1, 0) = param * Eigen::RowVector2d(var[1], var[0]);
        deriv(1, 1) = param * Eigen::RowVector2d(0.0, 2 * var[1]);
    }
    
    double IglUtils::computeRotAngle(const Eigen::RowVector2d& from, const Eigen::RowVector2d& to)
    {
        double angle = std::acos(std::max(-1.0, std::min(1.0, from.dot(to) / from.norm() / to.norm())));
        return ((from[0] * to[1] - from[1] * to[0] < 0.0) ? -angle : angle);
    }
    
    /////////////////////////////////////////////////////////////////
    // 2D line segments intersection checking code
    // based on Real-Time Collision Detection by Christer Ericson
    // (Morgan Kaufmaan Publishers, 2005 Elvesier Inc)
    double Signed2DTriArea(const Eigen::RowVector2d& a, const Eigen::RowVector2d& b, const Eigen::RowVector2d& c)
    {
        return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
    }
    
    bool IglUtils::Test2DSegmentSegment(const Eigen::RowVector2d& a, const Eigen::RowVector2d& b,
                                        const Eigen::RowVector2d& c, const Eigen::RowVector2d& d,
                                        double eps)
    {
        double eps_quad = 0.0, eps_sq = 0.0;
        if(eps) {
            eps = std::abs(eps);
            eps_sq = eps * eps * ((a-b).squaredNorm() + (c-d).squaredNorm()) / 2.0;
            eps_quad = eps_sq * eps_sq;
        }
        
        // signs of areas correspond to which side of ab points c and d are
        double a1 = Signed2DTriArea(a,b,d); // Compute winding of abd (+ or -)
        double a2 = Signed2DTriArea(a,b,c); // To intersect, must have sign opposite of a1
        
        // If c and d are on different sides of ab, areas have different signs
        if( a1 * a2 <= eps_quad ) // require unsigned x & y values.
        {
            double a3 = Signed2DTriArea(c,d,a); // Compute winding of cda (+ or -)
            double a4 = a3 + a2 - a1; // Since area is constant a1 - a2 = a3 - a4, or a4 = a3 + a2 - a1
            
            // Points a and b on different sides of cd if areas have different signs
            if( a3 * a4 <= eps_quad )
            {
                if((std::abs(a1) <= eps_sq) && (std::abs(a2) <= eps_sq)) {
                    // colinear
                    const Eigen::RowVector2d ab = b - a;
                    const double sqnorm_ab = ab.squaredNorm();
                    const Eigen::RowVector2d ac = c - a;
                    const Eigen::RowVector2d ad = d - a;
                    double coef_c = ac.dot(ab) / sqnorm_ab;
                    double coef_d = ad.dot(ab) / sqnorm_ab;
                    assert(coef_c != coef_d);
                    
                    if(coef_c > coef_d) {
                        std::swap(coef_c, coef_d);
                    }
                    
                    if((coef_c > 1.0 + eps) || (coef_d < -eps)) {
                        return false;
                    }
                    else {
                        return true;
                    }
                }
                else {
                    // Segments intersect.
                    return true;
                }
            }
        }
        
        // Segments not intersecting
        return false;
    }
    ////////////////////////////////////////////////////////////
    
    void IglUtils::addThickEdge(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& UV,
                                Eigen::MatrixXd& seamColor, const Eigen::RowVector3d& color,
                                const Eigen::RowVector3d& v0, const Eigen::RowVector3d& v1,
                                double halfWidth, double texScale,
                                bool UVorSurface, const Eigen::RowVector3d& normal)
    {
        if(UVorSurface) {
            const Eigen::RowVector3d e = v1 - v0;
            const Eigen::RowVector3d n = normal.normalized() * halfWidth;
            const Eigen::RowVector3d bn = e.cross(normal).normalized() * halfWidth;
            const int vAmt_old = V.rows();
            V.conservativeResize(V.rows() + 8, 3);
            V.bottomRows(8) << v0 - n - bn, v0 - n + bn, v0 + n + bn, v0 + n - bn,
                v1 - n - bn, v1 - n + bn, v1 + n + bn, v1 + n - bn;
            UV.conservativeResize(UV.rows() + 8, 2);
            UV.bottomRows(8) = Eigen::MatrixXd::Ones(8, 2) * texScale;
            F.conservativeResize(F.rows() + 6, 3);
            F.bottomRows(6) << vAmt_old + 2, vAmt_old + 1, vAmt_old + 5,
                vAmt_old + 2, vAmt_old + 5, vAmt_old + 6,
                vAmt_old + 3, vAmt_old + 2, vAmt_old + 6,
                vAmt_old + 3, vAmt_old + 6, vAmt_old + 7,
                vAmt_old, vAmt_old + 3, vAmt_old + 7,
                vAmt_old, vAmt_old + 7, vAmt_old + 4;
            seamColor.conservativeResize(seamColor.rows() + 6, 3);
            seamColor.bottomRows(6) << color, color, color, color, color, color;
        }
        else {
            const Eigen::RowVector3d e = v1 - v0;
            const Eigen::RowVector3d n = halfWidth * Eigen::RowVector3d(-e[1], e[0], 0.0).normalized();
            const int vAmt_old = V.rows();
            V.conservativeResize(V.rows() + 4, 3);
            V.bottomRows(4) << v0 - n, v0 + n, v1 + n, v1 - n;
            V.bottomRows(4).col(2) = Eigen::VectorXd::Ones(4) * halfWidth; // for depth test
            F.conservativeResize(F.rows() + 2, 3);
            F.bottomRows(2) << vAmt_old, vAmt_old + 1, vAmt_old + 2,
                vAmt_old, vAmt_old + 2, vAmt_old + 3;
            seamColor.conservativeResize(seamColor.rows() + 2, 3);
            seamColor.bottomRows(2) << color, color;
        }
    }
    
    void IglUtils::saveMesh_Seamster(const std::string& filePath,
                                     const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
    {
        assert((V.rows() > 0) && (V.cols() == 3));
        assert((F.rows() > 0) && (F.cols() == 3));
        
        std::ofstream outFile;
        outFile.open(filePath);
        assert(outFile.is_open());
        
        outFile << V.rows() << " " << F.rows() << std::endl;
        for(int vI = 0; vI < V.rows(); vI++) {
            outFile << vI + 1 << " " << V(vI, 0) << " " << V(vI, 1) << " " << V(vI, 2) << std::endl;
        }
        for(int triI = 0; triI < F.rows(); triI++) {
            outFile << triI + 1 << " " <<
                F(triI, 0) + 1 << " " << F(triI, 1) + 1 << " " << F(triI, 2) + 1 << std::endl;
        }
        
        outFile.close();
    }
    
    void IglUtils::saveTetMesh(const std::string& filePath,
                               const Eigen::MatrixXd& TV, const Eigen::MatrixXi& TT,
                               const Eigen::MatrixXi& F)
    {
        assert(TV.rows() > 0);
        assert(TV.cols() == 3);
        assert(TT.rows() > 0);
        assert(TT.cols() == 4);
        if(F.rows() > 0) {
            assert(F.cols() == 3);
        }
        
        FILE *out = fopen(filePath.c_str(), "w");
        assert(out);
        
        fprintf(out, "$MeshFormat\n4 0 8\n$EndMeshFormat\n");
        
        fprintf(out, "$Entities\n0 0 0 1\n");
        fprintf(out, "0 %le %le %le %le %le %le 0 0\n$EndEntities\n",
                TV.col(0).minCoeff(), TV.col(1).minCoeff(), TV.col(2).minCoeff(),
                TV.col(0).maxCoeff(), TV.col(1).maxCoeff(), TV.col(2).maxCoeff());
        
        fprintf(out, "$Nodes\n1 %lu\n0 3 0 %lu\n", TV.rows(), TV.rows());
        for(int vI = 0; vI < TV.rows(); vI++) {
            const Eigen::RowVector3d& v = TV.row(vI);
            fprintf(out, "%d %le %le %le\n", vI + 1, v[0], v[1], v[2]);
        }
        fprintf(out, "$EndNodes\n");
        
        fprintf(out, "$Elements\n1 %lu\n0 3 4 %lu\n", TT.rows(), TT.rows());
        for(int elemI = 0; elemI < TT.rows(); elemI++) {
            const Eigen::RowVector4i& tetVInd = TT.row(elemI);
            fprintf(out, "%d %d %d %d %d\n", elemI + 1,
                    tetVInd[0] + 1, tetVInd[1] + 1, tetVInd[2] + 1, tetVInd[3] + 1);
        }
        fprintf(out, "$EndElements\n");
        
        fprintf(out, "$Surface\n");
        fprintf(out, "%lu\n", F.rows());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            fprintf(out, "%d %d %d\n", triVInd[0] + 1, triVInd[1] + 1, triVInd[2] + 1);
        }
        fprintf(out, "$EndSurface\n");
        
        fclose(out);
    }
    void IglUtils::readTetMesh(const std::string& filePath,
                               Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                               Eigen::MatrixXi& F)
    {
        FILE *in = fopen(filePath.c_str(), "r");
        assert(in);
        
        char buf[BUFSIZ];
        while(fgets(buf, BUFSIZ, in)) {
            if(strncmp("$Nodes", buf, 6) == 0) {
                fgets(buf, BUFSIZ, in);
                int vAmt;
                sscanf(buf, "1 %d", &vAmt);
                TV.resize(vAmt, 3);
                fgets(buf, BUFSIZ, in);
                break;
            }
        }
        int bypass;
        for(int vI = 0; vI < TV.rows(); vI++) {
            fscanf(in, "%d %le %le %le\n", &bypass, &TV(vI, 0), &TV(vI, 1), &TV(vI, 2));
        }
        
        while(fgets(buf, BUFSIZ, in)) {
            if(strncmp("$Elements", buf, 9) == 0) {
                fgets(buf, BUFSIZ, in);
                int elemAmt;
                sscanf(buf, "1 %d", &elemAmt);
                TT.resize(elemAmt, 4);
                fgets(buf, BUFSIZ, in);
                break;
            }
        }
        for(int elemI = 0; elemI < TT.rows(); elemI++) {
            fscanf(in, "%d %d %d %d %d\n", &bypass,
                    &TT(elemI, 0), &TT(elemI, 1), &TT(elemI, 2), &TT(elemI, 3));
        }
        TT.array() -= 1;
        
        while(fgets(buf, BUFSIZ, in)) {
            if(strncmp("$Surface", buf, 7) == 0) {
                fgets(buf, BUFSIZ, in);
                int elemAmt;
                sscanf(buf, "%d", &elemAmt);
                F.resize(elemAmt, 3);
                break;
            }
        }
        for(int triI = 0; triI < F.rows(); triI++) {
            fscanf(in, "%d %d %d\n", &F(triI, 0), &F(triI, 1), &F(triI, 2));
        }
        if(F.rows() > 0) {
            F.array() -= 1;
        }
        else {
            //TODO: find surface triangles
        }
        
        std::cout << "tet mesh loaded with " << TV.rows() << " nodes, "
            << TT.rows() << " tets, and " << F.rows() << " surface tris." << std::endl;
        
        fclose(in);
    }
    
    void IglUtils::smoothVertField(const TriangleSoup<DIM>& mesh, Eigen::VectorXd& field)
    {
        assert(field.size() == mesh.V.rows());
        Eigen::VectorXd field_copy = field;
        for(int vI = 0; vI < field.size(); vI++) {
            for(const auto nbVI : mesh.vNeighbor[vI]) {
                field[vI] += field_copy[nbVI];
            }
            field[vI] /= mesh.vNeighbor[vI].size() + 1;
        }
    }
    
    void IglUtils::compute_dsigma_div_dx(const AutoFlipSVD<Eigen::Matrix<double, DIM, DIM>>& svd,
                                         const Eigen::Matrix<double, DIM, DIM>& A,
                                         Eigen::Matrix<double, DIM * (DIM + 1), DIM>& dsigma_div_dx)
    {
        for(int dimI = 0; dimI < DIM; dimI++) {
            Eigen::Matrix<double, DIM, DIM> dsigma_div_dF = svd.matrixU().col(dimI) * svd.matrixV().col(dimI).transpose();
            Eigen::Matrix<double, DIM * (DIM + 1), 1> result;
            IglUtils::dF_div_dx_mult(dsigma_div_dF, A, result);
            dsigma_div_dx.col(dimI) = result;
        }
    }
    
    void IglUtils::compute_dU_and_dV_div_dF(const AutoFlipSVD<Eigen::Matrix<double, DIM, DIM>>& svd,
                                            Eigen::Matrix<double, DIM * DIM, DIM * DIM>& dU_div_dF,
                                            Eigen::Matrix<double, DIM * DIM, DIM * DIM>& dV_div_dF)
    {
        assert(DIM == 2);
#if(DIM == 2)
        Eigen::Matrix2d coefMtr;
        coefMtr << svd.singularValues()[0], svd.singularValues()[1],
            svd.singularValues()[1], svd.singularValues()[0];
        const Eigen::LDLT<Eigen::Matrix2d>& solver = coefMtr.ldlt();
        assert(solver.info() == Eigen::Success);
        
        Eigen::Vector2d b;
        for(int rowI = 0; rowI < 2; rowI++) {
            for(int colI = 0; colI < 2; colI++) {
                b << svd.matrixU()(rowI, 1) * svd.matrixV()(colI, 0),
                    -svd.matrixU()(rowI, 0) * svd.matrixV()(colI, 1);
                const Eigen::Vector2d wij21 = solver.solve(b);
                
                dU_div_dF.block(rowI * 2 + colI, 0, 1, 2) = wij21[0] * svd.matrixU().col(1).transpose();
                dU_div_dF.block(rowI * 2 + colI, 2, 1, 2) = -wij21[0] * svd.matrixU().col(0).transpose();
                
                dV_div_dF.block(rowI * 2 + colI, 0, 1, 2) = -wij21[1] * svd.matrixV().col(1).transpose();
                dV_div_dF.block(rowI * 2 + colI, 2, 1, 2) = wij21[1] * svd.matrixV().col(0).transpose();
            }
        }
#endif
    }
    
    void IglUtils::compute_d2sigma_div_dF2(const AutoFlipSVD<Eigen::Matrix<double, DIM, DIM>>& svd,
                                           Eigen::Matrix<double, DIM * DIM, DIM * DIM * DIM>& d2sigma_div_dF2)
    {
        Eigen::Matrix<double, DIM * DIM, DIM * DIM> dU_div_dF, dV_div_dF;
        compute_dU_and_dV_div_dF(svd, dU_div_dF, dV_div_dF);
        
        for(int sigmaI = 0; sigmaI < DIM; sigmaI++) {
            for(int Fij = 0; Fij < DIM * DIM; Fij++) {
                const Eigen::Matrix<double, DIM, DIM>& d2sigma_div_dF2ij = dU_div_dF.block(Fij, sigmaI * DIM, 1, DIM).transpose() * svd.matrixV().col(sigmaI).transpose() +
                    svd.matrixU().col(sigmaI) * dV_div_dF.block(Fij, sigmaI * DIM, 1, DIM);
                
                d2sigma_div_dF2.block(Fij, sigmaI * DIM * DIM, 1, DIM) = d2sigma_div_dF2ij.row(0);
                d2sigma_div_dF2.block(Fij, sigmaI * DIM * DIM + DIM, 1, DIM) = d2sigma_div_dF2ij.row(1);
                if(DIM == 3) {
                    d2sigma_div_dF2.block(Fij, sigmaI * DIM * DIM + DIM * 2, 1, DIM) = d2sigma_div_dF2ij.row(2);
                }
            }
        }
    }
    
    void IglUtils::compute_d2sigma_div_dx2(const AutoFlipSVD<Eigen::Matrix<double, DIM, DIM>>& svd,
                                           const Eigen::Matrix<double, DIM, DIM>& A,
                                           Eigen::Matrix<double, DIM * (DIM + 1), DIM * (DIM + 1) * DIM>& d2sigma_div_dx2)
    {
        Eigen::Matrix<double, DIM * DIM, DIM * DIM * DIM> d2sigma_div_dF2;
        compute_d2sigma_div_dF2(svd, d2sigma_div_dF2);
        
        Eigen::Matrix<double, DIM * (DIM + 1), DIM * DIM> dF_div_dx;
        compute_dF_div_dx(A, dF_div_dx);
        
        for(int sigmaI = 0; sigmaI < DIM; sigmaI++) {
            for(int xI = 0; xI < DIM + 1; xI++) {
                for(int xJ = 0; xJ < DIM + 1; xJ++) {
                    d2sigma_div_dx2.block(xJ * DIM, sigmaI * DIM * (DIM + 1) + xI * DIM, DIM, DIM) = dF_div_dx.block(xJ * DIM, 0, DIM, DIM * DIM) *
                        d2sigma_div_dF2.block(0, DIM * DIM * sigmaI, DIM * DIM, DIM * DIM) *
                        dF_div_dx.block(xI * DIM, 0, DIM, DIM * DIM).transpose();
                }
            }
        }
    }
    
    void IglUtils::compute_dF_div_dx(const Eigen::Matrix<double, DIM, DIM>& A,
                                     Eigen::Matrix<double, DIM * (DIM + 1), DIM * DIM>& dF_div_dx)
    {
        assert(DIM == 2);
#if(DIM == 2)
        const double mA11mA21 = -A(0, 0) - A(1, 0);
        const double mA12mA22 = -A(0, 1) - A(1, 1);
        dF_div_dx <<
            mA11mA21, mA12mA22, 0.0, 0.0,
            0.0, 0.0, mA11mA21, mA12mA22,
            A(0, 0), A(0, 1), 0.0, 0.0,
            0.0, 0.0, A(0, 0), A(0, 1),
            A(1, 0), A(1, 1), 0.0, 0.0,
            0.0, 0.0, A(1, 0), A(1, 1);
#endif
    }
    void IglUtils::dF_div_dx_mult(const Eigen::Matrix<double, DIM, DIM>& right,
                                  const Eigen::Matrix<double, DIM, DIM>& A,
                                  Eigen::Matrix<double, DIM * (DIM + 1), 1>& result)
    {
#if(DIM == 2)
        const double _0000 = right(0, 0) * A(0, 0);
        const double _0010 = right(0, 0) * A(1, 0);
        const double _0101 = right(0, 1) * A(0, 1);
        const double _0111 = right(0, 1) * A(1, 1);
        const double _1000 = right(1, 0) * A(0, 0);
        const double _1010 = right(1, 0) * A(1, 0);
        const double _1101 = right(1, 1) * A(0, 1);
        const double _1111 = right(1, 1) * A(1, 1);
        
        result[2] = _0000 + _0101;
        result[3] = _1000 + _1101;
        result[4] = _0010 + _0111;
        result[5] = _1010 + _1111;
        result[0] = -result[2] - result[4];
        result[1] = -result[3] - result[5];
#else
        result[3] = A.row(0).dot(right.row(0));
        result[4] = A.row(0).dot(right.row(1));
        result[5] = A.row(0).dot(right.row(2));
        result[6] = A.row(1).dot(right.row(0));
        result[7] = A.row(1).dot(right.row(1));
        result[8] = A.row(1).dot(right.row(2));
        result[9] = A.row(2).dot(right.row(0));
        result[10] = A.row(2).dot(right.row(1));
        result[11] = A.row(2).dot(right.row(2));
        result[0] = - result[3] - result[6] - result[9];
        result[1] = - result[4] - result[7] - result[10];
        result[2] = - result[5] - result[8] - result[11];
#endif
    }
    
    void IglUtils::sampleSegment(const Eigen::RowVectorXd& vs,
                                 const Eigen::RowVectorXd& ve,
                                 double spacing,
                                 Eigen::MatrixXd& inBetween)
    {
        Eigen::RowVectorXd stepVec = (ve - vs);
        int segAmt = stepVec.norm() / spacing + 1;
        assert(segAmt > 1);
        stepVec /= segAmt;
        inBetween.resize(segAmt - 1, vs.size());
        for(int i = 0; i < segAmt - 1; i++) {
            inBetween.row(i) = vs + (i + 1) * stepVec;
        }
    }
    
    void IglUtils::findBorderVerts(const Eigen::MatrixXd& V,
                                   std::vector<std::vector<int>>& borderVerts)
    {
        // resize to match size
        Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
        Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
        Eigen::RowVectorXd range = topRight - bottomLeft;
        
        borderVerts.resize(2);
        for(int vI = 0; vI < V.rows(); vI++) {
            if(V(vI, 0) < bottomLeft[0] + range[0] / 100.0) {
                borderVerts[0].emplace_back(vI);
            }
            else if(V(vI, 0) > topRight[0] - range[0] / 100.0) {
                borderVerts[1].emplace_back(vI);
            }
        }
    }
    
}
