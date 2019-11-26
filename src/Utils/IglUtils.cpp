//
//  IglUtils.cpp
//  DOT
//
//  Created by Minchen Li on 8/30/17.
//

#include "IglUtils.hpp"
#include "Triplet.h"

#include "Timer.hpp"

#include <immintrin.h>
#include "SVD_EFTYCHIOS/PTHREAD_QUEUE.h"
#include "SVD_EFTYCHIOS/Singular_Value_Decomposition_Helper.h"

#include "SIMD_DOUBLE_MACROS.hpp"

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <set>

extern Timer timer_temp, timer_temp2;

extern double *a11,*a21,*a31,*a12,*a22,*a32,*a13,*a23,*a33;
extern double *u11,*u21,*u31,*u12,*u22,*u32,*u13,*u23,*u33;
extern double *v11,*v21,*v31,*v12,*v22,*v32,*v13,*v23,*v33;
extern double *sigma1,*sigma2,*sigma3;

namespace DOT {
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
            
//            // blue to black to green
//            {0x1e90ff,0x1f8df9,0x218af3,0x2287ee,0x2383e8,0x2480e2,0x247ddc,0x257ad7,0x2577d1,0x2674cb,0x2671c6,0x276ec0,0x276bbb,0x2768b5,0x2765b0,0x2762aa,0x275fa5,0x275ca0,0x26599a,0x265695,0x265490,0x26518a,0x254e85,0x254b80,0x24487b,0x244576,0x234371,0x22406c,0x223d67,0x213b62,0x20385d,0x1f3558,0x1f3353,0x1e304f,0x1d2d4a,0x1c2b45,0x1b2841,0x1a263c,0x182338,0x172133,0x161e2f,0x151c2a,0x141926,0x121722,0x11151e,0x0f121a,0x0c0f16,0x090b11,0x05070b,0x020204,0x010200,0x030601,0x050a02,0x060e03,0x081104,0x0a1405,0x0c1606,0x0d1807,0x0f1a08,0x101d09,0x111f09,0x12210a,0x12230b,0x12250b,0x13280c,0x132a0c,0x132c0d,0x142e0d,0x14310e,0x14330e,0x14360e,0x15380e,0x153a0e,0x153d0e,0x153f0e,0x15420e,0x15440e,0x15470e,0x15490e,0x154c0e,0x154e0d,0x15510d,0x14530d,0x14560d,0x14580c,0x135b0c,0x135d0b,0x12600b,0x12630a,0x11650a,0x106809,0x0f6a08,0x0e6d07,0x0c7006,0x0b7206,0x097505,0x077804,0x057b02,0x027d01,0x008000}
            
//            // light green to red
//            {0x00cc00,0x1bcb00,0x25c900,0x2cc900,0x33c800,0x3bc600,0x42c500,0x46c400,0x4ac300,0x4dc200,0x54c000,0x57be00,0x5abd00,0x5dbc00,0x62ba00,0x65b900,0x67b800,0x69b700,0x6eb600,0x70b500,0x72b300,0x74b200,0x78b100,0x7ab000,0x7baf00,0x7dad00,0x81ac00,0x82ab00,0x84aa00,0x87a700,0x88a700,0x8aa500,0x8da400,0x8ea300,0x90a200,0x929f00,0x949f00,0x959d00,0x979b00,0x989a00,0x9a9900,0x9c9700,0x9d9600,0x9f9500,0xa09400,0xa29100,0xa49000,0xa58f00,0xa68d00,0xa88c00,0xa98b00,0xab8800,0xac8700,0xad8600,0xaf8400,0xb08200,0xb18200,0xb28000,0xb47e00,0xb57d00,0xb67b00,0xb77a00,0xb97800,0xba7600,0xbb7400,0xbd7200,0xbd7000,0xbe6f00,0xbf6e00,0xc16b00,0xc26900,0xc36700,0xc46500,0xc56300,0xc66100,0xc76000,0xc85d00,0xc95b00,0xca5900,0xcb5700,0xcd5400,0xcd5300,0xce5000,0xcf4d00,0xd04a00,0xd14900,0xd24500,0xd34200,0xd34100,0xd43d00,0xd63900,0xd63600,0xd73200,0xd82e00,0xd92a00,0xda2400,0xda2100,0xdb1900,0xdc0f00,0xdd0000}
            
            // DeepSkyBlue, Green, Orange, Red
            {0x00bfff,0x0cbdf6,0x13baee,0x19b7e5,0x1bb5de,0x1eb3d3,0x20b0cc,0x21aec2,0x21acbb,0x21a9b2,0x20a7aa,0x20a4a3,0x1ea29a,0x1ca093,0x199c89,0x169b82,0x12987a,0x0c9572,0x05936b,0x029264,0x06915f,0x0a905b,0x0c8f57,0x0e8e52,0x108d4e,0x108b49,0x118a44,0x118a40,0x11883b,0x108836,0x108631,0x0e852b,0x0d8426,0x0b8420,0x088219,0x058211,0x038108,0x008000,0x1a8100,0x258300,0x338400,0x3a8600,0x438700,0x4c8800,0x528a00,0x588b00,0x608c00,0x678e00,0x6c8e00,0x719000,0x779100,0x7e9200,0x859300,0x8a9400,0x919500,0x979600,0x9c9700,0xa29800,0xa79900,0xad9a00,0xb29b00,0xb79c00,0xbc9d00,0xc19e00,0xc89f00,0xce9f00,0xd3a000,0xd7a100,0xdea100,0xe4a200,0xe9a300,0xf0a300,0xf4a400,0xfaa400,0xffa500,0xffa500,0xffa100,0xff9d00,0xff9900,0xff9400,0xff9000,0xff8d00,0xff8700,0xff8400,0xff7e00,0xff7a00,0xff7600,0xff7100,0xff6b00,0xff6500,0xff6100,0xff5b00,0xff5500,0xff4e00,0xff4700,0xff3f00,0xff3600,0xff2a00,0xff1d00,0xff0000}
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
    void IglUtils::writeDenseMatrixToFile(const std::string& filePath,
                                          const Eigen::MatrixXd& matrix,
                                          bool MATLAB)
    {
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                out << matrix.rows() << " " <<
                    matrix.cols() << " " <<
                    matrix.rows() * matrix.cols() << std::endl;
            }
            for(int rowI = 0; rowI < matrix.rows(); rowI++) {
                for(int colI = 0; colI < matrix.cols(); colI++) {
                    out << rowI + MATLAB << " "
                        << colI + MATLAB << " "
                        << matrix(rowI, colI) << std::endl;
                }
            }
            out.close();
        }
        else {
            std::cout << "writeDenseMatrixToFile failed! file open error!" << std::endl;
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
    
    void IglUtils::writeVectorToFile(const std::string& filePath,
                                     const Eigen::VectorXd& vec)
    {
        FILE *out = fopen(filePath.c_str(), "w");
        assert(out);
        for(int dI = 0; dI < vec.size(); ++dI) {
            fprintf(out, "%le\n", vec[dI]);
        }
        fclose(out);
    }
    void IglUtils::readVectorFromFile(const std::string& filePath,
                                      Eigen::VectorXd& vec)
    {
        FILE *in = fopen(filePath.c_str(), "r");
        assert(in);
        
        vec.resize(0);
        char buf[BUFSIZ];
        while(!feof(in) && fgets(buf, BUFSIZ, in)) {
            vec.conservativeResize(vec.size() + 1);
            sscanf(buf, "%le", &vec[vec.size() - 1]);
        }
        
        fclose(in);
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
    
    void IglUtils::findSurfaceTris(const Eigen::MatrixXi& TT, Eigen::MatrixXi& F)
    {
        //TODO: merge with below
        std::map<Triplet, int> tri2Tet;
        for(int elemI = 0; elemI < TT.rows(); elemI++) {
            const Eigen::RowVector4i& elemVInd = TT.row(elemI);
            tri2Tet[Triplet(elemVInd[0], elemVInd[2], elemVInd[1])] = elemI;
            tri2Tet[Triplet(elemVInd[0], elemVInd[3], elemVInd[2])] = elemI;
            tri2Tet[Triplet(elemVInd[0], elemVInd[1], elemVInd[3])] = elemI;
            tri2Tet[Triplet(elemVInd[1], elemVInd[2], elemVInd[3])] = elemI;
        }
        
        //TODO: parallelize
        F.conservativeResize(0, 3);
        for(const auto& triI : tri2Tet) {
            const int* triVInd = triI.first.key;
            // find dual triangle with reversed indices:
            auto finder = tri2Tet.find(Triplet(triVInd[2], triVInd[1], triVInd[0]));
            if(finder == tri2Tet.end()) {
                finder = tri2Tet.find(Triplet(triVInd[1], triVInd[0], triVInd[2]));
                if(finder == tri2Tet.end()) {
                    finder = tri2Tet.find(Triplet(triVInd[0], triVInd[2], triVInd[1]));
                    if(finder == tri2Tet.end()) {
                        int oldSize = F.rows();
                        F.conservativeResize(oldSize + 1, 3);
                        F(oldSize, 0) = triVInd[0];
                        F(oldSize, 1) = triVInd[1];
                        F(oldSize, 2) = triVInd[2];
                    }
                }
            }
        }
    }
    void IglUtils::buildSTri2Tet(const Eigen::MatrixXi& F, const Eigen::MatrixXi& SF,
                                 std::vector<int>& sTri2Tet)
    {
        //TODO: merge with above
        std::map<Triplet, int> tri2Tet;
        for(int elemI = 0; elemI < F.rows(); elemI++) {
            const Eigen::RowVector4i& elemVInd = F.row(elemI);
            tri2Tet[Triplet(elemVInd[0], elemVInd[2], elemVInd[1])] = elemI;
            tri2Tet[Triplet(elemVInd[0], elemVInd[3], elemVInd[2])] = elemI;
            tri2Tet[Triplet(elemVInd[0], elemVInd[1], elemVInd[3])] = elemI;
            tri2Tet[Triplet(elemVInd[1], elemVInd[2], elemVInd[3])] = elemI;
        }
        
        sTri2Tet.resize(SF.rows());
#ifdef USE_TBB
        tbb::parallel_for(0, (int)SF.rows(), 1, [&](int triI)
#else
        for(int triI = 0; triI < SF.rows(); triI++)
#endif
        {
            const Eigen::RowVector3i& triVInd = SF.row(triI);
            auto finder = tri2Tet.find(Triplet(triVInd.data()));
            if(finder == tri2Tet.end()) {
                finder = tri2Tet.find(Triplet(triVInd[1], triVInd[2], triVInd[0]));
                if(finder == tri2Tet.end()) {
                    finder = tri2Tet.find(Triplet(triVInd[2], triVInd[0], triVInd[1]));
                    assert(finder != tri2Tet.end());
                }
            }
            sTri2Tet[triI] = finder->second;
        }
#ifdef USE_TBB
        );
#endif
    }
    
    void IglUtils::saveTetMesh(const std::string& filePath,
                               const Eigen::MatrixXd& TV, const Eigen::MatrixXi& TT,
                               const Eigen::MatrixXi& p_F, bool findSurface)
    {
        assert(TV.rows() > 0);
        assert(TV.cols() == 3);
        assert(TT.rows() > 0);
        assert(TT.cols() == 4);
        
        Eigen::MatrixXi F_found;
        if(p_F.rows() > 0) {
            assert(p_F.cols() == 3);
        }
        else if(findSurface) {
            findSurfaceTris(TT, F_found);
        }
        const Eigen::MatrixXi& F = ((p_F.rows() > 0) ? p_F : F_found);
        
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
    bool IglUtils::readTetMesh(const std::string& filePath,
                               Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                               Eigen::MatrixXi& F, bool findSurface)
    {
        FILE *in = fopen(filePath.c_str(), "r");
        if(!in) {
            return false;
        }
        
        char buf[BUFSIZ];
        while((!feof(in)) && fgets(buf, BUFSIZ, in)) {
            if(strncmp("$Nodes", buf, 6) == 0) {
                fgets(buf, BUFSIZ, in);
                int vAmt;
                sscanf(buf, "1 %d", &vAmt);
                TV.resize(vAmt, 3);
                fgets(buf, BUFSIZ, in);
                break;
            }
        }
        assert(TV.rows() > 0);
        int bypass;
        for(int vI = 0; vI < TV.rows(); vI++) {
            fscanf(in, "%d %le %le %le\n", &bypass, &TV(vI, 0), &TV(vI, 1), &TV(vI, 2));
        }
        
        while((!feof(in)) && fgets(buf, BUFSIZ, in)) {
            if(strncmp("$Elements", buf, 9) == 0) {
                fgets(buf, BUFSIZ, in);
                int elemAmt;
                sscanf(buf, "1 %d", &elemAmt);
                TT.resize(elemAmt, 4);
                fgets(buf, BUFSIZ, in);
                break;
            }
        }
        assert(TT.rows() > 0);
        for(int elemI = 0; elemI < TT.rows(); elemI++) {
            fscanf(in, "%d %d %d %d %d\n", &bypass,
                    &TT(elemI, 0), &TT(elemI, 1), &TT(elemI, 2), &TT(elemI, 3));
        }
        TT.array() -= 1;
        
        while((!feof(in)) && fgets(buf, BUFSIZ, in)) {
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
        else if(findSurface) {
            // if no surface triangles information provided, then find
            findSurfaceTris(TT, F);
        }
        
        std::cout << "tet mesh loaded with " << TV.rows() << " nodes, "
            << TT.rows() << " tets, and " << F.rows() << " surface tris." << std::endl;
        
        fclose(in);
        
        return true;
    }
    void IglUtils::readNodeEle(const std::string& filePath,
                               Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                               Eigen::MatrixXi& F)
    {
        FILE *in = fopen((filePath + ".node").c_str(), "r");
        assert(in);
        
        int nN, nDim;
        fscanf(in, "%d %d 0 0", &nN, &nDim);
        std::cout << nN << " " << nDim << std::endl;
        assert(nN >= 4);
        assert(nDim == 3);
        
        int bypass;
        TV.conservativeResize(nN, nDim);
        for(int vI = 0; vI < nN; ++vI) {
            fscanf(in, "%d %le %le %le", &bypass,
                   &TV(vI, 0), &TV(vI, 1), &TV(vI, 2));
        }
        
        fclose(in);
        
        in = fopen((filePath + ".ele").c_str(), "r");
        assert(in);
        
        int nE, nDimp1;
        fscanf(in, "%d %d 0", &nE, &nDimp1);
        std::cout << nE << " " << nDimp1 << std::endl;
        assert(nE >= 0);
        assert(nDimp1 == 4);
        
        TT.conservativeResize(nE, nDimp1);
        for(int tI = 0; tI < nE; ++tI) {
            fscanf(in, "%d %d %d %d %d", &bypass,
                   &TT(tI, 0), &TT(tI, 1), &TT(tI, 2),  &TT(tI, 3));
        }
        
        fclose(in);
        
        findSurfaceTris(TT, F);
        
        std::cout << "tet mesh loaded with " << TV.rows() << " nodes, "
            << TT.rows() << " tets, and " << F.rows() << " surface tris." << std::endl;
    }
    
    void IglUtils::smoothVertField(const Mesh<DIM>& mesh, Eigen::VectorXd& field)
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
    
    void IglUtils::compute_dF_div_dx(const Eigen::Matrix<double, DIM, DIM>& A,
                                     Eigen::Matrix<double, DIM * (DIM + 1), DIM * DIM>& dF_div_dx)
    {
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
#else
        const double mA11mA21mA31 = -A(0, 0) - A(1, 0) - A(2, 0);
        const double mA12mA22mA32 = -A(0, 1) - A(1, 1) - A(2, 1);
        const double mA13mA23mA33 = -A(0, 2) - A(1, 2) - A(2, 2);
        dF_div_dx <<
            mA11mA21mA31, 0.0, 0.0, A(0, 0), 0.0, 0.0, A(1, 0), 0.0, 0.0, A(2, 0), 0.0, 0.0,
            mA12mA22mA32, 0.0, 0.0, A(0, 1), 0.0, 0.0, A(1, 1), 0.0, 0.0, A(2, 1), 0.0, 0.0,
            mA13mA23mA33, 0.0, 0.0, A(0, 2), 0.0, 0.0, A(1, 2), 0.0, 0.0, A(2, 2), 0.0, 0.0,
            0.0, mA11mA21mA31, 0.0, 0.0, A(0, 0), 0.0, 0.0, A(1, 0), 0.0, 0.0, A(2, 0), 0.0,
            0.0, mA12mA22mA32, 0.0, 0.0, A(0, 1), 0.0, 0.0, A(1, 1), 0.0, 0.0, A(2, 1), 0.0,
            0.0, mA13mA23mA33, 0.0, 0.0, A(0, 2), 0.0, 0.0, A(1, 2), 0.0, 0.0, A(2, 2), 0.0,
            0.0, 0.0, mA11mA21mA31, 0.0, 0.0, A(0, 0), 0.0, 0.0, A(1, 0), 0.0, 0.0, A(2, 0),
            0.0, 0.0, mA12mA22mA32, 0.0, 0.0, A(0, 1), 0.0, 0.0, A(1, 1), 0.0, 0.0, A(2, 1),
            0.0, 0.0, mA13mA23mA33, 0.0, 0.0, A(0, 2), 0.0, 0.0, A(1, 2), 0.0, 0.0, A(2, 2);
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
    
    void IglUtils::extractRotation(const Eigen::Matrix3d &A,
                                   Eigen::Quaterniond &q,
                                   const unsigned int maxIter)
    {
        for (unsigned int iter = 0; iter < maxIter; iter++)
        {
            Eigen::Matrix3d R = q.matrix();
            Eigen::Vector3d omega = ((R.col(0).cross(A.col(0)) +
                                      R.col(1).cross(A.col(1)) +
                                      R.col(2).cross(A.col(2))) *
                                     (1.0 / fabs(R.col(0).dot(A.col(0)) +
                                                 R.col(1).dot(A.col(1)) +
                                                 R.col(2).dot(A.col(2))) +
                                      1.0e-9));
            double w = omega.norm();
            if (w < 1.0e-9)
                break;
            q = Eigen::Quaterniond(Eigen::AngleAxisd(w, (1.0 / w) * omega)) * q;
            q.normalize();
        }
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
                                   std::vector<std::vector<int>>& borderVerts,
                                   double ratio)
    {
        // resize to match size
        Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
        Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
        Eigen::RowVectorXd range = topRight - bottomLeft;
        
        borderVerts.resize(2);
        for(int vI = 0; vI < V.rows(); vI++) {
            if(V(vI, 0) < bottomLeft[0] + range[0] * ratio) {
                borderVerts[0].emplace_back(vI);
            }
            else if(V(vI, 0) > topRight[0] - range[0] * ratio) {
                borderVerts[1].emplace_back(vI);
            }
        }
    }

    void IglUtils::computeSVD_SIMD(std::vector<Eigen::Matrix3d>& testF,
                                   std::vector<Eigen::Matrix3d>& U, std::vector<Eigen::Vector3d>& Sigma, std::vector<Eigen::Matrix3d>& V)
    {
        using T=double;
        int old_size=testF.size();
        int iter = std::ceil(testF.size() / 4.f) * 4;

        __m256i vBuffer= _mm256_set_epi64x(27, 18, 9, 0);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)iter/4-1, 1, [&](int i)
#else
        for (int i = 0; i < (int)iter/4-1; ++i)
#endif
        {
            __m256d vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36),vBuffer,8);
            _mm256_store_pd(&a11[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+1),vBuffer,8);
            _mm256_store_pd(&a21[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+2),vBuffer,8);
            _mm256_store_pd(&a31[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+3),vBuffer,8);
            _mm256_store_pd(&a12[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+4),vBuffer,8);
            _mm256_store_pd(&a22[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+5),vBuffer,8);
            _mm256_store_pd(&a32[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+6),vBuffer,8);
            _mm256_store_pd(&a13[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+7),vBuffer,8);
            _mm256_store_pd(&a23[i*4], vFourDouble);

            vFourDouble = _mm256_i64gather_pd((reinterpret_cast<double*>(testF.data())+i*36+8),vBuffer,8);
            _mm256_store_pd(&a33[i*4], vFourDouble);
        }
#ifdef USE_TBB
        );
#endif

	    for(int i=iter-4;i<old_size;++i)
	    {
	        a11[i]=testF[i](0,0);
            a21[i]=testF[i](1,0);
            a31[i]=testF[i](2,0);
            a12[i]=testF[i](0,1);
            a22[i]=testF[i](1,1);
            a32[i]=testF[i](2,1);
            a13[i]=testF[i](0,2);
            a23[i]=testF[i](1,2);
            a33[i]=testF[i](2,2);
        }
        for(int i=old_size;i<iter;++i) {
            a11[i]=1;
            a21[i]=0;
            a31[i]=0;
            a12[i]=0;
            a22[i]=1;
            a32[i]=0;
            a13[i]=0;
            a23[i]=0;
            a33[i]=1;
        }

        using namespace PhysBAM;
        using namespace Singular_Value_Decomposition;
        int NUMBER_OF_THREAD=12;
        Singular_Value_Decomposition_Size_Specific_Helper<T, 19380> test(
                a11,a21,a31,a12,a22,a32,a13,a23,a33,
                u11,u21,u31,u12,u22,u32,u13,u23,u33,
                v11,v21,v31,v12,v22,v32,v13,v23,v33,
                sigma1,sigma2,sigma3);


        std::vector<int> imins(NUMBER_OF_THREAD);
        std::vector<int> imax_plus_ones(NUMBER_OF_THREAD);
        int stride= (iter / (NUMBER_OF_THREAD*4)) * 4;
        for(int i=0;i<NUMBER_OF_THREAD;++i)
        {
            imins[i]=i*stride;
            imax_plus_ones[i]=imins[i]+stride;
        }
        imax_plus_ones[NUMBER_OF_THREAD-1]=iter;

        //for(int i=0;i<NUMBER_OF_THREAD;++i)
        //{
            //std::cout << i << " min " << imins[i] << ",max " << imax_plus_ones[i] << std::endl;
        //}
        //std::cout << "iter " << iter << std::endl;
        //exit(0);

#ifdef USE_TBB
        tbb::parallel_for(0, (int)NUMBER_OF_THREAD, 1, [&](int partition) 
#else
        for (int partition = 0; partition < (int)NUMBER_OF_THREAD; ++partition)
#endif
        {
            int imin = imins[partition];
            int imax_plus_one = imax_plus_ones[partition];
            test.Run_Index_Range(imin,imax_plus_one);
        }
#ifdef USE_TBB
        );
#endif

        U.resize(iter);
        V.resize(iter);
        Sigma.resize(iter);
#ifdef USE_TBB
        tbb::parallel_for(0, (int)old_size, 1, [&](int i) 
#else
        for (int i = 0; i < old_size; ++i)
#endif
        {
            U[i](0, 0) = (double)u11[i];
            U[i](1, 0) = (double)u21[i];
            U[i](2, 0) = (double)u31[i];
            U[i](0, 1) = (double)u12[i];
            U[i](1, 1) = (double)u22[i];
            U[i](2, 1) = (double)u32[i];
            U[i](0, 2) = (double)u13[i];
            U[i](1, 2) = (double)u23[i];
            U[i](2, 2) = (double)u33[i];

            V[i](0,0) = (double)v11[i];
            V[i](1,0) = (double)v21[i];
            V[i](2,0) = (double)v31[i];
            V[i](0,1) = (double)v12[i];
            V[i](1,1) = (double)v22[i];
            V[i](2,1) = (double)v32[i];
            V[i](0,2) = (double)v13[i];
            V[i](1,2) = (double)v23[i];
            V[i](2,2) = (double)v33[i];

            Sigma[i](0) = (double)sigma1[i];
            Sigma[i](1) = (double)sigma2[i];
            Sigma[i](2) = (double)sigma3[i];
        }
#ifdef USE_TBB
        );
#endif

//        if(U.size()!=iter){std::cout << "sth is wrong1" << U.size() << " vs " << iter << std::endl; }
//        if(V.size()!=iter){std::cout << "sth is wrong2" << V.size() << " vs " << iter << std::endl; }
//        if(Sigma.size()!=iter){std::cout << "sth is wrong3" << Sigma.size() << " vs " << iter << std::endl; }
        for(int i=old_size;i<iter;++i)
        {
            U[i]=Eigen::Matrix3d::Identity();
            V[i]=Eigen::Matrix3d::Identity();
            Sigma[i]=Eigen::Vector3d::Identity();
        }
    }

    // all inputs & outpus have size of ceiling_size
    void IglUtils::matrixVectorMatrixTProduct(const std::vector<Eigen::Matrix3d>& A,
                                              const std::vector<Eigen::Vector3d>& S,
                                              const std::vector<Eigen::Matrix3d>& B,
                                              std::vector<Eigen::Matrix3d>& C)
    {
        int ceiling_size = A.size();

        using T=double;
        __m256i vBuffer  = _mm256_set_epi64x(27, 18, 9, 0);
        __m256i vBuffer2 = _mm256_set_epi64x(9, 6, 3, 0);

#ifdef USE_TBB
        tbb::parallel_for(0, (int)ceiling_size/4-1, 1, [&](int i)
#else
        for (int i = 0; i<(int)ceiling_size/4-1; ++i)
#endif
        {
            T __attribute__ ((aligned(64))) A_buffer[3][3][4];
            for(int u=0;u<3;++u)
                for(int v=0;v<3;++v)
                {
                    __m256d vTmp = _mm256_i64gather_pd((reinterpret_cast<const double*>(A.data())+i*36+u*3+v),vBuffer,8);
                    _mm256_store_pd(&(A_buffer[v][u][0]), vTmp);
                }

            T __attribute__ ((aligned(64))) B_buffer[3][3][4];
            for(int u=0;u<3;++u)
                for(int v=0;v<3;++v)
                {
                    __m256d vTmp = _mm256_i64gather_pd((reinterpret_cast<const double*>(B.data())+i*36+u*3+v),vBuffer,8);
                    _mm256_store_pd(&(B_buffer[v][u][0]), vTmp);
                }

            T __attribute__ ((aligned(64))) S_buffer[3][4];
            for(int v=0;v<3;++v)
            {
                __m256d vTmp = _mm256_i64gather_pd((reinterpret_cast<const double*>(S.data())+i*12+v),vBuffer2,8);
                _mm256_store_pd(&(S_buffer[v][0]), vTmp);
            }

            T __attribute__ ((aligned(64))) C_buffer[3][3][4];

            MATRIX_TIMES_VECTOR_TIMES_MATRIX_TRANPOSE(A_buffer,S_buffer,B_buffer,C_buffer);

            for(int u=0;u<3;++u)
                for(int v=0;v<3;++v)
                    for(int w=0;w<4;++w)
                        C[i*4+w](u,v)=C_buffer[u][v][w];
        }
#ifdef USE_TBB
        );
#endif
    }
}
