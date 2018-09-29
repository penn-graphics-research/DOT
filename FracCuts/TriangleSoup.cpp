//
//  TriangleSoup.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "TriangleSoup.hpp"
#include "IglUtils.hpp"
#include "SymStretchEnergy.hpp"
#include "Optimizer.hpp"
#include "Timer.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/writeOBJ.h>
#include <igl/list_to_matrix.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include <fstream>

extern std::ofstream logFile;
extern Timer timer_step;

namespace FracCuts {
    
    template<int dim>
    TriangleSoup<dim>::TriangleSoup(void)
    {
        initSeamLen = 0.0;
    }
    
    template<int dim>
    TriangleSoup<dim>::TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                                    const Eigen::MatrixXd& Vt_mesh, double p_areaThres_AM)
    {
        assert(V_mesh.rows() > 0);
        assert(F_mesh.rows() > 0);
        
        V_rest = V_mesh;
        F = F_mesh;
        if(Vt_mesh.rows() == V_mesh.rows()) {
            V = Vt_mesh;
        }
        else {
            assert(Vt_mesh.rows() == 0);
            V = Eigen::MatrixXd::Zero(V_rest.rows(), dim);
            std::cout << "No Vt provided, initialized to all 0" << std::endl;
        }
        
        initSeamLen = 0.0;
        areaThres_AM = p_areaThres_AM;
        
        triWeight = Eigen::VectorXd::Ones(F.rows());
        computeFeatures(false, true);
        
        vertWeight = Eigen::VectorXd::Ones(V.rows());
    }
    
    void initCylinder(double r1_x, double r1_y, double r2_x, double r2_y, double height, int circle_res, int height_resolution,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXd * uv_coords_per_face = NULL,
        Eigen::MatrixXi * uv_coords_face_ids = NULL)
    {
        assert(DIM == 2);
#if(DIM == 2)
        int nvertices = circle_res * (height_resolution+1);
        int nfaces = 2*circle_res * height_resolution;
        
        V.resize(nvertices, 3);
        if(uv_coords_per_face) {
            uv_coords_per_face->resize(nvertices, 2);
        }
        F.resize(nfaces, 3);
        for (int j=0; j<height_resolution+1; j++) {
            for (int i=0; i<circle_res; i++)
            {
                double t = (double)j / (double)height_resolution;
                double h = height * t;
                double theta = i * 2*M_PI / circle_res;
                double r_x = r1_x * t + r2_x * (1-t);
                double r_y = r1_y * t + r2_y * (1-t);
                V.row(j*circle_res+i) = Eigen::Vector3d(r_x*cos(theta), height-h, r_y*sin(theta));
                if(uv_coords_per_face) {
                    uv_coords_per_face->row(j*circle_res+i) = Eigen::Vector2d(r_x*cos(theta), r_y*sin(theta));
                }
                
                if (j<height_resolution)
                {
                    int vl0 = j*circle_res+i;
                    int vl1 = j*circle_res+(i+1)%circle_res;
                    int vu0 = (j+1)*circle_res+i;
                    int vu1 = (j+1)*circle_res+(i+1)%circle_res;
                    F.row(2*(j*circle_res+i)+0) = Eigen::Vector3i(vl0, vl1, vu1);
                    F.row(2*(j*circle_res+i)+1) = Eigen::Vector3i(vu0, vl0, vu1);
                }
            }
        }
#endif
    }
    
    template<int dim>
    TriangleSoup<dim>::TriangleSoup(Primitive primitive, double size, int elemAmt)
    {
        assert(dim == 2);
#if(DIM == 2)
        assert(size > 0.0);
        assert(elemAmt > 0);
        
        switch(primitive)
        {
            case P_GRID: {
                double spacing = size / std::sqrt(elemAmt / 2.0);
                assert(size >= spacing);
                int gridSize = static_cast<int>(size / spacing) + 1;
                spacing = size / (gridSize - 1);
                V_rest.resize(gridSize * gridSize, 3);
                V.resize(gridSize * gridSize, 2);
                borderVerts_primitive.resize(2);
                for(int rowI = 0; rowI < gridSize; rowI++)
                {
                    for(int colI = 0; colI < gridSize; colI++)
                    {
                        int vI = rowI * gridSize + colI;
                        V_rest.row(vI) = Eigen::Vector3d(spacing * colI, spacing * rowI, 0.0);
                        V.row(vI) = spacing * Eigen::Vector2d(colI, rowI);
                        
                        if(colI == 0) {
                            borderVerts_primitive[0].emplace_back(vI);
                        }
                        else if(colI == gridSize - 1) {
                            borderVerts_primitive[1].emplace_back(vI);
                        }
                    }
                }
                
                F.resize((gridSize - 1) * (gridSize - 1) * 2, 3);
                for(int rowI = 0; rowI < gridSize - 1; rowI++)
                {
                    for(int colI = 0; colI < gridSize - 1; colI++)
                    {
                        int squareI = rowI * (gridSize - 1) + colI;
                        F.row(squareI * 2) = Eigen::Vector3i(
                            rowI * gridSize + colI, (rowI + 1) * gridSize + colI + 1, (rowI + 1) * gridSize + colI);
                        F.row(squareI * 2 + 1) = Eigen::Vector3i(
                            rowI * gridSize + colI, rowI * gridSize + colI + 1, (rowI + 1) * gridSize + colI + 1);
                    }
                }
                break;
            }
                
            case P_SQUARE: {
                double spacing = size / std::sqrt(elemAmt / 2.0);
                assert(size >= spacing);
                int gridSize = static_cast<int>(size / spacing) + 1;
                spacing = size / (gridSize - 1);
                
                Eigen::MatrixXd UV_bnds(gridSize * 4 - 4, 2);
                borderVerts_primitive.resize(2);
                int vI = 0;
                for(int rowI = 0; rowI < gridSize; rowI++)
                {
                    for(int colI = 0; colI < gridSize; colI++)
                    {
                        if((colI > 0) && (colI < gridSize - 1) &&
                           (rowI > 0) && (rowI < gridSize - 1))
                        {
                            continue;
                        }
                        
                        UV_bnds.row(vI) = spacing * Eigen::Vector2d(colI, rowI);
                        
                        if(colI == 0) {
                            borderVerts_primitive[0].emplace_back(vI);
                        }
                        else if(colI == gridSize - 1) {
                            borderVerts_primitive[1].emplace_back(vI);
                        }
                        
                        vI++;
                    }
                }
                
                Eigen::MatrixXi E(gridSize * 4 - 4, 2);
                for(int colI = 0; colI + 1 < gridSize; colI++) {
                    E.row(colI) << colI, colI + 1;
                }
                for(int rowI = 0; rowI + 1 < gridSize; rowI++) {
                    E.row(gridSize - 1 + rowI) <<
                        borderVerts_primitive[1][rowI],
                        borderVerts_primitive[1][rowI + 1];
                }
                for(int colI = 0; colI + 1 < gridSize; colI++) {
                    E.row(gridSize * 2 - 2 + colI) << UV_bnds.rows() - 1 - colI, UV_bnds.rows() - 2 - colI;
                }
                for(int rowI = 0; rowI + 1 < gridSize; rowI++) {
                    E.row(gridSize * 3 - 3 + rowI) <<
                        borderVerts_primitive[0][gridSize - 1 - rowI],
                        borderVerts_primitive[0][gridSize - 2 - rowI];
                }
                
                std::string flag("qQa" + std::to_string(size * size / elemAmt * 1.1));
                // "q" for high quality mesh generation
                // "Q" for quiet mode (no output)
                // "a" for area upper bound
                
                igl::triangle::triangulate(UV_bnds, E, Eigen::MatrixXd(), flag, V, F);
                
                V_rest.resize(V.rows(), 3);
                V_rest.leftCols(2) = V;
                V_rest.rightCols(1).setZero();
                
                break;
            }
                
            case P_SPIKES: {
                Eigen::MatrixXd UV_bnds_temp(7, 2);
                UV_bnds_temp <<
                    0.0, 0.0,
                    1.0, 0.0,
                    0.8, 0.7,
                    1.0, 1.0,
                    0.7, 0.9,
                    0.0, 1.0,
                    0.25, 0.4;
                UV_bnds_temp *= size;
                
                double spacing = std::sqrt(size * size * 0.725 / elemAmt * 4.0 / std::sqrt(3.0));
                Eigen::MatrixXd inBetween1, inBetween2, inBetween5, inBetween6;
                IglUtils::sampleSegment(UV_bnds_temp.row(1), UV_bnds_temp.row(2), spacing, inBetween1);
                IglUtils::sampleSegment(UV_bnds_temp.row(2), UV_bnds_temp.row(3), spacing, inBetween2);
                IglUtils::sampleSegment(UV_bnds_temp.row(5), UV_bnds_temp.row(6), spacing, inBetween5);
                IglUtils::sampleSegment(UV_bnds_temp.row(6), UV_bnds_temp.row(0), spacing, inBetween6);
                
                Eigen::MatrixXd UV_bnds(UV_bnds_temp.rows() + inBetween1.rows() + inBetween2.rows() +
                                        inBetween5.rows() + inBetween6.rows(), 2);
                UV_bnds <<
                    UV_bnds_temp.row(0),
                    UV_bnds_temp.row(1),
                    inBetween1,
                    UV_bnds_temp.row(2),
                    inBetween2,
                    UV_bnds_temp.row(3),
                    UV_bnds_temp.row(4),
                    UV_bnds_temp.row(5),
                    inBetween5,
                    UV_bnds_temp.row(6),
                    inBetween6;
                
                borderVerts_primitive.resize(2);
                
                borderVerts_primitive[0].emplace_back(0);
                for(int i = 0; i < inBetween6.rows(); i++) {
                    borderVerts_primitive[0].emplace_back(6 + inBetween1.rows() + inBetween2.rows() +
                                                          inBetween5.rows() + inBetween6.rows() - i);
                }
                borderVerts_primitive[0].emplace_back(6 + inBetween1.rows() + inBetween2.rows() +
                                                      inBetween5.rows());
                for(int i = 0; i < inBetween5.rows(); i++) {
                    borderVerts_primitive[0].emplace_back(6 + inBetween1.rows() + inBetween2.rows() +
                                                          inBetween5.rows() - 1 - i);
                }
                borderVerts_primitive[0].emplace_back(5 + inBetween1.rows() + inBetween2.rows());
                
                borderVerts_primitive[1].emplace_back(1);
                for(int i = 0; i < inBetween1.rows(); i++) {
                    borderVerts_primitive[1].emplace_back(1 + 1 + i);
                }
                borderVerts_primitive[1].emplace_back(2 + inBetween1.rows());
                for(int i = 0; i < inBetween2.rows(); i++) {
                    borderVerts_primitive[1].emplace_back(2 + inBetween1.rows() + 1 + i);
                }
                borderVerts_primitive[1].emplace_back(3 + inBetween1.rows() + inBetween2.rows());
                
                Eigen::MatrixXi E(UV_bnds.rows(), 2);
                for(int i = 0; i < E.rows(); i++) {
                    E.row(i) << i, i + 1;
                }
                E(E.rows() - 1, 1) = 0;
                
                std::string flag("qQa" + std::to_string(size * size * 0.725 / elemAmt * 1.1));
                // "q" for high quality mesh generation
                // "Q" for quiet mode (no output)
                // "a" for area upper bound
                
                igl::triangle::triangulate(UV_bnds, E, Eigen::MatrixXd(), flag, V, F);
                
                V_rest.resize(V.rows(), 3);
                V_rest.leftCols(2) = V;
                V_rest.rightCols(1).setZero();
                break;
            }
                
            case P_SHARKEY: {
                V.resize(406,2);
                V <<581,1904 , 580,1896 , 610,1872 , 658,1847 , 597,1837 , 577,1823 , 592,1732 , 598,1728 , 606,1735 , 610,1717 , 631,1723 , 617,1705 , 626,1680 , 637,1674 , 632,1667 , 609,1674 , 597,1504 , 589,1499 , 549,1520 , 504,1562 , 492,1564 , 491,1557 , 548,1508 , 607,1485 , 597,1401 , 454,1353 , 228,1301 , 220,1301 , 192,1336 , 175,1337 , 157,1306 , 138,1322 , 124,1322 , 111,1285 , 121,1066 , 154,1065 , 188,1038 , 323,1037 , 471,1061 , 591,1093 , 582,1051 , 509,980 , 473,954 , 436,946 , 453,914 , 398,887 , 358,877 , 312,879 , 311,867 , 300,869 , 271,890 , 253,931 , 246,930 , 239,886 , 232,883 , 202,939 , 200,966 , 185,963 , 177,972 , 176,1010 , 160,1010 , 151,999 , 163,949 , 180,935 , 213,868 , 207,831 , 220,794 , 234,770 , 264,761 , 269,745 , 263,737 , 284,696 , 310,698 , 315,707 , 313,725 , 284,779 , 305,798 , 315,801 , 319,789 , 356,806 , 476,889 , 499,872 , 495,799 , 505,749 , 524,707 , 482,682 , 454,654 , 420,599 , 413,604 , 420,618 , 398,631 , 388,589 , 330,648 , 267,673 , 219,683 , 149,683 , 122,678 , 119,669 , 172,641 , 224,598 , 317,548 , 356,505 , 384,438 , 393,433 , 402,436 , 413,461 , 427,546 , 473,579 , 450,599 , 472,639 , 503,670 , 541,690 , 603,662 , 589,605 , 590,533 , 542,500 , 466,422 , 428,356 , 417,287 , 438,286 , 466,304 , 607,430 , 644,436 , 679,457 , 679,397 , 602,372 , 556,324 , 541,283 , 539,223 , 556,150 , 580,116 , 638,78 , 711,67 , 766,80 , 822,128 , 849,195 , 853,263 , 830,322 , 794,364 , 758,386 , 701,400 , 701,650 , 803,661 , 811,686 , 826,691 , 854,683 , 956,620 , 983,617 , 993,623 , 992,636 , 961,658 , 960,707 , 893,719 , 913,774 , 857,795 , 889,856 , 837,868 , 936,906 , 958,932 , 960,974 , 944,982 , 945,954 , 929,934 , 896,931 , 872,941 , 843,962 , 808,1003 , 795,1066 , 903,1091 , 1002,1129 , 1024,1131 , 1023,1141 , 1061,1144 , 1087,1098 , 1112,1028 , 1162,1000 , 1183,999 , 1238,1012 , 1255,1028 , 1302,1047 , 1293,1058 , 1300,1070 , 1295,1094 , 1306,1103 , 1324,1101 , 1338,1125 , 1366,1120 , 1384,1131 , 1393,1122 , 1396,1094 , 1402,1094 , 1440,1113 , 1467,1107 , 1506,1124 , 1520,1134 , 1521,1153 , 1502,1198 , 1481,1214 , 1434,1320 , 1420,1313 , 1412,1330 , 1397,1321 , 1401,1340 , 1390,1344 , 1372,1329 , 1353,1338 , 1353,1354 , 1379,1364 , 1377,1386 , 1365,1392 , 1369,1406 , 1353,1416 , 1353,1428 , 1385,1441 , 1345,1522 , 1349,1545 , 1334,1576 , 1314,1604 , 1249,1605 , 1168,1561 , 970,1474 , 936,1450 , 938,1422 , 1007,1282 , 999,1269 , 1002,1233 , 988,1215 , 976,1224 , 952,1185 , 889,1132 , 833,1104 , 793,1099 , 793,1120 , 807,1149 , 795,1158 , 803,1317 , 799,1377 , 793,1398 , 785,1397 , 783,1453 , 772,1489 , 836,1506 , 872,1528 , 883,1548 , 876,1550 , 859,1532 , 825,1514 , 789,1505 , 771,1636 , 779,1708 , 805,1820 , 706,1839 , 677,650 , 679,473 , 619,522 , 617,538 , 623,634 , 632,647 , 666,650 , 669.37,1484.16 , 256.722,1205.87 , 710.262,878.775 , 718.791,241.321 , 577.5,1496.5 , 701,525 , 643.5,1871.5 , 1399,1108.48 , 246.841,858.039 , 252.926,906.516 , 224,1270.76 , 406.174,587.043 , 612.25,1887.75 , 660.961,1651.6 , 595.195,1896.62 , 286.678,725.492 , 299.888,833.205 , 678,561.5 , 671.5,691.582 , 677.362,618.002 , 519.5,1532.5 , 643.888,1699.71 , 569,1509.5 , 1422.52,1160.52 , 1307.64,1422 , 584.5,1777.5 , 799.599,745.203 , 701,462.5 , 1266.48,1140.08 , 1037.77,1254.11 , 804,1497.5 , 1459.94,1261.49 , 1292.52,1346 , 1446.97,1290.75 , 700.754,319.635 , 713.616,1109.5 , 1176.28,1078.88 , 1328.48,1261.68 , 596.763,762.763 , 604.494,1610.16 , 748.024,692.372 , 727.895,1342.13 , 1035.75,1192.31 , 720.017,1561.26 , 1301.91,1541.34 , 708.569,1750.71 , 1389.02,1260.99 , 312.104,1274.49 , 643.231,1780.52 , 115.161,1193.87 , 139.482,1130.91 , 642.891,185.864 , 1127.83,1384.35 , 1146.82,1219.61 , 679,427 , 701,431.25 , 701,415.625 , 701,446.875 , 685.185,412 , 685.426,442 , 678.5,517.25 , 620,586 , 701,587.5 , 677.681,589.751 , 701,556.25 , 701,618.75 , 637.57,1438.77 , 712.014,1420.46 , 716.467,1016.95 , 764.818,1026.93 , 746.172,1072.53 , 382.507,1161.01 , 717.531,1176.74 , 341,1327 , 284.5,1314 , 315.953,1306.58 , 408.442,1255.68 , 661.491,1369.44 , 566.071,1256.13 , 666.133,1286.34 , 599.629,1324.7 , 683.678,1329 , 327.459,1205.44 , 291.585,1122.11 , 378.918,1212.88 , 799,1237.5 , 651.037,1187.49 , 525.5,1377 , 510.035,1304.57 , 452.344,1294.45 , 400.904,1325.2 , 550.527,1333.79 , 615.073,1274.65 , 681.204,1233.46 , 616.319,942.686 , 752.497,946.107 , 684.119,955.917 , 644.661,1015.92 , 658.982,904.398 , 740.246,1231.28 , 761.813,1190.3 , 764.897,1279.07 , 762.206,1319.87 , 716.106,1290.73 , 487.465,1183.15 , 505.436,1242.61 , 397,1049 , 425.148,1109.59 , 351.43,1095.85 , 649.495,1122.15 , 572.012,988.241 , 569.564,1173.55 , 604.163,1216.38 , 521.415,1117.61 , 629.722,1062.74 , 435.268,1205.59 , 697.732,1062.69 , 561.25,1389 , 590.012,1362.57 , 561.42,1360.32 , 612.142,1031.34 , 625.439,1097.16 , 404.421,1188.59 , 726.02,1047.57 , 677.57,1095.63 , 680.385,1028.88 , 545.5,1015.5 , 602,1443 , 397.5,1340 , 472.446,1220.13 , 469.896,1259.32 , 444.884,1156.04 , 473.026,1116.67 , 412.412,1142.42 , 496.652,1146.34 , 440.179,1240.45 , 564.625,1129.11 , 531,1077 , 557.926,1096.53 , 495.943,1087.96 , 561,1085 , 574.463,1094.76 , 544.463,1086.76 , 603.01,1068.46 , 608.786,1146.69 , 641.561,1155.03 , 586.5,1072;
                
                F.resize(515,3);
                F <<99,94,98 , 100,92,99 , 92,100,91 , 93,99,92 , 96,97,95 , 93,94,99 , 95,97,98 , 95,98,94 , 298,302,272 , 102,103,104 , 104,105,102 , 102,105,101 , 299,261,277 , 91,101,106 , 10,307,8 , 88,90,91 , 270,88,91 , 91,100,101 , 106,101,105 , 311,312,291 , 274,74,75 , 274,69,70 , 66,75,267 , 66,67,75 , 68,75,67 , 69,75,68 , 274,75,69 , 72,73,274 , 297,82,83 , 274,73,74 , 277,261,297 , 91,106,270 , 275,76,77 , 190,191,266 , 78,79,77 , 70,71,274 , 52,53,268 , 77,79,275 , 280,272,304 , 275,75,76 , 342,35,36 , 48,275,79 , 90,88,89 , 117,118,119 , 127,310,293 , 117,120,116 , 120,117,119 , 116,121,115 , 128,129,310 , 293,126,127 , 310,127,128 , 121,116,120 , 115,254,114 , 115,121,254 , 254,255,114 , 254,121,122 , 255,320,114 , 87,106,108 , 88,270,87 , 185,296,287 , 107,108,106 , 108,86,87 , 2,273,1 , 253,264,319 , 109,85,86 , 110,85,109 , 108,109,86 , 308,30,33 , 248,272,302 , 240,302,259 , 156,285,154 , 112,113,256 , 256,257,112 , 113,320,256 , 82,297,81 , 50,267,49 , 85,110,84 , 84,111,297 , 267,65,66 , 111,84,110 , 277,297,112 , 204,305,201 , 83,84,297 , 112,297,111 , 291,296,205 , 4,307,3 , 249,304,272 , 129,130,310 , 3,251,265 , 55,63,64 , 55,64,54 , 64,267,54 , 65,267,64 , 143,285,299 , 62,57,58 , 62,63,57 , 250,304,249 , 55,56,57 , 63,55,57 , 125,293,124 , 267,53,54 , 308,34,309 , 46,47,48 , 79,46,48 , 14,15,298 , 260,308,309 , 52,268,51 , 283,303,219 , 218,219,303 , 268,50,51 , 79,45,46 , 50,53,267 , 48,49,275 , 261,355,357 , 308,260,269 , 61,59,60 , 35,309,34 , 32,33,31 , 30,31,33 , 269,30,308 , 58,59,61 , 37,342,36 , 342,309,35 , 29,30,28 , 269,26,27 , 28,30,27 , 269,306,333 , 269,260,306 , 367,37,365 , 61,62,58 , 199,201,305 , 42,43,44 , 185,287,183 , 42,44,80 , 45,79,80 , 44,45,80 , 42,80,81 , 388,374,363 , 306,335,332 , 263,17,281 , 41,42,81 , 314,316,318 , 257,277,112 , 41,369,385 , 336,377,339 , 355,356,353 , 23,17,263 , 386,24,325 , 16,23,259 , 263,281,22 , 16,17,23 , 16,259,302 , 16,302,298 , 274,71,72 , 297,357,81 , 279,22,18 , 20,21,19 , 279,19,21 , 271,273,2 , 5,284,4 , 284,6,8 , 12,280,11 , 131,310,130 , 280,12,13 , 4,284,307 , 284,8,307 , 10,8,9 , 8,6,7 , 307,251,3 , 2,3,265 , 232,331,294 , 22,281,18 , 257,258,277 , 126,293,125 , 310,132,133 , 124,293,140 , 314,313,315 , 258,252,277 , 252,324,141 , 123,253,122 , 286,264,253 , 283,311,291 , 254,122,253 , 262,133,134 , 262,310,133 , 134,135,262 , 262,135,136 , 197,305,282 , 293,138,139 , 293,137,138 , 262,136,137 , 139,140,293 , 141,142,299 , 354,165,166 , 141,299,277 , 252,278,324 , 277,252,141 , 329,232,294 , 106,87,270 , 231,167,230 , 382,328,329 , 232,329,231 , 27,30,269 , 299,142,143 , 402,379,373 , 152,285,144 , 344,360,358 , 152,154,285 , 296,291,312 , 342,330,341 , 156,261,285 , 167,168,230 , 156,165,354 , 156,164,165 , 163,164,156 , 247,240,289 , 272,14,298 , 156,154,155 , 2,265,271 , 145,152,144 , 143,144,285 , 220,311,219 , 49,267,275 , 328,166,167 , 362,358,360 , 295,176,177 , 158,162,157 , 312,301,172 , 145,146,150 , 157,163,156 , 286,253,123 , 162,163,157 , 160,161,159 , 191,188,266 , 161,162,158 , 75,275,267 , 150,152,145 , 153,154,152 , 146,149,150 , 149,146,147 , 148,149,147 , 312,287,296 , 26,269,333 , 321,322,276 , 288,225,301 , 158,159,161 , 150,151,152 , 175,176,295 , 295,287,312 , 182,295,178 , 180,178,179 , 177,178,295 , 174,175,295 , 311,283,219 , 312,288,301 , 295,182,287 , 182,180,181 , 180,182,178 , 173,174,295 , 304,307,10 , 185,183,184 , 189,190,266 , 193,282,192 , 189,266,188 , 182,183,287 , 230,168,229 , 53,50,268 , 238,239,326 , 197,290,305 , 172,173,312 , 234,232,233 , 262,293,310 , 237,238,236 , 300,236,238 , 282,191,192 , 259,23,325 , 352,358,362 , 227,228,226 , 226,171,301 , 228,171,226 , 228,229,169 , 206,291,205 , 299,285,261 , 187,305,296 , 288,224,225 , 226,301,225 , 262,137,293 , 221,222,220 , 169,171,228 , 234,331,232 , 229,168,169 , 222,223,311 , 13,14,272 , 240,247,302 , 302,247,248 , 240,259,326 , 239,240,326 , 272,280,13 , 307,304,251 , 304,250,251 , 10,280,304 , 280,10,11 , 0,1,273 , 248,249,272 , 247,289,246 , 246,241,245 , 289,241,246 , 242,243,244 , 242,244,245 , 245,241,242 , 220,222,311 , 361,236,300 , 19,279,18 , 209,283,206 , 204,296,305 , 187,185,186 , 187,296,185 , 282,305,187 , 288,223,224 , 203,204,201 , 312,173,295 , 292,305,290 , 205,296,204 , 191,282,188 , 170,171,169 , 196,193,195 , 197,282,196 , 194,195,193 , 200,201,199 , 292,198,199 , 305,292,199 , 187,188,282 , 196,282,193 , 215,216,303 , 288,311,223 , 214,303,283 , 211,212,283 , 215,303,214 , 217,218,303 , 217,303,216 , 206,283,291 , 207,208,209 , 214,283,212 , 209,210,211 , 283,209,211 , 212,213,214 , 207,209,206 , 201,202,203 , 301,171,172 , 311,288,312 , 132,310,131 , 318,316,123 , 317,315,313 , 313,124,317 , 315,317,140 , 286,123,316 , 140,317,124 , 123,313,318 , 314,318,313 , 264,276,319 , 114,320,113 , 276,323,321 , 324,322,321 , 276,264,323 , 322,324,278 , 325,336,326 , 326,300,238 , 259,325,326 , 325,24,336 , 383,373,375 , 354,327,355 , 329,328,167 , 354,328,327 , 231,329,167 , 329,294,375 , 403,371,370 , 306,260,341 , 300,326,336 , 344,235,360 , 341,343,306 , 306,334,333 , 306,332,334 , 332,333,334 , 347,364,337 , 335,348,349 , 336,24,377 , 300,336,340 , 388,389,394 , 371,403,345 , 338,340,339 , 331,352,345 , 347,346,25 , 351,339,337 , 339,340,336 , 359,358,331 , 342,341,260 , 343,341,330 , 342,260,309 , 398,391,38 , 330,392,381 , 306,343,335 , 362,340,338 , 359,344,358 , 352,351,371 , 364,370,337 , 348,347,25 , 378,339,377 , 350,337,339 , 349,348,25 , 394,335,374 , 387,332,349 , 335,349,332 , 346,347,350 , 337,350,347 , 351,337,371 , 338,339,351 , 351,352,338 , 404,331,345 , 353,369,81 , 261,354,355 , 354,261,156 , 328,354,166 , 355,327,384 , 355,353,357 , 375,384,382 , 373,379,356 , 81,357,353 , 297,261,357 , 352,331,358 , 331,234,359 , 344,359,234 , 235,361,360 , 300,340,362 , 236,361,235 , 362,361,300 , 362,338,352 , 362,360,361 , 390,366,391 , 393,395,370 , 389,347,348 , 370,364,363 , 365,38,366 , 37,367,342 , 366,392,367 , 366,367,365 , 330,342,367 , 380,402,373 , 331,368,294 , 81,369,41 , 356,369,353 , 370,363,393 , 368,403,380 , 371,345,352 , 337,370,371 , 390,391,393 , 400,395,397 , 384,375,373 , 380,373,383 , 335,343,374 , 330,381,343 , 383,375,294 , 384,373,356 , 350,378,346 , 376,378,377 , 24,376,377 , 346,378,376 , 339,378,350 , 40,369,379 , 369,356,379 , 395,403,370 , 402,380,39 , 381,392,390 , 374,343,381 , 328,382,327 , 329,375,382 , 294,368,383 , 383,368,380 , 355,384,356 , 382,384,327 , 369,40,385 , 325,23,386 , 349,25,387 , 363,364,388 , 389,388,364 , 347,389,364 , 394,348,335 , 363,374,390 , 381,390,374 , 38,391,366 , 393,391,372 , 367,392,330 , 366,390,392 , 393,372,395 , 390,393,363 , 388,394,374 , 348,394,389 , 401,372,396 , 403,39,380 , 398,372,391 , 372,397,395 , 399,400,397 , 38,396,398 , 372,398,396 , 401,396,399 , 39,395,400 , 399,39,400 , 397,372,401 , 399,397,401 , 405,40,402 , 379,402,40 , 403,395,39 , 404,403,368 , 331,404,368 , 345,403,404 , 402,39,405;
                //TODO: save into file
                
                // remeshing for different resolution
                Eigen::VectorXi bnd;
                igl::boundary_loop(F, bnd);
                Eigen::MatrixXi E(bnd.size(), 2);
                Eigen::MatrixXd UV_bnds(bnd.size(), 2);
                for(int i = 0; i < bnd.size(); i++) {
                    E.row(i) << i, i + 1;
                    UV_bnds.row(i) = V.row(bnd[i]);
                }
                E(bnd.size() - 1, 1) = 0;
                
                double area = 0.0;
                for(int triI = 0; triI < F.rows(); triI++) {
                    const Eigen::RowVector2d v01 = V.row(F(triI, 1)) - V.row(F(triI, 0));
                    const Eigen::RowVector2d v02 = V.row(F(triI, 2)) - V.row(F(triI, 0));
                    area += v01[0] * v02[1] - v01[1] * v02[0];
                }
                area /= 2.0;
                std::string flag("qQa" + std::to_string(area / elemAmt * 1.1));
                // "q" for high quality mesh generation
                // "Q" for quiet mode (no output)
                // "a" for area upper bound
                
                igl::triangle::triangulate(UV_bnds, E, Eigen::MatrixXd(), flag, V, F);
                
                // resize to match size
                Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
                Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
                Eigen::RowVectorXd range = topRight - bottomLeft;
                double scale = size / range[0];
                V *= scale;
                
                IglUtils::findBorderVerts(V, borderVerts_primitive);
                
                V_rest.resize(V.rows(), 3);
                V_rest.leftCols(2) = V;
                V_rest.rightCols(1).setZero();
                
                break;
            }
                
            case P_CYLINDER: {
                initCylinder(0.5, 0.5, 1.0, 1.0, 1.0, 20, 20, V_rest, F, &V);
                break;
            }
                
            default:
                assert(0 && "no such primitive to construct!");
                break;
        }
        
        triWeight = Eigen::VectorXd::Ones(F.rows());
        computeFeatures(false, true);
        initSeamLen = 0.0;
        
        vertWeight = Eigen::VectorXd::Ones(V.rows());
#endif
    }
    
    template<int dim>
    void TriangleSoup<dim>::computeLaplacianMtr(void)
    {
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V_rest, F, L);
        LaplacianMtr.resize(L.rows(), L.cols());
        LaplacianMtr.setZero();
        LaplacianMtr.reserve(L.nonZeros());
        for (int k = 0; k < L.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            {
                if((fixedVert.find(static_cast<int>(it.row())) == fixedVert.end()) &&
                   (fixedVert.find(static_cast<int>(it.col())) == fixedVert.end()))
                {
                    LaplacianMtr.insert(it.row(), it.col()) = -it.value();
                }
            }
        }
        for(const auto fixedVI : fixedVert) {
            LaplacianMtr.insert(fixedVI, fixedVI) = 1.0;
        }
        LaplacianMtr.makeCompressed();
    }
    
    template<int dim>
    void TriangleSoup<dim>::computeMassMatrix(const igl::MassMatrixType type)
    {
        const Eigen::MatrixXd & V = V_rest;
        Eigen::SparseMatrix<double>& M = massMatrix;
        
        typedef double Scalar;
        
        using namespace Eigen;
        using namespace std;
        using namespace igl;
        
        const int n = V.rows();
        const int m = F.rows();
        const int simplex_size = F.cols();
        
        MassMatrixType eff_type = type;
        // Use voronoi of for triangles by default, otherwise barycentric
        if(type == MASSMATRIX_TYPE_DEFAULT)
        {
            eff_type = (simplex_size == 3?MASSMATRIX_TYPE_VORONOI:MASSMATRIX_TYPE_BARYCENTRIC);
        }
        
        // Not yet supported
        assert(type!=MASSMATRIX_TYPE_FULL);
        
        Matrix<int,Dynamic,1> MI;
        Matrix<int,Dynamic,1> MJ;
        Matrix<Scalar,Dynamic,1> MV;
        if(simplex_size == 3)
        {
            // Triangles
            // edge lengths numbered same as opposite vertices
            Matrix<Scalar,Dynamic,3> l(m,3);
            // loop over faces
            for(int i = 0;i<m;i++)
            {
                l(i,0) = (V.row(F(i,1))-V.row(F(i,2))).norm();
                l(i,1) = (V.row(F(i,2))-V.row(F(i,0))).norm();
                l(i,2) = (V.row(F(i,0))-V.row(F(i,1))).norm();
            }
            Matrix<Scalar,Dynamic,1> dblA;
            doublearea(l,0.,dblA);
            
            //!!! weighted area
            for(int i = 0; i < m; i++) {
                dblA[i] *= triWeight[i];
            }
            
            switch(eff_type)
            {
                case MASSMATRIX_TYPE_BARYCENTRIC:
                    // diagonal entries for each face corner
                    MI.resize(m*3,1); MJ.resize(m*3,1); MV.resize(m*3,1);
                    MI.block(0*m,0,m,1) = F.col(0);
                    MI.block(1*m,0,m,1) = F.col(1);
                    MI.block(2*m,0,m,1) = F.col(2);
                    MJ = MI;
                    repmat(dblA,3,1,MV);
                    MV.array() /= 6.0;
                    break;
                case MASSMATRIX_TYPE_VORONOI:
                {
                    // diagonal entries for each face corner
                    // http://www.alecjacobson.com/weblog/?p=874
                    MI.resize(m*3,1); MJ.resize(m*3,1); MV.resize(m*3,1);
                    MI.block(0*m,0,m,1) = F.col(0);
                    MI.block(1*m,0,m,1) = F.col(1);
                    MI.block(2*m,0,m,1) = F.col(2);
                    MJ = MI;
                    
                    // Holy shit this needs to be cleaned up and optimized
                    Matrix<Scalar,Dynamic,3> cosines(m,3);
                    cosines.col(0) =
                    (l.col(2).array().pow(2)+l.col(1).array().pow(2)-l.col(0).array().pow(2))/(l.col(1).array()*l.col(2).array()*2.0);
                    cosines.col(1) =
                    (l.col(0).array().pow(2)+l.col(2).array().pow(2)-l.col(1).array().pow(2))/(l.col(2).array()*l.col(0).array()*2.0);
                    cosines.col(2) =
                    (l.col(1).array().pow(2)+l.col(0).array().pow(2)-l.col(2).array().pow(2))/(l.col(0).array()*l.col(1).array()*2.0);
                    Matrix<Scalar,Dynamic,3> barycentric = cosines.array() * l.array();
                    normalize_row_sums(barycentric,barycentric);
                    Matrix<Scalar,Dynamic,3> partial = barycentric;
                    partial.col(0).array() *= dblA.array() * 0.5;
                    partial.col(1).array() *= dblA.array() * 0.5;
                    partial.col(2).array() *= dblA.array() * 0.5;
                    Matrix<Scalar,Dynamic,3> quads(partial.rows(),partial.cols());
                    quads.col(0) = (partial.col(1)+partial.col(2))*0.5;
                    quads.col(1) = (partial.col(2)+partial.col(0))*0.5;
                    quads.col(2) = (partial.col(0)+partial.col(1))*0.5;
                    
                    quads.col(0) = (cosines.col(0).array()<0).select( 0.25*dblA,quads.col(0));
                    quads.col(1) = (cosines.col(0).array()<0).select(0.125*dblA,quads.col(1));
                    quads.col(2) = (cosines.col(0).array()<0).select(0.125*dblA,quads.col(2));
                    
                    quads.col(0) = (cosines.col(1).array()<0).select(0.125*dblA,quads.col(0));
                    quads.col(1) = (cosines.col(1).array()<0).select(0.25*dblA,quads.col(1));
                    quads.col(2) = (cosines.col(1).array()<0).select(0.125*dblA,quads.col(2));
                    
                    quads.col(0) = (cosines.col(2).array()<0).select(0.125*dblA,quads.col(0));
                    quads.col(1) = (cosines.col(2).array()<0).select(0.125*dblA,quads.col(1));
                    quads.col(2) = (cosines.col(2).array()<0).select( 0.25*dblA,quads.col(2));
                    
                    MV.block(0*m,0,m,1) = quads.col(0);
                    MV.block(1*m,0,m,1) = quads.col(1);
                    MV.block(2*m,0,m,1) = quads.col(2);
                    
                    break;
                }
                case MASSMATRIX_TYPE_FULL:
                    assert(false && "Implementation incomplete");
                    break;
                default:
                    assert(false && "Unknown Mass matrix eff_type");
            }
            
        }else if(simplex_size == 4)
        {
            assert(V.cols() == 3);
            if(eff_type != MASSMATRIX_TYPE_BARYCENTRIC) {
                std::cout << "switched to barycentric mass matrix for tetrahedra" << std::endl;
            }
            MI.resize(m*4,1); MJ.resize(m*4,1); MV.resize(m*4,1);
            MI.block(0*m,0,m,1) = F.col(0);
            MI.block(1*m,0,m,1) = F.col(1);
            MI.block(2*m,0,m,1) = F.col(2);
            MI.block(3*m,0,m,1) = F.col(3);
            MJ = MI;
            // loop over tets
            for(int i = 0;i<m;i++)
            {
                // http://en.wikipedia.org/wiki/Tetrahedron#Volume
                Matrix<Scalar,3,1> v0m3,v1m3,v2m3;
                v0m3.head(V.cols()) = V.row(F(i,0)) - V.row(F(i,3));
                v1m3.head(V.cols()) = V.row(F(i,1)) - V.row(F(i,3));
                v2m3.head(V.cols()) = V.row(F(i,2)) - V.row(F(i,3));
                Scalar v = fabs(v0m3.dot(v1m3.cross(v2m3)))/6.0;
                MV(i+0*m) = v/4.0;
                MV(i+1*m) = v/4.0;
                MV(i+2*m) = v/4.0;
                MV(i+3*m) = v/4.0;
            }
        }else
        {
            // Unsupported simplex size
            assert(false && "Unsupported simplex size");
        }
        sparse(MI,MJ,MV,n,n,M);
        
        massMatrix *= density;
    }
    
    template<int dim>
    void TriangleSoup<dim>::computeFeatures(bool multiComp, bool resetFixedV)
    {
        //TODO: if the mesh is multi-component, then fix more vertices
        if(resetFixedV) {
            fixedVert.clear();
            fixedVert.insert(0);
            isFixedVert.resize(0);
            isFixedVert.resize(V.rows(), false);
            isFixedVert[0] = true;
        }
        
        boundaryEdge.resize(cohE.rows());
        edgeLen.resize(cohE.rows());
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(cohE.row(cohI).minCoeff() >= 0) {
                boundaryEdge[cohI] = 0;
            }
            else {
                boundaryEdge[cohI] = 1;
            }
            edgeLen[cohI] = (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
        }

        restTriInv.resize(F.rows());
        triNormal.resize(F.rows(), 3);
        triArea.resize(F.rows());
        surfaceArea = 0.0;
        triAreaSq.resize(F.rows());
        e0SqLen.resize(F.rows());
        e1SqLen.resize(F.rows());
        e0dote1.resize(F.rows());
        e0SqLen_div_dbAreaSq.resize(F.rows());
        e1SqLen_div_dbAreaSq.resize(F.rows());
        e0dote1_div_dbAreaSq.resize(F.rows());
        vFLoc.resize(0);
        vFLoc.resize(V.rows());
        std::vector<Eigen::RowVector3d> vertNormals(V_rest.rows(), Eigen::RowVector3d::Zero());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::Matrix<int, 1, dim + 1>& triVInd = F.row(triI);
            
            vFLoc[triVInd[0]].insert(std::pair<int, int>(triI, 0));
            vFLoc[triVInd[1]].insert(std::pair<int, int>(triI, 1));
            vFLoc[triVInd[2]].insert(std::pair<int, int>(triI, 2));
            if(dim == 3) {
                vFLoc[triVInd[3]].insert(std::pair<int, int>(triI, 3));
            }
            
            const Eigen::Vector3d& P1 = V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = V_rest.row(triVInd[2]);
            
            Eigen::Matrix<double, dim, dim> X0;
#if(DIM == 2)
            Eigen::Vector3d x0_3D[3] = { P1, P2, P3 };
            Eigen::Vector2d x0[3];
            IglUtils::mapTriangleTo2D(x0_3D, x0);
            X0.col(0) = x0[1] - x0[0];
            X0.col(1) = x0[2] - x0[0];
#else
            const Eigen::Vector3d& P4 = V_rest.row(triVInd[3]);
            X0.col(0) = P2 - P1;
            X0.col(1) = P3 - P1;
            X0.col(2) = P4 - P1;
#endif
            restTriInv[triI] = X0.inverse();
            //TODO: support areaThres_AM
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            const Eigen::RowVector3d normalVec = P2m1.cross(P3m1).transpose();
            
            triNormal.row(triI) = normalVec.normalized(); //TODO: invalid in 3D
            triArea[triI] = X0.determinant() / dim / (dim - 1);
            if(triArea[triI] < areaThres_AM) {
                // air mesh triangle degeneracy prevention
                triArea[triI] = areaThres_AM;
                surfaceArea += areaThres_AM;
                triAreaSq[triI] = areaThres_AM * areaThres_AM;
                e0SqLen[triI] = e1SqLen[triI] = 4.0 / std::sqrt(3.0) * areaThres_AM;
                e0dote1[triI] = e0SqLen[triI] / 2.0;
                
                e0SqLen_div_dbAreaSq[triI] = e1SqLen_div_dbAreaSq[triI] = 2.0 / std::sqrt(3.0) / areaThres_AM;
                e0dote1_div_dbAreaSq[triI] = e0SqLen_div_dbAreaSq[triI] / 2.0;
            }
            else {
                surfaceArea += triArea[triI];
                triAreaSq[triI] = triArea[triI] * triArea[triI];
                e0SqLen[triI] = P2m1.squaredNorm();
                e1SqLen[triI] = P3m1.squaredNorm();
                e0dote1[triI] = P2m1.dot(P3m1);
                
                e0SqLen_div_dbAreaSq[triI] = e0SqLen[triI] / 2. / triAreaSq[triI];
                e1SqLen_div_dbAreaSq[triI] = e1SqLen[triI] / 2. / triAreaSq[triI];
                e0dote1_div_dbAreaSq[triI] = e0dote1[triI] / 2. / triAreaSq[triI];
            }
            vertNormals[triVInd[0]] += normalVec;
            vertNormals[triVInd[1]] += normalVec;
            vertNormals[triVInd[2]] += normalVec;
        }
        avgEdgeLen = igl::avg_edge_length(V_rest, F);
        virtualRadius = std::sqrt(surfaceArea / M_PI);
        for(auto& vNI : vertNormals) {
            vNI.normalize();
        }
        
        computeLaplacianMtr();
#ifndef STATIC_SOLVE
        computeMassMatrix(igl::MASSMATRIX_TYPE_VORONOI);
#endif
        
//        //!! for edge count minimization of separation energy
//        for(int cohI = 0; cohI < cohE.rows(); cohI++)
//        {
//            edgeLen[cohI] = avgEdgeLen;
//        }
        
        bbox.block(0, 0, 1, 3) = V_rest.row(0);
        bbox.block(1, 0, 1, 3) = V_rest.row(0);
        for(int vI = 1; vI < V_rest.rows(); vI++) {
            const Eigen::RowVector3d& v = V_rest.row(vI);
            for(int dimI = 0; dimI < 3; dimI++) {
                if(v[dimI] < bbox(0, dimI)) {
                    bbox(0, dimI) = v[dimI];
                }
                if(v[dimI] > bbox(1, dimI)) {
                    bbox(1, dimI) = v[dimI];
                }
            }
        }
        
        if(dim == 2) {
            edge2Tri.clear();
            vNeighbor.resize(0);
            vNeighbor.resize(V_rest.rows());
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::Matrix<int, 1, dim + 1>& triVInd = F.row(triI);
                for(int vI = 0; vI < 3; vI++) {
                    int vI_post = (vI + 1) % 3;
                    edge2Tri[std::pair<int, int>(triVInd[vI], triVInd[vI_post])] = triI;
                    vNeighbor[triVInd[vI]].insert(triVInd[vI_post]);
                    vNeighbor[triVInd[vI_post]].insert(triVInd[vI]);
                }
            }
        }
        else {
            vNeighbor.resize(0);
            vNeighbor.resize(V_rest.rows());
            for(int elemI = 0; elemI < F.rows(); elemI++) {
                const Eigen::Matrix<int, 1, dim + 1>& elemVInd = F.row(elemI);
                for(int vI = 0; vI < dim + 1; vI++) {
                    for(int vJ = vI + 1; vJ < dim + 1; vJ ++) {
                        vNeighbor[elemVInd[vI]].insert(elemVInd[vJ]);
                        vNeighbor[elemVInd[vJ]].insert(elemVInd[vI]);
                    }
                }
            }
        }

        cohEIndex.clear();
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            const Eigen::RowVector4i& cohEI = cohE.row(cohI);
            if(cohEI.minCoeff() >= 0) {
                cohEIndex[std::pair<int, int>(cohEI[0], cohEI[1])] = cohI;
                cohEIndex[std::pair<int, int>(cohEI[3], cohEI[2])] = -cohI - 1;
            }
        }
        
        // init fracture tail record
        fracTail.clear();
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            if(cohE(cohI, 0) == cohE(cohI, 2)) {
                fracTail.insert(cohE(cohI, 0));
            }
            else if(cohE(cohI, 1) == cohE(cohI, 3)) {
                fracTail.insert(cohE(cohI, 1));
            }
        }
        // tails of initial seams doesn't count as fracture tails for propagation
        for(int initSeamI = 0; initSeamI < initSeams.rows(); initSeamI++) {
            if(initSeams(initSeamI, 0) == initSeams(initSeamI, 2)) {
                fracTail.erase(initSeams(initSeamI, 0));
            }
            else if(initSeams(initSeamI, 1) == initSeams(initSeamI, 3)) {
                fracTail.erase(initSeams(initSeamI, 1));
            }
        }
    }
    
    template<int dim>
    void TriangleSoup<dim>::updateFeatures(void)
    {
        // assumption: triangle shapes stay the same
        
        const int nCE = static_cast<int>(boundaryEdge.size());
        boundaryEdge.conservativeResize(cohE.rows());
        edgeLen.conservativeResize(cohE.rows());
        for(int cohI = nCE; cohI < cohE.rows(); cohI++)
        {
            if(cohE.row(cohI).minCoeff() >= 0) {
                boundaryEdge[cohI] = 0;
            }
            else {
                boundaryEdge[cohI] = 1;
            }
            edgeLen[cohI] = (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
        }
        
        computeLaplacianMtr();
#ifndef STATIC_SOLVE
        computeMassMatrix(igl::MASSMATRIX_TYPE_VORONOI);
#endif
    }
    
    template<int dim>
    void TriangleSoup<dim>::resetFixedVert(const std::set<int>& p_fixedVert)
    {
        isFixedVert.resize(0);
        isFixedVert.resize(V.rows(), false);
        for(const auto& vI : p_fixedVert) {
            assert(vI < V.rows());
            isFixedVert[vI] = true;
        }
        
        fixedVert = p_fixedVert;
        computeLaplacianMtr();
    }
    template<int dim>
    void TriangleSoup<dim>::addFixedVert(int vI)
    {
        assert(vI < V.rows());
        fixedVert.insert(vI);
        isFixedVert[vI] = true;
    }
    template<int dim>
    void TriangleSoup<dim>::addFixedVert(const std::vector<int>& p_fixedVert)
    {
        for(const auto& vI : p_fixedVert) {
            assert(vI < V.rows());
            isFixedVert[vI] = true;
        }
        
        fixedVert.insert(p_fixedVert.begin(), p_fixedVert.end());
        computeLaplacianMtr();
    }
    
    template<int dim>
    bool TriangleSoup<dim>::checkInversion(int triI, bool mute) const
    {
        assert(triI < F.rows());
        
        const double eps = 0.0;//1.0e-20 * avgEdgeLen * avgEdgeLen;

        const Eigen::Matrix<int, 1, dim + 1>& triVInd = F.row(triI);
        
        Eigen::Matrix<double, dim, dim> e_u;
        e_u.col(0) = (V.row(triVInd[1]) - V.row(triVInd[0])).transpose();
        e_u.col(1) = (V.row(triVInd[2]) - V.row(triVInd[0])).transpose();
        if(dim == 3) {
            e_u.col(2) = (V.row(triVInd[3]) - V.row(triVInd[0])).transpose();
        }
        
        const double dbArea = e_u.determinant();
        if(dbArea < eps) {
            if(!mute) {
                std::cout << "***Element inversion detected: " << dbArea << " < " << eps << std::endl;
                std::cout << "mesh triangle count: " << F.rows() << std::endl;
                logFile << "***Element inversion detected: " << dbArea << " < " << eps << std::endl;
            }
            return false;
        }
        else {
            return true;
        }
    }
    template<int dim>
    bool TriangleSoup<dim>::checkInversion(bool mute, const std::vector<int>& triangles) const
    {
        if(triangles.empty()) {
            for(int triI = 0; triI < F.rows(); triI++)
            {
                if(!checkInversion(triI, mute)) {
                    return false;
                }
            }
        }
        else {
            for(const auto& triI : triangles)
            {
                if(!checkInversion(triI, mute)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    template<int dim>
    void TriangleSoup<dim>::save(const std::string& filePath, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                            const Eigen::MatrixXd UV, const Eigen::MatrixXi& FUV) const
    {
        assert(dim == 2);
#if(DIM == 2)
        std::ofstream out;
        out.open(filePath);
        assert(out.is_open());
        
        for(int vI = 0; vI < V.rows(); vI++) {
            const Eigen::RowVector3d& v = V.row(vI);
            out << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
        }
        
        for(int vI = 0; vI < UV.rows(); vI++) {
            const Eigen::RowVector2d& uv = UV.row(vI);
            out << "vt " << uv[0] << " " << uv[1] << std::endl;
        }
        
        if(FUV.rows() == F.rows()) {
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::RowVector3i& tri = F.row(triI);
                const Eigen::RowVector3i& tri_UV = FUV.row(triI);
                out << "f " << tri[0] + 1 << "/" << tri_UV[0] + 1 <<
                " " << tri[1] + 1 << "/" << tri_UV[1] + 1 <<
                " " << tri[2] + 1 << "/" << tri_UV[2] + 1 << std::endl;
            }
        }
        else {
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::RowVector3i& tri = F.row(triI);
                out << "f " << tri[0] + 1 << "/" << tri[0] + 1 <<
                " " << tri[1] + 1 << "/" << tri[1] + 1 <<
                " " << tri[2] + 1 << "/" << tri[2] + 1 << std::endl;
            }
        }
        
        out.close();
#endif
    }
    
    template<int dim>
    void TriangleSoup<dim>::save(const std::string& filePath) const
    {
        save(filePath, V_rest, F, V);
    }
    
    template<int dim>
    void TriangleSoup<dim>::saveAsMesh(const std::string& filePath, bool scaleUV,
                                       const Eigen::MatrixXi& SF) const
    {
#if(DIM == 2)
        const double thres = 1.0e-2;
        std::vector<int> dupVI2GroupI(V.rows());
        std::vector<std::set<int>> meshVGroup(V.rows());
        std::vector<int> dupVI2GroupI_3D(V_rest.rows());
        std::vector<std::set<int>> meshVGroup_3D(V_rest.rows());
        for(int dupI = 0; dupI < V.rows(); dupI++) {
            dupVI2GroupI[dupI] = dupI;
            meshVGroup[dupI].insert(dupI);
            dupVI2GroupI_3D[dupI] = dupI;
            meshVGroup_3D[dupI].insert(dupI);
        }
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            if(boundaryEdge[cohI]) {
                continue;
            }
            
            for(int pI = 0; pI < 2; pI++) {
                int groupI0_3D = dupVI2GroupI_3D[cohE(cohI, 0 + pI)];
                int groupI2_3D = dupVI2GroupI_3D[cohE(cohI, 2 + pI)];
                if(groupI0_3D != groupI2_3D) {
                    for(const auto& vI : meshVGroup_3D[groupI2_3D]) {
                        dupVI2GroupI_3D[vI] = groupI0_3D;
                        meshVGroup_3D[groupI0_3D].insert(vI);
                    }
                    meshVGroup_3D[groupI2_3D].clear();
                }
           
                // this is only for gluing the triangle soup
//                if((V.row(cohE(cohI, 0 + pI)) - V.row(cohE(cohI, 2 + pI))).norm() / avgEdgeLen > thres) {
                    continue;
//                }
                
                int groupI0 = dupVI2GroupI[cohE(cohI, 0 + pI)];
                int groupI2 = dupVI2GroupI[cohE(cohI, 2 + pI)];
                if(groupI0 != groupI2) {
                    for(const auto& vI : meshVGroup[groupI2]) {
                        dupVI2GroupI[vI] = groupI0;
                        meshVGroup[groupI0].insert(vI);
                    }
                    meshVGroup[groupI2].clear();
                }
            }
        }
        
        int meshVAmt = 0;
        for(int gI = 0; gI < meshVGroup.size(); gI++) {
            if(!meshVGroup[gI].empty()) {
                meshVAmt++;
            }
        }
        std::vector<int> groupI2meshVI(meshVGroup.size(), -1);
        Eigen::MatrixXd UV_mesh;
        UV_mesh.resize(meshVAmt, 2);
        
        int meshVAmt_3D = 0;
        for(int gI = 0; gI < meshVGroup_3D.size(); gI++) {
            if(!meshVGroup_3D[gI].empty()) {
                meshVAmt_3D++;
            }
        }
        std::vector<int> groupI2meshVI_3D(meshVGroup_3D.size(), -1);
        Eigen::MatrixXd V_mesh;
        V_mesh.resize(meshVAmt_3D, 3);
        
        int nextVI = 0;
        for(int gI = 0; gI < meshVGroup.size(); gI++) {
            if(meshVGroup[gI].empty()) {
                continue;
            }
            
            groupI2meshVI[gI] = nextVI;
            UV_mesh.row(nextVI) = Eigen::RowVector2d::Zero();
            for(const auto dupVI : meshVGroup[gI]) {
                UV_mesh.row(nextVI) += V.row(dupVI);
            }
            UV_mesh.row(nextVI) /= meshVGroup[gI].size();
            nextVI++;
        }
        int nextVI_3D = 0;
        for(int gI = 0; gI < meshVGroup_3D.size(); gI++) {
            if(meshVGroup_3D[gI].empty()) {
                continue;
            }
            
            groupI2meshVI_3D[gI] = nextVI_3D;
            V_mesh.row(nextVI_3D) = Eigen::RowVector3d::Zero();
            for(const auto dupVI : meshVGroup_3D[gI]) {
                V_mesh.row(nextVI_3D) += V_rest.row(dupVI);
            }
            V_mesh.row(nextVI_3D) /= meshVGroup_3D[gI].size();
            nextVI_3D++;
        }
        
        Eigen::MatrixXi F_mesh, FUV_mesh;
        F_mesh.resize(F.rows(), 3);
        FUV_mesh.resize(F.rows(), 3);
        for(int triI = 0; triI < F.rows(); triI++) {
            for(int vI = 0; vI < 3; vI++) {
                int groupI = dupVI2GroupI[F(triI, vI)];
                assert(groupI >= 0);
                int meshVI = groupI2meshVI[groupI];
                assert(meshVI >= 0);
                FUV_mesh(triI, vI) = meshVI;
                
                int groupI_3D = dupVI2GroupI_3D[F(triI, vI)];
                assert(groupI_3D >= 0);
                int meshVI_3D = groupI2meshVI_3D[groupI_3D];
                assert(meshVI_3D >= 0);
                F_mesh(triI, vI) = meshVI_3D;
            }
        }
        
        if(scaleUV) {
            const Eigen::VectorXd& u = UV_mesh.col(0);
            const Eigen::VectorXd& v = UV_mesh.col(1);
            const double uMin = u.minCoeff();
            const double vMin = v.minCoeff();
            const double uScale = u.maxCoeff() - uMin;
            const double vScale = v.maxCoeff() - vMin;
            const double scale = std::max(uScale, vScale);
            for(int uvI = 0; uvI < UV_mesh.rows(); uvI++) {
                UV_mesh(uvI, 0) = (UV_mesh(uvI, 0) - uMin) / scale;
                UV_mesh(uvI, 1) = (UV_mesh(uvI, 1) - vMin) / scale;
            }
        }
        
        save(filePath, V_mesh, F_mesh, UV_mesh, FUV_mesh);
#else
        IglUtils::saveTetMesh(filePath, V, F, SF);
#endif
    }
    
    template<int dim>
    void TriangleSoup<dim>::constructSubmesh(const Eigen::VectorXi& triangles,
                                        TriangleSoup& submesh,
                                        std::map<int, int>& globalVIToLocal,
                                        std::map<int, int>& globalTriIToLocal) const
    {
        assert(dim == 2);
#if(DIM == 2)
        Eigen::MatrixXi F_sub;
        F_sub.resize(triangles.size(), 3);
        Eigen::MatrixXd V_rest_sub, V_sub;
        globalVIToLocal.clear();
        for(int localTriI = 0; localTriI < triangles.size(); localTriI++) {
            int triI = triangles[localTriI];
            for(int vI = 0; vI < 3; vI++) {
                int globalVI = F(triI, vI);
                auto localVIFinder = globalVIToLocal.find(globalVI);
                if(localVIFinder == globalVIToLocal.end()) {
                    int localVI = static_cast<int>(V_rest_sub.rows());
                    V_rest_sub.conservativeResize(localVI + 1, 3);
                    V_rest_sub.row(localVI) = V_rest.row(globalVI);
                    V_sub.conservativeResize(localVI + 1, 2);
                    V_sub.row(localVI) = V.row(globalVI);
                    F_sub(localTriI, vI) = localVI;
                    globalVIToLocal[globalVI] = localVI;
                }
                else {
                    F_sub(localTriI, vI) = localVIFinder->second;
                }
            }
            globalTriIToLocal[triI] = localTriI;
        }
        submesh = TriangleSoup(V_rest_sub, F_sub, V_sub);
        
        std::set<int> fixedVert_sub;
        for(const auto& fixedVI : fixedVert) {
            auto finder = globalVIToLocal.find(fixedVI);
            if(finder != globalVIToLocal.end()) {
                fixedVert_sub.insert(finder->second);
            }
        }
        submesh.resetFixedVert(fixedVert_sub);
#endif
    }
    
    template<int dim>
    bool TriangleSoup<dim>::isBoundaryVert(int vI, int vI_neighbor,
                                           std::vector<int>& tri_toSep,
                                           std::pair<int, int>& boundaryEdge,
                                           bool toBound) const
    {
        assert(dim == 2);
#if(DIM == 2)
        //        const auto inputEdgeTri = edge2Tri.find(toBound ? std::pair<int, int>(vI, vI_neighbor) :
        //                                                std::pair<int, int>(vI_neighbor, vI));
        //        assert(inputEdgeTri != edge2Tri.end());
        
        tri_toSep.resize(0);
        auto finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_neighbor, vI):
                                    std::pair<int, int>(vI, vI_neighbor));
        if(finder == edge2Tri.end()) {
            boundaryEdge.first = (toBound ? vI : vI_neighbor);
            boundaryEdge.second = (toBound ? vI_neighbor : vI);
            return true;
        }
        
        int vI_new = vI_neighbor;
        do {
            tri_toSep.emplace_back(finder->second);
            const Eigen::RowVector3i& triVInd = F.row(finder->second);
            for(int i = 0; i < 3; i++) {
                if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                    vI_new = triVInd[i];
                    break;
                }
            }
            
            if(vI_new == vI_neighbor) {
                return false;
            }
            
            finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_new, vI) :
                                   std::pair<int, int>(vI, vI_new));
            if(finder == edge2Tri.end()) {
                boundaryEdge.first = (toBound ? vI : vI_new);
                boundaryEdge.second = (toBound ? vI_new : vI);
                return true;
            }
        } while(1);
#endif
    }
    
    template<int dim>
    bool TriangleSoup<dim>::isBoundaryVert(int vI) const
    {
        assert(vNeighbor.size() == V.rows());
        assert(vI < vNeighbor.size());
        
        for(const auto vI_neighbor : vNeighbor[vI]) {
            if((edge2Tri.find(std::pair<int, int>(vI, vI_neighbor)) == edge2Tri.end()) ||
               (edge2Tri.find(std::pair<int, int>(vI_neighbor, vI)) == edge2Tri.end()))
            {
                return true;
            }
        }
        
        return false;
    }
    
    template<int dim>
    void TriangleSoup<dim>::compute2DInwardNormal(int vI, Eigen::RowVector2d& normal) const
    {
        assert(dim == 2);
#if(DIM == 2)
        std::vector<int> incTris[2];
        std::pair<int, int> boundaryEdge[2];
        if(!isBoundaryVert(vI, *vNeighbor[vI].begin(), incTris[0], boundaryEdge[0], 0)) {
            return;
        }
        isBoundaryVert(vI, *vNeighbor[vI].begin(), incTris[1], boundaryEdge[1], 1);
        assert(!(incTris[0].empty() && incTris[1].empty()));
        
        Eigen::RowVector2d boundaryEdgeDir[2] = {
            (V.row(boundaryEdge[0].first) - V.row(boundaryEdge[0].second)).normalized(),
            (V.row(boundaryEdge[1].second) - V.row(boundaryEdge[1].first)).normalized(),
        };
        normal = (boundaryEdgeDir[0] + boundaryEdgeDir[1]).normalized();
        if(boundaryEdgeDir[1][0] * normal[1] - boundaryEdgeDir[1][1] * normal[0] < 0.0) {
            normal *= -1.0;
        }
#endif
    }
    
    template class TriangleSoup<DIM>;
    
}
