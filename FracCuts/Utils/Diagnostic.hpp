//
//  Diagnostic.hpp
//  FracCuts
//
//  Created by Minchen Li on 1/31/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Diagnostic_hpp
#define Diagnostic_hpp

#include "FixedCoRotEnergy.hpp"

#include "TriangleSoup.hpp"
#include "GIF.hpp"

#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <cstdio>

extern std::string outputFolderPath;
extern igl::opengl::glfw::Viewer viewer;
extern bool viewUV;
extern bool showTexture;
extern int showDistortion;
extern Eigen::MatrixXd faceColors_default;
extern double texScale;
extern bool showFixedVerts;

extern GifWriter GIFWriter;
extern uint32_t GIFDelay; //*10ms
extern double GIFScale;

extern std::vector<const FracCuts::TriangleSoup<DIM>*> triSoup;
extern std::vector<FracCuts::Energy<DIM>*> energyTerms;
extern std::vector<double> energyParams;
extern FracCuts::Optimizer<DIM>* optimizer;

extern void updateViewerData(void);
extern bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier);
extern bool preDrawFunc(igl::opengl::glfw::Viewer& viewer);
extern bool postDrawFunc(igl::opengl::glfw::Viewer& viewer);
extern void saveScreenshot(const std::string& filePath, double scale, bool writeGIF, bool writePNG);

namespace FracCuts{
    class Diagnostic
    {
    public:
        static void run(int argc, char *argv[])
        {
            if(argc > 2) {
                int diagMode = 0;
                diagMode = std::stoi(argv[2]);
                switch(diagMode) {
                    case 6: {
                        // output ExpInfo into js variables for local web visualization
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        FILE *outFile = fopen((resultsFolderPath + "/data_ExpInfo.js").c_str(), "w");
                        assert(outFile);
                        
                        std::string resultsFolderName;
                        int endI_substr = resultsFolderPath.find_last_of('/');
                        if(endI_substr == std::string::npos) {
                            resultsFolderName = resultsFolderPath;
                        }
                        else {
                            if(endI_substr == resultsFolderPath.length() - 1) {
                                endI_substr = resultsFolderPath.find_last_of('/', endI_substr - 1);
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 2);
                            }
                            else {
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 1);
                            }
                        }
                        
                        fprintf(outFile, "var %s = [\n", resultsFolderName.c_str());
                        
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string resultName(buf);
                            std::string infoFilePath(resultsFolderPath + '/' + resultName + "/info.txt");
                            std::ifstream infoFile(infoFilePath);
                            if(infoFile.is_open()) {
                                fprintf(outFile, "\tnew ExpInfo(");
                                
                                std::string bypass;
                                int vertAmt, faceAmt;
                                infoFile >> vertAmt >> faceAmt;
                                fprintf(outFile, "%d, %d, ", vertAmt, faceAmt);
                                
                                int innerIterNum, outerIterNum;
                                double lambda_init, lambda_end;
                                infoFile >> innerIterNum >> outerIterNum >>
                                    bypass >> bypass >>
                                    lambda_init >> lambda_end;
                                fprintf(outFile, "%d, %d, 0, 0, %lf, %lf, ",
                                        innerIterNum, outerIterNum, lambda_init, lambda_end);
                                
                                double time, duration;
                                infoFile >> bypass >> bypass >> time >> duration;
                                for(int wordI = 0; wordI < 13; wordI++) {
                                    infoFile >> bypass;
                                }
                                fprintf(outFile, "0.0, 0.0, %lf, [], [], ", time);
                                
                                double E_d, E_s;
                                infoFile >> E_d >> E_s;
                                fprintf(outFile, "%lf, %lf, ", E_d, E_s);
                                
                                double l2Stretch, lInfStretch, l2Shear, lInfCompress;
                                infoFile >> l2Stretch >> lInfStretch >> l2Shear >> lInfCompress;
                                fprintf(outFile, "%lf, %lf, %lf, ", l2Stretch, lInfStretch, l2Shear);
                                
                                fprintf(outFile, "\"%s\"", buf);
                                
                                fprintf(outFile, "),\n");
                                
                                infoFile.close();
                            }
                            else {
                                std::cout << "can't open " << infoFilePath << std::endl;
                            }
                        }
                        fprintf(outFile, "];\n");
                        
                        fclose(dirList);
                        fclose(outFile);
                        std::cout << "output finished" << std::endl;
                        
                        break;
                    }
                        
                    case 7: {
                        // check whether oscillation detected or converged exactly
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        std::string resultsFolderName;
                        int endI_substr = resultsFolderPath.find_last_of('/');
                        if(endI_substr == std::string::npos) {
                            resultsFolderName = resultsFolderPath;
                        }
                        else {
                            if(endI_substr == resultsFolderPath.length() - 1) {
                                endI_substr = resultsFolderPath.find_last_of('/', endI_substr - 1);
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 2);
                            }
                            else {
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 1);
                            }
                        }
                        
                        char buf[BUFSIZ];
                        int oscAmt = 0, exactConvAmt = 0, rollbackAmt_osc = 0, rollbackAmt_conv = 0;
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string resultName(buf);
                            std::string infoFilePath(resultsFolderPath + '/' + resultName + "/log.txt");
                            FILE *infoFile = fopen(infoFilePath.c_str(), "r");
                            if(infoFile) {
                                std::vector<std::string> lines;
                                char line[BUFSIZ];
                                while((!feof(infoFile)) && fgets(line, BUFSIZ, infoFile)) {
                                    lines.emplace_back(line);
                                }
                                
                                bool oscillation = false, issue = true;
                                for(int i = 0; (i < 10) && (i < lines.size()); i++) {
                                    if(lines[lines.size() - 1 - i].find("oscillation") != std::string::npos) {
                                        issue = false;
                                        oscillation = true;
                                        oscAmt++;
                                        break;
                                    }
                                    if(lines[lines.size() - 1 - i].find("all converged") != std::string::npos) {
                                        issue = false;
                                        exactConvAmt++;
                                        break;
                                    }
                                }
                                
                                for(int i = 0; (i < 10) && (i < lines.size()); i++) {
                                    if(lines[lines.size() - 1 - i].find("rolled back") != std::string::npos) {
                                        if(oscillation) {
                                            rollbackAmt_osc++;
                                        }
                                        else {
                                            rollbackAmt_conv++;
                                        }
                                    }
                                }
                                
                                if(issue) {
                                    std::cout << resultName << std::endl;
                                }
                                
                                fclose(infoFile);
                            }
                            else {
                                std::cout << "can't open " << infoFilePath << std::endl;
                            }
                        }
                        
                        fclose(dirList);
                        std::cout << oscAmt << " oscillation (" << rollbackAmt_osc << " rollbacks), " <<
                            exactConvAmt << " exact convergence (" << rollbackAmt_conv << " rollbacks)." << std::endl;
                        
                        break;
                    }
                        
                    case 9: {
                        //TODO: support tet mesh
                        // visualize ADMM inner iterations
                        if(argc < 5) {
                            std::cout << "not enough command line arguments" << std::endl;
                            break;
                        }
                        
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        // for rendering:
                        energyTerms.emplace_back(new FracCuts::SymStretchEnergy<DIM>());
                        energyParams.emplace_back(1.0);
                        triSoup.resize(2);
                        viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
                        viewer.callback_key_down = &key_down;
                        viewer.callback_pre_draw = &preDrawFunc;
                        viewer.callback_post_draw = &postDrawFunc;
                        viewer.data().show_lines = true;
                        viewer.core.orthographic = true;
                        viewer.core.camera_zoom *= 1.5;
                        viewer.core.animation_max_fps = 60.0;
                        viewer.data().point_size = 16.0f;
                        viewer.data().show_overlay = true;
                        viewer.core.is_animating = true;
                        viewer.launch_init(true, false);
                        showDistortion = -1;
                        showFixedVerts = false;
                        showTexture = false;
                        GIFDelay = 40;
                        
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            int timestepI = std::stoi(argv[4]);
                            GifBegin(&GIFWriter, (resultsFolderPath + '/' + std::string(buf) + "/timestep" +
                                                  std::to_string(timestepI) + "/" + "subdomains.gif").c_str(),
                                     GIFScale * (viewer.core.viewport[2] - viewer.core.viewport[0]),
                                     GIFScale * (viewer.core.viewport[3] - viewer.core.viewport[1]), GIFDelay);
                            for(int ADMMIterI = 0; true; ADMMIterI++) {
                                int subdomainI = 0;
                                std::string meshPath(resultsFolderPath + '/' + std::string(buf) + "/timestep" +
                                                     std::to_string(timestepI) + "/" +
                                                     std::to_string(ADMMIterI) + "_subdomain0.obj");
                                Eigen::MatrixXd V, UV, N;
                                Eigen::MatrixXi F, FUV, FN;
                                Eigen::MatrixXd V_, UV_;
                                Eigen::MatrixXi F_, FUV_;
                                Eigen::VectorXd faceLabel;
                                while(igl::readOBJ(meshPath, V, UV, N, F, FUV, FN)) {
                                    F.array() += V_.rows();
                                    F_.conservativeResize(F_.rows() + F.rows(), 3);
                                    F_.bottomRows(F.rows()) = F;
                                    if(FUV.rows() > 0) {
                                        FUV.array() += UV_.rows();
                                        FUV_.conservativeResize(FUV_.rows() + FUV.rows(), 3);
                                        FUV_.bottomRows(FUV.rows()) = FUV;
                                    }
                                    V_.conservativeResize(V_.rows() + V.rows(), 3);
                                    V_.bottomRows(V.rows()) = V;
                                    UV_.conservativeResize(UV_.rows() + V.rows(), 2);
                                    UV_.bottomRows(V.rows()) = V.leftCols(2);
                                    faceLabel.conservativeResize(faceLabel.rows() + F.rows());
                                    faceLabel.bottomRows(F.rows()).setConstant(subdomainI);
                                    
                                    subdomainI++;
                                    meshPath = resultsFolderPath + '/' + std::string(buf) +
                                        "/timestep" + std::to_string(timestepI) + "/" +
                                        std::to_string(ADMMIterI) + "_subdomain" +
                                        std::to_string(subdomainI) + ".obj";
                                }
                                if(subdomainI == 0) {
                                    break;
                                }
                                std::cout << subdomainI << " subdomain meshes loaded" << std::endl;
                                TriangleSoup<DIM> resultMesh(V_, F_, UV_);
                                
                                triSoup[0] = triSoup[1] = &resultMesh;
                                if(ADMMIterI == 0) {
                                    texScale = 10.0 / (triSoup[0]->bbox.row(1) -
                                                       triSoup[0]->bbox.row(0)).maxCoeff();
                                }
                                optimizer = new FracCuts::Optimizer<DIM>(*triSoup[0], energyTerms, energyParams, 0, false, false);
                                igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, faceLabel, true, faceColors_default);
                                updateViewerData();
                                viewer.launch_rendering(false);
                                saveScreenshot(resultsFolderPath + '/' + std::string(buf) + "/timestep" +
                                               std::to_string(timestepI) + "/" + std::to_string(ADMMIterI) +
                                               "_subdomains.png", 0.5, true, true);
                                
                                std::cout << buf << " processed" << std::endl;
                                delete optimizer;
                            }
                            GifEnd(&GIFWriter);
                        }
                        
                        fclose(dirList);
                        
                        break;
                    }
                        
                    case 10: { // a unit test for computing p
                        Energy<DIM>* e = new FixedCoRotEnergy<DIM>;
                        
                        Eigen::Matrix<double, DIM, DIM> F;
                        F.setIdentity();
                        AutoFlipSVD<Eigen::Matrix<double, DIM, DIM>> svd(F, Eigen::ComputeFullU |
                                                                         Eigen::ComputeFullV);
                        
                        Eigen::Matrix<double, DIM, DIM> dE_div_dF;
                        e->compute_dE_div_dF(F, svd, dE_div_dF);
                        
                        std::cout << dE_div_dF << std::endl;
                        
                        delete e;
                        
                        break;
                    }
                        
                    case 100: {
                        FILE *in = fopen(argv[3], "r");
                        if(in) {
                            char buf[BUFSIZ];
                            Eigen::VectorXi count;
                            while((!feof(in)) && fgets(buf, BUFSIZ, in)) {
                                std::string line(buf);
                                if((line.find("stat") != std::string::npos) && (line.find("conjugate_residual iterations")
                                    != std::string::npos))
                                {
                                    int numStart = line.find('"', line.find("value")) + 1;
                                    count.conservativeResize(count.size() + 1);
                                    count[count.size() - 1] = stoi(line.substr(numStart, line.find('"', numStart) - numStart));
                                }
                            }
                            std::cout << "avg = " << double(count.sum()) / count.size() << std::endl;
                            std::cout << "min = " << count.minCoeff() << std::endl;
                            std::cout << "max = " << count.maxCoeff() << std::endl;
                            std::cout << "sum = " << count.sum() << std::endl;
                        }
                        else {
                            std::cout << "can't open file " << argv[3] << std::endl;
                        }
                        break;
                    }
                        
                    default:
                        std::cout << "No diagMode " << diagMode << std::endl;
                        break;
                }
            }
            else {
                std::cout << "Please enter diagMode!" << std::endl;
            }
        }
    };
}

#endif /* Diagnostic_hpp */
