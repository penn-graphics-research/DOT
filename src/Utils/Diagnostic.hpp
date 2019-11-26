//
//  Diagnostic.hpp
//  DOT
//
//  Created by Minchen Li on 1/31/18.
//

#ifndef Diagnostic_hpp
#define Diagnostic_hpp

#include "FixedCoRotEnergy.hpp"
#include "StableNHEnergy.hpp"

#include "Mesh.hpp"
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
extern Eigen::MatrixXi SF;
extern std::vector<int> sTri2Tet;
extern bool viewUV;
extern int showDistortion;
extern Eigen::MatrixXd faceColors_default;
extern bool showFixedVerts;

extern GifWriter GIFWriter;
extern uint32_t GIFDelay; //*10ms
extern double GIFScale;

extern std::vector<const DOT::Mesh<DIM>*> triSoup;
extern std::vector<DOT::Energy<DIM>*> energyTerms;
extern std::vector<double> energyParams;
extern DOT::Optimizer<DIM>* optimizer;

extern void updateViewerData(void);
extern bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier);
extern bool preDrawFunc(igl::opengl::glfw::Viewer& viewer);
extern bool postDrawFunc(igl::opengl::glfw::Viewer& viewer);
extern void saveScreenshot(const std::string& filePath, double scale, bool writeGIF, bool writePNG);

namespace DOT{
    class Diagnostic
    {
    public:
        static void run(int argc, char *argv[])
        {
            if(argc > 2) {
                int diagMode = 0;
                diagMode = std::stoi(argv[2]);
                switch(diagMode) {
                    case 11: { // a unit test for computing dE/dsigma
                        std::vector<Energy<DIM>*> e;
                        e.emplace_back(new FixedCoRotEnergy<DIM>);
                        e.emplace_back(new StableNHEnergy<DIM>);
                        for(const auto eI : e) {
                            eI->unitTest_dE_div_dsigma();
                            eI->unitTest_d2E_div_dsigma2();
                            eI->unitTest_BLeftCoef();
                            eI->unitTest_dE_div_dF();
                            eI->unitTest_dP_div_dF();
                            delete eI;
                        }
                        break;
                    }
                        
                    case 13: { // extract sysE from log.txt
                        if(argc < 4) {
                            std::cout << "please input log file path" << std::endl;
                        }
                        else {
                            FILE *in = fopen(argv[3], "r");
                            if(in) {
                                const std::string filePath = std::string(argv[3]);
                                const std::string fileFolderPath = filePath.substr(0, filePath.find_last_of('/') + 1);
                                
                                FILE *out = fopen((fileFolderPath + "sysE.txt").c_str(), "w+");
                                assert(out);
                                
                                char buf[BUFSIZ];
                                while((!feof(in)) && fgets(buf, BUFSIZ, in)) {
                                    std::string line(buf);
                                    if((line.find("sysE") != std::string::npos)) {
                                        double sysE;
                                        sscanf(buf, "sysE = %le", &sysE);
                                        fprintf(out, "%le\n", sysE);
                                    }
                                }
                                
                                fclose(out);
                                fclose(in);
                            }
                            else {
                                std::cout << "can't open log file" << std::endl;
                            }
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
