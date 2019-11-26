#include "Types.hpp"
#include "IglUtils.hpp"
#include "Config.hpp"
#include "Optimizer.hpp"
#include "ADMMTimeStepper.hpp"
#include "ADMMDDTimeStepper.hpp"
#include "LBFGSTimeStepper.hpp"
#include "DOTTimeStepper.hpp"
#include "FixedCoRotEnergy.hpp"
#include "GIF.hpp"
#include "Timer.hpp"

#include "Diagnostic.hpp"
#include "MeshProcessing.hpp"

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>

#include <sys/stat.h> // for mkdir

#include <fstream>
#include <string>
#include <ctime>


// optimization/simulation
DOT::Config config;
std::vector<const DOT::Mesh<DIM>*> triSoup;
int vertAmt_input;
DOT::Optimizer<DIM>* optimizer;
std::vector<DOT::Energy<DIM>*> energyTerms;
std::vector<double> energyParams;
bool saveStatus = true;
bool savePNG = false;
bool offlineMode = false;
bool autoSwitch = false;
bool optimization_on = false;
int iterNum = 0;
int converged = 0;
bool outerLoopFinished = false;

std::ofstream logFile;
std::string outputFolderPath = "output/";

// visualization
igl::opengl::glfw::Viewer viewer;
const int channel_initial = 0;
const int channel_result = 1;
const int channel_findExtrema = 2;
int viewChannel = channel_result;
Eigen::MatrixXi SF;
std::vector<int> sTri2Tet;
int showDistortion = 2; // 0: don't show; 1: energy value; 2: other scalar field;
int showDistortion_init = showDistortion;
bool isLighting = true;
bool showFixedVerts = true; //TODO: add key control
float fracTailSize = 20.0f;
Eigen::MatrixXd faceColors_default;
bool viewCurrent = true; // view current shape or rest shape

std::vector<bool> isSurfNode;
std::vector<int> tetIndToSurf;
std::vector<int> surfIndToTet;
Eigen::MatrixXd V_surf;
Eigen::MatrixXi F_surf;

bool saveInfo_postDraw = false;
std::string infoName = "";

GifWriter GIFWriter;
uint32_t GIFDelay = 10; //*10ms
int GIFStep = 1;
double GIFScale = 0.6;

// timer
double secPast = 0.0;
time_t lastStart_world;
Timer timer, timer_step, timer_temp3;

// SIMD
double *a11,*a21,*a31,*a12,*a22,*a32,*a13,*a23,*a33;
double *u11,*u21,*u31,*u12,*u22,*u32,*u13,*u23,*u33;
double *v11,*v21,*v31,*v12,*v22,*v32,*v13,*v23,*v33;
double *sigma1,*sigma2,*sigma3;

double *Gmu, *Glambda, *Gsigma0, *Gsigma1, *Gsigma2;


void saveInfo(bool writePNG = true, bool writeGIF = true, int writeMesh = 1);

void proceedOptimization(int proceedNum = 1)
{
    for(int proceedI = 0; (proceedI < proceedNum) && (!converged); proceedI++) {
        // PNG output only works under online rendering mode
        saveInfo(false, true, false);
        if(savePNG) {
            infoName = std::to_string(iterNum);
            saveScreenshot(outputFolderPath + infoName + ".png",
                            1.0, false, true);
        }
        saveInfo(false, false, 3);

        showDistortion = showDistortion_init;
        
        std::cout << "Timestep" << iterNum << ":" << std::endl;
        
        if (!config.tol.empty()) {
            if(iterNum < config.tol.size()) {
                optimizer->setRelGL2Tol(config.tol[iterNum]);
            }
            else {
                optimizer->setRelGL2Tol(config.tol.back());
            }
        }
        else {
            optimizer->setRelGL2Tol();
        }
        
        converged = optimizer->solve(1);
        if(converged == 2) {
            showDistortion = 0;
            converged = 0;
            logFile << "!!! maxIter reached for timeStep" << iterNum << std::endl;
        }
        iterNum = optimizer->getIterNum();

#ifdef FIRST_TIME_STEP
        converged = true;
#endif
    }
}

void updateViewerData_distortion(void)
{
    Eigen::MatrixXd color_distortionVis;
    
    switch(showDistortion) {
        case 1: { // show elasticity energy value
            Eigen::VectorXd distortionPerElem;
            energyTerms[0]->getEnergyValPerElem(*triSoup[viewChannel], distortionPerElem, true);
            const Eigen::VectorXd& visRange = energyTerms[0]->getVisRange_energyVal();
            DOT::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis,
                                                 visRange[0], visRange[1]);
            break;
        }
            
        case 2: { // show other triangle-based scalar fields
            Eigen::VectorXd l2StretchPerElem;
            Eigen::VectorXd faceWeight;
//            faceWeight.resize(triSoup[viewChannel]->F.rows());
//            for(int fI = 0; fI < triSoup[viewChannel]->F.rows(); fI++) {
//                const Eigen::RowVector3i& triVInd = triSoup[viewChannel]->F.row(fI);
//                faceWeight[fI] = (triSoup[viewChannel]->vertWeight[triVInd[0]] +
//                                  triSoup[viewChannel]->vertWeight[triVInd[1]] +
//                                  triSoup[viewChannel]->vertWeight[triVInd[2]]) / 3.0;
//            }
#if(DIM == 2)
            optimizer->getFaceFieldForVis(faceWeight);
#else
            Eigen::VectorXd faceWeight_tet;
            optimizer->getFaceFieldForVis(faceWeight_tet);
            faceWeight.conservativeResize(SF.rows());
            for(int sfI = 0; sfI < SF.rows(); sfI++) {
                faceWeight[sfI] = faceWeight_tet[sTri2Tet[sfI]];
            }
#endif
//            DOT::IglUtils::mapScalarToColor(faceWeight, color_distortionVis,
//                faceWeight.minCoeff(), faceWeight.maxCoeff());
            igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, faceWeight, true, color_distortionVis);
            color_distortionVis.array() += std::min(0.2, 1.0 - color_distortionVis.maxCoeff());
            break;
        }
    
        case 0: {
#if(DIM == 2)
            color_distortionVis = Eigen::MatrixXd::Ones(triSoup[viewChannel]->F.rows(), 3);
#else
            color_distortionVis = Eigen::MatrixXd::Ones(SF.rows(), 3);
#endif
            color_distortionVis.col(2).setZero();
            break;
        }
            
        default:
            assert(0 && "unknown distortion visualization option!");
            break;
    }
    
    viewer.data().set_colors(color_distortionVis);
}

void updateViewerData(void)
{
    Eigen::MatrixXd UV_vis = triSoup[viewChannel]->V;
    Eigen::MatrixXi F_vis = ((DIM == 2) ? triSoup[viewChannel]->F : SF);
    if(viewCurrent) {
        if(DIM == 2) {
            UV_vis.conservativeResize(UV_vis.rows(), 3);
            UV_vis.rightCols(1) = Eigen::VectorXd::Zero(UV_vis.rows());
        }
        viewer.core.align_camera_center(triSoup[viewChannel]->V_rest, F_vis);
        
        if((UV_vis.rows() != viewer.data().V.rows()) ||
           (F_vis.rows() != viewer.data().F.rows()))
        {
            viewer.data().clear();
        }
        viewer.data().set_mesh(UV_vis, F_vis);
        
        viewer.data().show_texture = false;
#if(DIM == 2)
        viewer.core.lighting_factor = 0.0;
#else
        if(isLighting) {
            viewer.core.lighting_factor = 0.6;
        }
        else {
            viewer.core.lighting_factor = 0.0;
        }
#endif

        viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 0.0));
        if(showFixedVerts) {
            for(const auto& fixedVI : triSoup[viewChannel]->fixedVert) {
                viewer.data().add_points(UV_vis.row(fixedVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
//        Eigen::VectorXi sharedVerts;
//        optimizer->getSharedVerts(sharedVerts);
//        for(int svI = 0; svI < sharedVerts.size(); svI++) {
//            viewer.data().add_points(UV_vis.row(sharedVerts[svI]), Eigen::RowVector3d(1.0, 0.3, 0.3));
//        }
    }
    else {    
    Eigen::MatrixXd V_vis = triSoup[viewChannel]->V_rest;
    viewer.core.align_camera_center(V_vis, F_vis);
    
    if((V_vis.rows() != viewer.data().V.rows()) ||
        (UV_vis.rows() != viewer.data().V_uv.rows()) ||
        (F_vis.rows() != viewer.data().F.rows()))
    {
        viewer.data().clear();
    }
    viewer.data().set_mesh(V_vis, F_vis);
    
    viewer.data().show_texture = false;
    
    if(isLighting) {
        viewer.core.lighting_factor = 1.0;
    }
    else {
        viewer.core.lighting_factor = 0.0;
    }
    
    viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 0.0));
    if(showFixedVerts) {
        for(const auto& fixedVI : triSoup[viewChannel]->fixedVert) {
            viewer.data().add_points(V_vis.row(fixedVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
        }
    }
//        Eigen::VectorXi sharedVerts;
//        optimizer->getSharedVerts(sharedVerts);
//        for(int svI = 0; svI < sharedVerts.size(); svI++) {
//            viewer.data().add_points(V_vis.row(sharedVerts[svI]), Eigen::RowVector3d(1.0, 0.3, 0.3));
//        }
        }
    updateViewerData_distortion();
    
    viewer.data().compute_normals();
}

void saveScreenshot(const std::string& filePath, double scale = 1.0, bool writeGIF = false, bool writePNG = true)
{
    if(offlineMode) {
        return;
    }
    
    if(writeGIF) {
        scale = GIFScale;
    }
    viewer.data().point_size = fracTailSize * scale;
    
    int width = static_cast<int>(scale * (viewer.core.viewport[2] - viewer.core.viewport[0]));
    int height = static_cast<int>(scale * (viewer.core.viewport[3] - viewer.core.viewport[1]));
    
    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);
    
    // Draw the scene in the buffers
    viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);
    
    if(writePNG) {
        // Save it to a PNG
        igl::png::writePNG(R, G, B, A, filePath);
    }
    
    if(writeGIF && (iterNum % GIFStep == 0)) {
        std::vector<uint8_t> img(width * height * 4);
        for(int rowI = 0; rowI < width; rowI++) {
            for(int colI = 0; colI < height; colI++) {
                int indStart = (rowI + (height - 1 - colI) * width) * 4;
                img[indStart] = R(rowI, colI);
                img[indStart + 1] = G(rowI, colI);
                img[indStart + 2] = B(rowI, colI);
                img[indStart + 3] = A(rowI, colI);
            }
        }
        GifWriteFrame(&GIFWriter, img.data(), width, height, GIFDelay);
    }
    
    viewer.data().point_size = fracTailSize;
}

void saveInfo(bool writePNG, bool writeGIF, int writeMesh)
{
    saveScreenshot(outputFolderPath + infoName + ".png", 0.5, writeGIF, writePNG);
    switch(writeMesh) {
        case 1:
        case 2:
//            triSoup[channel_result]->save(outputFolderPath + infoName + "_triSoup.obj");
            triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + "_mesh" +
                                            ((DIM == 2) ? ".obj" : ".msh"), false, SF);
        break;
            
        case 3: // save status
            optimizer->saveStatus();
            break;
    
        default:
            break;
    }
}

void saveInfoForPresent(const std::string fileName = "info.txt")
{
    std::ofstream file;
    file.open(outputFolderPath + fileName);
    assert(file.is_open());
    
    file << vertAmt_input << " " <<
        triSoup[channel_initial]->F.rows() << std::endl;
    
    file << iterNum << " " << optimizer->getInnerIterAmt() << " 0 0 " << 1.0 - energyParams[0] << std::endl;
    
    timer.print(file);
    timer_step.print(file);
    timer_temp3.print(file);
    
    double distortion=0.0;
    //energyTerms[0]->computeEnergyVal(*triSoup[channel_result], distortion);
    file << distortion << " " << 0.0 << std::endl;
    
    file.close();
}

void toggleOptimization(void)
{
    optimization_on = !optimization_on;
    if(optimization_on) {
        if(converged) {
            optimization_on = false;
            std::cout << "optimization converged." << std::endl;
        }
        else {
            if(iterNum == 0) {
                if (!offlineMode) {
                    GifBegin(&GIFWriter, (outputFolderPath + "anim.gif").c_str(),
                            GIFScale * (viewer.core.viewport[2] - viewer.core.viewport[0]),
                            GIFScale * (viewer.core.viewport[3] - viewer.core.viewport[1]), GIFDelay);
                }
                
                saveScreenshot(outputFolderPath + "0.png", 0.5, true);
            }
            std::cout << "start/resume optimization, press again to pause." << std::endl;
            viewer.core.is_animating = true;
            
            time(&lastStart_world);
        }
    }
    else {
        std::cout << "pause optimization, press again to resume." << std::endl;
        viewer.core.is_animating = false;
        std::cout << "World Time:\nTime past: " << secPast << "s." << std::endl;
        secPast += difftime(time(NULL), lastStart_world);
    }
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if((key >= '0') && (key <= '9')) {
        int changeToChannel = key - '0';
        if((changeToChannel < triSoup.size()) && (viewChannel != changeToChannel)) {
            viewChannel = changeToChannel;
        }
    }
    else {
        switch (key)
        {
            case ' ': {
                proceedOptimization();
                viewChannel = channel_result;
                break;
            }
                
            case '/': {
                toggleOptimization();
                break;
            }
                
            case 'b':
            case 'B': {
                isLighting = !isLighting;
                break;
            }
                
            case 'o':
            case 'O': {
                infoName = std::to_string(iterNum);
                saveInfo(true, false, true);
                break;
            }

            case 'r':
            case 'R': {
                viewCurrent = !viewCurrent;
                break;
            }
                
            default:
                break;
        }
    }
    
    updateViewerData();

    return false;
}

bool postDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    if(autoSwitch && (iterNum == 0)) {
        toggleOptimization();
    }
    
    if(saveInfo_postDraw) {
        saveInfo_postDraw = false;
        saveInfo(outerLoopFinished, true, outerLoopFinished);
        // Note that the content saved in the screenshots are depends on where updateViewerData() is called
        if(outerLoopFinished) {
//            triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + "_mesh_01UV.obj", true);
        }
    }
    
    if(outerLoopFinished) { //var name change!!!
        if (!offlineMode) {
            GifEnd(&GIFWriter);
        }
        saveInfoForPresent();
        if(autoSwitch) {
            exit(0);
        }
        else {
            viewer.core.is_animating = false;
            outerLoopFinished = false;
        }
    }
    
    return false;
}

void converge_preDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    infoName = "finalResult";
    
    secPast += difftime(time(NULL), lastStart_world);
    updateViewerData();
    
    optimization_on = false;
    viewer.core.is_animating = false;
    std::cout << "optimization converged, with " << optimizer->getInnerIterAmt() <<
        " inner iterations in " << secPast << "s." << std::endl;
    logFile << "optimization converged, with " << optimizer->getInnerIterAmt() <<
        " inner iterations in " << secPast << "s." << std::endl;
    outerLoopFinished = true;
}

bool preDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    static bool initViewerData = true;
    if(initViewerData) {
        updateViewerData();
        initViewerData = false;
    }
    
    if(optimization_on)
    {
        if(offlineMode) {
            while(!converged) {
                proceedOptimization();
            }
        }
        else {
            proceedOptimization();
        }
        
//        viewChannel = channel_result;
        updateViewerData();
        
        if(converged) {
            saveInfo_postDraw = true;
            converge_preDrawFunc(viewer);
        }
    }
    return false;
}

void initSIMD(const DOT::Mesh<DIM>* temp)
{
    using T=double;
    int size = std::ceil(temp->F.rows() / 4.f) * 4;
    void* buffers_raw; int buffers_return=0;
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a11=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a21=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a31=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a12=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a22=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a32=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a13=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a23=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); a33=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u11=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u21=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u31=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u12=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u22=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u32=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u13=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u23=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); u33=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v11=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v21=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v31=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v12=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v22=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v32=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v13=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v23=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); v33=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); sigma1=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); sigma2=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return=posix_memalign(&buffers_raw,64,size*8); sigma3=reinterpret_cast<T*>(buffers_raw);
    if(buffers_return!=0) exit(0);
    
    buffers_return = posix_memalign(&buffers_raw, 64, size * 8);        Gmu = reinterpret_cast<T *>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return = posix_memalign(&buffers_raw, 64, size * 8);        Glambda = reinterpret_cast<T *>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return = posix_memalign(&buffers_raw, 64, size * 8);        Gsigma0 = reinterpret_cast<T *>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return = posix_memalign(&buffers_raw, 64, size * 8);        Gsigma1 = reinterpret_cast<T *>(buffers_raw);
    if(buffers_return!=0) exit(0);
    buffers_return = posix_memalign(&buffers_raw, 64, size * 8);        Gsigma2 = reinterpret_cast<T *>(buffers_raw);
    if(buffers_return!=0) exit(0);
}

int main(int argc, char *argv[])
{
    int progMode = 0;
    if(argc > 1) {
        progMode = std::stoi(argv[1]);
    }
    switch(progMode) {
        case 0:
            // optimization mode
            std::cout << "Manually start optimization mode" << std::endl;
            break;
            
        case 10:
            // autoswitch optimization mode
            autoSwitch = true;
            std::cout << "Auto-start optimization mode" << std::endl;
            break;
            
        case 11:
            // autoswitch optimization mode
            autoSwitch = true;
            savePNG = true;
            std::cout << "Auto-start optimization mode with screenshots saved" << std::endl;
            break;
            
        case 100: {
            // offline optimization mode
            autoSwitch = true;
            offlineMode = true;
            std::cout << "Offline optimization mode without visualization" << std::endl;
            break;
        }
            
        case 1: {
            // diagnostic mode
            DOT::Diagnostic::run(argc, argv);
            return 0;
        }
            
        case 2: {
            // mesh processing mode
            DOT::MeshProcessing::run(argc, argv);
            return 0;
        }
            
        default: {
            std::cout<< "No progMode " << progMode << std::endl;
            return 0;
        }
    }
    
    // Optimization mode
    
    std::string meshFileName;
    if(argc > 2) {
        meshFileName = std::string(argv[2]);
    }
    else {
        std::cout << "please input script file path" << std::endl;
        exit(0);
    }

    std::string meshFilePath;
    // The input mesh file name needs to be a global file path
    meshFilePath = meshFileName;
    meshFileName = meshFileName.substr(meshFileName.find_last_of('/') + 1);
    std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
    // Load mesh
    Eigen::MatrixXd V, UV, N;
    Eigen::MatrixXi F, FUV, FN;
    const std::string suffix = meshFilePath.substr(meshFilePath.find_last_of('.'));
    bool loadSucceed = false;
    std::vector<std::vector<int>> borderVerts_primitive;
    if(suffix == ".txt") {
        loadSucceed = !config.loadFromFile(meshFilePath);
        if(loadSucceed) {
            if(config.shapeType == DOT::Primitive::P_INPUT) {
                assert(DIM == 3); //TODO: extend to 2D cases and absorb above
                
                int suffixI = config.inputShapePath.find_last_of('.');
                if(suffixI == std::string::npos) {
                    DOT::IglUtils::readNodeEle(config.inputShapePath, V, F, SF);
                }
                else {
                    const std::string meshFileSuffix = config.inputShapePath.substr(suffixI);
                    if(meshFileSuffix == ".msh") {
                        DOT::IglUtils::readTetMesh(config.inputShapePath, V, F, SF);
                    }
                    else {
                        assert(0 && "unsupported tet mesh file format!");
                    }
                }

                if(config.rotDeg != 0.0) {
                    const Eigen::Matrix3d rotMtr =
                            Eigen::AngleAxis<double>(config.rotDeg / 180.0 * M_PI,
                                                     config.rotAxis).toRotationMatrix();
#ifdef USE_TBB
                    tbb::parallel_for(0, (int)V.rows(), 1, [&](int vI)
#else
                    for(int vI = 0; vI < V.rows(); ++vI)
#endif
                    {
                        V.row(vI) = (rotMtr * V.row(vI).transpose()).transpose();
                    }
#ifdef USE_TBB
                    );
#endif
                }
                
                V *= config.size / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
                V.rowwise() -= V.colwise().minCoeff();
                //TODO: resampling according to config.resolution?
                UV = V.leftCols(DIM);

#ifdef FIRST_TIME_STEP

                Eigen::RowVector3d maxCoords = V.colwise().maxCoeff();
                Eigen::RowVector3d minCoords = V.colwise().minCoeff();
                Eigen::RowVector3d coordsRange = maxCoords - minCoords;
#ifdef USE_TBB
                tbb::parallel_for(0, (int)V.rows(), 1, [&](int vI)
#else
                for(int vI = 0; vI < V.rows(); ++vI)
#endif
                {
                    // nonuniform scale
                    UV(vI, 0) *= 1.0 + 0.1 * (UV(vI, 0) - minCoords[0]) / coordsRange[0];

                    double yRatio = (UV(vI, 1) - minCoords[1]) / coordsRange[1];
                    UV(vI, 1) += yRatio * 0.1 * UV(vI, 2);

                    UV(vI, 2) *= 1.0 - 0.1 * (UV(vI, 2) - minCoords[2]) / coordsRange[2];
                }
#ifdef USE_TBB
                );
#endif

#endif // FIRST_TIME_STEP

                // nonuniform scale
//                UV.col(0) *= 1.1;
//                UV.col(1) *= 1.2;
//                UV.col(2) *= 1.3;
                // shear
//                UV.col(0) += 0.1 * UV.col(1);

                DOT::IglUtils::findBorderVerts(V, borderVerts_primitive, config.handleRatio);
                
                DOT::IglUtils::buildSTri2Tet(F, SF, sTri2Tet);
            }
            else {
                DOT::Mesh<DIM> primitive(config.shapeType,
                                                      config.size, config.resolution,
                                                      config.YM, config.PR, config.rho);
                V = primitive.V_rest;
                V *= config.size / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
                V.rowwise() -= V.colwise().minCoeff();
                UV = V.leftCols(DIM);

                // nonuniform scale
//                UV.col(0) *= 1.1;
//                UV.col(1) *= 1.2;
                // shear
//                UV.col(0) += UV.col(1) * 0.1;

                F = primitive.F;
                borderVerts_primitive = primitive.borderVerts_primitive;
            }
        }
    }
    else {
        std::cout << "unkown script file format!" << std::endl;
        return -1;
    }
    
    if(!loadSucceed) {
        std::cout << "failed to load mesh!" << std::endl;
        return -1;
    }
    vertAmt_input = V.rows();
    
    // construct mesh data structure
    DOT::Mesh<DIM> *temp = new DOT::Mesh<DIM>(V, F, UV,
                                                                config.YM, config.PR,
                                                                config.rho);
    temp->computeBoundaryVert(SF);
    // primitive test cases
    if((suffix == ".txt") || (suffix == ".primitive")) {
        temp->borderVerts_primitive = borderVerts_primitive;
    }
    triSoup.emplace_back(temp);

    if(config.blockSize > 0) {
//        config.partitionAmt = std::ceil(temp->V_rest.rows() / config.blockSize);
        config.partitionAmt = temp->V_rest.rows() / config.blockSize + 1;
        if(config.partitionAmt == 1) {
            exit(0);
        }
    }
    
    {
        // for output surface mesh
        isSurfNode.resize(0);
        isSurfNode.resize(temp->V.rows(), false);
        for(int tI = 0; tI < SF.rows(); ++tI) {
            isSurfNode[SF(tI, 0)] = true;
            isSurfNode[SF(tI, 1)] = true;
            isSurfNode[SF(tI, 2)] = true;
        }
        
        tetIndToSurf.resize(0);
        tetIndToSurf.resize(temp->V.rows(), -1);
        surfIndToTet.resize(0);
        surfIndToTet.resize(temp->V.rows(), -1);
        int sVI = 0;
        for(int vI = 0; vI < isSurfNode.size(); ++vI) {
            if(isSurfNode[vI]) {
                tetIndToSurf[vI] = sVI;
                surfIndToTet[sVI] = vI;
                ++sVI;
            }
        }
        
        V_surf.resize(sVI, 3);
        F_surf.resize(SF.rows(), 3);
        for(int tI = 0; tI < SF.rows(); ++tI) {
            F_surf(tI, 0) = tetIndToSurf[SF(tI, 0)];
            F_surf(tI, 1) = tetIndToSurf[SF(tI, 1)];
            F_surf(tI, 2) = tetIndToSurf[SF(tI, 2)];
        }
    }

#ifdef USE_SIMD
    initSIMD(temp);
#endif
    
    std::string startDS = "Sim";
    
    std::string folderTail = "";
    if(argc > 3) {
        //TODO: remove all '_'
        folderTail += argv[3];
    }
    
    // create output folder
    mkdir(outputFolderPath.c_str(), 0777);
    if((suffix == ".txt") || (suffix == ".primitive")) {
        config.appendInfoStr(outputFolderPath);
        outputFolderPath += folderTail;
    }
    else {
        outputFolderPath += meshName + "_" +startDS + folderTail;
    }
    mkdir(outputFolderPath.c_str(), 0777);
    config.saveToFile(outputFolderPath + "/config.txt");
    
    // create log file
    outputFolderPath += '/';
    logFile.open(outputFolderPath + "log.txt");
    if(!logFile.is_open()) {
        std::cout << "failed to create log file, please ensure output directory is created successfully!" << std::endl;
        return -1;
    }
    
    // setup timer
    timer.new_activity("descent");
    
    timer_step.new_activity("matrixComputation");
    timer_step.new_activity("matrixAssembly");
    timer_step.new_activity("symbolicFactorization");
    timer_step.new_activity("numericalFactorization");
    timer_step.new_activity("backSolve");
    timer_step.new_activity("lineSearch_other");
    timer_step.new_activity("modifyGrad");//previously boundarySplit
    timer_step.new_activity("modifySearchDir");//previously interiorSplit
    timer_step.new_activity("updateHistory");//previously cornerMerge
    timer_step.new_activity("lineSearch_eVal");
    timer_step.new_activity("fullyImplicit_eComp");
    timer_step.new_activity("solve_extraComp");
    timer_step.new_activity("compGrad");
    timer_step.new_activity("CCD");
    
    timer_temp3.new_activity("init");
    timer_temp3.new_activity("initPrimal");
    timer_temp3.new_activity("initDual");
    timer_temp3.new_activity("initWeights");
    timer_temp3.new_activity("initCons");
    timer_temp3.new_activity("subdSolve");
    timer_temp3.new_activity("consSolve");
    
    // * Our approach
    energyParams.emplace_back(1.0);
    switch(config.energyType) {
        case DOT::ET_SNH:
            energyTerms.emplace_back(new DOT::StableNHEnergy<DIM>());
            break;
            
        case DOT::ET_FCR:
            energyTerms.emplace_back(new DOT::FixedCoRotEnergy<DIM>());
            break;
    }
//        energyTerms.back()->checkEnergyVal(*triSoup[0]);
//        energyTerms.back()->checkGradient(*triSoup[0]);
//        energyTerms.back()->checkHessian(*triSoup[0], true);
    
    switch (config.timeStepperType) {
        case DOT::TST_NEWTON:
            optimizer = new DOT::Optimizer<DIM>(*triSoup[0], energyTerms, energyParams, false, config);
            break;
            
        case DOT::TST_ADMM:
            optimizer = new DOT::ADMMTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, false, config);
            break;
            
        case DOT::TST_ADMMDD:
            optimizer = new DOT::ADMMDDTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, false, config);
            break;
            
        case DOT::TST_LBFGS:
            optimizer = new DOT::LBFGSTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, DOT::D0T_PD, false, config);
            break;
            
        case DOT::TST_LBFGSH:
            optimizer = new DOT::LBFGSTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, DOT::D0T_H, false, config);
            break;
            
        case DOT::TST_LBFGSHI:
            optimizer = new DOT::LBFGSTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, DOT::D0T_HI, false, config);
            break;
            
        case DOT::TST_LBFGSJH:
            optimizer = new DOT::LBFGSTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, DOT::D0T_JH, false, config);
            break;
            
        case DOT::TST_DOT:
        case DOT::TST_LBFGS_GSDD:
            optimizer = new DOT::DOTTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, false, config);
            break;
    }
    optimizer->setTime(config.duration, config.dt);
    
    optimizer->precompute();
    optimizer->setAllowEDecRelTol(false);

    double delay_10ms = std::min(10.0, optimizer->getDt() * 100.0);
    GIFStep = static_cast<int>(std::ceil(3.0 / delay_10ms));
    GIFDelay = static_cast<int>(delay_10ms * GIFStep);

    triSoup.emplace_back(&optimizer->getResult());
    
    if(config.disableCout) {
        std::cout << "cout will be disabled from now on..." << std::endl;
        std::cout.setstate(std::ios_base::failbit);
    }
    
    if(offlineMode) {
        while(true) {
            preDrawFunc(viewer);
            postDrawFunc(viewer);
        }
    }
    else {
        // Setup viewer and launch
        viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
        viewer.callback_key_down = &key_down;
        viewer.callback_pre_draw = &preDrawFunc;
        viewer.callback_post_draw = &postDrawFunc;
        viewer.data().show_lines = true;
        viewer.core.orthographic = config.orthographic;
        viewer.core.camera_zoom *= config.zoom;
        viewer.core.animation_max_fps = 60.0;
        viewer.data().point_size = fracTailSize;
        viewer.data().show_overlay = true;
#if(DIM == 3)
        if(!config.orthographic) {
            viewer.core.trackball_angle = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_4 / 2.0, Eigen::Vector3f::UnitX()));
        }
#endif
        viewer.launch();
    }

    
    // Before exit
    logFile.close();
    for(auto& eI : energyTerms) {
        delete eI;
    }
    delete optimizer;
    delete triSoup[0];
}
