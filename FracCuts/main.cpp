#include "Types.hpp"
#include "IglUtils.hpp"
#include "Config.hpp"
#include "Optimizer.hpp"
#include "ADMMTimeStepper.hpp"
#include "DADMMTimeStepper.hpp"
#include "ADMMDDTimeStepper.hpp"
#include "SymStretchEnergy.hpp"
#include "ARAPEnergy.hpp"
#include "NeoHookeanEnergy.hpp"
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
FracCuts::Config config;
FracCuts::MethodType methodType;
std::vector<const FracCuts::TriangleSoup<DIM>*> triSoup;
int vertAmt_input;
FracCuts::Optimizer<DIM>* optimizer;
std::vector<FracCuts::Energy<DIM>*> energyTerms;
std::vector<double> energyParams;
bool offlineMode = false;
bool autoSwitch = false;
bool contactHandling = false;
double lambda_init;
bool optimization_on = false;
int iterNum = 0;
int converged = 0;
bool outerLoopFinished = false;

std::ofstream logFile;
std::string outputFolderPath = "output/";
const std::string meshFolder = "/Users/mincli/Desktop/meshes/";

// visualization
igl::opengl::glfw::Viewer viewer;
const int channel_initial = 0;
const int channel_result = 1;
const int channel_findExtrema = 2;
int viewChannel = channel_result;
Eigen::MatrixXi SF;
std::vector<int> sTri2Tet;
bool viewUV = true; // view UV or 3D model
double texScale = 1.0;
bool showSeam = false;
Eigen::MatrixXd seamColor;
bool showBoundary = false;
int showDistortion = 2; // 0: don't show; 1: energy value; 2: other scalar field;
int showDistortion_init = showDistortion;
Eigen::MatrixXd faceColors_default;
bool showTexture = true; // show checkerboard
bool isLighting = true;
bool showFracTail = true; //!!! frac tail info not initialized correctly
bool showFixedVerts = true; //TODO: add key control
float fracTailSize = 20.0f;

bool saveInfo_postDraw = false;
std::string infoName = "";
bool isCapture3D = false;
int capture3DI = 0;

GifWriter GIFWriter;
uint32_t GIFDelay = 10; //*10ms
double GIFScale = 0.4;

// timer
double secPast = 0.0;
time_t lastStart_world;
Timer timer, timer_step, timer_temp, timer_temp2, timer_temp3;


void saveInfo(bool writePNG = true, bool writeGIF = true, bool writeMesh = true);

void proceedOptimization(int proceedNum = 1)
{
    for(int proceedI = 0; (proceedI < proceedNum) && (!converged); proceedI++) {
//        infoName = std::to_string(iterNum);
        if((!offlineMode) && (methodType == FracCuts::MT_NOCUT)) {
            saveInfo(false, true, false); //!!! output mesh for making video, PNG output only works under online rendering mode
        }
        showDistortion = showDistortion_init;
        std::cout << "Iteration" << iterNum << ":" << std::endl;
        converged = optimizer->solve(1);
        if(converged == 2) {
            showDistortion = 0;
            converged = 0;
            logFile << "maxIter reached for timeStep" << iterNum << std::endl;
        }
        iterNum = optimizer->getIterNum();
    }
}

void updateViewerData_meshEdges(void)
{
    viewer.data().show_lines = !showSeam;
    
    viewer.data().set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
    if(showSeam) {
        // only draw air mesh edges
        if(optimizer->isScaffolding() && viewUV && (viewChannel == channel_result)) {
            const Eigen::MatrixXd V_airMesh = optimizer->getAirMesh().V * texScale;
            for(int triI = 0; triI < optimizer->getAirMesh().F.rows(); triI++) {
                const Eigen::RowVector3i& triVInd = optimizer->getAirMesh().F.row(triI);
                for(int eI = 0; eI < 3; eI++) {
                    viewer.data().add_edges(V_airMesh.row(triVInd[eI]), V_airMesh.row(triVInd[(eI + 1) % 3]),
                                          Eigen::RowVector3d::Zero());
                }
            }
        }
    }
}

void updateViewerData_seam(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& UV)
{
    if(showSeam) {
        const double seamDistThres = 1.0e-2;
        Eigen::VectorXd seamScore;
        seamScore.setZero();
        
        const Eigen::VectorXd cohIndices = Eigen::VectorXd::LinSpaced(triSoup[viewChannel]->cohE.rows(),
                                                                0, triSoup[viewChannel]->cohE.rows() - 1);
        Eigen::MatrixXd color;
//        FracCuts::IglUtils::mapScalarToColor(cohIndices, color, 0, cohIndices.rows() - 1, 1);
        color.resize(cohIndices.size(), 3);
        color.rowwise() = Eigen::RowVector3d(1.0, 0.5, 0.0);
        
        //TODO: seamscore only for autocuts
        seamColor.resize(0, 3);
        double seamThickness = (viewUV ? (triSoup[viewChannel]->virtualRadius * 0.0007 / viewer.core.camera_zoom * texScale) :
                                (triSoup[viewChannel]->virtualRadius * 0.006));
        for(int eI = 0; eI < triSoup[viewChannel]->cohE.rows(); eI++) {
            const Eigen::RowVector4i& cohE = triSoup[viewChannel]->cohE.row(eI);
            const auto finder = triSoup[viewChannel]->edge2Tri.find(std::pair<int, int>(cohE[0], cohE[1]));
            assert(finder != triSoup[viewChannel]->edge2Tri.end());
            const Eigen::RowVector3d& sn = triSoup[viewChannel]->triNormal.row(finder->second);
            if((seamScore[eI] > seamDistThres) || (methodType != FracCuts::MT_AUTOCUTS)) {
                // seam edge
                FracCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[0]), V.row(cohE[1]), seamThickness, texScale, !viewUV, sn);
                if(viewUV) {
                    FracCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[2]), V.row(cohE[3]), seamThickness, texScale, !viewUV, sn);
                }
            }
            else if((seamScore[eI] < 0.0) && showBoundary) {
                // boundary edge
                //TODO: debug!
                FracCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[0]), V.row(cohE[1]), seamThickness, texScale, !viewUV, sn);
            }
        }
    }
}

void updateViewerData_distortion(void)
{
    Eigen::MatrixXd color_distortionVis;
    
    switch(showDistortion) {
        case 1: { // show SD energy value
            Eigen::VectorXd distortionPerElem;
            energyTerms[0]->getEnergyValPerElem(*triSoup[viewChannel], distortionPerElem, true);
            const Eigen::VectorXd& visRange = energyTerms[0]->getVisRange_energyVal();
            FracCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis,
                                                 visRange[0], visRange[1]);
            break;
        }
            
        case 2: { // show other triangle-based scalar fields
            Eigen::VectorXd l2StretchPerElem;
//            triSoup[viewChannel]->computeL2StretchPerElem(l2StretchPerElem);
//            dynamic_cast<FracCuts::SymStretchEnergy*>(energyTerms[0])->getDivGradPerElem(*triSoup[viewChannel], l2StretchPerElem);
//            std::cout << l2StretchPerElem << std::endl; //DEBUG
//            FracCuts::IglUtils::mapScalarToColor(l2StretchPerElem, color_distortionVis, 1.0, 2.0);
//            FracCuts::IglUtils::mapScalarToColor(l2StretchPerElem, color_distortionVis,
//                l2StretchPerElem.minCoeff(), l2StretchPerElem.maxCoeff());
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
//            FracCuts::IglUtils::mapScalarToColor(faceWeight, color_distortionVis,
//                faceWeight.minCoeff(), faceWeight.maxCoeff());
            igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, faceWeight, true, color_distortionVis);
            color_distortionVis.array() += std::min(0.2, 1.0 - color_distortionVis.maxCoeff());
            break;
        }
    
        case 0: {
            color_distortionVis = Eigen::MatrixXd::Ones(triSoup[viewChannel]->F.rows(), 3);
            color_distortionVis.col(2).setZero();
            break;
        }
            
        case -1: {
            assert(faceColors_default.rows() == ((DIM == 3) ? SF.rows() :
                                                 triSoup[viewChannel]->F.rows()));
            color_distortionVis = faceColors_default;
            break;
        }
            
        default:
            assert(0 && "unknown distortion visualization option!");
            break;
    }
    
    if(optimizer->isScaffolding() && viewUV && (viewChannel == channel_result)) {
        optimizer->getScaffold().augmentFColorwithAirMesh(color_distortionVis);
    }
    
    if(showSeam) {
        color_distortionVis.conservativeResize(color_distortionVis.rows() + seamColor.rows(), 3);
        color_distortionVis.bottomRows(seamColor.rows()) = seamColor;
    }
    viewer.data().set_colors(color_distortionVis);
}

void updateViewerData(void)
{
    Eigen::MatrixXd UV_vis = triSoup[viewChannel]->V * texScale;
    Eigen::MatrixXi F_vis = ((DIM == 2) ? triSoup[viewChannel]->F : SF);
    if(viewUV) {
        if(optimizer->isScaffolding() && (viewChannel == channel_result)) {
            optimizer->getScaffold().augmentUVwithAirMesh(UV_vis, texScale);
            optimizer->getScaffold().augmentFwithAirMesh(F_vis);
        }
        if(DIM == 2) {
            UV_vis.conservativeResize(UV_vis.rows(), 3);
            UV_vis.rightCols(1) = Eigen::VectorXd::Zero(UV_vis.rows());
        }
        viewer.core.align_camera_center(triSoup[viewChannel]->V_rest * texScale, F_vis);
        updateViewerData_seam(UV_vis, F_vis, UV_vis);
        
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

        updateViewerData_meshEdges();
        
        viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 0.0));
        if(showFracTail) {
            for(const auto& tailVI : triSoup[viewChannel]->fracTail) {
                viewer.data().add_points(UV_vis.row(tailVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
        if(showFixedVerts) {
            for(const auto& fixedVI : triSoup[viewChannel]->fixedVert) {
                viewer.data().add_points(UV_vis.row(fixedVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
        Eigen::VectorXi sharedVerts;
        optimizer->getSharedVerts(sharedVerts);
        for(int svI = 0; svI < sharedVerts.size(); svI++) {
            viewer.data().add_points(UV_vis.row(sharedVerts[svI]), Eigen::RowVector3d(1.0, 0.3, 0.3));
        }
    }
    else {
        Eigen::MatrixXd V_vis = triSoup[viewChannel]->V_rest;
        viewer.core.align_camera_center(V_vis, F_vis);
        updateViewerData_seam(V_vis, F_vis, UV_vis);
        
        if((V_vis.rows() != viewer.data().V.rows()) ||
           (UV_vis.rows() != viewer.data().V_uv.rows()) ||
           (F_vis.rows() != viewer.data().F.rows()))
        {
            viewer.data().clear();
        }
        viewer.data().set_mesh(V_vis, F_vis);
        
        if(showTexture) {
            viewer.data().set_uv(UV_vis);
            viewer.data().show_texture = true;
        }
        else {
            viewer.data().show_texture = false;
        }
        
        if(isLighting) {
            viewer.core.lighting_factor = 1.0;
        }
        else {
            viewer.core.lighting_factor = 0.0;
        }
        
        updateViewerData_meshEdges();
        
        viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 0.0));
        if(showFracTail) {
            for(const auto& tailVI : triSoup[viewChannel]->fracTail) {
                viewer.data().add_points(V_vis.row(tailVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
        if(showFixedVerts) {
            for(const auto& fixedVI : triSoup[viewChannel]->fixedVert) {
                viewer.data().add_points(V_vis.row(fixedVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
        Eigen::VectorXi sharedVerts;
        optimizer->getSharedVerts(sharedVerts);
        for(int svI = 0; svI < sharedVerts.size(); svI++) {
            viewer.data().add_points(V_vis.row(sharedVerts[svI]), Eigen::RowVector3d(1.0, 0.3, 0.3));
        }
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
    
    if(writeGIF) {
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

void saveInfo(bool writePNG, bool writeGIF, bool writeMesh)
{
    saveScreenshot(outputFolderPath + infoName + ".png", 0.5, writeGIF, writePNG);
    if(writeMesh) {
//        triSoup[channel_result]->save(outputFolderPath + infoName + "_triSoup.obj");
        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + "_mesh" +
                                            ((DIM == 2) ? ".obj" : ".msh"), false, SF);
    }
}

void saveInfoForPresent(const std::string fileName = "info.txt")
{
    std::ofstream file;
    file.open(outputFolderPath + fileName);
    assert(file.is_open());
    
    file << vertAmt_input << " " <<
        triSoup[channel_initial]->F.rows() << std::endl;
    
    file << iterNum << " " << optimizer->getInnerIterAmt() << " 0 0 " << lambda_init << " " << 1.0 - energyParams[0] << std::endl;
    
    timer.print(file);
    timer_step.print(file);
    timer_temp.print(file);
    timer_temp2.print(file);
    timer_temp3.print(file);
    
    double distortion;
    energyTerms[0]->computeEnergyVal(*triSoup[channel_result], distortion);
    file << distortion << " " << 0.0 << std::endl;
    
    file << "initialSeams " << triSoup[channel_result]->initSeams.rows() << std::endl;
    file << triSoup[channel_result]->initSeams << std::endl;
    
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
                GifBegin(&GIFWriter, (outputFolderPath + "anim.gif").c_str(),
                         GIFScale * (viewer.core.viewport[2] - viewer.core.viewport[0]),
                         GIFScale * (viewer.core.viewport[3] - viewer.core.viewport[1]), GIFDelay);
                
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
                
            case 'u':
            case 'U': {
                viewUV = !viewUV;
                break;
            }
                
            case 's':
            case 'S': {
                showSeam = !showSeam;
                break;
            }
                
            case 'e':
            case 'E': {
                showBoundary = !showBoundary;
                break;
            }
                
            case 'd':
            case 'D': {
                showDistortion++;
                if(showDistortion > 2) {
                    showDistortion = 0;
                }
                break;
            }
                
            case 'c':
            case 'C': {
                showTexture = !showTexture;
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
                
            case 'p':
            case 'P': {
                showFracTail = !showFracTail;
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
        if(!isCapture3D) {
            viewer.core.is_animating = true;
            isCapture3D = true;
        }
        else {
            if(capture3DI < 2) {
                // take screenshot
                std::cout << "Taking screenshot for 3D View " << capture3DI / 2 << std::endl;
                std::string filePath = outputFolderPath + "3DView" + std::to_string(capture3DI / 2) +
                    ((capture3DI % 2 == 0) ? "_seam.png" : "_distortion.png");
                saveScreenshot(filePath, 0.5);
                capture3DI++;
            }
            else {
                GifEnd(&GIFWriter);
                saveInfoForPresent();
                if(autoSwitch) {
                    exit(0);
                }
                else {
                    viewer.core.is_animating = false;
                    isCapture3D = false;
                    outerLoopFinished = false;
                }
            }
        }
    }
    
    return false;
}

void converge_preDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    infoName = "finalResult";
    
    secPast += difftime(time(NULL), lastStart_world);
    updateViewerData();
    
    optimizer->flushEnergyFileOutput();
    optimizer->flushGradFileOutput();
    
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
            
            switch(methodType) {
                case FracCuts::MT_NOCUT: {
                    converge_preDrawFunc(viewer);
                    break;
                }
            }
        }
    }
    else {
        if(isCapture3D && (capture3DI < 2)) {
            // change view accordingly
            double rotDeg = ((capture3DI < 8) ? (M_PI_2 * (capture3DI / 2)) : M_PI_2);
            Eigen::Vector3f rotAxis = Eigen::Vector3f::UnitY();
            if((capture3DI / 2) == 4) {
                rotAxis = Eigen::Vector3f::UnitX();
            }
            else if((capture3DI / 2) == 5) {
                rotAxis = -Eigen::Vector3f::UnitX();
            }
            viewer.core.trackball_angle = Eigen::Quaternionf(Eigen::AngleAxisf(rotDeg, rotAxis));
            viewChannel = channel_result;
            viewUV = false;
            showSeam = true;
            showBoundary = false;
            isLighting = false;
            showTexture = capture3DI % 2;
            showDistortion = 2 - capture3DI % 2;
            updateViewerData();
        }
    }
    return false;
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
            std::cout << "Optimization mode" << std::endl;
            break;
            
        case 10:
            // autoswitch optimization mode
            autoSwitch = true;
            std::cout << "Auto-switch optimization mode" << std::endl;
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
            FracCuts::Diagnostic::run(argc, argv);
            return 0;
        }
            
        case 2: {
            // mesh processing mode
            FracCuts::MeshProcessing::run(argc, argv);
            return 0;
        }
            
        default: {
            std::cout<< "No progMode " << progMode << std::endl;
            return 0;
        }
    }
    
    // Optimization mode
    
    std::string meshFileName("cone2.0.obj");
    if(argc > 2) {
        meshFileName = std::string(argv[2]);
    }
    std::string meshFilePath;
    if(meshFileName.at(0) == '/') {
        std::cout << "The input mesh file name is gloabl mesh file path." << std::endl;
        meshFilePath = meshFileName;
        meshFileName = meshFileName.substr(meshFileName.find_last_of('/') + 1);
    }
    else {
        meshFilePath = meshFolder + meshFileName;
    }
    std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
    // Load mesh
    Eigen::MatrixXd V, UV, N;
    Eigen::MatrixXi F, FUV, FN;
    const std::string suffix = meshFilePath.substr(meshFilePath.find_last_of('.'));
    bool loadSucceed = false;
    std::vector<std::vector<int>> borderVerts_primitive;
    if(suffix == ".off") {
        loadSucceed = igl::readOFF(meshFilePath, V, F);
    }
    else if(suffix == ".obj") {
        loadSucceed = igl::readOBJ(meshFilePath, V, UV, N, F, FUV, FN);
    }
    else if(suffix == ".primitive") {
        loadSucceed = !config.loadFromFile(meshFilePath);
        if(loadSucceed) {
            if(config.shapeType == FracCuts::Primitive::P_INPUT) {
                assert(DIM == 3); //TODO: extend to 2D cases and absorb above
                
                FracCuts::IglUtils::readTetMesh(config.inputShapePath, V, F, SF);
                
                V *= config.size / (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
                //TODO: resampling according to config.resolution?
                UV = V.leftCols(DIM);
//                UV.col(0) *= 1.1;
//                UV.col(1) *= 1.2;
//                UV.col(2) *= 1.3;
                FracCuts::IglUtils::findBorderVerts(V, borderVerts_primitive);
                
                FracCuts::IglUtils::buildSTri2Tet(F, SF, sTri2Tet);
            }
            else {
                FracCuts::TriangleSoup<DIM> primitive(config.shapeType,
                                                      config.size, config.resolution);
                V = primitive.V_rest;
                UV = primitive.V;
//                UV.col(0) *= 1.1;
//                UV.col(1) *= 1.2;
                F = primitive.F;
                borderVerts_primitive = primitive.borderVerts_primitive;
            }
        }
    }
    else {
        std::cout << "unkown mesh file format!" << std::endl;
        return -1;
    }
    if(!loadSucceed) {
        std::cout << "failed to load mesh!" << std::endl;
        return -1;
    }
    vertAmt_input = V.rows();
    
    // construct mesh data structure
    FracCuts::TriangleSoup<DIM> *temp = new FracCuts::TriangleSoup<DIM>(V, F, UV);
    // primitive test cases
    if(suffix == ".primitive") {
        temp->borderVerts_primitive = borderVerts_primitive;
    }
    triSoup.emplace_back(temp);
    
    // Set lambda
    double lambda = 0.0;
    if(argc > 3) {
        lambda = std::stod(argv[3]);
        if((lambda != lambda) || (lambda < 0.0) || (lambda > 1.0)) {
            std::cout << "Overwrite invalid lambda " << lambda << " to 0.0" << std::endl;
            lambda = 0.0;
        }
    }
    else {
        std::cout << "Use default lambda = " << lambda << std::endl;
    }
    lambda_init = lambda;
    
    // Set delta
    double delta = 4.0;
    if(argc > 4) {
        delta = std::stod(argv[4]);
        if((delta != delta) || (delta < 0.0)) {
            std::cout << "Overwrite invalid delta " << delta << " to 4" << std::endl;
            delta = 4.0;
        }
    }
    else {
        std::cout << "Use default delta = " << delta << std::endl;
    }
    
    if(argc > 5) {
        methodType = FracCuts::MethodType(std::stoi(argv[5]));
    }
    else {
        methodType = FracCuts::MT_NOCUT;
        std::cout << "Use default method: simulation" << std::endl;
    }
    
    std::string startDS;
    switch (methodType) {
        case FracCuts::MT_NOCUT:
            lambda = 0.0;
            startDS = "Sim";
            break;
            
        default:
            std::cout << "method type not valid!" << std::endl;
            exit(-1);
            break;
    }
    
    std::string folderTail = "";
    if(argc > 6) {
        //TODO: remove all '_'
        folderTail += argv[6];
    }
#ifdef STATIC_SOLVE
    folderTail += "QuasiStatic";
#endif
    
    // create output folder
    mkdir(outputFolderPath.c_str(), 0777);
    if(suffix == ".primitive") {
        config.appendInfoStr(outputFolderPath);
        outputFolderPath += folderTail;
    }
    else {
        outputFolderPath += meshName + "_" +startDS + folderTail;
    }
    mkdir(outputFolderPath.c_str(), 0777);
    
    // create log file
    outputFolderPath += '/';
    logFile.open(outputFolderPath + "log.txt");
    if(!logFile.is_open()) {
        std::cout << "failed to create log file, please ensure output directory is created successfully!" << std::endl;
        return -1;
    }
    
    // setup timer
    timer.new_activity("topology");
    timer.new_activity("descent");
    timer.new_activity("scaffolding");
    timer.new_activity("energyUpdate");
    
    timer_step.new_activity("matrixComputation");
    timer_step.new_activity("matrixAssembly");
    timer_step.new_activity("symbolicFactorization");
    timer_step.new_activity("numericalFactorization");
    timer_step.new_activity("backSolve");
    timer_step.new_activity("lineSearch");
    
    timer_step.new_activity("boundarySplit");
    timer_step.new_activity("interiorSplit");
    timer_step.new_activity("cornerMerge");
    
    timer_temp.new_activity("SVD");
    timer_temp.new_activity("derivComp");
    timer_temp.new_activity("SPD");
    timer_temp.new_activity("blk2Mtr");
    timer_temp.new_activity("inertiaE");
    timer_temp.new_activity("inertiaG");
    timer_temp.new_activity("inertiaH");
    
    timer_temp2.new_activity("compA");
    timer_temp2.new_activity("compB");
    timer_temp2.new_activity("comp_dP_div_dF");
    timer_temp2.new_activity("comp_d2E_div_dx2");
    timer_temp2.new_activity("eVal");
    timer_temp2.new_activity("grad_pre");
    timer_temp2.new_activity("grad_dE_div_dF");
    timer_temp2.new_activity("grad_dE_div_dx");
    timer_temp2.new_activity("grad_add");
    
    timer_temp3.new_activity("init");
    timer_temp3.new_activity("initPrimal");
    timer_temp3.new_activity("initDual");
    timer_temp3.new_activity("initWeights");
    timer_temp3.new_activity("initCons");
    timer_temp3.new_activity("subdSolve");
    timer_temp3.new_activity("consSolve");
    
    // * Our approach
    if(lambda != 1.0) {
        energyParams.emplace_back(1.0 - lambda);
        switch(config.energyType) {
            case FracCuts::ET_NH:
                energyTerms.emplace_back(new FracCuts::NeoHookeanEnergy<DIM>(config.YM, config.PR));
                break;
                
            case FracCuts::ET_FCR:
                energyTerms.emplace_back(new FracCuts::FixedCoRotEnergy<DIM>(config.YM, config.PR));
                break;
                
            case FracCuts::ET_SD:
                energyTerms.emplace_back(new FracCuts::SymStretchEnergy<DIM>());
                break;
                
            case FracCuts::ET_ARAP:
                energyTerms.emplace_back(new FracCuts::ARAPEnergy<DIM>());
                break;
        }
//        energyTerms.back()->checkEnergyVal(*triSoup[0]);
//        energyTerms.back()->checkGradient(*triSoup[0]);
//        energyTerms.back()->checkHessian(*triSoup[0], true);
    }
    
    assert(lambda == 0.0);
    switch (config.timeStepperType) {
        case FracCuts::TST_NEWTON:
            optimizer = new FracCuts::Optimizer<DIM>(*triSoup[0], energyTerms, energyParams, 0, false, contactHandling, Eigen::MatrixXd(), Eigen::MatrixXi(), Eigen::VectorXi(), config);
            break;
            
        case FracCuts::TST_ADMM:
            optimizer = new FracCuts::ADMMTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, 0, false, contactHandling, Eigen::MatrixXd(), Eigen::MatrixXi(), Eigen::VectorXi(), config);
            break;
            
        case FracCuts::TST_DADMM:
            optimizer = new FracCuts::DADMMTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, 0, false, contactHandling, Eigen::MatrixXd(), Eigen::MatrixXi(), Eigen::VectorXi(), config);
            break;
            
        case FracCuts::TST_ADMMDD:
            optimizer = new FracCuts::ADMMDDTimeStepper<DIM>(*triSoup[0], energyTerms, energyParams, 0, false, contactHandling, Eigen::MatrixXd(), Eigen::MatrixXi(), Eigen::VectorXi(), config);
            break;
    }
    optimizer->setTime(config.duration, config.dt);
    
    //TODO: bijectivity for other mode?
    optimizer->precompute();
    optimizer->setAllowEDecRelTol(false);
#ifndef STATIC_SOLVE
    GIFDelay = optimizer->getDt() * 100;
#endif
    triSoup.emplace_back(&optimizer->getResult());
    triSoup.emplace_back(&optimizer->getData_findExtrema()); // for visualizing UV map for finding extrema
    
//    //TEST: regional seam placement, Zhongshi
//    std::ifstream vWFile("/Users/mincli/Desktop/output_FracCuts/" + meshName + "_RSP.txt");
//    if(vWFile.is_open()) {
//        double revLikelihood;
//        for(int vI = 0; vI < optimizer->getResult().vertWeight.size(); vI++) {
//            if(vWFile.eof()) {
//                std::cout << "# of weights less than # of V for regional seam placement, " <<
//                    "reset vertWeight to all 1.0" << std::endl;
//                optimizer->getResult().vertWeight = Eigen::VectorXd::Ones(optimizer->getResult().V.rows());
//                vWFile.close();
//                break;
//            }
//            else {
//                vWFile >> revLikelihood;
//                if(revLikelihood < 0.0) {
//                    revLikelihood = 0.0;
//                }
//                else if(revLikelihood > 1.0) {
//                    revLikelihood = 1.0;
//                }
//                optimizer->getResult().vertWeight[vI] = 1.0 + 10.0 * revLikelihood;
//            }
//        }
//        vWFile.close();
//        std::cout << "regional seam placement weight loaded" << std::endl;
//    }
    
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
        viewer.core.orthographic = true;
    //    viewer.core.camera_zoom *= 1.9;
        viewer.core.animation_max_fps = 60.0;
        viewer.data().point_size = fracTailSize;
        viewer.data().show_overlay = true;
#if(DIM == 3)
        viewer.core.trackball_angle = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_4, Eigen::Vector3f::UnitX()));
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
