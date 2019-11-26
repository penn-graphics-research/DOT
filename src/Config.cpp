//
//  Config.cpp
//  DOT
//
//  Created by Minchen Li on 7/12/18.
//

#include "Config.hpp"
#include "IglUtils.hpp"

#include <fstream>
#include <sstream>
#include <ctime>

namespace DOT {
    
    const std::vector<std::string> Config::energyTypeStrs = {
        "SNH", "FCR"
    };
    const std::vector<std::string> Config::timeIntegrationTypeStrs = {
        "BE"
    };
    const std::vector<std::string> Config::timeStepperTypeStrs = {
        "Newton", "ADMM", "ADMMDD",
        "LBFGS", "LBFGSH", "LBFGSHI", "LBFGSJH",
        "DOT",
        "GSDD"
    };
    const std::vector<std::string> Config::shapeTypeStrs = {
        "grid", "square", "rectangle", "spikes", "Sharkey", "cylinder", "input"
    };
    
    Config::Config(void) :
    resolution(100), size(1.0), duration(10.0), dt(0.025), inexactSolve(0),
    rho(1.0), YM(100.0), PR(0.4), shapeType(P_GRID), partitionAmt(-1), blockSize(-1), maxIter_APD(1000), warmStart(2), withGravity(true),
    orthographic(false), zoom(1.0), restart(false), disableCout(false), rotDeg(0.0), handleRatio(0.01)
    {}
    
    Config::~Config(void)
    {
    }
    
    int Config::loadFromFile(const std::string& filePath)
    {
        std::ifstream file(filePath);
        if(file.is_open()) {
            std::string line;
            while(std::getline(file, line)) {
                std::stringstream ss(line);
                std::string token;
                ss >> token;
                if(token == "energy") {
                    std::string type;
                    ss >> type;
                    energyType = getEnergyTypeByStr(type);
                }
                else if(token == "timeIntegration") {
                    std::string type;
                    ss >> type;
                    timeIntegrationType = getTimeIntegrationTypeByStr(type);
                }
                else if(token == "timeStepper") {
                    std::string type;
                    ss >> type;
                    timeStepperType = getTimeStepperTypeByStr(type);
                    if((timeStepperType == TST_ADMMDD) ||
                       (timeStepperType == TST_DOT) ||
                       (timeStepperType == TST_LBFGSJH) ||
                       (timeStepperType == TST_LBFGS_GSDD))
                    {
                        ss >> partitionAmt;
                        if(partitionAmt < 0) {
                            ss >> blockSize;
                            assert(blockSize >= 3);
                        }
                        else if(partitionAmt < 2) {
                            partitionAmt = 4;
                            std::cout << "use default partition amount: " <<
                                partitionAmt << std::endl;
                        }
                    }
                    else if(timeStepperType == TST_ADMM)
                    {
                        ss >> maxIter_APD;
                        if(maxIter_APD < 1) {
                            maxIter_APD = 10;
                            std::cout << "use default maxIter for ADMM PD: " <<
                            maxIter_APD << std::endl;
                        }
                    }
                }
                else if(token == "inexactSolve") {
                    ss >> inexactSolve;
                    assert(inexactSolve >= 0);
                }
                else if(token == "resolution") {
                    ss >> resolution;
                }
                else if(token == "size") {
                    ss >> size;
                }
                else if(token == "time") {
                    ss >> duration >> dt;
                }
                else if(token == "density") {
                    ss >> rho;
                    assert(rho > 0.0);
                }
                else if(token == "stiffness") {
                    ss >> YM >> PR;
                }
                else if(token == "turnOffGravity") {
                    withGravity = false;
                }
                else if(token == "script") {
                    std::string type;
                    ss >> type;
                    animScriptType = AnimScripter<DIM>::getAnimScriptTypeByStr(type);
                }
                else if(token == "shape") {
                    std::string type;
                    ss >> type;
                    shapeType = getShapeTypeByStr(type);
                    if(shapeType == P_INPUT) {
                        ss >> inputShapePath;
                    }
                }
                else if(token == "tol") {
                    int amt;
                    ss >> amt;
                    assert(amt >= 0);
                    tol.resize(amt);
                    for(auto& tolI : tol) {
                        file >> tolI;
                    }
                }
                else if(token == "warmStart") {
                    ss >> warmStart;
                    assert(warmStart >= 0);
                }
                else if(token == "view") {
                    std::string type;
                    ss >> type;
                    if(type == "orthographic") {
                        orthographic = true;
                    }
                    else if (type == "perspective"){
                        orthographic = false;
                    }
                    else {
                        orthographic = false;
                        std::cout << "use default perspective view" << std::endl;
                    }
                }
                else if(token == "zoom") {
                    ss >> zoom;
                    assert(zoom > 0.0);
                }
                
                else if(token == "appendStr") {
                    ss >> appendStr;
                }
                
                else if(token == "restart") {
                    restart = true;
                    ss >> statusPath;
                }
                
                else if(token == "disableCout") {
                    disableCout = true;
                }

                else if(token == "rotateModel") {
                    assert(DIM == 3);
                    ss >> rotAxis[0] >> rotAxis[1] >> rotAxis[2] >> rotDeg;
                }

                else if(token == "handleRatio") {
                    ss >> handleRatio;
                    assert((handleRatio > 0) && (handleRatio < 0.5));
                }
                
                else if(token == "tuning") {
                    int amt;
                    ss >> amt;
                    assert(amt >= 0);
                    tuning.resize(amt);
                    for(auto& tuneI : tuning) {
                        file >> tuneI;
                    }
                }
            }
            
            file.close();
            
            if(timeStepperType == TST_ADMM) {
                if(warmStart != 2) {
                    warmStart = 2;
                    std::cout << "forcing xHat warm start for ADMM PD" << std::endl;
                }
            }
            
            return 0;
        }
        else {
            return -1;
        }
    }
    void Config::saveToFile(const std::string& filePath)
    {
        std::ofstream file(filePath);
        assert(file.is_open());
        
        file << "energy " << getStrByEnergyType(energyType) << std::endl;
        
        file << "timeIntegration " << getStrByTimeIntegrationType(timeIntegrationType) << std::endl;
        
        file << "timeStepper " << getStrByTimeStepperType(timeStepperType);
        if((timeStepperType == TST_ADMMDD) ||
           (timeStepperType == TST_DOT) ||
           (timeStepperType == TST_LBFGSJH) ||
           (timeStepperType == TST_LBFGS_GSDD))
        {
            if(blockSize > 0) {
                file << " -1 " << blockSize;
            }
            else {
                file << " " << partitionAmt;
            }
        }
        else if(timeStepperType == TST_ADMM)
        {
            file << " " << maxIter_APD;
        }
        file << std::endl;
        
        file << "inexactSolve " << inexactSolve << std::endl;
        
        file << "warmStart " << warmStart << std::endl;
        
        file << "resolution " << resolution << std::endl;
        
        file << "size " << size << std::endl;
        
        file << "time " << duration << " " << dt << std::endl;
        
        file << "density " << rho << std::endl;
        
        file << "stiffness " << YM << " " << PR << std::endl;
        
        if(!withGravity) {
            file << "turnOffGravity" << std::endl;
        }
        
        file << "script " << AnimScripter<DIM>::getStrByAnimScriptType(animScriptType) << std::endl;

        if(handleRatio != 0.01) {
            file << "handleRatio " << handleRatio << std::endl;
        }
        
        file << "shape " << getStrByShapeType(shapeType);
        if(shapeType == P_INPUT) {
            file << " " << inputShapePath;
        }
        file << std::endl;

        if(rotDeg != 0.0) {
            file << "rotateModel " << rotAxis[0] << " " << rotAxis[1] << " " << rotAxis[2] << " " << rotDeg << std::endl;
        }
        
        if(restart) {
            file << "restart " << statusPath << std::endl;
        }
        
        if(!tuning.empty()) {
            file << "tuning " << tuning.size() << std::endl;
            for(const auto& tuneI : tuning) {
                file << tuneI << std::endl;
            }
        }
        
        file << "view " << (orthographic ? "orthographic": "perspective") << std::endl;
        
        file << "zoom " << zoom << std::endl;
        
        if(appendStr.length() > 0) {
            file << "appendStr " << appendStr << std::endl;
        }
        
        if(disableCout) {
            file << "disableCout" << std::endl;
        }
        
        if(!tol.empty()) {
            file << "tol " << tol.size() << std::endl;
            for(const auto& tolI : tol) {
                file << tolI << std::endl;
            }
        }
        
        file.close();
    }
    
    void Config::appendInfoStr(std::string& inputStr) const
    {
        std::string shapeName;
        if(shapeType == P_INPUT) {
            std::string fileName = inputShapePath.substr(inputShapePath.find_last_of('/') + 1);
            shapeName = fileName.substr(0, fileName.find_last_of('.'));
        }
        else {
            shapeName = getStrByShapeType(shapeType);
        }
        
//        inputStr += (shapeName + "_" +
//                     AnimScripter<DIM>::getStrByAnimScriptType(animScriptType) + "_" +
//                     getStrByEnergyType(energyType) + "_" +
//                     IglUtils::rtos(YM) + "_" + IglUtils::rtos(PR) + "_" +
//                     getStrByTimeStepperType(timeStepperType) +
//                     ((timeStepperType == TST_ADMMDD) ? std::to_string(partitionAmt) : "") +
//                     "_" + IglUtils::rtos(dt) + "_" + std::to_string(resolution) +
//                     (appendStr.length() ? ("_" + appendStr) : ""));
        
        inputStr += (shapeName + "_" +
                     AnimScripter<DIM>::getStrByAnimScriptType(animScriptType) + "_" +
                     getStrByEnergyType(energyType) + "_" +
                     getStrByTimeIntegrationType(timeIntegrationType) + "_" +
                     getStrByTimeStepperType(timeStepperType) +
                     (inexactSolve ? "i": "") +
                     (((timeStepperType == TST_ADMMDD) ||
                       (timeStepperType == TST_DOT) ||
                       (timeStepperType == TST_LBFGSJH) ||
                       (timeStepperType == TST_LBFGS_GSDD)) ?
                      std::to_string(partitionAmt) : "") +
                     ((timeStepperType == TST_ADMM) ? std::to_string(maxIter_APD) : "")
                     + "_");
        
        time_t rawTime = std::time(NULL);
        char buf[BUFSIZ];
        std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", std::localtime(&rawTime));
        inputStr += buf;
        
        if(appendStr.length()) {
            inputStr += "_" + appendStr;
        }
    }
    
    EnergyType Config::getEnergyTypeByStr(const std::string& str)
    {
        for(int i = 0; i < energyTypeStrs.size(); i++) {
            if(str == energyTypeStrs[i]) {
                return EnergyType(i);
            }
        }
        std::cout << "use default energy type: NH" << std::endl;
        return ET_SNH;
    }
    std::string Config::getStrByEnergyType(EnergyType energyType)
    {
        assert(energyType < energyTypeStrs.size());
        return energyTypeStrs[energyType];
    }
    TimeIntegrationType Config::getTimeIntegrationTypeByStr(const std::string& str)
    {
        for(int i = 0; i < timeIntegrationTypeStrs.size(); i++) {
            if(str == timeIntegrationTypeStrs[i]) {
                return TimeIntegrationType(i);
            }
        }
        std::cout << "use default time integration type: BE" << std::endl;
        return TIT_BE;
    }
    std::string Config::getStrByTimeIntegrationType(TimeIntegrationType timeIntegrationType)
    {
        assert(timeIntegrationType < timeIntegrationTypeStrs.size());
        return timeIntegrationTypeStrs[timeIntegrationType];
    }
    TimeStepperType Config::getTimeStepperTypeByStr(const std::string& str)
    {
        for(int i = 0; i < timeStepperTypeStrs.size(); i++) {
            if(str == timeStepperTypeStrs[i]) {
                return TimeStepperType(i);
            }
        }
        std::cout << "use default time stepper type: Newton" << std::endl;
        return TST_NEWTON;
    }
    std::string Config::getStrByTimeStepperType(TimeStepperType timeStepperType)
    {
        assert(timeStepperType < timeStepperTypeStrs.size());
        return timeStepperTypeStrs[timeStepperType];
    }
    Primitive Config::getShapeTypeByStr(const std::string& str)
    {
        for(int i = 0; i < shapeTypeStrs.size(); i++) {
            if(str == shapeTypeStrs[i]) {
                return Primitive(i);
            }
        }
        std::cout << "use default shape: grid" << std::endl;
        return P_GRID;
    }
    std::string Config::getStrByShapeType(Primitive shapeType)
    {
        assert(shapeType < shapeTypeStrs.size());
        return shapeTypeStrs[shapeType];
    }
    
}
