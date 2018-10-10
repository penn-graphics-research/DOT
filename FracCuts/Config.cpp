//
//  Config.cpp
//  FracCuts
//
//  Created by Minchen Li on 7/12/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "Config.hpp"
#include "IglUtils.hpp"

#include <fstream>
#include <sstream>

namespace FracCuts {
    
    const std::vector<std::string> Config::energyTypeStrs = {
        "NH", "FCR", "SD", "ARAP"
    };
    const std::vector<std::string> Config::timeStepperTypeStrs = {
        "Newton", "ADMM", "DADMM", "ADMMDD"
    };
    const std::vector<std::string> Config::shapeTypeStrs = {
        "grid", "square", "spikes", "Sharkey", "cylinder", "input"
    };
    
    Config::Config(void) :
    resolution(100), size(1.0), duration(10.0), dt(0.025),
    YM(100.0), PR(0.4), shapeType(P_GRID), partitionAmt(-1)
    {}
    
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
                else if(token == "timeStepper") {
                    std::string type;
                    ss >> type;
                    timeStepperType = getTimeStepperTypeByStr(type);
                    if(timeStepperType == TST_ADMMDD) {
                        ss >> partitionAmt;
                        if(partitionAmt < 2) {
                            partitionAmt = 4;
                            std::cout << "use default partition amount: " <<
                                partitionAmt << std::endl;
                        }
                    }
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
                else if(token == "stiffness") {
                    ss >> YM >> PR;
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
                        assert(std::getline(file, line));
                        sscanf(line.c_str(), "%le", &tolI);
                    }
                }
            }
            
            file.close();
            
            return 0;
        }
        else {
            return -1;
        }
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
        
        inputStr += (shapeName + "_" +
                     AnimScripter<DIM>::getStrByAnimScriptType(animScriptType) + "_" +
                     getStrByEnergyType(energyType) + "_" +
                     IglUtils::rtos(YM) + "_" + IglUtils::rtos(PR) + "_" +
                     getStrByTimeStepperType(timeStepperType) +
                     ((timeStepperType == TST_ADMMDD) ? std::to_string(partitionAmt) : "") +
                     "_" + IglUtils::rtos(dt) + "_" + std::to_string(resolution));
    }
    
    EnergyType Config::getEnergyTypeByStr(const std::string& str)
    {
        for(int i = 0; i < energyTypeStrs.size(); i++) {
            if(str == energyTypeStrs[i]) {
                return EnergyType(i);
            }
        }
        std::cout << "use default energy type: NH" << std::endl;
        return ET_NH;
    }
    std::string Config::getStrByEnergyType(EnergyType energyType)
    {
        assert(energyType < energyTypeStrs.size());
        return energyTypeStrs[energyType];
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
