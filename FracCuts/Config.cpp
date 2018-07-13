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
    const std::vector<std::string> Config::integratorTypeStrs = {
        "Newton", "ADMM", "DADMM"
    };
    
    Config::Config(void) :
    resolution(100), size(1.0), duration(10.0), dt(0.025), YM(100.0), PR(0.4)
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
                else if(token == "integrator") {
                    std::string type;
                    ss >> type;
                    integratorType = getIntegratorTypeByStr(type);
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
                    animScriptType = AnimScripter::getAnimScriptTypeByStr(type);
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
        inputStr += (AnimScripter::getStrByAnimScriptType(animScriptType) + "_" +
                     getStrByEnergyType(energyType) + "_" +
                     IglUtils::rtos(YM) + "_" + IglUtils::rtos(PR)) + "_" +
                     getStrByIntegratorType(integratorType) + "_" +
                     IglUtils::rtos(dt) + "_" + std::to_string(resolution);
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
    IntegratorType Config::getIntegratorTypeByStr(const std::string& str)
    {
        for(int i = 0; i < integratorTypeStrs.size(); i++) {
            if(str == integratorTypeStrs[i]) {
                return IntegratorType(i);
            }
        }
        std::cout << "use default integrator type: Newton" << std::endl;
        return IT_NEWTON;
    }
    std::string Config::getStrByIntegratorType(IntegratorType integratorType)
    {
        assert(integratorType < integratorTypeStrs.size());
        return integratorTypeStrs[integratorType];
    }
    
}
