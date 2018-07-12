//
//  Config.cpp
//  FracCuts
//
//  Created by Minchen Li on 7/12/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "Config.hpp"

#include <fstream>
#include <sstream>

namespace FracCuts {
    
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
    
    EnergyType Config::getEnergyTypeByStr(const std::string& str) {
        if(str == "NH") {
            return ET_NH;
        }
        else if(str == "FCR") {
            return ET_FCR;
        }
        else if(str == "SD") {
            return ET_SD;
        }
        else if(str == "ARAP") {
            return ET_ARAP;
        }
        else {
            std::cout << "use default energy type: NH" << std::endl;
            return ET_NH;
        }
    }
    IntegratorType Config::getIntegratorTypeByStr(const std::string& str) {
        if(str == "Newton") {
            return IT_NEWTON;
        }
        else if(str == "ADMM") {
            return IT_ADMM;
        }
        else if(str == "DADMM") {
            return IT_DADMM;
        }
        else {
            std::cout << "use default integrator type: Newton" << std::endl;
            return IT_NEWTON;
        }
    }
    
}
