//
//  Config.hpp
//  FracCuts
//
//  Created by Minchen Li on 7/12/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Config_hpp
#define Config_hpp

#include "AnimScripter.hpp"

#include <iostream>
#include <map>

namespace FracCuts {
    
    enum EnergyType {
        ET_NH,
        ET_FCR,
        ET_SD,
        ET_ARAP
    };
    
    enum IntegratorType {
        IT_NEWTON,
        IT_ADMM,
        IT_DADMM
    };
    
    class Config
    {
    public:
        EnergyType energyType;
        IntegratorType integratorType;
        int resolution;
        double size;
        double duration, dt;
        double YM, PR;
        AnimScriptType animScriptType;
        
    public:
        Config(void);
        int loadFromFile(const std::string& filePath);
        
    public:
        static EnergyType getEnergyTypeByStr(const std::string& str);
        static IntegratorType getIntegratorTypeByStr(const std::string& str);
    };
    
}

#endif /* Config_hpp */
