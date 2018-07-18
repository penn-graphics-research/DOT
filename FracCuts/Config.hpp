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
    
    enum TimeStepperType {
        TST_NEWTON,
        TST_ADMM,
        TST_DADMM,
        TST_ADMMDD
    };
    
    class Config
    {
    public:
        EnergyType energyType;
        TimeStepperType timeStepperType;
        int resolution;
        double size;
        double duration, dt;
        double YM, PR;
        AnimScriptType animScriptType;
        
    public:
        static const std::vector<std::string> energyTypeStrs;
        static const std::vector<std::string> timeStepperTypeStrs;
        
    public:
        Config(void);
        int loadFromFile(const std::string& filePath);
        
    public:
        void appendInfoStr(std::string& inputStr) const;
        
    public:
        static EnergyType getEnergyTypeByStr(const std::string& str);
        static std::string getStrByEnergyType(EnergyType energyType);
        static TimeStepperType getTimeStepperTypeByStr(const std::string& str);
        static std::string getStrByTimeStepperType(TimeStepperType timeStepperType);
    };
    
}

#endif /* Config_hpp */
