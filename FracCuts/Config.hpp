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
    
    enum ConstraintSolverType {
        CST_PENALTY,
        CST_QP
    };
    
    class Config
    {
    public:
        EnergyType energyType;
        TimeStepperType timeStepperType;
        Primitive shapeType;
        std::string inputShapePath;
        int resolution;
        double size;
        double duration, dt;
        double YM, PR;
        AnimScriptType animScriptType;
        int partitionAmt;
        std::vector<double> tol;
        bool orthographic;
        bool isConstrained;
        ConstraintSolverType constraintSolverType;
        
        // ground
        bool ground;
        bool groundFriction;
        double groundY;
        double groundRelStiff;
        
        std::string appendStr;
        
    public:
        static const std::vector<std::string> energyTypeStrs;
        static const std::vector<std::string> timeStepperTypeStrs;
        static const std::vector<std::string> shapeTypeStrs;
        static const std::vector<std::string> constraintSolverTypeStrs;
        
    public:
        Config(void);
        int loadFromFile(const std::string& filePath);
        void saveToFile(const std::string& filePath);
        
    public:
        void appendInfoStr(std::string& inputStr) const;
        
    public:
        static EnergyType getEnergyTypeByStr(const std::string& str);
        static std::string getStrByEnergyType(EnergyType energyType);
        static TimeStepperType getTimeStepperTypeByStr(const std::string& str);
        static std::string getStrByTimeStepperType(TimeStepperType timeStepperType);
        static Primitive getShapeTypeByStr(const std::string& str);
        static std::string getStrByShapeType(Primitive shapeType);
        static ConstraintSolverType getConstraintSolverTypeByStr(const std::string& str);
        static std::string getStrByConstraintSolverType(ConstraintSolverType constraintSolverType);
    };
    
}

#endif /* Config_hpp */
