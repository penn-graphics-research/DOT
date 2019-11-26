//
//  Config.hpp
//  DOT
//
//  Created by Minchen Li on 7/12/18.
//

#ifndef Config_hpp
#define Config_hpp

#include "AnimScripter.hpp"

#include <iostream>
#include <map>

namespace DOT {
    
    enum EnergyType {
        ET_SNH,
        ET_FCR
    };
    
    enum TimeIntegrationType {
        TIT_BE
    };
    
    enum TimeStepperType {
        TST_NEWTON,
        TST_ADMM,
        TST_ADMMDD,
        
        TST_LBFGS,
        TST_LBFGSH,
        TST_LBFGSHI,
        TST_LBFGSJH,
        
        TST_DOT,
        TST_LBFGS_GSDD
    };
    
    class Config
    {
    public:
        EnergyType energyType;
        double rho;
        double YM, PR;
        bool withGravity;
        
        TimeIntegrationType timeIntegrationType;
        TimeStepperType timeStepperType;
        double duration, dt;
        int inexactSolve;
        std::vector<double> tol;
        int maxIter_APD;
        int warmStart;
        int partitionAmt, blockSize;
        
        Primitive shapeType;
        std::string inputShapePath;
        int resolution;
        double size;
        Eigen::Vector3d rotAxis;
        double rotDeg;
        
        AnimScriptType animScriptType;
        double handleRatio;

        bool orthographic;
        double zoom;
        
        bool restart;
        std::string statusPath;
        bool disableCout;
        
        std::string appendStr;
        
        std::vector<double> tuning; // the parameter that is currently tuning
        
    public:
        static const std::vector<std::string> energyTypeStrs;
        static const std::vector<std::string> timeIntegrationTypeStrs;
        static const std::vector<std::string> timeStepperTypeStrs;
        static const std::vector<std::string> shapeTypeStrs;
        
    public:
        Config(void);
        ~Config(void);
        int loadFromFile(const std::string& filePath);
        void saveToFile(const std::string& filePath);
        
    public:
        void appendInfoStr(std::string& inputStr) const;
        
    public:
        static EnergyType getEnergyTypeByStr(const std::string& str);
        static std::string getStrByEnergyType(EnergyType energyType);
        static TimeIntegrationType getTimeIntegrationTypeByStr(const std::string& str);
        static std::string getStrByTimeIntegrationType(TimeIntegrationType timeIntegrationType);
        static TimeStepperType getTimeStepperTypeByStr(const std::string& str);
        static std::string getStrByTimeStepperType(TimeStepperType timeStepperType);
        static Primitive getShapeTypeByStr(const std::string& str);
        static std::string getStrByShapeType(Primitive shapeType);
    };
    
}

#endif /* Config_hpp */
