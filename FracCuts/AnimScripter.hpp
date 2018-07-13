//
//  AnimScripter.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef AnimScripter_hpp
#define AnimScripter_hpp

#include "TriangleSoup.hpp"

#include <cstdio>
#include <map>

namespace FracCuts {
    
    enum AnimScriptType {
        AST_NULL,
        AST_HANG,
        AST_STRETCH,
        AST_SQUASH,
        AST_BEND,
        AST_ONEPOINT,
        AST_RANDOM
    };
    
    class AnimScripter
    {
    protected:
        AnimScriptType animScriptType;
        
        std::vector<std::vector<int>> handleVerts;
        
        std::map<int, Eigen::Vector2d> velocity_handleVerts;
        
        std::map<int, double> angVel_handleVerts;
        std::map<int, Eigen::Vector2d> rotCenter_handleVerts;
        
    protected:
        static const std::vector<std::string> animScriptTypeStrs;
        
    public:
        AnimScripter(AnimScriptType p_animScriptType = AST_NULL);
        
    public:
        void initAnimScript(TriangleSoup& mesh);
        void stepAnimScript(TriangleSoup& mesh, double dt);
        
    public:
        void setAnimScriptType(AnimScriptType p_animScriptType);
        
    public:
        static AnimScriptType getAnimScriptTypeByStr(const std::string& str);
        static std::string getStrByAnimScriptType(AnimScriptType animScriptType);
    };
    
}

#endif /* AnimScripter_hpp */
