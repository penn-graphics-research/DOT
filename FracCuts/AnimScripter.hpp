//
//  AnimScripter.hpp
//  FracCuts
//
//  Created by Minchen Li on 6/20/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef AnimScripter_hpp
#define AnimScripter_hpp

#include "Energy.hpp"
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
        AST_RANDOM,
        AST_FALL
    };
    
    template<int dim>
    class AnimScripter
    {
    protected:
        AnimScriptType animScriptType;
        
        std::vector<std::vector<int>> handleVerts;
        
        std::map<int, Eigen::Matrix<double, dim, 1>> velocity_handleVerts;
        
        std::map<int, double> angVel_handleVerts;
        std::map<int, Eigen::Matrix<double, dim, 1>> rotCenter_handleVerts;
        
    protected:
        static const std::vector<std::string> animScriptTypeStrs;
        
    public:
        AnimScripter(AnimScriptType p_animScriptType = AST_NULL);
        
    public:
        void initAnimScript(TriangleSoup<dim>& mesh);
        void stepAnimScript(TriangleSoup<dim>& mesh, double dt,
                            const std::vector<Energy<dim>*>& energyTerms);
        
    public:
        void setAnimScriptType(AnimScriptType p_animScriptType);
        //TODO: set velocity according to shape size
        
    public:
        static AnimScriptType getAnimScriptTypeByStr(const std::string& str);
        static std::string getStrByAnimScriptType(AnimScriptType animScriptType);
    };
    
}

#endif /* AnimScripter_hpp */
