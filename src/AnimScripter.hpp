//
//  AnimScripter.hpp
//  DOT
//
//  Created by Minchen Li on 6/20/18.
//

#ifndef AnimScripter_hpp
#define AnimScripter_hpp

#include "Energy.hpp"
#include "Mesh.hpp"

#include <cstdio>
#include <map>

namespace DOT {
    
    enum AnimScriptType {
        AST_NULL,
        AST_SCALEF,
        AST_HANG,
        AST_STRETCH,
        AST_SQUASH,
        AST_STRETCHNSQUASH,
        AST_BEND,
        AST_TWIST,
        AST_TWISTNSTRETCH,
        AST_TWISTNSNS,
        AST_TWISTNSNS_OLD,
        AST_RUBBERBANDPULL,
        AST_ONEPOINT,
        AST_RANDOM,
        AST_FALL,
    };
    
    template<int dim>
    class AnimScripter
    {
    protected:
        AnimScriptType animScriptType;
        
        std::vector<std::vector<int>> handleVerts;
        
        std::map<int, Eigen::Matrix<double, dim, 1>> velocity_handleVerts;
        std::pair<int, Eigen::Matrix<double, dim, 2>> velocityTurningPoints;
        
        std::map<int, double> angVel_handleVerts;
        std::map<int, Eigen::Matrix<double, dim, 1>> rotCenter_handleVerts;
        
    protected:
        static const std::vector<std::string> animScriptTypeStrs;
        
    public:
        AnimScripter(AnimScriptType p_animScriptType = AST_NULL);
        
    public:
        void initAnimScript(Mesh<dim>& mesh);
        int stepAnimScript(Mesh<dim>& mesh, double dt,
                           const std::vector<Energy<dim>*>& energyTerms);
        
    public:
        void setAnimScriptType(AnimScriptType p_animScriptType);
        //TODO: set velocity according to shape size
        
        const std::vector<std::vector<int>>& getHandleVerts(void) const;
        
    public:
        static AnimScriptType getAnimScriptTypeByStr(const std::string& str);
        static std::string getStrByAnimScriptType(AnimScriptType animScriptType);
    };
    
}

#endif /* AnimScripter_hpp */
